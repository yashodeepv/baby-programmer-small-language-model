"""render.py - A flow as an English request, with every span accounted for.

The unit of training data is a sampled flow; the English is derived from it.
While writing, the renderer records which characters came from which step:

    clause            the whole fragment a step produced -- its intent
    value:<name>      text the encoder must copy verbatim: a path, an event
                      name, a service or operation, an entity, a name being
                      assigned, a response key, a validation message
    expr:<field>      the English for one expression -- what the logic model
                      turns back into IR2 (rule, cond, assign.total, args.amount,
                      to, message, data.total, response.order_id)

Those spans are the labels: nothing is annotated by hand.

Three styles, as in the arm corpus; `HELD_OUT_STYLE` is kept out of training
so robustness to rewording can be measured.

Structure is carried by wording, which is what lets the encoder recover the
edges: "stop with an error" means an end step on an error edge, "fail the
flow" means no error edge, "if ... ; otherwise ..." a branch and its arms,
"at the same time" a parallel.
"""

from contextlib import contextmanager

from flow.check.dataflow import check_dataflow
from flow.check.diagnostics import Report
from flow.check.typing import type_of
from flow.gen.english import say, words
from flow.gen.order import canonical_order
from flow.ir2 import graph as g
from flow.ir2.expr import Str
from flow.ir2.types import BOOL, RecordT

N_STYLES = 3
HELD_OUT_STYLE = 2

T = {
    'http':      ('When a {method} request arrives at {path}', 'On {method} {path}',
                  'Whenever {method} {path} is called'),
    'event':     ('When the {name} event arrives', 'On a {name} event', 'Whenever {name} happens'),
    'schedule':  ('On the schedule "{cron}"', 'Every time the cron "{cron}" fires',
                  'Following the schedule "{cron}"'),
    'then':      ('Then', 'Next,', 'After that,'),
    'validate':  ('check that {rule}; if not, {fail}', 'make sure {rule}, otherwise {fail}',
                  'require that {rule}, or else {fail}'),
    'validate_hard': ('check that {rule}, failing the flow with "{message}" otherwise',
                      'make sure {rule} (fail the whole flow with "{message}" if not)',
                      'require that {rule}, or the flow fails with "{message}"'),
    'fail_reason': ('stop with an error saying {reason}', 'stop with the error {reason}',
                    'end with an error: {reason}'),
    'fail_bare':  ('stop with an error "{message}"', 'stop with the error "{message}"',
                   'end with an error: "{message}"'),
    'fail_plain': ('stop with an error', 'stop and report an error', 'end with an error'),
    'assign':    ('work out {name} as {expr}', 'let {name} be {expr}', 'compute {name} = {expr}'),
    'assign_sep': (', and ', '; ', ', then '),
    'call':      ('call {service}.{op} with {args}', 'use {service}.{op}, passing {args}',
                  'ask {service}.{op} for a result, giving {args}'),
    'arg':       ('{param} = {expr}', '{expr} as {param}', '{param}: {expr}'),
    'retry':     (', retrying {times}', ', with {n} {retries}', ', trying again {times} if it fails'),
    'on_error':  ('; if it fails, {fail}', ', and if that fails, {fail}', '; on failure, {fail}'),
    'branch':    ('if {cond}, {yes}', 'when {cond}, {yes}', 'in case {cond}, {yes}'),
    'otherwise': ('; otherwise, {no}', ', else {no}', '; if not, {no}'),
    'stop':      ('stop', 'finish there', 'end the flow'),
    'parallel':  ('at the same time, {items}', 'in parallel, {items}', 'simultaneously {items}'),
    'par_last':  (' and ', '; ', ', and '),
    'create':    ('save a new {entity} with {data}', 'create {an_entity} record where {data}',
                  'store {an_entity}: {data}'),
    'update':    ('update the {entity} with {data}', 'change the {entity} record so that {data}',
                  'modify the {entity}: {data}'),
    'delete':    ('delete the {entity} whose {data}', 'remove the {entity} record where {data}',
                  'erase the {entity} with {data}'),
    'field':     ('{f} = {expr}', '{f} is {expr}', '{f} {expr}'),
    'email':     ('email {to} saying {msg}', 'send an email to {to}: {msg}', 'notify {to} by email with {msg}'),
    'slack':     ('post {msg} to slack channel {to}', 'send a slack message to {to}: {msg}',
                  'tell {to} on slack: {msg}'),
    'sms':       ('text {to} saying {msg}', 'send an sms to {to}: {msg}', 'message {to} by sms with {msg}'),
    'webhook':   ('call the webhook {to} with {msg}', 'post {msg} to the webhook {to}',
                  'hit the webhook {to} with {msg}'),
    'respond':   ('finally, respond with {resp}', 'reply with {resp}', 'return {resp}'),
    'resp_item': ('{k} = {expr}', '{k} set to {expr}', '{k} as {expr}'),
    'finish':    ('finish', 'that is all', 'stop there'),
    'sync':      ('The caller waits for the result.', 'Answer the caller directly.',
                  'Respond while the caller waits.'),
    'async':     ('Run it in the background.', 'The caller does not wait.', 'Handle it asynchronously.'),
}


def times(n):
    return {1: 'once', 2: 'twice'}.get(n, f'{n} times')


# --------------------------------------------------------------------------
# Writing with spans
# --------------------------------------------------------------------------

class Writer:
    def __init__(self):
        self.buf = []
        self.pos = 0
        self.spans = []

    def add(self, text):
        self.buf.append(text)
        self.pos += len(text)

    @contextmanager
    def mark(self, step, role):
        start = self.pos
        yield
        self.spans.append({'step': step, 'role': role, 'start': start, 'end': self.pos})

    def value(self, text, step, role):
        with self.mark(step, role):
            self.add(text)

    def text(self):
        return ''.join(self.buf)


class Namer:
    """Real English names for the values visible at one step."""

    def __init__(self, scope, noun, flow):
        self.scope, self.noun, self.flow = scope, noun, flow
        self.lasts = {}
        for top, t in scope.items():
            if isinstance(t, RecordT):
                for n, _ in t.fields:
                    self.lasts[n] = self.lasts.get(n, 0) + 1

    def owner(self, top):
        if top == 'input':
            return f'in the {self.noun}'
        s = self.flow.step(top)
        if isinstance(s, g.Call):
            return f'from {s.service}.{s.op}'
        return 'computed above'

    def ref(self, path):
        if len(path) == 1:
            return f'the {self.noun}' if path[0] == 'input' else f'the result of {path[0]}'
        phrase = f'the {words(path[-1])}'
        if self.lasts.get(path[-1], 0) > 1:
            phrase += ' ' + self.owner(path[0])
        return phrase

    def type(self, path):
        t = self.scope.get(path[0])
        for n in path[1:]:
            t = t.field(n) if isinstance(t, RecordT) else None
        return t

    def is_bool(self, path):
        return self.type(path) == BOOL

    def field(self, name):
        return words(name)

    def num(self, v):
        return str(v)

    def text(self, v):
        return f'"{v}"'


# --------------------------------------------------------------------------
# The renderer
# --------------------------------------------------------------------------

class _Render:
    def __init__(self, flow, style, noun, state_mode):
        self.flow, self.style, self.noun = flow, style, noun
        self.state_mode = state_mode
        self.w = Writer()
        self.done = set()
        order = canonical_order(flow)
        self.order = order
        self.scopes = check_dataflow(flow, order, Report())

    def t(self, key):
        return T[key][self.style % N_STYLES]

    def namer(self, sid, extra=None):
        scope = dict(self.scopes[sid])
        if extra:
            scope.update(extra)
        return Namer(scope, self.noun, self.flow)

    def expr(self, step, field, e, namer):
        with self.w.mark(step.id, f'expr:{field}'):
            self.w.add(say(e, self.style, namer))

    def template(self, tmpl, fill):
        """Write `tmpl`, calling fill[name]() at each {name}, plain text between."""
        i = 0
        while i < len(tmpl):
            j = tmpl.find('{', i)
            if j < 0:
                self.w.add(tmpl[i:])
                break
            self.w.add(tmpl[i:j])
            k = tmpl.index('}', j)
            fill[tmpl[j + 1:k]]()
            i = k + 1

    def items(self, parts, sep, last_sep=None):
        for i, part in enumerate(parts):
            if i:
                self.w.add(last_sep if last_sep is not None and i == len(parts) - 1 else sep)
            part()

    # -- step fragments ---------------------------------------------------
    def fail(self, end, message=None, message_of=None):
        """The phrase for an error end reached from `message_of` (validate / call)."""
        self.done.add(end.id)
        with self.w.mark(end.id, 'clause'):
            rest = dict(end.response or {})
            reason = rest.pop('reason', None)
            if reason is not None:
                def r():
                    with self.w.mark(end.id, 'expr:response.reason'):
                        if message_of is not None and isinstance(reason, Str):
                            # The validate's message is the reason's text:
                            # one span inside the quotes serves both.
                            self.w.add('"')
                            self.w.value(reason.v, message_of, 'value:message')
                            self.w.add('"')
                        else:
                            self.w.add(say(reason, self.style, self.namer(end.id)))
                self.template(self.t('fail_reason'), {'reason': r})
            elif message is not None:
                self.template(self.t('fail_bare'), {
                    'message': lambda: self.w.value(message, message_of, 'value:message')})
            else:
                self.w.add(self.t('fail_plain'))
            if rest:
                self.w.add(', responding with ')
                self.resp_items(end, rest, self.namer(end.id))

    def frag(self, sid):
        """Write one step's fragment (and any steps it says inline)."""
        s = self.flow.step(sid)
        self.done.add(sid)
        nm = self.namer(sid)
        with self.w.mark(sid, 'clause'):
            getattr(self, f'f_{s.kind}')(s, nm)

    def f_validate(self, s, nm):
        err = self.flow.next(s.id, 'error')
        rule = lambda: self.expr(s, 'rule', s.rule, nm)
        if err:
            self.template(self.t('validate'), {
                'rule': rule, 'fail': lambda: self.fail(self.flow.step(err), s.message, s.id)})
        else:
            self.template(self.t('validate_hard'), {
                'rule': rule, 'message': lambda: self.w.value(s.message, s.id, 'value:message')})

    def f_compute(self, s, nm):
        extra = {}
        parts = []
        fields = []
        for name, e in s.assign:
            def part(name=name, e=e, scope_extra=dict(extra)):
                inner = self.namer(s.id, scope_extra)
                self.template(self.t('assign'), {
                    'name': lambda: self.w.value(name, s.id, f'value:name.{name}'),
                    'expr': lambda: self._assign_expr(s, name, e, inner)})
            parts.append(part)
            fields.append((name, type_of(e, {**self.scopes[s.id], **extra})))
            extra = {s.id: RecordT(tuple(fields))}
        self.items(parts, self.t('assign_sep'))

    def _assign_expr(self, s, name, e, nm):
        if type_of(e, nm.scope) == BOOL:
            self.w.add('whether ')
        self.expr(s, f'assign.{name}', e, nm)

    def f_call(self, s, nm):
        def args():
            parts = []
            for p, e in s.args.items():
                def part(p=p, e=e):
                    self.template(self.t('arg'), {
                        'param': lambda: self.w.add(p),
                        'expr': lambda: self.expr(s, f'args.{p}', e, nm)})
                parts.append(part)
            self.items(parts, ', ')
        self.template(self.t('call'), {
            'service': lambda: self.w.value(s.service, s.id, 'value:service'),
            'op': lambda: self.w.value(s.op, s.id, 'value:op'),
            'args': args})
        if s.retries:
            self.w.add(self.t('retry').format(times=times(s.retries), n=s.retries,
                                              retries='retry' if s.retries == 1 else 'retries'))
        err = self.flow.next(s.id, 'error')
        if err:
            end = self.flow.step(err)
            self.template(self.t('on_error'), {'fail': lambda: self.fail(end)})

    def arm(self, sid):
        s = self.flow.step(sid)
        if isinstance(s, g.End) and not (s.status == 'ok' and s.response):
            if s.status == 'error':
                self.fail(s, message=None)
            else:
                self.done.add(sid)
                with self.w.mark(sid, 'clause'):
                    self.w.add(self.t('stop'))
        else:
            self.frag(sid)

    def f_branch(self, s, nm):
        yes, no = self.flow.next(s.id, 'true'), self.flow.next(s.id, 'false')
        self.template(self.t('branch'), {
            'cond': lambda: self.expr(s, 'cond', s.cond, nm),
            'yes': lambda: self.arm(yes)})
        if len(self.flow.in_edges(no)) == 1:                 # an arm, not the merge point
            self.template(self.t('otherwise'), {'no': lambda: self.arm(no)})

    def f_parallel(self, s, nm):
        branches = [e.dst for e in self.flow.out_edges(s.id) if e.on == 'branch']
        self.done.add(s.join)
        self.template(self.t('parallel'), {
            'items': lambda: self.items([lambda b=b: self.frag(b) for b in branches],
                                        ', ', self.t('par_last'))})

    def data(self, s, nm):
        parts = []
        for f, e in s.data.items():
            def part(f=f, e=e):
                self.template(self.t('field'), {
                    'f': lambda: self.w.add(f),
                    'expr': lambda: self.expr(s, f'data.{f}', e, nm)})
            parts.append(part)
        self.items(parts, ', ')

    def f_persist(self, s, nm):
        def an_entity():
            self.w.add('an ' if s.entity[0] in 'aeiou' else 'a ')
            self.w.value(s.entity, s.id, 'value:entity')
        self.template(self.t(s.op), {
            'entity': lambda: self.w.value(s.entity, s.id, 'value:entity'),
            'an_entity': an_entity,
            'data': lambda: self.data(s, nm)})

    def f_notify(self, s, nm):
        self.template(self.t(s.channel), {
            'to': lambda: self.expr(s, 'to', s.to, nm),
            'msg': lambda: self.expr(s, 'message', s.message, nm)})

    def resp_items(self, s, items, nm):
        parts = []
        for k, e in items.items():
            def part(k=k, e=e):
                self.template(self.t('resp_item'), {
                    'k': lambda: self.w.value(k, s.id, f'value:key.{k}'),
                    'expr': lambda: self.expr(s, f'response.{k}', e, nm)})
            parts.append(part)
        self.items(parts, ', ', ' and ')

    def f_end(self, s, nm):
        if s.response:
            self.template(self.t('respond'), {'resp': lambda: self.resp_items(s, s.response, nm)})
        else:
            self.w.add(self.t('finish'))

    def f_join(self, s, nm):
        pass

    # -- the request ------------------------------------------------------
    def trigger(self, trig):
        with self.w.mark(trig.id, 'clause'):
            fill = {k: (lambda k=k, v=v: self.w.value(v, trig.id, f'value:{k}'))
                    for k, v in trig.config.items()}
            self.template(self.t(trig.source), fill)

    def run(self):
        trig = self.flow.trigger()
        self.done.add(trig.id)
        self.trigger(trig)
        first = True
        for sid in self.order:
            if sid in self.done or isinstance(self.flow.step(sid), g.Join):
                continue
            if first:
                self.w.add(', ')
                first = False
            else:
                self.w.add('. ' + self.t('then') + ' ')
            self.frag(sid)
        self.w.add('.')
        if self.state_mode:
            self.w.add(' ' + self.t(self.flow.mode))
        return self.w.text(), self.w.spans


def render(flow, style=0, noun='request', state_mode=True):
    """Returns (text, spans). Spans are {step, role, start, end} over `text`."""
    return _Render(flow, style, noun, state_mode).run()
