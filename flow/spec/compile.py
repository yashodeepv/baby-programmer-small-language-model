"""compile.py - A Baby Spec document -> IR2 flows, or line-numbered errors.

    result = compile_spec(text)                 # prose left as intent
    result = compile_spec(text, intent=hook)    # hook(prose) -> exact expression text, or None

Structure is compiled deterministically. Each step keyword builds exactly
one IR2 construct (LANGUAGE.md §11): an `If` becomes a branch whose sides
rejoin at the next step; an `In parallel` becomes a parallel step and a
join; a stop becomes an end. The compiled flows then go through the IR2
checker, and its findings are mapped back to spec lines and rule names, so
an analyst gets one kind of error message whichever stage caught it.

Prose values are recorded in `result.intents`. Without an intent hook they
stay IR2 `Invalid` placeholders; the flow is structurally complete but
can't run until the logic model (or the hook) fills them.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from flow.check import check
from flow.check.typing import TypeErr, type_of
from flow.ir2 import graph as g
from flow.ir2.expr import Invalid, Str
from flow.ir2.types import UNKNOWN, RecordT
from flow.spec.document import SpecError, parse_document
from flow.spec.text import split_clauses, split_top, strip_period, unquote
from flow.spec.types import TypeSyntaxError, parse_fields
from flow.spec.values import Resolver, literal, value

NAME = r'[a-z][a-z0-9_]*'
STRING = r'"(?:[^"\\]|\\.)*"'
RESERVED = {'input', 'true', 'false', 'and', 'or', 'not', 'if', 'else', 'where', 'each'} \
    | {f'a{i}' for i in range(10)}
CHANNELS = {'Email': 'email', 'Slack': 'slack', 'SMS': 'sms', 'Webhook': 'webhook'}

# IR2 checker code -> the spec rule an analyst should read about
CODE_RULE = {
    'unknown-ref': 'resolves', 'unknown-field': 'resolves', 'type-mismatch': 'types',
    'parse-error': 'types', 'missing-field': 'entity-fields', 'branch-escapes': 'parallel-branch',
    'unreachable': 'reachable', 'response-ignored': 'respond-sync', 'unknown-service': 'known-service',
    'unknown-op': 'known-service', 'unknown-entity': 'known-entity', 'empty-compute': 'syntax',
}


@dataclass
class Intent:
    line: int
    flow: str
    step: str
    field: str
    text: str


@dataclass
class Compiled:
    system: str = ''
    description: str = ''
    flows: List[g.Flow] = field(default_factory=list)
    errors: List[SpecError] = field(default_factory=list)
    warnings: List[SpecError] = field(default_factory=list)
    intents: List[Intent] = field(default_factory=list)
    lines: Dict[Tuple[str, str], int] = field(default_factory=dict)     # (flow, step id) -> line

    @property
    def ok(self):
        return not self.errors


class _FlowCompiler:
    def __init__(self, out, fdoc, constants, entities, services, intent):
        self.out, self.fdoc = out, fdoc
        self.constants, self.entities, self.services = constants, entities, services
        self.intent_hook = intent
        self.steps, self.edges = [], []
        self.counts = {}
        self.reserved = set()
        self.mode = 'sync'
        self.r = Resolver(constants)
        self.placeholders = set()        # (step, field) holding an already-reported error

    # -- plumbing ---------------------------------------------------------
    def err(self, rule, line, msg):
        self.out.errors.append(SpecError(rule, line, msg))

    def new_id(self, stem):
        while True:
            self.counts[stem] = self.counts.get(stem, 0) + 1
            sid = f'{stem}{self.counts[stem]}'
            if sid not in self.reserved:
                self.reserved.add(sid)
                return sid

    def add(self, step, node_line, pending, title=''):
        step.title = title
        self.steps.append(step)
        self.out.lines[(self.fdoc.name, step.id)] = node_line
        for src, on in pending:
            self.edges.append(g.Edge(src, step.id, on))
        return step

    def val(self, raw, line, step_id, fld):
        """A value slot. An error is reported and replaced by a placeholder, so
        the rest of the spec is still checked and every error shows at once."""
        try:
            e, prose = value(raw, self.r, line, self.intent_hook)
        except SpecError as err:
            self.out.errors.append(err)
            self.placeholders.add((step_id, fld))
            return Invalid(raw.strip(), 'error')
        if prose:
            self.out.intents.append(Intent(line, self.fdoc.name, step_id, fld, raw.strip()))
        return e

    def args(self, raw, line):
        """'order_id, amount = total' -> [(key, raw value)]"""
        out, seen = [], set()
        for part in split_top(raw, ','):
            if not part:
                continue
            m = re.fullmatch(rf'({NAME})\s*(?:=\s*(.+))?', part, re.S)
            if not m:
                raise SpecError('syntax', line, f'expected "name" or "name = value", got "{part}"')
            k = m.group(1)
            if k in seen:
                raise SpecError('syntax', line, f'"{k}" is given twice')
            seen.add(k)
            out.append((k, m.group(2) if m.group(2) is not None else k))
        return out

    def error_end(self, message, responding, line):
        resp = None
        if self.mode == 'sync':
            resp = {'reason': Str(message)}
        if responding:
            if self.mode != 'sync':
                raise SpecError('respond-sync', line, '"responding with" in an async flow: nobody receives it')
            sid = self.new_id('fail')
            end = g.End(id=sid, status='error', response=resp)
            for k, raw in self.args(responding, line):
                if k == 'reason':
                    raise SpecError('syntax', line, '"reason" is set by the error message')
                end.response[k] = self.val(raw, line, sid, f'response.{k}')
            return end
        return g.End(id=self.new_id('fail'), status='error', response=resp)

    # -- header -----------------------------------------------------------
    def header(self):
        f = self.fdoc
        if f.trigger is None:
            raise SpecError('layout', f.line, f'flow {f.name} has no "Trigger:" line')
        if f.mode is None:
            raise SpecError('layout', f.line, f'flow {f.name} has no "Mode:" line')
        line, text = f.trigger
        m = (re.fullmatch(rf'(GET|POST|PUT|DELETE)\s+(/\S*)(?:\s+with\s+(.*))?', text)
             or re.fullmatch(rf'event\s+({NAME})(?:\s+with\s+(.*))?', text)
             or re.fullmatch(rf'schedule\s+({STRING})(?:\s+with\s+(.*))?', text))
        if not m:
            raise SpecError('syntax', line, 'a trigger is "<METHOD> /path with ...", '
                            '"event <name> with ..." or "schedule "<cron>" with ..."')
        try:
            if text.startswith('event'):
                source, config, fields = 'event', {'name': m.group(1)}, m.group(2)
            elif text.startswith('schedule'):
                source, config, fields = 'schedule', {'cron': unquote(m.group(1))}, m.group(2)
            else:
                source, config, fields = 'http', {'method': m.group(1), 'path': m.group(2)}, m.group(3)
            payload = RecordT(parse_fields(fields or '', self.entities))
        except TypeSyntaxError as e:
            raise SpecError('known-type', line, str(e))
        mline, mode = f.mode
        if mode not in ('sync', 'async'):
            raise SpecError('syntax', mline, 'Mode is "sync" or "async"')
        if mode == 'sync' and source != 'http':
            raise SpecError('sync-http', mline, f'a {source} trigger has no caller to wait: use Mode: async')
        self.mode = mode
        self.r.add_record('input', payload)
        trig = g.Trigger(id='start', source=source, config=config, payload=payload)
        self.reserved.add('start')
        self.add(trig, line, [])
        return [('start', 'next')]

    # -- names declared by steps -----------------------------------------
    def prescan(self, nodes, payload_fields):
        """Reserve compute and call names, so auto-ids never take them."""
        for n in nodes:
            m = re.fullmatch(rf'Compute ({NAME}):', n.text)
            c = re.match(rf'Call ({NAME})\.({NAME})(?: as ({NAME}))?\b', n.text)
            name = m.group(1) if m else (c.group(3) or c.group(2)) if c else None
            if name:
                if name in self.reserved:
                    how = ' (name the call with "as <name>")' if c and not c.group(3) else ''
                    self.err('unique-names', n.line, f'step name "{name}" is already used{how}')
                elif name in RESERVED or name in payload_fields:
                    self.err('unique-names', n.line, f'"{name}" can\'t name a step: it is reserved '
                             f'or a payload field')
                self.reserved.add(name)
            self.prescan(n.children, payload_fields)
            if n.otherwise:
                self.prescan(n.otherwise, payload_fields)

    # -- steps ------------------------------------------------------------
    def build(self, nodes, pending):
        for n in nodes:
            if not pending:
                self.err('reachable', n.line, 'no path reaches this step: the flow has already stopped')
                return []
            try:
                pending = self.step(n, pending)
            except SpecError as e:
                self.out.errors.append(e)          # skip this step, keep checking the rest
        return pending

    def step(self, n, pending):
        t = n.text
        if n.items and not t.startswith('Compute '):
            self.err('syntax', n.items[0][0], '"- " lines belong to a Compute step')
        if (n.children or n.otherwise is not None) and not (t.startswith('If ') or t == 'In parallel:'):
            self.err('syntax', n.line, 'only If and In parallel have nested steps')
        if t.startswith('Validate:'):
            return self.s_validate(n, pending)
        if re.fullmatch(rf'Compute ({NAME}):', t):
            return self.s_compute(n, pending)
        if t.startswith('Call '):
            return self.s_call(n, pending)
        if re.fullmatch(r'If (.+):', t):
            return self.s_if(n, pending)
        if t == 'In parallel:':
            return self.s_parallel(n, pending)
        m = re.fullmatch(rf'(Save|Update) ({NAME}):\s*(.+)', strip_period(t), re.S)
        if m:
            return self.s_persist(n, pending, 'create' if m.group(1) == 'Save' else 'update',
                                  m.group(2), m.group(3))
        m = re.fullmatch(rf'Delete ({NAME}) where (.+)', strip_period(t), re.S)
        if m:
            return self.s_persist(n, pending, 'delete', m.group(1), m.group(2))
        m = re.fullmatch(r'(Email|Slack|SMS|Webhook) (.+)', strip_period(t), re.S)
        if m:
            return self.s_notify(n, pending, CHANNELS[m.group(1)], m.group(2))
        if t.startswith('Respond with') or t.startswith('Stop'):
            return self.s_end(n, pending)
        raise SpecError('syntax', n.line, f'unknown step "{t[:40]}": a step starts with Validate:, '
                        'Compute, Call, If, In parallel:, Save, Update, Delete, Email, Slack, SMS, '
                        'Webhook, Respond with or Stop')

    def s_validate(self, n, pending):
        parts = split_clauses(n.text[len('Validate:'):].strip(), ['Otherwise'])
        if len(parts) != 2:
            raise SpecError('syntax', n.line, 'expected "Validate: <condition>. Otherwise ..."')
        cond, other = parts
        stop = re.fullmatch(rf'Otherwise stop with error ({STRING})(?:, responding with (.+))?', other, re.S)
        fail = re.fullmatch(rf'Otherwise fail with ({STRING})', other)
        if not (stop or fail):
            raise SpecError('syntax', n.line, 'after the condition: "Otherwise stop with error "..."" '
                            'or "Otherwise fail with "...""')
        sid = self.new_id('validate')
        message = unquote((stop or fail).group(1))
        v = self.add(g.Validate(id=sid, rule=self.val(cond, n.line, sid, 'rule'), message=message),
                     n.line, pending, n.title)
        if stop:
            end = self.error_end(message, stop.group(2), n.line)
            self.add(end, n.line, [(v.id, 'error')])
        return [(v.id, 'next')]

    def s_compute(self, n, pending):
        name = re.fullmatch(rf'Compute ({NAME}):', n.text).group(1)
        if not n.items:
            raise SpecError('syntax', n.line, 'a Compute step lists its values as "- name: value" lines')
        self.r.add_root(name, RecordT(()))
        assign, seen = [], set()
        for line, item in n.items:
            m = re.fullmatch(rf'({NAME}):\s*(.+)', item, re.S)
            if not m:
                raise SpecError('syntax', line, f'expected "- name: value", got "- {item}"')
            var = m.group(1)
            if var in seen:
                raise SpecError('unique-names', line, f'"{var}" is computed twice in {name}')
            seen.add(var)
            e = self.val(m.group(2), line, name, f'assign.{var}')
            try:
                t = UNKNOWN if isinstance(e, Invalid) else type_of(e, self.r.type_scope())
            except TypeErr:
                t = UNKNOWN                    # reported by the IR2 checker, at this line
            assign.append((var, e))
            self.r.add_field(name, var, t)
        self.add(g.Compute(id=name, assign=assign), n.line, pending, n.title)
        return [(name, 'next')]

    def s_call(self, n, pending):
        parts = split_clauses(n.text, ['Retry', 'On failure'])
        m = re.fullmatch(rf'Call ({NAME})\.({NAME})(?: as ({NAME}))?(?: with (.+))?', parts[0], re.S)
        if not m:
            raise SpecError('syntax', n.line, 'expected "Call <service>.<operation> [as <name>] with ..."')
        service, op, name = m.group(1), m.group(2), m.group(3) or m.group(2)
        sig = self.services.get(service, {}).get(op)
        if sig is None:
            raise SpecError('known-service', n.line, f'{service}.{op} is not declared under ## Services')
        given = dict(self.args(m.group(4) or '', n.line))
        params = [p for p, _ in sig.input.fields]
        for p in given:
            if p not in params:
                self.err('call-args', n.line, f'{service}.{op} has no parameter "{p}" '
                         f'(parameters: {", ".join(params) or "none"})')
        missing = [p for p in params if p not in given]
        if missing:
            self.err('call-args', n.line, f'{service}.{op} needs {", ".join(missing)}')
        retries, failure = 0, None
        for clause in parts[1:]:
            r = re.fullmatch(r'Retry (\d+) times?', clause)
            f = re.fullmatch(rf'On failure stop with error ({STRING})(?:, responding with (.+))?', clause, re.S)
            if r:
                retries = int(r.group(1))
                if retries > 3:
                    self.err('retry-range', n.line, 'Retry is between 0 and 3')
                    retries = 3
            elif f:
                failure = f
            else:
                raise SpecError('syntax', n.line, f'unexpected "{clause}": after a call, only '
                                '"Retry n times." and "On failure stop with error "..."."')
        args = {}
        for p in params:
            if p in given:
                args[p] = self.val(given[p], n.line, name, f'args.{p}')
            else:
                args[p] = Invalid('', 'error')
                self.placeholders.add((name, f'args.{p}'))
        call = self.add(g.Call(id=name, service=service, op=op, args=args, retries=retries),
                        n.line, pending, n.title)
        if failure:
            end = self.error_end(unquote(failure.group(1)), failure.group(2), n.line)
            self.add(end, n.line, [(call.id, 'error')])
        self.r.add_record(name, sig.output)
        return [(call.id, 'next')]

    def s_if(self, n, pending):
        cond = re.fullmatch(r'If (.+):', n.text, re.S).group(1)
        if not n.children:
            raise SpecError('syntax', n.line, 'an If has lettered steps under it')
        sid = self.new_id('if')
        b = self.add(g.Branch(id=sid, cond=self.val(cond, n.line, sid, 'cond')), n.line, pending, n.title)
        yes = self.build(n.children, [(b.id, 'true')])
        if n.otherwise is not None:
            if not n.otherwise:
                raise SpecError('syntax', n.otherwise_line, '"Otherwise:" has lettered steps under it')
            no = self.build(n.otherwise, [(b.id, 'false')])
        else:
            no = [(b.id, 'false')]
        return yes + no

    def s_parallel(self, n, pending):
        if len(n.children) < 2:
            raise SpecError('parallel-branch', n.line, 'In parallel has at least two lettered steps')
        pid, jid = self.new_id('parallel'), self.new_id('join')
        self.add(g.Parallel(id=pid, join=jid), n.line, pending, n.title)
        exits = []
        for c in n.children:
            t = c.text
            bad = (c.children or c.otherwise is not None or t.startswith(('Stop', 'Respond', 'If ', 'In parallel'))
                   or (t.startswith('Call ') and 'On failure' in t)
                   or (t.startswith('Validate:') and 'Otherwise stop' in t))
            if bad:
                self.err('parallel-branch', c.line, 'a parallel branch is one step that can\'t stop the flow')
                continue
            exits += self.build([c], [(pid, 'branch')])
        self.add(g.Join(id=jid), n.line, exits)
        return [(jid, 'next')]

    def s_persist(self, n, pending, op, entity, raw):
        rec = self.entities.get(entity)
        if rec is None:
            raise SpecError('known-entity', n.line, f'"{entity}" is not declared under ## Entities')
        given = dict(self.args(raw, n.line))
        names = rec.names()
        want = names[:1] if op == 'delete' else names
        extra = [k for k in given if k not in want]
        missing = [k for k in want if k not in given]
        sid = self.new_id('save')
        if extra or missing:
            what = f'the key field "{names[0]}"' if op == 'delete' else f'every field ({", ".join(names)})'
            self.err('entity-fields', n.line, f'{op} {entity} takes {what}'
                     + (f'; missing {", ".join(missing)}' if missing else '')
                     + (f'; not fields: {", ".join(extra)}' if extra else ''))
        data = {}
        for k in want:
            if k in given:
                data[k] = self.val(given[k], n.line, sid, f'data.{k}')
            else:
                data[k] = Invalid('', 'error')
                self.placeholders.add((sid, f'data.{k}'))
        self.add(g.Persist(id=sid, entity=entity, op=op, data=data), n.line, pending, n.title)
        return [(sid, 'next')]

    def s_notify(self, n, pending, channel, rest):
        parts = split_top(rest, ': ')
        if len(parts) < 2:
            raise SpecError('syntax', n.line, f'expected "{n.text.split()[0]} <to>: <message>"')
        to, msg = parts[0], ': '.join(parts[1:])
        sid = self.new_id('notify')
        self.add(g.Notify(id=sid, channel=channel, to=self.val(to, n.line, sid, 'to'),
                          message=self.val(msg, n.line, sid, 'message')), n.line, pending, n.title)
        return [(sid, 'next')]

    def s_end(self, n, pending):
        t = strip_period(n.text)
        if t == 'Stop':
            self.add(g.End(id=self.new_id('stop'), status='ok'), n.line, pending, n.title)
            return []
        m = re.fullmatch(rf'Stop with error ({STRING})(?:, responding with (.+))?', t, re.S)
        if m:
            end = self.error_end(unquote(m.group(1)), m.group(2), n.line)
            self.add(end, n.line, pending, n.title)
            return []
        m = re.fullmatch(r'Respond with (.+)', t, re.S)
        if m:
            if self.mode != 'sync':
                raise SpecError('respond-sync', n.line, 'Respond with in an async flow: nobody receives it')
            sid = self.new_id('done')
            resp = {k: self.val(raw, n.line, sid, f'response.{k}') for k, raw in self.args(m.group(1), n.line)}
            self.add(g.End(id=sid, status='ok', response=resp), n.line, pending, n.title)
            return []
        raise SpecError('syntax', n.line, 'expected "Stop.", "Stop with error "...".", or "Respond with ..."')

    # -- the flow ---------------------------------------------------------
    def compile(self):
        pending = self.header()
        self.prescan(self.fdoc.steps, set(self.r.roots['input'].names()))
        if not self.fdoc.steps:
            raise SpecError('layout', self.fdoc.line, f'flow {self.fdoc.name} has no steps')
        pending = self.build(self.fdoc.steps, pending)
        if pending:
            last = self.fdoc.steps[-1].line
            self.add(g.End(id=self.new_id('stop'), status='ok'), last, pending)
        return g.Flow(name=self.fdoc.name, mode=self.mode, steps=self.steps, edges=self.edges,
                      services={s: dict(ops) for s, ops in self.services.items()},
                      entities=dict(self.entities), intent=self.fdoc.intent)


def _declarations(doc, out):
    constants, entities, services = {}, {}, {}
    for line, text in doc.constants:
        m = re.fullmatch(r'([A-Z][A-Z0-9_]*)\s*=\s*(.+)', text)
        if not m:
            out.errors.append(SpecError('syntax', line, 'expected "- NAME = value" (NAME in capitals)'))
            continue
        if m.group(1) in constants:
            out.errors.append(SpecError('unique-names', line, f'constant {m.group(1)} is declared twice'))
            continue
        try:
            constants[m.group(1)] = literal(m.group(2), line)
        except SpecError as e:
            out.errors.append(e)
    for line, text in doc.entities:
        m = re.fullmatch(rf'({NAME}):\s*(.+)', text)
        if not m:
            out.errors.append(SpecError('syntax', line, 'expected "- <entity>: <field> <type>, ..."'))
            continue
        if m.group(1) in entities:
            out.errors.append(SpecError('unique-names', line, f'entity {m.group(1)} is declared twice'))
            continue
        try:
            entities[m.group(1)] = RecordT(parse_fields(m.group(2), entities))
        except TypeSyntaxError as e:
            out.errors.append(SpecError('known-type', line, str(e)))
    for line, text in doc.services:
        m = re.fullmatch(rf'({NAME})\.({NAME})\((.*?)\)\s*->\s*\((.*)\)', text)
        if not m:
            out.errors.append(SpecError('syntax', line, 'expected "- service.op(param type, ...) -> (field type, ...)"'))
            continue
        svc, op = m.group(1), m.group(2)
        if op in services.get(svc, {}):
            out.errors.append(SpecError('unique-names', line, f'{svc}.{op} is declared twice'))
            continue
        try:
            sig = g.Signature(RecordT(parse_fields(m.group(3), entities)),
                              RecordT(parse_fields(m.group(4), entities)))
            services.setdefault(svc, {})[op] = sig
        except TypeSyntaxError as e:
            out.errors.append(SpecError('known-type', line, str(e)))
    return constants, entities, services


def compile_spec(text, intent=None):
    out = Compiled()
    doc, errors = parse_document(text)
    out.errors.extend(errors)
    out.system, out.description = doc.system, doc.description
    if not doc.system:
        return out
    constants, entities, services = _declarations(doc, out)
    names = set()
    for fdoc in doc.flows:
        if fdoc.name in names:
            out.errors.append(SpecError('unique-names', fdoc.line, f'flow {fdoc.name} is declared twice'))
            continue
        names.add(fdoc.name)
        fc = _FlowCompiler(out, fdoc, constants, entities, services, intent)
        try:
            flow = fc.compile()
        except SpecError as e:
            out.errors.append(e)
            continue
        out.flows.append(flow)
        _check_ir2(flow, out, fc.placeholders)
    if not doc.flows:
        out.errors.append(SpecError('layout', doc.system_line, 'a system has at least one "## Flow:"'))
    seen, unique = set(), []
    for e in out.errors:                       # the same fault can surface twice; say it once
        key = (e.line, e.rule, e.message)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    out.errors = unique
    return out


def _check_ir2(flow, out, placeholders):
    """Run the IR2 checker and report its findings against spec lines."""
    quiet = placeholders | {(i.step, i.field) for i in out.intents if i.flow == flow.name}
    rep = check(flow)
    for d in rep.items:
        if d.code == 'parse-error' and (d.step, d.field) in quiet:
            continue                           # intent awaiting the logic model, or already reported
        line = out.lines.get((flow.name, d.step), 0)
        where = f'{d.field}: ' if d.field else ''
        e = SpecError(CODE_RULE.get(d.code, 'check'), line, f'{where}{d.message}')
        (out.errors if d.level == 'error' else out.warnings).append(e)
