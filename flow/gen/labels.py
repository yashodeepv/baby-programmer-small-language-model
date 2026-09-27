"""labels.py - Training targets for both models, read off a rendered flow.

Flow encoder (M1), one example per flow:

    text      the request
    mode      sync | async | abstain -- abstain when the text does not say and
              nothing in it implies it; the model is taught to ask
    slots     one per step in canonical order:
                kind      n-way over the step catalog
                settings  n-way values: source, method, retries, op, channel, status
                copy      {name: [start, end]} -- characters to copy verbatim
                intent    [start, end] of the step's clause
                exprs     {field: [start, end]} -- what the logic model reads
    edges     [src slot, dst slot, label]

Logic model (M2), one example per expression:

    intent    the English for the expression, as it appears in the request
    scope     the names visible at that point, with types
    target    the IR2 s-expression

plus an anonymised copy of all three. Step ids become v1.., field names
f1.., numbers n1.., strings s1.. -- consistently across intent, scope and
target, with the mapping kept to undo it. The model then learns structure,
not a vocabulary of business names, and the closed tokenizer still covers
everything it sees.
"""

import dataclasses
import re

from flow.check.dataflow import check_dataflow
from flow.check.diagnostics import Report
from flow.check.typing import type_of
from flow.gen.english import say
from flow.gen.order import canonical_order
from flow.gen.render import Namer
from flow.ir2 import graph as g
from flow.ir2.expr import ArrLoop, Bool, Const, Loop, Pluck, Ref, Str, children
from flow.ir2.sexpr import emit
from flow.ir2.types import ListT, Prim, RecordT


# --------------------------------------------------------------------------
# M1
# --------------------------------------------------------------------------

def settings(s):
    if isinstance(s, g.Trigger):
        out = {'source': s.source}
        if s.source == 'http':
            out['method'] = s.config.get('method')
        return out
    if isinstance(s, g.Call):
        return {'retries': s.retries}
    if isinstance(s, g.Persist):
        return {'op': s.op}
    if isinstance(s, g.Notify):
        return {'channel': s.channel}
    if isinstance(s, g.End):
        return {'status': s.status, 'responds': s.response is not None}
    return {}


def m1_example(flow, text, spans, mode):
    order = canonical_order(flow)
    idx = {sid: i for i, sid in enumerate(order)}
    slots = []
    for sid in order:
        s = flow.step(sid)
        mine = [sp for sp in spans if sp['step'] == sid]
        clause = [sp for sp in mine if sp['role'] == 'clause']
        slots.append({
            'id': sid,
            'kind': s.kind,
            'settings': settings(s),
            'copy': {sp['role'][6:]: [sp['start'], sp['end']] for sp in mine
                     if sp['role'].startswith('value:')},
            'intent': [clause[0]['start'], clause[0]['end']] if clause else None,
            'exprs': {sp['role'][5:]: [sp['start'], sp['end']] for sp in mine
                      if sp['role'].startswith('expr:')},
        })
    edges = [[idx[e.src], idx[e.dst], e.on] for e in flow.edges]
    edges.sort()
    return {'text': text, 'mode': mode, 'slots': slots, 'edges': edges}


# --------------------------------------------------------------------------
# M2
# --------------------------------------------------------------------------

def scope_text(scope):
    """input{order_id:str items:list[{sku:str amount:int}]} pricing{total:int}"""
    def t(x):
        if isinstance(x, Prim):
            return x.name
        if isinstance(x, ListT):
            return f'list[{t(x.elem)}]'
        if isinstance(x, RecordT):
            return '{' + ' '.join(f'{n}:{t(ft)}' for n, ft in x.fields) + '}'
        raise TypeError(x)
    return ' '.join(f'{name}{t(ty)}' for name, ty in scope.items())


class Anon:
    """One consistent renaming of names, numbers and strings for one example."""

    def __init__(self, scope):
        self.v, self.f, self.n, self.s = {'input': 'input'}, {}, {}, {}

        def fields(ty):
            if isinstance(ty, ListT):
                fields(ty.elem)
            elif isinstance(ty, RecordT):
                for n, ft in ty.fields:
                    self.f.setdefault(n, f'f{len(self.f) + 1}')
                    fields(ft)

        for name, ty in scope.items():
            self.v.setdefault(name, f'v{len(self.v)}')
            fields(ty)

    def literals(self, expr):
        stack = [expr]
        while stack:
            e = stack.pop(0)
            if isinstance(e, Const):
                self.n.setdefault(e.v, f'n{len(self.n) + 1}')
            elif isinstance(e, Str):
                self.s.setdefault(e.v, f's{len(self.s) + 1}')
            if isinstance(e, (ArrLoop, Loop)):
                if e.pred and e.pred[0] not in ('even', 'odd'):
                    self.n.setdefault(e.pred[1], f'n{len(self.n) + 1}')
                if e.mapf in ('addk', 'mulk', 'constk'):
                    self.n.setdefault(e.k, f'n{len(self.n) + 1}')
            stack.extend(children(e))

    def scope(self, scope):
        def t(x):
            if isinstance(x, Prim):
                return x.name
            if isinstance(x, ListT):
                return f'list[{t(x.elem)}]'
            return '{' + ' '.join(f'{self.f[n]}:{t(ft)}' for n, ft in x.fields) + '}'
        return ' '.join(f'{self.v[name]}{t(ty)}' for name, ty in scope.items())

    def target(self, expr):
        text = emit(_rename(expr, self))
        text = re.sub(r'"(?:[^"\\]|\\.)*"', lambda m: self.s[_unquote(m.group(0))], text)
        return re.sub(r'(?<![\w.])-?\d+(?![\w.])', lambda m: self.n[int(m.group(0))], text)

    def mapping(self):
        inv = lambda d: {v: k for k, v in d.items()}
        return {'names': inv(self.v) | inv(self.f), 'numbers': inv(self.n), 'strings': inv(self.s)}


def _unquote(q):
    return q[1:-1].replace('\\"', '"').replace('\\\\', '\\')


def _rename(e, a):
    """The same expression with step ids and field names renamed."""
    if isinstance(e, Ref):
        return Ref((a.v[e.path[0]],) + tuple(a.f[n] for n in e.path[1:]))
    if isinstance(e, Pluck):
        return Pluck(_rename(e.a, a), a.f[e.field])
    if not dataclasses.is_dataclass(e):
        return e
    changes = {}
    for fld in dataclasses.fields(e):
        v = getattr(e, fld.name)
        if isinstance(v, tuple) and v and dataclasses.is_dataclass(v[0]):
            changes[fld.name] = tuple(_rename(x, a) for x in v)
        elif dataclasses.is_dataclass(v):
            changes[fld.name] = _rename(v, a)
    return dataclasses.replace(e, **changes) if changes else e


class AnonNamer(Namer):
    """Placeholder English: 'the f3', 'the f3 of v2', 'n1', 's1'."""

    def __init__(self, scope, anon):
        Namer.__init__(self, scope, 'request', None)
        self.anon = anon

    def ref(self, path):
        if len(path) == 1:
            return f'the {self.anon.v[path[0]]}'
        phrase = f'the {self.anon.f[path[-1]]}'
        if self.lasts.get(path[-1], 0) > 1:
            phrase += f' of {self.anon.v[path[0]]}'
        return phrase

    def field(self, name):
        return self.anon.f[name]

    def num(self, v):
        return self.anon.n[v]

    def text(self, v):
        return self.anon.s[v]


def _expressions(flow, scopes):
    """(step, field, expr, scope) for every expression, compute assigns in order."""
    for s in flow.steps:
        scope = scopes.get(s.id, {})
        if isinstance(s, g.Compute):
            fields = []
            for name, e in s.assign:
                inner = dict(scope)
                if fields:
                    inner[s.id] = RecordT(tuple(fields))
                yield s, f'assign.{name}', e, inner
                fields.append((name, type_of(e, inner)))
            continue
        pairs = []
        if isinstance(s, g.Validate):
            pairs = [('rule', s.rule)]
        elif isinstance(s, g.Branch):
            pairs = [('cond', s.cond)]
        elif isinstance(s, g.Call):
            pairs = [(f'args.{k}', v) for k, v in s.args.items()]
        elif isinstance(s, g.Persist):
            pairs = [(f'data.{k}', v) for k, v in s.data.items()]
        elif isinstance(s, g.Notify):
            pairs = [('to', s.to), ('message', s.message)]
        elif isinstance(s, g.End) and s.response:
            pairs = [(f'response.{k}', v) for k, v in s.response.items()]
        for field, e in pairs:
            yield s, field, e, scope


def m2_examples(flow, text, spans, style):
    scopes = check_dataflow(flow, canonical_order(flow), Report())
    where = {(sp['step'], sp['role'][5:]): sp for sp in spans if sp['role'].startswith('expr:')}
    out = []
    for s, field, e, scope in _expressions(flow, scopes):
        sp = where.get((s.id, field))
        if sp is None:
            raise ValueError(f'no rendered span for {s.id}.{field}')
        anon = Anon(scope)
        anon.literals(e)
        out.append({
            'step': s.id, 'field': field, 'kind': s.kind,
            'intent': text[sp['start']:sp['end']],
            'scope': scope_text(scope),
            'target': emit(e),
            # A bare name or literal: most call arguments, saved fields and
            # response values. Plentiful and easy -- a trainer should
            # down-weight these or the logic model mostly learns to copy.
            'trivial': isinstance(e, (Ref, Const, Str, Bool)),
            'anon': {
                'intent': say(e, style, AnonNamer(scope, anon)),
                'scope': anon.scope(scope),
                'target': anon.target(e),
                'mapping': anon.mapping(),
            },
        })
    return out


def deanonymize(target, mapping):
    """Undo `Anon.target`: placeholders back to names, numbers and strings."""
    def sub(m):
        tok = m.group(0)
        if tok[0] == 's':
            v = mapping['strings'][tok]
            return '"' + v.replace('\\', '\\\\').replace('"', '\\"') + '"'
        if tok[0] == 'n':
            return str(mapping['numbers'][tok])
        return mapping['names'][tok]
    return re.sub(r'(?<![\w"])[svnf]\d+(?![\w"])', sub, target)
