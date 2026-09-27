"""io.py - JSON form of a flow: what the UI loads, edits and saves.

    {
      "name": "order_checkout", "title": "...", "intent": "...", "mode": "sync",
      "services": {"payments": {"charge": {"input": {...}, "output": {...}}}},
      "entities": {"order": {"order_id": "str", "total": "int"}},
      "steps": [
        {"id": "start", "kind": "trigger", "source": "http",
         "config": {"method": "POST", "path": "/orders"}, "payload": {...}},
        {"id": "pricing", "kind": "compute",
         "assign": {"subtotal": "(sum (pluck input.items amount))"}},
        ...
      ],
      "edges": [{"from": "start", "to": "pricing"}, {"from": "a", "to": "b", "on": "error"}],
      "scenarios": {...}
    }

Expressions are IR2 s-expression strings (flow/ir2/sexpr.py). One that does
not parse loads as `Invalid` and is reported by the checker, so a flow with a
half-edited step still opens. Structural mistakes -- an unknown step kind, a
missing required field -- raise FormatError. `scenarios` belongs to the
simulator and is ignored here.
"""

import json

from flow.ir2 import graph as g
from flow.ir2.expr import Invalid
from flow.ir2.sexpr import ParseError, emit, parse
from flow.ir2.types import dump_type, parse_type


class FormatError(ValueError):
    pass


def _expr(text, where):
    if not isinstance(text, str):
        raise FormatError(f'{where}: expected an expression string, got {text!r}')
    try:
        return parse(text)
    except ParseError as e:
        return Invalid(text, str(e))


def _emit(node):
    return node.text if isinstance(node, Invalid) else emit(node)


def _need(obj, key, where):
    if key not in obj:
        raise FormatError(f'{where}: missing "{key}"')
    return obj[key]


def _step(obj, i):
    where = f'steps[{i}]'
    sid = _need(obj, 'id', where)
    kind = _need(obj, 'kind', where)
    where = f'step "{sid}"'
    if kind not in g.STEP_CLASSES:
        raise FormatError(f'{where}: unknown kind {kind!r}; expected one of {", ".join(g.KINDS)}')
    common = dict(id=sid, title=obj.get('title', ''), intent=obj.get('intent', ''))
    exprs = lambda d, name: {k: _expr(v, f'{where}.{name}.{k}') for k, v in d.items()}
    try:
        if kind == 'trigger':
            return g.Trigger(**common, source=_need(obj, 'source', where),
                             config=dict(obj.get('config', {})),
                             payload=parse_type(_need(obj, 'payload', where)))
        if kind == 'call':
            return g.Call(**common, service=_need(obj, 'service', where),
                          op=_need(obj, 'op', where),
                          args=exprs(obj.get('args', {}), 'args'),
                          retries=int(obj.get('retries', 0)))
        if kind == 'compute':
            return g.Compute(**common, assign=list(exprs(_need(obj, 'assign', where), 'assign').items()))
        if kind == 'validate':
            return g.Validate(**common, rule=_expr(_need(obj, 'rule', where), f'{where}.rule'),
                              message=obj.get('message', ''))
        if kind == 'branch':
            return g.Branch(**common, cond=_expr(_need(obj, 'cond', where), f'{where}.cond'))
        if kind == 'parallel':
            return g.Parallel(**common, join=_need(obj, 'join', where))
        if kind == 'join':
            return g.Join(**common)
        if kind == 'persist':
            return g.Persist(**common, entity=_need(obj, 'entity', where),
                             op=_need(obj, 'op', where),
                             data=exprs(_need(obj, 'data', where), 'data'))
        if kind == 'notify':
            return g.Notify(**common, channel=_need(obj, 'channel', where),
                            to=_expr(_need(obj, 'to', where), f'{where}.to'),
                            message=_expr(_need(obj, 'message', where), f'{where}.message'))
        if kind == 'end':
            resp = obj.get('response')
            return g.End(**common, status=obj.get('status', 'ok'),
                         response=None if resp is None else exprs(resp, 'response'))
    except ValueError as e:                  # a bad type in payload, a bad int
        if isinstance(e, FormatError):
            raise
        raise FormatError(f'{where}: {e}') from e
    raise AssertionError(kind)


def load_flow(obj):
    """dict (parsed JSON) -> Flow."""
    services = {}
    for sname, ops in obj.get('services', {}).items():
        services[sname] = {}
        for oname, sig in ops.items():
            where = f'services.{sname}.{oname}'
            try:
                services[sname][oname] = g.Signature(parse_type(_need(sig, 'input', where)),
                                                     parse_type(_need(sig, 'output', where)))
            except ValueError as e:
                raise FormatError(f'{where}: {e}') from e
    entities = {}
    for ename, t in obj.get('entities', {}).items():
        try:
            entities[ename] = parse_type(t)
        except ValueError as e:
            raise FormatError(f'entities.{ename}: {e}') from e
    steps = [_step(s, i) for i, s in enumerate(_need(obj, 'steps', 'flow'))]
    edges = []
    for i, e in enumerate(_need(obj, 'edges', 'flow')):
        edges.append(g.Edge(_need(e, 'from', f'edges[{i}]'), _need(e, 'to', f'edges[{i}]'),
                            e.get('on', 'next')))
    return g.Flow(name=_need(obj, 'name', 'flow'), mode=obj.get('mode', 'sync'),
                  steps=steps, edges=edges, services=services, entities=entities,
                  title=obj.get('title', ''), intent=obj.get('intent', ''))


def load(path):
    """Read a flow file. Returns (flow, raw dict) -- the raw dict keeps scenarios."""
    with open(path) as f:
        raw = json.load(f)
    return load_flow(raw), raw


def dump_step(s):
    out = {'id': s.id, 'kind': s.kind}
    if s.title:
        out['title'] = s.title
    if s.intent:
        out['intent'] = s.intent
    exprs = lambda d: {k: _emit(v) for k, v in d.items()}
    if isinstance(s, g.Trigger):
        out.update(source=s.source, config=dict(s.config), payload=dump_type(s.payload))
    elif isinstance(s, g.Call):
        out.update(service=s.service, op=s.op, args=exprs(s.args))
        if s.retries:
            out['retries'] = s.retries
    elif isinstance(s, g.Compute):
        out['assign'] = exprs(dict(s.assign))
    elif isinstance(s, g.Validate):
        out.update(rule=_emit(s.rule), message=s.message)
    elif isinstance(s, g.Branch):
        out['cond'] = _emit(s.cond)
    elif isinstance(s, g.Parallel):
        out['join'] = s.join
    elif isinstance(s, g.Persist):
        out.update(entity=s.entity, op=s.op, data=exprs(s.data))
    elif isinstance(s, g.Notify):
        out.update(channel=s.channel, to=_emit(s.to), message=_emit(s.message))
    elif isinstance(s, g.End):
        out['status'] = s.status
        if s.response is not None:
            out['response'] = exprs(s.response)
    return out


def dump_flow(flow):
    """Flow -> dict, ready for json.dump. load_flow(dump_flow(f)) == f."""
    out = {'name': flow.name}
    if flow.title:
        out['title'] = flow.title
    if flow.intent:
        out['intent'] = flow.intent
    out['mode'] = flow.mode
    if flow.services:
        out['services'] = {s: {o: {'input': dump_type(sig.input), 'output': dump_type(sig.output)}
                               for o, sig in ops.items()}
                           for s, ops in flow.services.items()}
    if flow.entities:
        out['entities'] = {e: dump_type(t) for e, t in flow.entities.items()}
    out['steps'] = [dump_step(s) for s in flow.steps]
    out['edges'] = [{'from': e.src, 'to': e.dst, **({'on': e.on} if e.on != 'next' else {})}
                    for e in flow.edges]
    return out
