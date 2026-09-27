"""dataflow.py - Which names each step can see, and do its expressions fit?

Walks the steps in topological order. The scope entering a step is the
intersection of what every incoming edge carries -- a value produced on only
one side of a branch is not available after the paths merge -- except at a
join, where every parallel branch has run and their outputs combine.

An edge carries the scope its source step saw, plus the source's output --
but a call's error edge carries no output, because the call did not return.
"""

from flow.check.typing import TypeErr, type_of
from flow.ir2 import graph as g
from flow.ir2.types import BOOL, STR, UNKNOWN, RecordT


def _typed(rep, step, field, e, scope, want=None):
    try:
        t = type_of(e, scope)
    except TypeErr as err:
        if err.code != 'suppressed':
            rep.error(err.code, str(err), step.id, field)
        return UNKNOWN
    if want is not None and t not in (want, UNKNOWN):
        rep.error('type-mismatch', f'must be {want}, got {t}', step.id, field)
    return t


def _fields(rep, step, field, exprs, rec, scope, exact=True):
    given, wanted = set(exprs), set(rec.names())
    for k in sorted(wanted - given) if exact else ():
        rep.error('missing-field', f'no value given for {k!r}', step.id, field)
    for k in sorted(given - wanted):
        rep.error('unknown-field', f'{k!r} is not a field (fields: {", ".join(rec.names())})',
                  step.id, field)
    for k, e in exprs.items():
        if k in wanted:
            _typed(rep, step, f'{field}.{k}', e, scope, rec.field(k))


def _check_step(flow, step, scope, rep):
    """Report problems in one step. Returns the record it adds to scope, or None."""
    if isinstance(step, g.Trigger):
        return step.payload
    if isinstance(step, g.Call):
        ops = flow.services.get(step.service)
        if ops is None:
            rep.error('unknown-service', f'no service {step.service!r} is declared', step.id, 'service')
            return UNKNOWN
        sig = ops.get(step.op)
        if sig is None:
            rep.error('unknown-op', f'{step.service} has no operation {step.op!r} '
                      f'(operations: {", ".join(ops)})', step.id, 'op')
            return UNKNOWN
        _fields(rep, step, 'args', step.args, sig.input, scope)
        return sig.output
    elif isinstance(step, g.Compute):
        if not step.assign:
            rep.error('empty-compute', 'a compute step assigns at least one name', step.id)
        seen, inner, fields = set(), dict(scope), []
        for name, e in step.assign:
            if name in seen:
                rep.error('duplicate-name', f'{name!r} is assigned twice', step.id, f'assign.{name}')
            seen.add(name)
            fields.append((name, _typed(rep, step, f'assign.{name}', e, inner)))
            inner[step.id] = RecordT(tuple(fields))
        return RecordT(tuple(fields)) if fields else UNKNOWN
    elif isinstance(step, g.Validate):
        _typed(rep, step, 'rule', step.rule, scope, BOOL)
    elif isinstance(step, g.Branch):
        _typed(rep, step, 'cond', step.cond, scope, BOOL)
    elif isinstance(step, g.Persist):
        rec = flow.entities.get(step.entity)
        if rec is None:
            rep.error('unknown-entity', f'no entity {step.entity!r} is declared', step.id, 'entity')
            return None
        if step.op == 'delete':
            if not step.data:
                rep.error('missing-field', 'delete needs at least one field to match on',
                          step.id, 'data')
            _fields(rep, step, 'data', step.data, rec, scope, exact=False)
        else:
            _fields(rep, step, 'data', step.data, rec, scope)
    elif isinstance(step, g.Notify):
        _typed(rep, step, 'to', step.to, scope, STR)
        _typed(rep, step, 'message', step.message, scope, STR)
    elif isinstance(step, g.End):
        for k, e in (step.response or {}).items():
            _typed(rep, step, f'response.{k}', e, scope)
        trig = flow.trigger()
        if step.response is not None and (flow.mode == 'async' or
                                          (trig and trig.source != 'http')):
            why = 'the flow is async' if flow.mode == 'async' else f'a {trig.source} trigger'
            rep.warn('response-ignored', f'nobody receives this response: {why} has no waiting caller',
                     step.id, 'response')
    return None


def check_dataflow(flow, order, rep):
    """Adds findings to `rep`. Returns {step id: scope entering it}."""
    carried = {}        # (src, dst, on) -> scope dict
    scopes = {}
    for sid in order:
        step = flow.step(sid)
        ins = [carried[(e.src, e.dst, e.on)] for e in flow.in_edges(sid)
               if (e.src, e.dst, e.on) in carried]
        if not ins:
            scope = {}
        elif isinstance(step, g.Join):
            scope = {}
            for s in ins:
                scope.update(s)
        else:
            scope = {k: t for k, t in ins[0].items()
                     if all(s.get(k) == t for s in ins[1:])}
        scopes[sid] = scope

        out = _check_step(flow, step, scope, rep)
        name = 'input' if isinstance(step, g.Trigger) else sid
        for e in flow.out_edges(sid):
            s = dict(scope)
            if out is not None and e.on != 'error':
                s[name] = out
            carried[(e.src, e.dst, e.on)] = s
    return scopes
