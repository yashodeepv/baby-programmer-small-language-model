"""python.py - IR2 flow -> a self-contained Python module."""

from flow.ir2 import graph as g
from flow.ir2.expr import (All, Any, Arr, ArrLoop, Bin, Bool, Cmp, Const, Fmt, Index, Invalid, Len,
                           Loop, Not, Pluck, Ref, Sel, Str, Un)

RUNTIME = '''
class ServiceError(Exception):
    """Raise from `services` when a call fails."""


class _Fault(Exception):
    pass


class _Failed(Exception):
    pass


_M = 0xFFFFFFFF


def _s32(x):
    x &= _M
    return x - (1 << 32) if x & 0x80000000 else x


def _div(a, b):
    if b == 0:
        raise _Fault('division by zero')
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a, b):
    return a - _div(a, b) * b


def _idx(xs, i):
    if not 0 <= i < len(xs):
        raise _Fault(f'index {i} is outside a list of {len(xs)}')
    return xs[i]


def _fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def _reduce(op, vals):
    if op == 'sum':
        return _s32(sum(vals))
    if op == 'product':
        acc = 1
        for v in vals:
            acc = _s32(acc * v)
        return acc
    if op == 'count':
        return len(vals)
    if op == 'min':
        return _s32(min(vals)) if vals else 0
    return _s32(max(vals)) if vals else 0


def _walk(steps, sid, stop, env, io):
    while sid != stop:
        try:
            nxt = steps[sid](env, io)
        except _Fault as f:
            raise _Failed(f'{sid}: {f}')
        if isinstance(nxt, dict):
            return nxt
        sid = nxt
    return None


class _IO:
    def __init__(self, services, store, notify):
        self.services, self.store, self.notify = services, store, notify
'''

_BIN = {'add': '+', 'sub': '-', 'mul': '*'}
_BITS = {'and': '&', 'orr': '|', 'eor': '^'}
_CMP = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}


def expr(e):
    """IR2 expression -> Python expression (reading values from `env`)."""
    if isinstance(e, Const):
        return str(e.v)
    if isinstance(e, Str):
        return repr(e.v)
    if isinstance(e, Bool):
        return 'True' if e.v else 'False'
    if isinstance(e, Ref):
        return 'env' + ''.join(f'[{p!r}]' for p in e.path)
    if isinstance(e, Bin):
        a, b = expr(e.a), expr(e.b)
        if e.op in _BIN:
            return f'_s32({a} {_BIN[e.op]} {b})'
        if e.op in _BITS:
            return f'_s32(({a} & _M) {_BITS[e.op]} ({b} & _M))'
        if e.op == 'lsl':
            return f'_s32(({a} & _M) << ({b} & 31))'
        if e.op == 'lsr':
            return f'_s32(({a} & _M) >> ({b} & 31))'
        if e.op == 'div':
            return f'_s32(_div({a}, {b}))'
        if e.op == 'mod':
            return f'_s32(_mod({a}, {b}))'
    if isinstance(e, Un):
        return f'_s32(-{expr(e.a)})' if e.op == 'neg' else f'_s32(abs({expr(e.a)}))'
    if isinstance(e, Cmp):
        return f'({expr(e.a)} {_CMP[e.op]} {expr(e.b)})'
    if isinstance(e, Sel):
        return f'({expr(e.a)} if {expr(e.c)} else {expr(e.b)})'
    if isinstance(e, Not):
        return f'(not {expr(e.a)})'
    if isinstance(e, All):
        return '(' + ' and '.join(expr(x) for x in e.items) + ')'
    if isinstance(e, Any):
        return '(' + ' or '.join(expr(x) for x in e.items) + ')'
    if isinstance(e, Len):
        return f'len({expr(e.a)})'
    if isinstance(e, Pluck):
        return f'[r[{e.field!r}] for r in {expr(e.a)}]'
    if isinstance(e, Fmt):
        return "''.join([" + ', '.join(f'_fmt({expr(p)})' for p in e.parts) + '])'
    if isinstance(e, Arr):
        return '[' + ', '.join(expr(x) for x in e.items) + ']'
    if isinstance(e, Index):
        return f'_idx({expr(e.arr)}, {expr(e.idx)})'
    if isinstance(e, (ArrLoop, Loop)):
        if isinstance(e, Loop):
            if e.op == 'countdown':
                return '0'
            src = f'range({expr(e.lo)}, {expr(e.hi)} + 1)'
        else:
            src = expr(e.arr)
        m = {None: 'v', 'square': 'v * v', 'double': 'v * 2', 'addk': f'v + {e.k}',
             'mulk': f'v * {e.k}', 'constk': str(e.k)}[e.mapf]
        cond = ''
        if e.pred:
            kind, k = e.pred
            cond = {'even': ' if v % 2 == 0', 'odd': ' if v % 2 != 0', 'gt': f' if v > {k}',
                    'lt': f' if v < {k}', 'divk': f' if {k} != 0 and v % {k} == 0'}[kind]
        return f'_reduce({e.op!r}, [{m} for v in {src}{cond}])'
    if isinstance(e, Invalid):
        raise ValueError(f'expression {e.text!r} is not compiled yet ({e.error})')
    raise TypeError(f'python: cannot compile {e!r}')


def _fn(sid):
    return f's_{sid}'


def _record(d):
    return '{' + ', '.join(f'{k!r}: {expr(v)}' for k, v in d.items()) + '}'


def step(flow, s):
    nxt = lambda on='next': repr(flow.next(s.id, on))
    lines = [f'def {_fn(s.id)}(env, io):']
    if s.title:
        lines.append(f'    # {s.title}')
    body = []
    if isinstance(s, (g.Trigger, g.Join)):
        body.append(f'return {nxt()}')
    elif isinstance(s, g.Compute):
        body.append(f'env[{s.id!r}] = {{}}')
        for name, e in s.assign:
            body.append(f'env[{s.id!r}][{name!r}] = {expr(e)}')
        body.append(f'return {nxt()}')
    elif isinstance(s, g.Validate):
        body.append(f'if {expr(s.rule)}:')
        body.append(f'    return {nxt()}')
        if flow.next(s.id, 'error'):
            body.append(f'return {nxt("error")}')
        else:
            body.append(f'raise _Failed({(s.id + ": " + (s.message or "validation failed"))!r})')
    elif isinstance(s, g.Branch):
        body.append(f'return {nxt("true")} if {expr(s.cond)} else {nxt("false")}')
    elif isinstance(s, g.Call):
        body.append(f'args = {_record(s.args)}')
        body.append('last = None')
        body.append(f'for attempt in range({s.retries + 1}):')
        body.append('    try:')
        body.append(f'        env[{s.id!r}] = io.services({s.service!r}, {s.op!r}, args, step={s.id!r})')
        body.append(f'        return {nxt()}')
        body.append('    except ServiceError as e:')
        body.append('        last = e')
        if flow.next(s.id, 'error'):
            body.append(f'return {nxt("error")}')
        else:
            body.append(f"raise _Failed({s.id + ': ' + s.service + '.' + s.op + ' failed: '!r} + str(last))")
    elif isinstance(s, g.Parallel):
        branches = [e.dst for e in flow.out_edges(s.id) if e.on == 'branch']
        body.append(f'for branch in {branches!r}:')
        body.append(f'    _walk(STEPS, branch, {s.join!r}, env, io)')
        body.append(f'return {s.join!r}')
    elif isinstance(s, g.Persist):
        body.append(f'io.store({s.entity!r}, {s.op!r}, {_record(s.data)})')
        body.append(f'return {nxt()}')
    elif isinstance(s, g.Notify):
        body.append(f'io.notify({s.channel!r}, {expr(s.to)}, {expr(s.message)})')
        body.append(f'return {nxt()}')
    elif isinstance(s, g.End):
        resp = _record(s.response) if s.response is not None else 'None'
        err = repr(f'{s.id}: {s.title or "ended with error"}') if s.status == 'error' else 'None'
        body.append(f"return {{'status': {s.status!r}, 'response': {resp}, 'error': {err}}}")
    return '\n'.join(lines + ['    ' + b for b in body])


def to_python(flow):
    head = [f'"""{flow.name} - generated by baby-programmer from IR2. Do not edit by hand.']
    if flow.intent:
        head.append('')
        head.append(flow.intent)
    head.append('"""')
    parts = ['\n'.join(head), RUNTIME.strip()]
    parts += [step(flow, s) for s in flow.steps]
    table = ',\n'.join(f'    {s.id!r}: {_fn(s.id)}' for s in flow.steps)
    parts.append(f'STEPS = {{\n{table},\n}}')
    parts.append(f'''def run(input, services, store, notify):
    """Run {flow.name} once. See the module docstring of flow.codegen for the contract."""
    env = {{'input': input}}
    try:
        return _walk(STEPS, {flow.trigger().id!r}, None, env, _IO(services, store, notify))
    except _Failed as f:
        return {{'status': 'error', 'response': None, 'error': str(f)}}''')
    return '\n\n\n'.join(parts) + '\n'
