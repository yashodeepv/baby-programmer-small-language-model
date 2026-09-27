"""python.py - IR2 flow -> Python.

    to_python(flow)                   one self-contained module
    flow_function(flow, name)         just the flow as a function, for a project

The flow is written as ordinary nested code when it structures cleanly
(structure.py), which is every flow the spec language produces; otherwise
as a small state machine, one function per step. Both forms are verified
against the simulator the same way.
"""

import keyword

from flow.codegen.structure import (Action, CallStmt, Check, IfStmt, Par, Ret, Unstructured,
                                    structure)
from flow.ir2 import graph as g
from flow.ir2.expr import (All, Any, Arr, ArrLoop, Bin, Bool, Cmp, Const, Fmt, Index, Invalid, Len,
                           Loop, Not, Pluck, Ref, Sel, Str, Un, children)

RUNTIME = '''
class ServiceError(Exception):
    """Raise from a service when a call fails."""


class Failed(Exception):
    """The flow fails with this message."""


class CallFailed(Failed):
    """A service call failed on every attempt."""


class _Fault(Exception):
    pass


_M = 0xFFFFFFFF


def ok(response=None):
    return {'status': 'ok', 'response': response, 'error': None}


def failed(error, response=None):
    return {'status': 'error', 'response': response, 'error': error}


def call(io, service, op, args, step, retries=0):
    """Call a service, retrying; raises CallFailed after the last attempt."""
    last = None
    for _ in range(retries + 1):
        try:
            return io.services(service, op, args, step=step)
        except ServiceError as e:
            last = e
    raise CallFailed(f'{step}: {service}.{op} failed: {last}')


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


class IO:
    """What a flow may touch: services, the store, notifications."""

    def __init__(self, services, store, notify):
        self.services, self.store, self.notify = services, store, notify
        self.at = None                       # the step running, for fault messages
'''

_HELPERS = {'request', 'io', 'ok', 'failed', 'call', 'Failed', 'CallFailed', 'ServiceError', 'IO', 'len',
            'abs', 'sum', 'min', 'max', 'str', 'range'}
_BIN = {'add': '+', 'sub': '-', 'mul': '*'}
_BITS = {'and': '&', 'orr': '|', 'eor': '^'}
_CMP = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}


def var(sid):
    """A step id as a Python variable name."""
    return sid + '_' if keyword.iskeyword(sid) or sid in _HELPERS else sid


def local_ref(path):
    head = 'request' if path[0] == 'input' else var(path[0])
    return head + ''.join(f'[{p!r}]' for p in path[1:])


def env_ref(path):
    return 'env' + ''.join(f'[{p!r}]' for p in path)


def expr(e, ref=local_ref):
    """IR2 expression -> Python expression. `ref` spells a value's path."""
    E = lambda x: expr(x, ref)
    if isinstance(e, Const):
        return str(e.v)
    if isinstance(e, Str):
        return repr(e.v)
    if isinstance(e, Bool):
        return 'True' if e.v else 'False'
    if isinstance(e, Ref):
        return ref(e.path)
    if isinstance(e, Bin):
        a, b = E(e.a), E(e.b)
        if e.op in _BIN:
            return f'_s32({a} {_BIN[e.op]} {b})'
        if e.op in _BITS:
            return f'_s32(({a} & _M) {_BITS[e.op]} ({b} & _M))'
        return {'lsl': f'_s32(({a} & _M) << ({b} & 31))', 'lsr': f'_s32(({a} & _M) >> ({b} & 31))',
                'div': f'_s32(_div({a}, {b}))', 'mod': f'_s32(_mod({a}, {b}))'}[e.op]
    if isinstance(e, Un):
        return f'_s32(-{E(e.a)})' if e.op == 'neg' else f'_s32(abs({E(e.a)}))'
    if isinstance(e, Cmp):
        return f'{E(e.a)} {_CMP[e.op]} {E(e.b)}'
    if isinstance(e, Sel):
        return f'({E(e.a)} if {E(e.c)} else {E(e.b)})'
    if isinstance(e, Not):
        return f'not ({E(e.a)})'
    if isinstance(e, All):
        return ' and '.join(f'({E(x)})' for x in e.items)
    if isinstance(e, Any):
        return ' or '.join(f'({E(x)})' for x in e.items)
    if isinstance(e, Len):
        return f'len({E(e.a)})'
    if isinstance(e, Pluck):
        return f'[r[{e.field!r}] for r in {E(e.a)}]'
    if isinstance(e, Fmt):
        # literal text stays a plain string; only values go through _fmt
        return "''.join([" + ', '.join(repr(p.v) if isinstance(p, Str) else f'_fmt({E(p)})'
                                       for p in e.parts) + '])'
    if isinstance(e, Arr):
        return '[' + ', '.join(E(x) for x in e.items) + ']'
    if isinstance(e, Index):
        return f'_idx({E(e.arr)}, {E(e.idx)})'
    if isinstance(e, (ArrLoop, Loop)):
        if isinstance(e, Loop):
            if e.op == 'countdown':
                return '0'
            src = f'range({E(e.lo)}, {E(e.hi)} + 1)'
        else:
            src = E(e.arr)
        m = {None: 'v', 'square': 'v * v', 'double': 'v * 2', 'addk': f'v + {e.k}',
             'mulk': f'v * {e.k}', 'constk': str(e.k)}[e.mapf]
        cond = ''
        if e.pred:
            kind, k = e.pred
            cond = {'even': ' if v % 2 == 0', 'odd': ' if v % 2 != 0', 'gt': f' if v > {k}',
                    'lt': f' if v < {k}', 'divk': f' if {k} != 0 and v % {k} == 0'}[kind]
        if m == 'v' and not cond and not isinstance(e, Loop):
            return f'_reduce({e.op!r}, {src})'
        return f'_reduce({e.op!r}, [{m} for v in {src}{cond}])'
    if isinstance(e, Invalid):
        raise ValueError(f'expression {e.text!r} is not compiled yet ({e.error})')
    raise TypeError(f'python: cannot compile {e!r}')


def _record(d, ref=local_ref):
    return '{' + ', '.join(f'{k!r}: {expr(v, ref)}' for k, v in d.items()) + '}'


def _step_exprs(s):
    if isinstance(s, g.Compute):
        return [e for _, e in s.assign]
    out = [getattr(s, a) for a in ('rule', 'cond', 'to', 'message') if hasattr(s, a)]
    for a in ('args', 'data', 'response'):
        out += list((getattr(s, a, None) or {}).values())
    return out


def _can_fault(s):
    stack = _step_exprs(s)
    while stack:
        e = stack.pop()
        if isinstance(e, Index) or (isinstance(e, Bin) and e.op in ('div', 'mod')):
            return True
        stack.extend(children(e))
    return False


# --------------------------------------------------------------------------
# Structured form
# --------------------------------------------------------------------------

def _ret(s):
    resp = _record(s.response) if s.response is not None else None
    if s.status == 'ok':
        return f'return ok({resp})' if resp else 'return ok()'
    err = repr(f'{s.id}: {s.title or "ended with error"}')
    return f'return failed({err}, {resp})' if resp else f'return failed({err})'


def _body(stmts, pad):
    out = []
    P = lambda line: out.append(pad + line)
    for st in stmts:
        s = st.step
        if getattr(s, 'title', ''):
            P(f'# {s.title}')
        if _can_fault(s):
            P(f'io.at = {s.id!r}')
        if isinstance(st, Action):
            if isinstance(s, g.Compute):
                P(f'{var(s.id)} = {{}}')
                for name, e in s.assign:
                    P(f'{var(s.id)}[{name!r}] = {expr(e)}')
            elif isinstance(s, g.Persist):
                P(f'io.store({s.entity!r}, {s.op!r}, {_record(s.data)})')
            else:
                P(f'io.notify({s.channel!r}, {expr(s.to)}, {expr(s.message)})')
        elif isinstance(st, Check):
            P(f'if not ({expr(s.rule)}):')
            if st.on_fail is None:
                P(f'    raise Failed({(s.id + ": " + (s.message or "validation failed"))!r})')
            else:
                out += _body(st.on_fail, pad + '    ')
        elif isinstance(st, CallStmt):
            line = (f'{var(s.id)} = call(io, {s.service!r}, {s.op!r}, {_record(s.args)}, '
                    f'step={s.id!r}' + (f', retries={s.retries})' if s.retries else ')'))
            if st.on_fail is None:
                P(line)
            else:
                P('try:')
                P('    ' + line)
                P('except CallFailed:')
                out += _body(st.on_fail, pad + '    ')
        elif isinstance(st, IfStmt):
            then, orelse, neg = st.then, st.orelse, st.negate
            if not then:
                then, orelse, neg = orelse, [], not neg
            cond = expr(s.cond)
            P(f'if not ({cond}):' if neg else f'if {cond}:')
            out += _body(then, pad + '    ')
            if orelse:
                P('else:')
                out += _body(orelse, pad + '    ')
        elif isinstance(st, Par):
            P('# in parallel (run one after another here, in this order)')
            for branch in st.branches:
                out += _body(branch, pad)
        elif isinstance(st, Ret):
            P(_ret(s))
    return out


def flow_function(flow, name=None):
    """The flow as `def <name>(request, io):` in structured form. Raises Unstructured."""
    body = structure(flow)
    name = name or var(flow.name)
    lines = [f'def {name}(request, io):']
    doc = flow.intent or f'The {flow.name} flow.'
    lines.append(f'    """{doc}"""')
    lines += _body(body, '    ')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# State-machine form (fallback)
# --------------------------------------------------------------------------

def _machine(flow):
    name = var(flow.name)
    parts = []
    for s in flow.steps:
        nxt = lambda on='next', s=s: repr(flow.next(s.id, on))
        b = []
        if _can_fault(s):
            b.append(f'io.at = {s.id!r}')
        if isinstance(s, (g.Trigger, g.Join)):
            b.append(f'return {nxt()}')
        elif isinstance(s, g.Compute):
            b.append(f'env[{s.id!r}] = {{}}')
            b += [f'env[{s.id!r}][{n!r}] = {expr(e, env_ref)}' for n, e in s.assign]
            b.append(f'return {nxt()}')
        elif isinstance(s, g.Validate):
            b.append(f'if {expr(s.rule, env_ref)}:')
            b.append(f'    return {nxt()}')
            b.append(f'return {nxt("error")}' if flow.next(s.id, 'error') else
                     f'raise Failed({(s.id + ": " + (s.message or "validation failed"))!r})')
        elif isinstance(s, g.Branch):
            b.append(f'return {nxt("true")} if {expr(s.cond, env_ref)} else {nxt("false")}')
        elif isinstance(s, g.Call):
            c = (f'env[{s.id!r}] = call(io, {s.service!r}, {s.op!r}, {_record(s.args, env_ref)}, '
                 f'step={s.id!r}, retries={s.retries})')
            if flow.next(s.id, 'error'):
                b += ['try:', '    ' + c, 'except CallFailed:', f'    return {nxt("error")}']
            else:
                b.append(c)
            b.append(f'return {nxt()}')
        elif isinstance(s, g.Parallel):
            branches = [e.dst for e in flow.out_edges(s.id) if e.on == 'branch']
            b += [f'for branch in {branches!r}:', f'    _walk(branch, {s.join!r}, env, io)',
                  f'return {s.join!r}']
        elif isinstance(s, g.Persist):
            b += [f'io.store({s.entity!r}, {s.op!r}, {_record(s.data, env_ref)})', f'return {nxt()}']
        elif isinstance(s, g.Notify):
            b += [f'io.notify({s.channel!r}, {expr(s.to, env_ref)}, {expr(s.message, env_ref)})',
                  f'return {nxt()}']
        elif isinstance(s, g.End):
            b.append(_ret_env(s))
        parts.append(f'def _s_{s.id}(env, io):\n' + '\n'.join('    ' + x for x in b))
    table = ',\n'.join(f'    {s.id!r}: _s_{s.id}' for s in flow.steps)
    parts.append(f'_STEPS = {{\n{table},\n}}')
    parts.append('''def _walk(sid, stop, env, io):
    while sid != stop:
        nxt = _STEPS[sid](env, io)
        if isinstance(nxt, dict):
            return nxt
        sid = nxt
    return None''')
    parts.append(f'''def {name}(request, io):
    """{flow.intent or f'The {flow.name} flow.'} (state-machine form)"""
    return _walk({flow.trigger().id!r}, None, {{'input': request}}, io)''')
    return '\n\n\n'.join(parts)


def _ret_env(s):
    resp = _record(s.response, env_ref) if s.response is not None else 'None'
    if s.status == 'ok':
        return f'return ok({resp})'
    return f'return failed({(s.id + ": " + (s.title or "ended with error"))!r}, {resp})'


# --------------------------------------------------------------------------
# Whole module
# --------------------------------------------------------------------------

def entry(name):
    return f'''def run(input, services, store, notify):
    """Run the flow once.

    services(service, op, args, step) returns the output record or raises ServiceError;
    store(entity, op, data); notify(channel, to, message).
    Returns {{'status': 'ok' | 'error', 'response': ..., 'error': ...}}.
    """
    io = IO(services, store, notify)
    try:
        return {name}(input, io)
    except Failed as e:
        return failed(str(e))
    except _Fault as e:
        return failed(f'{{io.at}}: {{e}}')'''


def to_python(flow, style='auto'):
    """One self-contained module. style: 'auto' (structured, else state machine),
    'structured' (raise if it can't be), or 'machine'."""
    code = None
    if style in ('auto', 'structured'):
        try:
            code = flow_function(flow)
        except Unstructured:
            if style == 'structured':
                raise
    if code is None:
        code = _machine(flow)
    head = f'"""{flow.name} - generated by baby-programmer from IR2. Do not edit by hand."""'
    return '\n\n\n'.join([head, RUNTIME.strip(), code, entry(var(flow.name))]) + '\n'
