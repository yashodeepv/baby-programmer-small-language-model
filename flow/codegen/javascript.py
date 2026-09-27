"""javascript.py - IR2 flow -> JavaScript (ES modules, Node 18+ or a browser).

    to_javascript(flow)            one self-contained module
    flow_function(flow, name)      just the flow as an async function, for a project

Structured nested code when the flow structures cleanly (structure.py),
otherwise a state machine. Integer semantics match the simulator exactly:
results wrap to signed 32 bits (`| 0`, `Math.imul`), division truncates
toward zero, and reductions accumulate in BigInt so a sum of squares can't
lose precision before it is wrapped.
"""

import json

from flow.codegen.python import _can_fault
from flow.codegen.structure import (Action, CallStmt, Check, IfStmt, Par, Ret, Unstructured,
                                    structure)
from flow.ir2 import graph as g
from flow.ir2.expr import (All, Any, Arr, ArrLoop, Bin, Bool, Cmp, Const, Fmt, Index, Invalid, Len,
                           Loop, Not, Pluck, Ref, Sel, Str, Un)

RUNTIME = '''
export class ServiceError extends Error {}
export class Failed extends Error {}
export class CallFailed extends Failed {}
export class Fault extends Error {}

export const ok = (response = null) => ({ status: 'ok', response, error: null });
export const failed = (error, response = null) => ({ status: 'error', response, error });

/** Call a service, retrying; throws CallFailed after the last attempt. */
export async function call(io, service, op, args, step, retries = 0) {
  let last = null;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await io.services(service, op, args, step);
    } catch (e) {
      if (!(e instanceof ServiceError)) throw e;
      last = e;
    }
  }
  throw new CallFailed(`${step}: ${service}.${op} failed: ${last.message}`);
}

export const s32 = (x) => typeof x === 'bigint' ? Number(BigInt.asIntN(32, x)) : x | 0;
export const div = (a, b) => { if (b === 0) throw new Fault('division by zero'); return Math.trunc(a / b) | 0; };
export const mod = (a, b) => { if (b === 0) throw new Fault('division by zero'); return (a % b) | 0; };
export const idx = (xs, i) => {
  if (i < 0 || i >= xs.length) throw new Fault(`index ${i} is outside a list of ${xs.length}`);
  return xs[i];
};
export const fmt = (v) => typeof v === 'boolean' ? (v ? 'true' : 'false') : String(v);
export const range = (lo, hi) => { const out = []; for (let i = lo; i <= hi; i++) out.push(i); return out; };
export const reduce = (op, vals) => {
  if (op === 'count') return vals.length;
  if (op === 'sum') return s32(vals.reduce((a, v) => a + BigInt(v), 0n));
  if (op === 'product') return vals.reduce((a, v) => s32(BigInt(a) * BigInt(v)), 1);
  if (vals.length === 0) return 0;
  const big = vals.map(BigInt);
  return s32(big.reduce((a, v) => (op === 'min' ? (v < a ? v : a) : (v > a ? v : a))));
};
'''

RESERVED = {
    'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger', 'default', 'delete', 'do', 'else',
    'enum', 'export', 'extends', 'false', 'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof',
    'new', 'null', 'return', 'super', 'switch', 'this', 'throw', 'true', 'try', 'typeof', 'var', 'void',
    'while', 'with', 'yield', 'let', 'static', 'await', 'async', 'implements', 'interface', 'package',
    'private', 'protected', 'public', 'arguments', 'eval', 'undefined', 'NaN', 'Infinity',
    # helper names used inside generated flows
    'request', 'io', 'ok', 'failed', 'call', 'Failed', 'CallFailed', 'ServiceError', 'Fault',
    's32', 'div', 'mod', 'idx', 'fmt', 'range', 'reduce', 'env',
}
_CMP = {'eq': '===', 'ne': '!==', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}


def var(sid):
    return sid + '_' if sid in RESERVED else sid


def local_ref(path):
    head = 'request' if path[0] == 'input' else var(path[0])
    return head + ''.join(f'.{p}' if p not in RESERVED else f'[{json.dumps(p)}]' for p in path[1:])


def env_ref(path):
    return 'env' + ''.join(f'[{json.dumps(p)}]' for p in path)


def expr(e, ref=local_ref):
    E = lambda x: expr(x, ref)
    if isinstance(e, Const):
        return str(e.v)
    if isinstance(e, Str):
        return json.dumps(e.v)
    if isinstance(e, Bool):
        return 'true' if e.v else 'false'
    if isinstance(e, Ref):
        return ref(e.path)
    if isinstance(e, Bin):
        a, b = E(e.a), E(e.b)
        return {
            'add': f'(({a} + {b}) | 0)', 'sub': f'(({a} - {b}) | 0)', 'mul': f'Math.imul({a}, {b})',
            'and': f'({a} & {b})', 'orr': f'({a} | {b})', 'eor': f'({a} ^ {b})',
            'lsl': f'({a} << ({b} & 31))', 'lsr': f'(({a} >>> ({b} & 31)) | 0)',
            'div': f'div({a}, {b})', 'mod': f'mod({a}, {b})',
        }[e.op]
    if isinstance(e, Un):
        return f'((-{E(e.a)}) | 0)' if e.op == 'neg' else f'(Math.abs({E(e.a)}) | 0)'
    if isinstance(e, Cmp):
        return f'{E(e.a)} {_CMP[e.op]} {E(e.b)}'
    if isinstance(e, Sel):
        return f'({E(e.c)} ? {E(e.a)} : {E(e.b)})'
    if isinstance(e, Not):
        return f'!({E(e.a)})'
    if isinstance(e, All):
        return ' && '.join(f'({E(x)})' for x in e.items)
    if isinstance(e, Any):
        return ' || '.join(f'({E(x)})' for x in e.items)
    if isinstance(e, Len):
        return f'{E(e.a)}.length'
    if isinstance(e, Pluck):
        return f'{E(e.a)}.map((r) => r[{json.dumps(e.field)}])'
    if isinstance(e, Fmt):
        return '[' + ', '.join(json.dumps(p.v) if isinstance(p, Str) else f'fmt({E(p)})'
                               for p in e.parts) + "].join('')"
    if isinstance(e, Arr):
        return '[' + ', '.join(E(x) for x in e.items) + ']'
    if isinstance(e, Index):
        return f'idx({E(e.arr)}, {E(e.idx)})'
    if isinstance(e, (ArrLoop, Loop)):
        if isinstance(e, Loop):
            if e.op == 'countdown':
                return '0'
            chain = f'range({E(e.lo)}, {E(e.hi)})'
        else:
            chain = E(e.arr)
        k = e.k
        if e.pred:
            kind, kk = e.pred
            test = {'even': 'v % 2 === 0', 'odd': 'v % 2 !== 0', 'gt': f'v > {kk}', 'lt': f'v < {kk}',
                    'divk': f'{kk} !== 0 && v % {kk} === 0'}[kind]
            chain += f'.filter((v) => {test})'
        if e.mapf:
            m = {'square': 'BigInt(v) * BigInt(v)', 'double': 'BigInt(v) * 2n', 'addk': f'BigInt(v) + {k}n',
                 'mulk': f'BigInt(v) * {k}n', 'constk': f'{k}n'}[e.mapf]
            chain += f'.map((v) => {m})'
        return f'reduce({json.dumps(e.op)}, {chain})'
    if isinstance(e, Invalid):
        raise ValueError(f'expression {e.text!r} is not compiled yet ({e.error})')
    raise TypeError(f'javascript: cannot compile {e!r}')


def _record(d, ref=local_ref):
    return '{ ' + ', '.join(f'{json.dumps(k)}: {expr(v, ref)}' for k, v in d.items()) + ' }'


def _ret(s, ref=local_ref):
    resp = _record(s.response, ref) if s.response is not None else None
    if s.status == 'ok':
        return f'return ok({resp or ""});'
    err = json.dumps(f'{s.id}: {s.title or "ended with error"}')
    return f'return failed({err}, {resp});' if resp else f'return failed({err});'


# --------------------------------------------------------------------------
# Structured form
# --------------------------------------------------------------------------

def _body(stmts, pad):
    out = []
    P = lambda line: out.append(pad + line)
    for st in stmts:
        s = st.step
        if getattr(s, 'title', ''):
            P(f'// {s.title}')
        if _can_fault(s):
            P(f'io.at = {json.dumps(s.id)};')
        if isinstance(st, Action):
            if isinstance(s, g.Compute):
                P(f'{var(s.id)} = {{}};')
                for name, e in s.assign:
                    P(f'{var(s.id)}[{json.dumps(name)}] = {expr(e)};')
            elif isinstance(s, g.Persist):
                P(f'await io.store({json.dumps(s.entity)}, {json.dumps(s.op)}, {_record(s.data)});')
            else:
                P(f'await io.notify({json.dumps(s.channel)}, {expr(s.to)}, {expr(s.message)});')
        elif isinstance(st, Check):
            P(f'if (!({expr(s.rule)})) {{')
            if st.on_fail is None:
                P(f'  throw new Failed({json.dumps(s.id + ": " + (s.message or "validation failed"))});')
            else:
                out += _body(st.on_fail, pad + '  ')
            P('}')
        elif isinstance(st, CallStmt):
            line = (f'{var(s.id)} = await call(io, {json.dumps(s.service)}, {json.dumps(s.op)}, '
                    f'{_record(s.args)}, {json.dumps(s.id)}' + (f', {s.retries});' if s.retries else ');'))
            if st.on_fail is None:
                P(line)
            else:
                P('try {')
                P('  ' + line)
                P('} catch (e) {')
                P('  if (!(e instanceof CallFailed)) throw e;')
                out += _body(st.on_fail, pad + '  ')
                P('}')
        elif isinstance(st, IfStmt):
            then, orelse, neg = st.then, st.orelse, st.negate
            if not then:
                then, orelse, neg = orelse, [], not neg
            cond = expr(s.cond)
            P(f'if (!({cond})) {{' if neg else f'if ({cond}) {{')
            out += _body(then, pad + '  ')
            if orelse:
                P('} else {')
                out += _body(orelse, pad + '  ')
            P('}')
        elif isinstance(st, Par):
            P('// in parallel (run one after another here, in this order)')
            for branch in st.branches:
                out += _body(branch, pad)
        elif isinstance(st, Ret):
            P(_ret(s))
    return out


def _declared(stmts, acc):
    for st in stmts:
        s = st.step
        if isinstance(s, (g.Compute, g.Call)):
            acc.append(var(s.id))
        for sub in ('then', 'orelse', 'on_fail'):
            _declared(getattr(st, sub, None) or [], acc)
        for b in getattr(st, 'branches', []):
            _declared(b, acc)
    return acc


def flow_function(flow, name=None, export=False):
    """The flow as `async function <name>(request, io)`, structured. Raises Unstructured."""
    body = structure(flow)
    name = name or var(flow.name)
    lines = [f'/** {flow.intent or f"The {flow.name} flow."} */',
             f'{"export " if export else ""}async function {name}(request, io) {{']
    names = _declared(body, [])
    if names:
        lines.append(f'  let {", ".join(dict.fromkeys(names))};')
    lines += _body(body, '  ')
    lines.append('}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# State-machine form (fallback)
# --------------------------------------------------------------------------

def _machine(flow, name, export=False):
    parts = []
    for s in flow.steps:
        nxt = lambda on='next', s=s: json.dumps(flow.next(s.id, on))
        b = []
        if _can_fault(s):
            b.append(f'io.at = {json.dumps(s.id)};')
        if isinstance(s, (g.Trigger, g.Join)):
            b.append(f'return {nxt()};')
        elif isinstance(s, g.Compute):
            b.append(f'env[{json.dumps(s.id)}] = {{}};')
            b += [f'env[{json.dumps(s.id)}][{json.dumps(n)}] = {expr(e, env_ref)};' for n, e in s.assign]
            b.append(f'return {nxt()};')
        elif isinstance(s, g.Validate):
            b.append(f'if ({expr(s.rule, env_ref)}) return {nxt()};')
            b.append(f'return {nxt("error")};' if flow.next(s.id, 'error') else
                     f'throw new Failed({json.dumps(s.id + ": " + (s.message or "validation failed"))});')
        elif isinstance(s, g.Branch):
            b.append(f'return ({expr(s.cond, env_ref)}) ? {nxt("true")} : {nxt("false")};')
        elif isinstance(s, g.Call):
            c = (f'env[{json.dumps(s.id)}] = await call(io, {json.dumps(s.service)}, {json.dumps(s.op)}, '
                 f'{_record(s.args, env_ref)}, {json.dumps(s.id)}, {s.retries});')
            if flow.next(s.id, 'error'):
                b += ['try {', '  ' + c, '} catch (e) {', '  if (!(e instanceof CallFailed)) throw e;',
                      f'  return {nxt("error")};', '}']
            else:
                b.append(c)
            b.append(f'return {nxt()};')
        elif isinstance(s, g.Parallel):
            branches = [e.dst for e in flow.out_edges(s.id) if e.on == 'branch']
            b.append(f'for (const branch of {json.dumps(branches)}) await walk(branch, {json.dumps(s.join)}, env, io);')
            b.append(f'return {json.dumps(s.join)};')
        elif isinstance(s, g.Persist):
            b += [f'await io.store({json.dumps(s.entity)}, {json.dumps(s.op)}, {_record(s.data, env_ref)});',
                  f'return {nxt()};']
        elif isinstance(s, g.Notify):
            b += [f'await io.notify({json.dumps(s.channel)}, {expr(s.to, env_ref)}, {expr(s.message, env_ref)});',
                  f'return {nxt()};']
        elif isinstance(s, g.End):
            b.append(_ret(s, env_ref))
        parts.append(f'async function s_{s.id}(env, io) {{\n' + '\n'.join('  ' + x for x in b) + '\n}')
    table = ',\n'.join(f'  {json.dumps(s.id)}: s_{s.id}' for s in flow.steps)
    parts.append(f'const STEPS = {{\n{table},\n}};')
    parts.append('''async function walk(sid, stop, env, io) {
  while (sid !== stop) {
    const next = await STEPS[sid](env, io);
    if (next !== null && typeof next === 'object') return next;
    sid = next;
  }
  return null;
}''')
    parts.append(f'''/** {flow.intent or f"The {flow.name} flow."} (state-machine form) */
{"export " if export else ""}async function {name}(request, io) {{
  return walk({json.dumps(flow.trigger().id)}, null, {{ input: request }}, io);
}}''')
    return '\n\n'.join(parts)


def flow_code(flow, name=None, export=False, style='auto'):
    name = name or var(flow.name)
    if style in ('auto', 'structured'):
        try:
            return flow_function(flow, name, export)
        except Unstructured:
            if style == 'structured':
                raise
    return _machine(flow, name, export)


# --------------------------------------------------------------------------
# Whole module
# --------------------------------------------------------------------------

def entry(name):
    return f'''/** Run the flow once. services(service, op, args, step) resolves to the output
 *  record or throws ServiceError; store(entity, op, data); notify(channel, to, message). */
export async function run(input, services, store, notify) {{
  const io = {{ services, store, notify, at: null }};
  try {{
    return await {name}(input, io);
  }} catch (e) {{
    if (e instanceof Failed) return failed(e.message);
    if (e instanceof Fault) return failed(`${{io.at}}: ${{e.message}}`);
    throw e;
  }}
}}'''


def to_javascript(flow, style='auto'):
    head = f'// {flow.name} - generated by baby-programmer from IR2. Do not edit by hand.'
    name = var(flow.name)
    return '\n\n'.join([head, RUNTIME.strip(), flow_code(flow, name, style=style), entry(name)]) + '\n'
