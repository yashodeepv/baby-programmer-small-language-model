"""javascript.py - IR2 flow -> a self-contained ES module (Node 18+ or a browser).

Integer semantics match the simulator exactly: every result is wrapped to
signed 32 bits (`| 0`, `Math.imul`), division truncates toward zero, and
reductions accumulate in BigInt so a sum of squares can't lose precision
before it is wrapped.
"""

import json

from flow.ir2 import graph as g
from flow.ir2.expr import (All, Any, Arr, ArrLoop, Bin, Bool, Cmp, Const, Fmt, Index, Invalid, Len,
                           Loop, Not, Pluck, Ref, Sel, Str, Un)

RUNTIME = '''
export class ServiceError extends Error {}
class Fault extends Error {}
class Failed extends Error {}

const s32 = (x) => typeof x === 'bigint' ? Number(BigInt.asIntN(32, x)) : x | 0;
const div = (a, b) => { if (b === 0) throw new Fault('division by zero'); return Math.trunc(a / b) | 0; };
const mod = (a, b) => { if (b === 0) throw new Fault('division by zero'); return (a % b) | 0; };
const idx = (xs, i) => {
  if (i < 0 || i >= xs.length) throw new Fault(`index ${i} is outside a list of ${xs.length}`);
  return xs[i];
};
const fmt = (v) => typeof v === 'boolean' ? (v ? 'true' : 'false') : String(v);
const range = (lo, hi) => { const out = []; for (let i = lo; i <= hi; i++) out.push(i); return out; };
const reduce = (op, vals) => {
  if (op === 'count') return vals.length;
  if (op === 'sum') return s32(vals.reduce((a, v) => a + BigInt(v), 0n));
  if (op === 'product') return vals.reduce((a, v) => s32(BigInt(a) * BigInt(v)), 1);
  if (vals.length === 0) return 0;
  const big = vals.map(BigInt);
  return s32(big.reduce((a, v) => (op === 'min' ? (v < a ? v : a) : (v > a ? v : a))));
};

async function walk(steps, sid, stop, env, io) {
  while (sid !== stop) {
    let next;
    try {
      next = await steps[sid](env, io);
    } catch (e) {
      if (e instanceof Fault) throw new Failed(`${sid}: ${e.message}`);
      throw e;
    }
    if (next !== null && typeof next === 'object') return next;
    sid = next;
  }
  return null;
}
'''

_CMP = {'eq': '===', 'ne': '!==', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}


def expr(e):
    if isinstance(e, Const):
        return str(e.v)
    if isinstance(e, Str):
        return json.dumps(e.v)
    if isinstance(e, Bool):
        return 'true' if e.v else 'false'
    if isinstance(e, Ref):
        return 'env' + ''.join(f'[{json.dumps(p)}]' for p in e.path)
    if isinstance(e, Bin):
        a, b = expr(e.a), expr(e.b)
        return {
            'add': f'(({a} + {b}) | 0)', 'sub': f'(({a} - {b}) | 0)', 'mul': f'Math.imul({a}, {b})',
            'and': f'({a} & {b})', 'orr': f'({a} | {b})', 'eor': f'({a} ^ {b})',
            'lsl': f'({a} << ({b} & 31))', 'lsr': f'(({a} >>> ({b} & 31)) | 0)',
            'div': f'div({a}, {b})', 'mod': f'mod({a}, {b})',
        }[e.op]
    if isinstance(e, Un):
        return f'((-{expr(e.a)}) | 0)' if e.op == 'neg' else f'(Math.abs({expr(e.a)}) | 0)'
    if isinstance(e, Cmp):
        return f'({expr(e.a)} {_CMP[e.op]} {expr(e.b)})'
    if isinstance(e, Sel):
        return f'({expr(e.c)} ? {expr(e.a)} : {expr(e.b)})'
    if isinstance(e, Not):
        return f'(!{expr(e.a)})'
    if isinstance(e, All):
        return '(' + ' && '.join(expr(x) for x in e.items) + ')'
    if isinstance(e, Any):
        return '(' + ' || '.join(expr(x) for x in e.items) + ')'
    if isinstance(e, Len):
        return f'{expr(e.a)}.length'
    if isinstance(e, Pluck):
        return f'{expr(e.a)}.map((r) => r[{json.dumps(e.field)}])'
    if isinstance(e, Fmt):
        return '[' + ', '.join(f'fmt({expr(p)})' for p in e.parts) + "].join('')"
    if isinstance(e, Arr):
        return '[' + ', '.join(expr(x) for x in e.items) + ']'
    if isinstance(e, Index):
        return f'idx({expr(e.arr)}, {expr(e.idx)})'
    if isinstance(e, (ArrLoop, Loop)):
        if isinstance(e, Loop):
            if e.op == 'countdown':
                return '0'
            src = f'range({expr(e.lo)}, {expr(e.hi)})'
        else:
            src = expr(e.arr)
        k = e.k
        m = {None: 'v', 'square': 'BigInt(v) * BigInt(v)', 'double': 'BigInt(v) * 2n',
             'addk': f'BigInt(v) + {k}n', 'mulk': f'BigInt(v) * {k}n', 'constk': f'{k}n'}[e.mapf]
        chain = src
        if e.pred:
            kind, kk = e.pred
            test = {'even': 'v % 2 === 0', 'odd': 'v % 2 !== 0', 'gt': f'v > {kk}', 'lt': f'v < {kk}',
                    'divk': f'{kk} !== 0 && v % {kk} === 0'}[kind]
            chain += f'.filter((v) => {test})'
        if e.mapf:
            chain += f'.map((v) => {m})'
        return f'reduce({json.dumps(e.op)}, {chain})'
    if isinstance(e, Invalid):
        raise ValueError(f'expression {e.text!r} is not compiled yet ({e.error})')
    raise TypeError(f'javascript: cannot compile {e!r}')


def _fn(sid):
    return f's_{sid}'


def _record(d):
    return '{ ' + ', '.join(f'{json.dumps(k)}: {expr(v)}' for k, v in d.items()) + ' }'


def step(flow, s):
    nxt = lambda on='next': json.dumps(flow.next(s.id, on))
    head = f'async function {_fn(s.id)}(env, io) {{'
    body = [f'// {s.title}'] if s.title else []
    if isinstance(s, (g.Trigger, g.Join)):
        body.append(f'return {nxt()};')
    elif isinstance(s, g.Compute):
        body.append(f'env[{json.dumps(s.id)}] = {{}};')
        for name, e in s.assign:
            body.append(f'env[{json.dumps(s.id)}][{json.dumps(name)}] = {expr(e)};')
        body.append(f'return {nxt()};')
    elif isinstance(s, g.Validate):
        body.append(f'if ({expr(s.rule)}) return {nxt()};')
        if flow.next(s.id, 'error'):
            body.append(f'return {nxt("error")};')
        else:
            body.append(f'throw new Failed({json.dumps(s.id + ": " + (s.message or "validation failed"))});')
    elif isinstance(s, g.Branch):
        body.append(f'return {expr(s.cond)} ? {nxt("true")} : {nxt("false")};')
    elif isinstance(s, g.Call):
        body.append(f'const args = {_record(s.args)};')
        body.append('let last = null;')
        body.append(f'for (let attempt = 0; attempt < {s.retries + 1}; attempt++) {{')
        body.append('  try {')
        body.append(f'    env[{json.dumps(s.id)}] = await io.services({json.dumps(s.service)}, '
                    f'{json.dumps(s.op)}, args, {json.dumps(s.id)});')
        body.append(f'    return {nxt()};')
        body.append('  } catch (e) {')
        body.append('    if (!(e instanceof ServiceError)) throw e;')
        body.append('    last = e;')
        body.append('  }')
        body.append('}')
        if flow.next(s.id, 'error'):
            body.append(f'return {nxt("error")};')
        else:
            prefix = f'{s.id}: {s.service}.{s.op} failed: '
            body.append(f'throw new Failed({json.dumps(prefix)} + last.message);')
    elif isinstance(s, g.Parallel):
        branches = [e.dst for e in flow.out_edges(s.id) if e.on == 'branch']
        body.append(f'for (const branch of {json.dumps(branches)}) '
                    f'await walk(STEPS, branch, {json.dumps(s.join)}, env, io);')
        body.append(f'return {json.dumps(s.join)};')
    elif isinstance(s, g.Persist):
        body.append(f'await io.store({json.dumps(s.entity)}, {json.dumps(s.op)}, {_record(s.data)});')
        body.append(f'return {nxt()};')
    elif isinstance(s, g.Notify):
        body.append(f'await io.notify({json.dumps(s.channel)}, {expr(s.to)}, {expr(s.message)});')
        body.append(f'return {nxt()};')
    elif isinstance(s, g.End):
        resp = _record(s.response) if s.response is not None else 'null'
        err = json.dumps(f'{s.id}: {s.title or "ended with error"}') if s.status == 'error' else 'null'
        body.append(f'return {{ status: {json.dumps(s.status)}, response: {resp}, error: {err} }};')
    return '\n'.join([head] + ['  ' + b for b in body] + ['}'])


def to_javascript(flow):
    head = [f'// {flow.name} - generated by baby-programmer from IR2. Do not edit by hand.']
    if flow.intent:
        head.append(f'// {flow.intent}')
    parts = ['\n'.join(head), RUNTIME.strip()]
    parts += [step(flow, s) for s in flow.steps]
    table = ',\n'.join(f'  {json.dumps(s.id)}: {_fn(s.id)}' for s in flow.steps)
    parts.append(f'const STEPS = {{\n{table},\n}};')
    parts.append(f'''/** Run {flow.name} once. services(service, op, args, step) resolves to the output
 *  record or throws ServiceError; store(entity, op, data); notify(channel, to, message). */
export async function run(input, services, store, notify) {{
  const env = {{ input }};
  try {{
    return await walk(STEPS, {json.dumps(flow.trigger().id)}, null, env, {{ services, store, notify }});
  }} catch (e) {{
    if (e instanceof Failed) return {{ status: 'error', response: null, error: e.message }};
    throw e;
  }}
}}''')
    return '\n\n'.join(parts) + '\n'
