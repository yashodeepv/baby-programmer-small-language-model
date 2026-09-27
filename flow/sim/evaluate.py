"""evaluate.py - Reference semantics of IR2 expressions.

Integer arithmetic, comparisons, maps, predicates and reductions call the
arm oracle's own tables (arm/ir.py: _BIN, _CMP, _MAP, _keep, _reduce, s32),
not copies of them: an expression the assembly model writes must mean exactly
the same thing inside a flow. Only the new nodes are defined here.

`div` / `mod` truncate toward zero, as in Java and C; dividing by zero raises
FlowFault, which fails the flow like any other runtime fault.

`legacy=(args, arr)` supplies the arm register leaves (a0.., (arr), (len)) so
the whole assembly corpus can be evaluated here and compared with arm's
`evaluate`. Flows never use it; the checker rejects those leaves.
"""

from flow.arm_bridge import arm_ir
from flow.ir2.expr import (All, Any, Arg, Arr, ArrLoop, Bin, Bool, Cmp, Const,
                           Fmt, Index, InArr, InLen, Invalid, Len, Loop, Not,
                           Pluck, Ref, Sel, Str, Un)

s32 = arm_ir.s32


class FlowFault(Exception):
    """A runtime failure inside an expression (division by zero, bad index)."""


def _div(a, b):
    if b == 0:
        raise FlowFault('division by zero')
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _mod(a, b):
    return a - _div(a, b) * b


def _fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def evaluate(node, env, legacy=None):
    """`env` maps top-level names (input, step ids) to runtime values."""
    E = lambda n: evaluate(n, env, legacy)

    if isinstance(node, Const):
        return s32(node.v)
    if isinstance(node, (Str, Bool)):
        return node.v
    if isinstance(node, Ref):
        v = env[node.path[0]]
        for name in node.path[1:]:
            v = v[name]
        return v
    if isinstance(node, Invalid):
        raise FlowFault(f'unparsed expression {node.text!r}')

    if isinstance(node, Bin):
        a, b = E(node.a), E(node.b)
        if node.op == 'div':
            return s32(_div(a, b))
        if node.op == 'mod':
            return s32(_mod(a, b))
        return s32(arm_ir._BIN[node.op](a, b))
    if isinstance(node, Un):
        v = E(node.a)
        return s32(-v) if node.op == 'neg' else s32(abs(v))
    if isinstance(node, Cmp):
        return bool(arm_ir._CMP[node.op](E(node.a), E(node.b)))
    if isinstance(node, Sel):
        return E(node.a) if E(node.c) else E(node.b)
    if isinstance(node, Not):
        return not E(node.a)
    if isinstance(node, All):
        return all(E(x) for x in node.items)
    if isinstance(node, Any):
        return any(E(x) for x in node.items)
    if isinstance(node, Len):
        return len(E(node.a))
    if isinstance(node, Pluck):
        return [r[node.field] for r in E(node.a)]
    if isinstance(node, Fmt):
        return ''.join(_fmt(E(x)) for x in node.parts)

    if isinstance(node, Loop):
        if node.op == 'countdown':
            return 0
        lo, hi = E(node.lo), E(node.hi)
        vals = [arm_ir._MAP[node.mapf](i, node.k)
                for i in range(lo, hi + 1) if arm_ir._keep(node.pred, i)]
        return s32(arm_ir._reduce(node.op, vals))
    if isinstance(node, ArrLoop):
        vals = [arm_ir._MAP[node.mapf](v, node.k)
                for v in E(node.arr) if arm_ir._keep(node.pred, v)]
        return s32(arm_ir._reduce(node.op, vals))
    if isinstance(node, Arr):
        return [E(x) for x in node.items]
    if isinstance(node, Index):
        xs, i = E(node.arr), E(node.idx)
        if not 0 <= i < len(xs):
            raise FlowFault(f'index {i} is outside a list of {len(xs)}')
        return xs[i]

    if legacy is not None:
        args, arr = legacy
        if isinstance(node, Arg):
            return s32(args[node.i])
        if isinstance(node, InArr):
            return list(arr or [])
        if isinstance(node, InLen):
            return len(arr or [])
    raise TypeError(f'cannot evaluate {node!r}')
