"""
lower.py - IR -> AArch64. The second, independent implementation of the IR.

Register allocation is a deterministic function of program STRUCTURE, not a
random draw. That distinction is the whole point:

  * Randomised registers (the original generator) made the choice unpredictable
    from the question -- pure noise in the training target.
  * A fixed register everywhere (the step-4 fix) removed the noise but made the
    store block so uniform that the model stopped tracking position, and
    t_stack_array_index collapsed to emitting one constant offset.

Allocation threads an explicit next-free index. The invariant is that a node
lowered with destination `dst` and free index `f` may only touch w{f}..w9, and
`dst` is always either w0 or a register below f -- so evaluating the right
operand can never clobber the left one already sitting in dst.

An earlier version derived the register from tree depth instead. That collides:
lowering `and(251, w1)` into w1 allocated its right operand at the same depth,
emitting `mov w1,#251 / mov w1,w11 / and w1,w1,w1` and silently losing the left
operand. Parity testing missed it because no original template ever nested an
expression inside a comparison; the compositional grammar hit it immediately.

Conventions:
  w0            result
  w0..w2        incoming arguments
  w10..w12      arguments copied at entry, so temporaries cannot clobber them
  w1..w9        expression temporaries, allocated by depth
  sp            arrays, 16-byte aligned
"""

from ir import (Arg, Arr, ArrLoop, Bin, Cmp, Const, InArr, InLen, Index,
                Loop, Program, Sel, Un)

# ABI. With an input array the caller passes pointer in x0 and length in w1,
# so scalar arguments start at w2 and are parked above the array registers.
ARG_BASE  = 10          # scalar args parked here when there is no input array
ARR_PTR   = 'x10'       # parked array pointer
ARR_LEN   = 'w11'       # parked array length
ARR_ARG_BASE = 12       # scalar args parked here when there IS an input array
TMP_LO    = 1           # expression temporaries live in w1..w9
TMP_HI    = 9

_SETCC = {'eq': 'eq', 'ne': 'ne', 'lt': 'lt', 'le': 'le', 'gt': 'gt', 'ge': 'ge'}
# Branch taken when the comparison is FALSE, used to skip the taken arm.
_INVBR = {'eq': 'b.ne', 'ne': 'b.eq', 'lt': 'b.ge', 'le': 'b.gt',
          'gt': 'b.le', 'ge': 'b.lt'}


class Ctx:
    """Emission context: instruction list, label counter, stack reservation."""

    def __init__(self):
        self.out = []
        self.n_label = 0
        self.stack = 0
        self.arg_base = ARG_BASE

    def emit(self, line):
        self.out.append(line)

    def label(self, stem):
        self.n_label += 1
        return f'{stem}{self.n_label}'


def _mov_imm(ctx, dst, v):
    """Materialise a 32-bit constant. movz gives 16 bits; movk adds the rest."""
    v &= 0xFFFFFFFF
    lo, hi = v & 0xFFFF, (v >> 16) & 0xFFFF
    ctx.emit(f'mov {dst}, #{lo}')
    if hi:
        ctx.emit(f'movk {dst}, #{hi}, lsl #16')


def _tmp(i):
    if not TMP_LO <= i <= TMP_HI:
        raise ValueError(f'out of temporaries (w{i}); expression too deep')
    return f'w{i}'


def _map_inline(ctx, reg, mapf, k, free):
    """Apply the map axis to a value already in `reg`."""
    if mapf is None:
        return
    if mapf == 'square':
        ctx.emit(f'mul {reg}, {reg}, {reg}')
    elif mapf == 'double':
        ctx.emit(f'add {reg}, {reg}, {reg}')
    elif mapf == 'addk':
        t = _tmp(free)
        _mov_imm(ctx, t, k)
        ctx.emit(f'add {reg}, {reg}, {t}')
    elif mapf == 'mulk':
        t = _tmp(free)
        _mov_imm(ctx, t, k)
        ctx.emit(f'mul {reg}, {reg}, {t}')
    elif mapf == 'constk':
        _mov_imm(ctx, reg, k)
    else:
        raise ValueError(f'unknown map {mapf}')


def _pred_skip(ctx, reg, pred, skip, free):
    """Emit a branch to `skip` when `reg` fails the predicate."""
    if pred is None:
        return
    kind, k = pred
    t = _tmp(free)
    if kind in ('even', 'odd'):
        ctx.emit(f'and {t}, {reg}, #1')
        ctx.emit(f'cmp {t}, #{0 if kind == "even" else 1}')
        ctx.emit(f'b.ne {skip}')
    elif kind in ('gt', 'lt'):
        _mov_imm(ctx, t, k)
        ctx.emit(f'cmp {reg}, {t}')
        ctx.emit(f'{"b.le" if kind == "gt" else "b.ge"} {skip}')
    elif kind == 'divk':
        # r = reg - (reg / k) * k, without a divide: repeated subtraction is
        # out of scope, so divk is restricted to powers of two at generation.
        mask = k - 1
        ctx.emit(f'and {t}, {reg}, #{mask}')
        ctx.emit(f'cmp {t}, #0')
        ctx.emit(f'b.ne {skip}')
    else:
        raise ValueError(f'unknown predicate {kind}')


def lower_expr(ctx, node, dst, free=TMP_LO):
    """Emit code leaving `node`'s value in `dst`."""
    if isinstance(node, Const):
        _mov_imm(ctx, dst, node.v)
        return

    if isinstance(node, Arg):
        ctx.emit(f'mov {dst}, w{ctx.arg_base + node.i}')
        return

    if isinstance(node, InLen):
        ctx.emit(f'mov {dst}, {ARR_LEN}')
        return

    if isinstance(node, Bin):
        lower_expr(ctx, node.a, dst, free)
        rhs = _tmp(free)
        lower_expr(ctx, node.b, rhs, free + 1)
        if node.op in ('lsl', 'lsr') and isinstance(node.b, Const):
            ctx.emit(f'{node.op} {dst}, {dst}, #{node.b.v & 31}')
        else:
            ctx.emit(f'{node.op} {dst}, {dst}, {rhs}')
        return

    if isinstance(node, Un):
        lower_expr(ctx, node.a, dst, free)
        if node.op == 'neg':
            ctx.emit(f'neg {dst}, {dst}')
        else:
            end = ctx.label('abs')
            ctx.emit(f'cmp {dst}, #0')
            ctx.emit(f'b.ge {end}')
            ctx.emit(f'neg {dst}, {dst}')
            ctx.emit(f'{end}:')
        return

    if isinstance(node, Cmp):
        lower_expr(ctx, node.a, dst, free)
        rhs = _tmp(free)
        lower_expr(ctx, node.b, rhs, free + 1)
        ctx.emit(f'cmp {dst}, {rhs}')
        ctx.emit(f'cset {dst}, {_SETCC[node.op]}')
        return

    if isinstance(node, Sel):
        c = node.c
        lhs = _tmp(free)
        lower_expr(ctx, c.a, lhs, free + 1)
        rhs = _tmp(free + 1)
        lower_expr(ctx, c.b, rhs, free + 2)
        other, end = ctx.label('else'), ctx.label('end')
        ctx.emit(f'cmp {lhs}, {rhs}')
        ctx.emit(f'{_INVBR[c.op]} {other}')
        lower_expr(ctx, node.a, dst, free)
        ctx.emit(f'b {end}')
        ctx.emit(f'{other}:')
        lower_expr(ctx, node.b, dst, free)
        ctx.emit(f'{end}:')
        return

    if isinstance(node, Loop):
        _lower_loop(ctx, node, dst, free)
        return

    if isinstance(node, Index):
        base = _materialise_array(ctx, node.arr, free)
        if isinstance(node.idx, Const):
            ctx.emit(f'ldr {dst}, [sp, #{base + node.idx.v * 4}]')
        else:
            t = _tmp(free)
            lower_expr(ctx, node.idx, t, free + 1)
            ctx.emit(f'add {t}, {t}, #{base // 4}')
            ctx.emit(f'ldr {dst}, [sp, {t}, uxtw #2]')
        return

    if isinstance(node, ArrLoop):
        _lower_arrloop(ctx, node, dst, free)
        return

    raise TypeError(f'cannot lower {node!r}')


def _init_acc(ctx, dst, op, have=None):
    _mov_imm(ctx, dst, 1 if op == 'product' else 0)
    if op in ('min', 'max') and have:
        _mov_imm(ctx, have, 0)          # nothing seen yet


def _accumulate(ctx, dst, val, op, have=None):
    if op == 'sum':
        ctx.emit(f'add {dst}, {dst}, {val}')
    elif op == 'product':
        ctx.emit(f'mul {dst}, {dst}, {val}')
    elif op == 'count':
        ctx.emit(f'add {dst}, {dst}, #1')
    elif op in ('min', 'max'):
        # First kept value seeds the accumulator; an empty selection leaves it
        # at 0, which is what the oracle defines.
        cmp_l, after = ctx.label('seed'), ctx.label('keep')
        ctx.emit(f'cmp {have}, #0')
        ctx.emit(f'b.ne {cmp_l}')
        ctx.emit(f'mov {dst}, {val}')
        _mov_imm(ctx, have, 1)
        ctx.emit(f'b {after}')
        ctx.emit(f'{cmp_l}:')
        ctx.emit(f'cmp {val}, {dst}')
        ctx.emit(f'{"b.ge" if op == "min" else "b.le"} {after}')
        ctx.emit(f'mov {dst}, {val}')
        ctx.emit(f'{after}:')
    else:
        raise ValueError(f'unknown reduce op {op}')


def _lower_loop(ctx, node, dst, free):
    if node.op == 'countdown':
        lower_expr(ctx, node.hi, dst, free)
        top, end = ctx.label('loop'), ctx.label('done')
        ctx.emit(f'{top}:')
        ctx.emit(f'cmp {dst}, #0')
        ctx.emit(f'b.eq {end}')
        ctx.emit(f'sub {dst}, {dst}, #1')
        ctx.emit(f'b {top}')
        ctx.emit(f'{end}:')
        return

    i, lim, val = _tmp(free), _tmp(free + 1), _tmp(free + 2)
    have = _tmp(free + 3) if node.op in ('min', 'max') else None
    scratch = free + (4 if have else 3)
    lower_expr(ctx, node.lo, i, scratch)
    lower_expr(ctx, node.hi, lim, scratch)
    _init_acc(ctx, dst, node.op, have)

    top, end, nxt = ctx.label('loop'), ctx.label('done'), ctx.label('skip')
    ctx.emit(f'{top}:')
    ctx.emit(f'cmp {i}, {lim}')
    ctx.emit(f'b.gt {end}')
    ctx.emit(f'mov {val}, {i}')
    _pred_skip(ctx, val, node.pred, nxt, scratch)
    _map_inline(ctx, val, node.mapf, node.k, scratch)
    _accumulate(ctx, dst, val, node.op, have)
    ctx.emit(f'{nxt}:')
    ctx.emit(f'add {i}, {i}, #1')
    ctx.emit(f'b {top}')
    ctx.emit(f'{end}:')


def _materialise_array(ctx, arr, free):
    """Store the array to the stack; returns its byte offset from sp."""
    n = len(arr.items)
    need = (n * 4 + 15) // 16 * 16
    base = ctx.stack
    ctx.stack += need
    t = _tmp(free)
    for k, item in enumerate(arr.items):
        lower_expr(ctx, item, t, free + 1)
        ctx.emit(f'str {t}, [sp, #{base + k * 4}]')
    return base


def _lower_arrloop(ctx, node, dst, free):
    i, lim, val = _tmp(free), _tmp(free + 1), _tmp(free + 2)
    have = _tmp(free + 3) if node.op in ('min', 'max') else None
    scratch = free + (4 if have else 3)

    from_input = isinstance(node.arr, InArr)
    if from_input:
        # Walk the caller's buffer: length is dynamic, so the loop bound comes
        # from a register rather than a constant. This is what makes the same
        # program work at any input size.
        _mov_imm(ctx, i, 0)
        ctx.emit(f'mov {lim}, {ARR_LEN}')
        load = f'ldr {val}, [{ARR_PTR}, {i}, uxtw #2]'
    else:
        base = _materialise_array(ctx, node.arr, free)
        n = len(node.arr.items)
        _mov_imm(ctx, i, base // 4)
        _mov_imm(ctx, lim, base // 4 + n)
        load = f'ldr {val}, [sp, {i}, uxtw #2]'
    _init_acc(ctx, dst, node.op, have)

    top, end, nxt = ctx.label('loop'), ctx.label('done'), ctx.label('skip')
    ctx.emit(f'{top}:')
    ctx.emit(f'cmp {i}, {lim}')
    ctx.emit(f'b.ge {end}')
    ctx.emit(load)
    _pred_skip(ctx, val, node.pred, nxt, scratch)
    _map_inline(ctx, val, node.mapf, node.k, scratch)
    _accumulate(ctx, dst, val, node.op, have)
    ctx.emit(f'{nxt}:')
    ctx.emit(f'add {i}, {i}, #1')
    ctx.emit(f'b {top}')
    ctx.emit(f'{end}:')


def uses_input_array(node):
    stack = [node.body if isinstance(node, Program) else node]
    while stack:
        n = stack.pop()
        if isinstance(n, (InArr, InLen)):
            return True
        for f in getattr(n, '__dataclass_fields__', {}):
            v = getattr(n, f)
            if isinstance(v, tuple):
                stack.extend(x for x in v if hasattr(x, '__dataclass_fields__'))
            elif hasattr(v, '__dataclass_fields__'):
                stack.append(v)
    return False


def lower(prog):
    """IR Program -> list of assembly lines (labels unindented, no `ret`)."""
    ctx = Ctx()
    arr = uses_input_array(prog)
    if arr:
        ctx.arg_base = ARR_ARG_BASE
        ctx.emit(f'mov {ARR_PTR}, x0')
        ctx.emit(f'mov {ARR_LEN}, w1')
        for i in range(prog.n_args):             # scalars follow ptr and length
            ctx.emit(f'mov w{ARR_ARG_BASE + i}, w{2 + i}')
    else:
        for i in range(prog.n_args):             # park args clear of temporaries
            ctx.emit(f'mov w{ARG_BASE + i}, w{i}')
    body = Ctx()
    body.arg_base = ctx.arg_base
    body.n_label = ctx.n_label
    lower_expr(body, prog.body, 'w0')

    lines = list(ctx.out)
    if body.stack:
        need = (body.stack + 15) // 16 * 16
        lines.append(f'sub sp, sp, #{need}')
        lines += body.out
        lines.append(f'add sp, sp, #{need}')
    else:
        lines += body.out
    return lines
