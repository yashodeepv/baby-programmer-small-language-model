"""
grammar.py - Sample program STRUCTURES, not instances of 17 hand-written ones.

The corpus this replaces taught 17 shapes very well and composition not at all:
asked to sum the *even* integers, the model emitted the ordinary sum loop and
dropped the word, because no training example ever had "even" change the code.
Here each axis is an independent choice that does change the code, so the
qualifier has to be read.

    reduce(op, map(mapf, filter(pred, source)))

`shape_of` names the grammar cell a program came from, ignoring its constants.
Phase 3 splits on that name to hold out whole COMBINATIONS -- unseen
compositions of seen parts -- which is the measurement the old corpus could
not make.

Totality is a hard constraint: every program must terminate and be defined for
every argument vector, or verification is measuring the harness instead of the
model. Hence countdown keeps a positive constant bound, array indices stay
constant, and divk is restricted to powers of two (the lowering masks, because
there is no divide in the instruction subset).
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir import (Arg, Arr, ArrLoop, Bin, Cmp, Const as C, InArr, InLen,
                Index, Loop, Program, Sel, Un)

REDUCE = ['sum', 'product', 'count', 'min', 'max']
MAPS   = [None, 'square', 'double', 'addk', 'mulk', 'constk']
PREDS  = [None, 'even', 'odd', 'gt', 'lt', 'divk']
BINOPS = ['add', 'sub', 'mul', 'and', 'orr', 'eor', 'lsl', 'lsr']
CMPOPS = ['eq', 'ne', 'lt', 'le', 'gt', 'ge']
DIVK   = [2, 4, 8, 16]              # powers of two only; lowering masks


# --------------------------------------------------------------------------
# Structural identity
# --------------------------------------------------------------------------

def shape_of(n):
    """Canonical name of the grammar cell, with constants abstracted away."""
    if isinstance(n, Program):
        return shape_of(n.body)
    if isinstance(n, C):
        return 'C'
    if isinstance(n, Arg):
        return 'A'
    if isinstance(n, Bin):
        return f'{n.op}({shape_of(n.a)},{shape_of(n.b)})'
    if isinstance(n, Un):
        return f'{n.op}({shape_of(n.a)})'
    if isinstance(n, Cmp):
        return f'cmp.{n.op}({shape_of(n.a)},{shape_of(n.b)})'
    if isinstance(n, Sel):
        return f'sel({shape_of(n.c)},{shape_of(n.a)},{shape_of(n.b)})'
    if isinstance(n, Loop):
        p = n.pred[0] if n.pred else '-'
        return f'loop[{n.op},{n.mapf or "-"},{p}]'
    if isinstance(n, InArr):
        return 'inarr'
    if isinstance(n, InLen):
        return 'inlen'
    if isinstance(n, Arr):
        return f'arr{len(n.items)}'
    if isinstance(n, Index):
        return f'index({shape_of(n.arr)})'
    if isinstance(n, ArrLoop):
        p = n.pred[0] if n.pred else '-'
        return f'arrloop[{n.op},{n.mapf or "-"},{p}]'
    raise TypeError(n)


def depth_of(n):
    if isinstance(n, Bin):
        return 1 + max(depth_of(n.a), depth_of(n.b))
    if isinstance(n, Un):
        return 1 + depth_of(n.a)
    return 0


def cell_of(n):
    """The GRAMMAR CELL a program came from -- coarser than shape_of.

    shape_of encodes the whole subtree, which makes almost every conditional
    and expression unique: 91% of shapes occur exactly once. A cell that
    appears once cannot be held out, so splitting on shape_of would make the
    Phase 3 compositional split meaningless. The cell records the axis choices
    (which reduce, which map, which predicate, which comparison) and abstracts
    the rest, giving a few hundred cells with enough examples each to both
    train on and hold out.
    """
    if isinstance(n, Program):
        return cell_of(n.body)
    if isinstance(n, Loop):
        return f'loop[{n.op},{n.mapf or "-"},{n.pred[0] if n.pred else "-"}]'
    if isinstance(n, ArrLoop):
        fam = 'inarrloop' if isinstance(n.arr, InArr) else 'arrloop'
        return f'{fam}[{n.op},{n.mapf or "-"},{n.pred[0] if n.pred else "-"}]'
    if isinstance(n, Index):
        return f'index[{len(n.arr.items)}]'
    if isinstance(n, Sel):
        return f'sel[{n.c.op}]'
    if isinstance(n, Bin):
        return f'expr[{n.op},d{depth_of(n)}]'
    if isinstance(n, Un):
        return f'un[{n.op}]'
    if isinstance(n, InLen):
        return 'inlen'
    if isinstance(n, C):
        return 'const'
    if isinstance(n, Arg):
        return 'arg'
    raise TypeError(n)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def _leaf(rng, nargs, small=False):
    if nargs and rng.random() < 0.45:
        return Arg(rng.randrange(nargs))
    return C(rng.randint(1, 20) if small else rng.randint(0, 500))


def expr(rng, depth, nargs):
    """A small arithmetic tree. Depth is capped to stay inside the temp pool."""
    if depth <= 0 or rng.random() < 0.35:
        return _leaf(rng, nargs)
    op = rng.choice(BINOPS)
    if op in ('lsl', 'lsr'):
        return Bin(op, expr(rng, depth - 1, nargs), C(rng.randint(1, 8)))
    return Bin(op, expr(rng, depth - 1, nargs), expr(rng, depth - 1, nargs))


def _pred(rng):
    kind = rng.choice(PREDS)
    if kind is None:
        return None
    if kind == 'divk':
        return ('divk', rng.choice(DIVK))
    if kind in ('gt', 'lt'):
        return (kind, rng.randint(2, 40))
    return (kind, 0)


def _map(rng, op):
    # `count` ignores the mapped value, so a map there would be invisible in the
    # answer while still appearing in the question. Keep the cell honest.
    if op == 'count':
        return None, 0
    mapf = rng.choice(MAPS)
    k = rng.randint(2, 12) if mapf in ('addk', 'mulk', 'constk') else 0
    return mapf, k


# --------------------------------------------------------------------------
# Top-level forms
# --------------------------------------------------------------------------

def range_loop(rng, nargs):
    op = rng.choice(REDUCE)
    mapf, k = _map(rng, op)
    pred = _pred(rng)
    if nargs >= 2 and rng.random() < 0.35:
        lo, hi = Arg(0), Arg(1)              # empty ranges are exercised here
    else:
        a = rng.randint(1, 40)
        span = rng.randint(2, 60) if op != 'product' else rng.randint(1, 4)
        lo, hi = C(a), C(a + span)
    return Loop(op, lo, hi, mapf=mapf, pred=pred, k=k)


def input_array_loop(rng, nargs):
    """Reduce over the CALLER's array. Identical program at any length -- which
    is what makes the train-short / test-long question answerable."""
    op = rng.choice(REDUCE)
    mapf, k = _map(rng, op)
    if op == 'product':
        mapf, k = None, 0            # products of mapped values overflow fast
    return ArrLoop(op, InArr(), mapf=mapf, pred=_pred(rng), k=k)


def array_len(rng, nargs):
    return InLen()


def array_loop(rng, nargs):
    op = rng.choice(REDUCE)
    mapf, k = _map(rng, op)
    n = rng.randint(3, 5)
    hi = 40 if op != 'product' else 6
    items = tuple(C(rng.randint(1, hi)) for _ in range(n))
    return ArrLoop(op, Arr(items), mapf=mapf, pred=_pred(rng), k=k)


def array_index(rng, nargs):
    n = rng.randint(3, 5)
    items = tuple(C(rng.randint(1, 90)) for _ in range(n))
    return Index(Arr(items), C(rng.randrange(n)))


def conditional(rng, nargs):
    c = Cmp(rng.choice(CMPOPS), expr(rng, 1, nargs), expr(rng, 1, nargs))
    return Sel(c, expr(rng, 1, nargs), expr(rng, 1, nargs))


def unary(rng, nargs):
    return Un(rng.choice(['neg', 'abs']), expr(rng, 1, nargs))


def countdown(rng, nargs):
    return Loop('countdown', C(0), C(rng.randint(3, 2000)))


FORMS = [
    (range_loop,       0.24),
    (input_array_loop, 0.22),      # the new capability: variable-length input
    (array_loop,       0.10),
    (conditional,      0.14),
    (lambda r, n: expr(r, 3, n), 0.17),
    (array_index,      0.06),
    (unary,            0.04),
    (array_len,        0.02),
    (countdown,        0.01),
]


def sample_program(rng):
    nargs = rng.choice([0, 0, 1, 2, 2])
    r = rng.random()
    acc = 0.0
    for fn, w in FORMS:
        acc += w
        if r <= acc:
            body = fn(rng, nargs)
            break
    else:
        body = expr(rng, 3, nargs)

    # An unused argument would make the question mention an input the code
    # never reads; count what the tree actually references.
    used = _max_arg(body)
    return Program(body, n_args=used + 1, shape=shape_of(body),
                   phrase=rng.randrange(16))


def _max_arg(n, best=-1):
    if isinstance(n, Arg):
        return max(best, n.i)
    for f in getattr(n, '__dataclass_fields__', {}):
        v = getattr(n, f)
        if isinstance(v, tuple):
            for x in v:
                if hasattr(x, '__dataclass_fields__'):
                    best = _max_arg(x, best)
        elif hasattr(v, '__dataclass_fields__'):
            best = _max_arg(v, best)
    return best


def sample(n, seed=0):
    rng = random.Random(seed)
    return [sample_program(rng) for _ in range(n)]
