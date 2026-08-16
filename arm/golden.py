"""
golden.py - Hand-written ground truth. The one thing not derived from the IR.

lower.py and ir.evaluate are independent implementations, so a codegen bug
shows up as a mismatch between them. What that cannot catch is a shared
misunderstanding: if the IR's *meaning* is wrong, the assembly and the oracle
inherit the same mistake, agree with each other, and verification goes green on
a program that answers the wrong question.

Every expected value below was worked out by hand from the English description,
NOT by running ir.evaluate. If the oracle drifts, these fail. That is the point.

    .venv/bin/python arm/golden.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir import (Arg, Arr, ArrLoop, Bin, Cmp, Const, InArr, InLen, Index,
                Loop, Program, Sel, Un, evaluate, question)

C = Const

# (description, node, args, expected-by-hand)
GOLDEN = [
    ("the constant 42",                       C(42),                      (), 42),
    ("the first argument",                    Arg(0),                     (7,), 7),
    ("9 + 12",                                Bin('add', C(9), C(12)),    (), 21),
    ("100 - 30",                              Bin('sub', C(100), C(30)),  (), 70),
    ("6 * 7",                                 Bin('mul', C(6), C(7)),     (), 42),
    ("12 AND 10",                             Bin('and', C(12), C(10)),   (), 8),
    ("12 OR 10",                              Bin('orr', C(12), C(10)),   (), 14),
    ("12 XOR 10",                             Bin('eor', C(12), C(10)),   (), 6),
    ("3 << 4",                                Bin('lsl', C(3), C(4)),     (), 48),
    ("200 >> 3",                              Bin('lsr', C(200), C(3)),   (), 25),
    ("negate 5",                              Un('neg', C(5)),            (), -5),
    ("abs(-5)",                               Un('abs', Un('neg', C(5))), (), 5),
    ("13 > 24 as 0/1",                        Cmp('gt', C(13), C(24)),    (), 0),
    ("3 <= 3 as 0/1",                         Cmp('le', C(3), C(3)),      (), 1),

    # max(w0,w1) written as a select
    ("max of 12 and 188",
     Sel(Cmp('ge', Arg(0), Arg(1)), Arg(0), Arg(1)), (12, 188), 188),
    ("max of 200 and 5",
     Sel(Cmp('ge', Arg(0), Arg(1)), Arg(0), Arg(1)), (200, 5), 200),
    ("min of 12 and 188",
     Sel(Cmp('le', Arg(0), Arg(1)), Arg(0), Arg(1)), (12, 188), 12),

    # loops over a range
    ("sum 1..10",                     Loop('sum', C(1), C(10)),   (), 55),
    ("sum 5..3 (empty range)",        Loop('sum', C(5), C(3)),    (), 0),
    ("product 2..5",                  Loop('product', C(2), C(5)), (), 120),
    ("product 3..3 (single)",         Loop('product', C(3), C(3)), (), 3),
    ("count 1..10",                   Loop('count', C(1), C(10)), (), 10),
    ("countdown from 7 leaves 0",     Loop('countdown', C(0), C(7)), (), 0),
    ("sum w0..w1 with 1,10",          Loop('sum', Arg(0), Arg(1)), (1, 10), 55),
    ("sum w0..w1 with 5,3 (empty)",   Loop('sum', Arg(0), Arg(1)), (5, 3), 0),

    # the compositional axes -- the whole reason for this redesign
    ("sum of EVEN integers 2..20",
     Loop('sum', C(2), C(20), pred=('even', 0)),   (), 110),
    ("sum of ODD integers 1..9",
     Loop('sum', C(1), C(9), pred=('odd', 0)),     (), 25),
    ("count of ODD integers 1..10",
     Loop('count', C(1), C(10), pred=('odd', 0)),  (), 5),
    ("sum of integers 1..10 greater than 5",
     Loop('sum', C(1), C(10), pred=('gt', 5)),     (), 40),
    ("sum of integers 1..10 less than 4",
     Loop('sum', C(1), C(10), pred=('lt', 4)),     (), 6),
    ("sum of integers 1..20 divisible by 4",
     Loop('sum', C(1), C(20), pred=('divk', 4)),   (), 60),
    ("sum of SQUARES 1..5",
     Loop('sum', C(1), C(5), mapf='square'),       (), 55),
    ("sum of DOUBLES 1..5",
     Loop('sum', C(1), C(5), mapf='double'),       (), 30),
    ("sum of 1..4 each increased by 10",
     Loop('sum', C(1), C(4), mapf='addk', k=10),   (), 50),
    ("sum of 1..4 each multiplied by 3",
     Loop('sum', C(1), C(4), mapf='mulk', k=3),    (), 30),
    ("7 added once per integer 1..6 (repeated addition)",
     Loop('sum', C(1), C(6), mapf='constk', k=7),  (), 42),
    ("sum of the squares of the EVEN integers 2..6",
     Loop('sum', C(2), C(6), mapf='square', pred=('even', 0)), (), 56),

    # arrays
    ("element 2 of [5,9,2,7]",
     Index(Arr((C(5), C(9), C(2), C(7))), C(2)),   (), 2),
    ("element 0 of [5,9,2,7]",
     Index(Arr((C(5), C(9), C(2), C(7))), C(0)),   (), 5),
    ("element 3 of [5,9,2,7]",
     Index(Arr((C(5), C(9), C(2), C(7))), C(3)),   (), 7),
    ("sum of [5,9,2,7]",
     ArrLoop('sum', Arr((C(5), C(9), C(2), C(7)))), (), 23),
    ("product of [2,3,4]",
     ArrLoop('product', Arr((C(2), C(3), C(4)))),  (), 24),
    ("how many of [5,9,2,7] exceed 4",
     ArrLoop('count', Arr((C(5), C(9), C(2), C(7))), pred=('gt', 4)), (), 3),
    ("sum of the EVEN elements of [5,9,2,7]",
     ArrLoop('sum', Arr((C(5), C(9), C(2), C(7))), pred=('even', 0)), (), 2),
]


# Input-array cases. Expected values worked out by hand from the description,
# including the empty input -- which is where an off-by-one in the loop bound
# or a min/max sentinel disagreement actually shows up.
ARRAY_GOLDEN = [
    ("sum of the array",            ArrLoop('sum', InArr()),   [4, 9, 2],   15),
    ("sum of the empty array",      ArrLoop('sum', InArr()),   [],           0),
    ("sum of a single element",     ArrLoop('sum', InArr()),   [7],          7),
    ("product of the array",        ArrLoop('product', InArr()), [2, 3, 4],  24),
    ("product of the empty array",  ArrLoop('product', InArr()), [],          1),
    ("count of the array",          ArrLoop('count', InArr()), [5, 5, 5, 5], 4),
    ("largest of the array",        ArrLoop('max', InArr()),   [3, 11, 7],  11),
    ("largest, all negative",       ArrLoop('max', InArr()),   [-9, -2, -5], -2),
    ("largest of the empty array",  ArrLoop('max', InArr()),   [],           0),
    ("smallest of the array",       ArrLoop('min', InArr()),   [3, 11, 7],   3),
    ("smallest, all negative",      ArrLoop('min', InArr()),   [-9, -2, -5], -9),
    ("sum of the EVEN elements",    ArrLoop('sum', InArr(), pred=('even', 0)),
                                                               [1, 2, 3, 4],  6),
    ("count of elements over 10",   ArrLoop('count', InArr(), pred=('gt', 10)),
                                                          [4, 20, 11, 9],     2),
    ("largest EVEN element",        ArrLoop('max', InArr(), pred=('even', 0)),
                                                          [3, 8, 5, 12, 7],  12),
    ("no element passes the filter", ArrLoop('max', InArr(), pred=('gt', 100)),
                                                          [3, 8, 5],          0),
    ("sum of the squares",          ArrLoop('sum', InArr(), mapf='square'),
                                                               [1, 2, 3],    14),
    ("the array length",            InLen(),                   [5, 5, 5],     3),
]


def check_arrays():
    """Both views must reproduce the hand-computed array results."""
    from verify_ir import verify_cases
    from lower import lower
    bad = []
    for desc, node, arr, want in ARRAY_GOLDEN:
        got = evaluate(node, (), arr)
        if got != want:
            bad.append((desc, 'oracle', want, got))
    cases = [(lower(Program(n, 0, d)), [((), a, w & 0xFFFFFFFF)])
             for d, n, a, w in ARRAY_GOLDEN]
    res, errs = verify_cases(cases)
    for (desc, _, _, _), r in zip(ARRAY_GOLDEN, res):
        if r != (1, 1):
            bad.append((desc, 'compiled', 'pass', r))
    print(f'array cases vs hand-written: '
          f'{len(ARRAY_GOLDEN) * 2 - len(bad)}/{len(ARRAY_GOLDEN) * 2} agree')
    for d, which, w, g in bad:
        print(f'  MISMATCH [{which}] {d}: want {w}, got {g}')
    for b, e in errs:
        print(f'  BUILD ERROR near {b}: {e}')
    return bad


# s-expression surface. These strings were written by hand from the English
# description, NOT produced by sexpr.emit -- so a bug in the emitter, or an
# ambiguity in the surface syntax, shows up here rather than silently training
# the model on a target that means something else.
SEXPR_GOLDEN = [
    ("the sum of the even integers from 2 to 20",
     "(sum (rng 2 20) :even)",                       110),
    ("the product of the integers from 2 to 5",
     "(product (rng 2 5))",                          120),
    ("how many odd integers from 1 to 10",
     "(count (rng 1 10) :odd)",                        5),
    ("the sum of the squares of the integers from 1 to 5",
     "(sum (rng 1 5) :sq)",                           55),
    ("the sum of 1 to 4, each increased by 10",
     "(sum (rng 1 4) :add 10)",                       50),
    ("the sum of the integers 1 to 20 divisible by 4",
     "(sum (rng 1 20) :divk 4)",                      60),
    ("zero after counting down from 7",
     "(down 7)",                                       0),
    ("9 plus 12",                  "(add 9 12)",      21),
    ("(21 plus 95) times 42",      "(mul (add 21 95) 42)", 4872),
    ("the absolute value of -5",   "(abs -5)",         5),
    ("1 if 13 is greater than 24, else 0",
     "(gt 13 24)",                                     0),
    ("element at index 2 of [5, 9, 2, 7]",
     "(idx (lit 5 9 2 7) 2)",                          2),
    ("the sum of the elements of [5, 9, 2, 7]",
     "(sum (lit 5 9 2 7))",                           23),
]

ARRAY_SEXPR_GOLDEN = [
    ("the sum of the elements of the array",  "(sum (arr))",        [4, 9, 2], 15),
    ("the largest even element of the array", "(max (arr) :even)",  [3, 8, 5, 12], 12),
    ("the smallest element of the array",     "(min (arr))",        [9, 4, 7],  4),
    ("how many elements exceed 10",           "(count (arr) :gt 10)", [4, 20, 11, 9], 2),
    ("the length of the array",               "(len)",              [5, 5, 5],  3),
]


def check_sexpr():
    """Hand-written s-expressions must parse to IR that means what the English says,
    and must be exactly what the emitter produces."""
    from sexpr import emit, parse
    bad = []
    for desc, text, want in SEXPR_GOLDEN:
        node = parse(text)
        got = evaluate(node, (), None)
        if got != want:
            bad.append((desc, 'value', want, got))
        if emit(node) != text:
            bad.append((desc, 'emit', text, emit(node)))
    for desc, text, arr, want in ARRAY_SEXPR_GOLDEN:
        node = parse(text)
        got = evaluate(node, (), arr)
        if got != want:
            bad.append((desc, 'value', want, got))
        if emit(node) != text:
            bad.append((desc, 'emit', text, emit(node)))
    n = (len(SEXPR_GOLDEN) + len(ARRAY_SEXPR_GOLDEN)) * 2
    print(f's-expressions vs hand-written: {n - len(bad)}/{n} agree')
    for d, w, want, got in bad:
        print(f'  MISMATCH [{w}] {d}: want {want!r}, got {got!r}')
    return bad


# The renderer is the third view, and the only one execution cannot check: if
# the question says something other than what the code does, the oracle and the
# assembly still agree and verification goes green on a mislabelled example.
# These expected strings are hand-written from the intended meaning.
RENDER_GOLDEN = [
    (Bin('mul', Bin('add', C(21), C(95)), C(42)), '(21 plus 95) times 42'),
    (Bin('add', C(21), Bin('mul', C(95), C(42))), '21 plus (95 times 42)'),
    (Bin('sub', Bin('sub', C(100), C(20)), C(30)), '(100 minus 20) minus 30'),
    (Loop('sum', C(2), C(20), pred=('even', 0)),
     'the sum of the even integers from 2 to 20'),
    (Loop('sum', C(1), C(5), mapf='square'),
     'the sum of the squares of the integers from 1 to 5'),
    (Loop('count', C(1), C(10), pred=('odd', 0)),
     'how many of the odd integers from 1 to 10'),
    (Index(Arr((C(5), C(9))), C(1)), 'element at index 1 of [5, 9]'),
]


def check_renderer():
    bad = []
    from ir import show
    for node, want in RENDER_GOLDEN:
        got = show(node)
        if got != want:
            bad.append((want, got))
    print(f'question text vs hand-written: '
          f'{len(RENDER_GOLDEN) - len(bad)}/{len(RENDER_GOLDEN)} agree')
    for w, g in bad:
        print(f'  MISMATCH  want {w!r}')
        print(f'            got  {g!r}')
    return bad


def check_oracle():
    """The oracle must reproduce every hand-computed value."""
    bad = []
    for desc, node, args, want in GOLDEN:
        try:
            got = evaluate(node, args)
        except Exception as e:
            got = f'raised {e}'
        if got != want:
            bad.append((desc, want, got))
    print(f'oracle vs hand-written: {len(GOLDEN) - len(bad)}/{len(GOLDEN)} agree')
    for d, w, g in bad:
        print(f'  MISMATCH  {d}: hand says {w}, oracle says {g}')
    return bad


def check_lowering():
    """The compiled assembly must reproduce every hand-computed value too."""
    from verify_ir import verify
    progs, wants = [], []
    for desc, node, args, want in GOLDEN:
        n = 1 + max([a.i for a in _args_of(node)], default=-1)
        progs.append(Program(node, n_args=n, shape=desc))
        wants.append((args, want))

    # Pin each program to exactly its golden argument vector.
    import verify_ir
    orig = verify_ir.build_cases
    verify_ir.build_cases = lambda ps, rng, sizes=None: [
        (p, [(wants[i][0], None, wants[i][1] & 0xFFFFFFFF)])
        for i, p in enumerate(ps)]
    try:
        results, errors = verify(progs)
    finally:
        verify_ir.build_cases = orig

    bad = [(p.shape, r) for p, r in zip(progs, results) if r != (1, 1)]
    print(f'compiled code vs hand-written: {len(progs) - len(bad)}/{len(progs)} agree')
    for name, r in bad:
        print(f'  MISMATCH  {name}: {r}')
    for base, err in errors:
        print(f'  BUILD ERROR near case {base}: {err}')
    return bad


def _args_of(node):
    out = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Arg):
            out.append(n)
        for f in getattr(n, '__dataclass_fields__', {}):
            v = getattr(n, f)
            if isinstance(v, tuple):
                stack.extend(x for x in v if hasattr(x, '__dataclass_fields__'))
            elif hasattr(v, '__dataclass_fields__'):
                stack.append(v)
    return out


if __name__ == '__main__':
    a = check_oracle()
    b = check_lowering()
    c = check_renderer()
    d = check_arrays()
    e = check_sexpr()
    print()
    print('GOLDEN SET PASSES' if not (a or b or c or d or e) else 'GOLDEN SET FAILS')
    sys.exit(1 if (a or b or c or d or e) else 0)
