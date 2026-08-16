"""
split.py - Hold out COMBINATIONS, not constants.

Every split so far varied the numbers in a question while the structure stayed
in training, so "held out" only ever meant "unseen constants". That cannot
answer the question the project actually cares about: does it generalise to a
kind of problem it has not seen?

Here whole grammar cells are removed from training -- say every
`loop[product,square,odd]` -- under one binding rule: each individual axis value
must still appear in training somewhere else. `product`, `square` and `odd` are
each seen many times; only their COMBINATION is new. Otherwise the eval would
merely be testing an unknown word, which is not composition.

Three eval sets fall out:

    seen    same cells as training, new constants     (comparable to before)
    combo   held-out cells: unseen mix of seen parts  (the real target)
    depth   deeper expressions than any in training   (structural extrapolation)
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grammar
from grammar import cell_of, depth_of
from ir import Arr, ArrLoop, Bin, Loop, Program, Const as C

TRAIN_MAX_DEPTH = 3          # expression depth ceiling during training

# Input-array lengths. Training never sees an array longer than 8; the size
# eval uses 12 and 16. The correct program is IDENTICAL at every length, so
# unlike the depth test this is a fair question -- a gap here means the model
# learned a length rather than an algorithm.
TRAIN_SIZES = (0, 1, 2, 5, 8)
TEST_SIZES  = (12, 16)


def _loop_cells():
    """All (family, op, mapf, pred) cells the grammar can produce."""
    out = []
    for fam in ('loop', 'arrloop'):
        for op in grammar.REDUCE:
            maps = [None] if op == 'count' else grammar.MAPS
            for mf in maps:
                for pr in grammar.PREDS:
                    out.append((fam, op, mf, pr))
    return out


def choose_holdout(frac=0.15, seed=0):
    """Pick cells to hold out, keeping every axis value covered in training."""
    cells = _loop_cells()
    rng = random.Random(seed)
    rng.shuffle(cells)

    target = int(len(cells) * frac)
    held = []
    for cand in cells:
        if len(held) >= target:
            break
        trial = held + [cand]
        remaining = [c for c in cells if c not in trial]
        # every axis value must survive in the training half
        ok = True
        for idx in (0, 1, 2, 3):
            need = {c[idx] for c in cells}
            have = {c[idx] for c in remaining}
            if need != have:
                ok = False
                break
        if ok:
            held.append(cand)

    return {f'{fam}[{op},{mf or "-"},{pr or "-"}]' for fam, op, mf, pr in held}


# --------------------------------------------------------------------------
# Targeted construction, so held-out cells can be sampled directly
# --------------------------------------------------------------------------

def make_cell_program(cell, rng):
    """Build a program that belongs to exactly `cell`."""
    fam, rest = cell.split('[')
    op, mf, pr = rest.rstrip(']').split(',')
    mapf = None if mf == '-' else mf
    k = rng.randint(2, 12) if mapf in ('addk', 'mulk', 'constk') else 0
    if pr == '-':
        pred = None
    elif pr == 'divk':
        pred = ('divk', rng.choice(grammar.DIVK))
    elif pr in ('gt', 'lt'):
        pred = (pr, rng.randint(2, 40))
    else:
        pred = (pr, 0)

    if fam == 'loop':
        a = rng.randint(1, 40)
        span = rng.randint(2, 60) if op != 'product' else rng.randint(1, 4)
        body = Loop(op, C(a), C(a + span), mapf=mapf, pred=pred, k=k)
    else:
        n = rng.randint(3, 5)
        hi = 40 if op != 'product' else 6
        items = tuple(C(rng.randint(1, hi)) for _ in range(n))
        body = ArrLoop(op, Arr(items), mapf=mapf, pred=pred, k=k)

    return Program(body, n_args=0, shape=grammar.shape_of(body),
                   phrase=rng.randrange(16))


# --------------------------------------------------------------------------
# The three sets
# --------------------------------------------------------------------------

def train_programs(n, held, seed=42, max_depth=TRAIN_MAX_DEPTH):
    """Sample training programs, rejecting held-out cells and deep expressions.

    max_depth is the Phase 4 knob. Phase 3 trained at <=3 and scored 6.4% at
    depth 4, with the model emitting systematically SHORTER programs -- it had
    learned a maximum nesting depth rather than recursion. Raising this tests
    whether that is a data limit (train on varied depths, extrapolate further)
    or a capacity limit.
    """
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        p = grammar.sample_program(rng)
        if cell_of(p) in held:
            continue
        if depth_of(p.body) > max_depth:
            continue
        out.append(p)
    return out


def eval_seen(n, held, train_questions, seed=777, max_depth=TRAIN_MAX_DEPTH):
    """Cells the model trained on, questions it has not seen."""
    from ir import question
    rng = random.Random(seed)
    out, guard = [], 0
    while len(out) < n and guard < n * 400:
        guard += 1
        p = grammar.sample_program(rng)
        if cell_of(p) in held or depth_of(p.body) > max_depth:
            continue
        if question(p) in train_questions:
            continue
        out.append(p)
    return out


def eval_combo(n, held, seed=778):
    """Held-out cells: an unseen combination of parts seen individually."""
    rng = random.Random(seed)
    cells = sorted(held)
    return [make_cell_program(cells[i % len(cells)], rng) for i in range(n)]


def eval_paraphrase(n, held, train_questions, seed=782):
    """Trained cells and trained constants, but a WORDING never seen.

    Every other split varies what is asked. This varies only how it is asked,
    which is the one thing the old corpus could not test at all -- it had a
    single wording per operation, so "meaning survives rephrasing" was never
    demonstrated and never learned.
    """
    from ir import question
    rng = random.Random(seed)
    out, guard = [], 0
    while len(out) < n and guard < n * 400:
        guard += 1
        p = grammar.sample_program(rng)
        if cell_of(p) in held or depth_of(p.body) > TRAIN_MAX_DEPTH:
            continue
        p.style = grammar.HELDOUT_STYLE          # the unseen wording
        if question(p) in train_questions:
            continue
        out.append(p)
    return out


def eval_size(n, held, seed=781):
    """Programs over the input array, to be verified at unseen lengths."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        p = grammar.sample_program(rng)
        if cell_of(p) in held:
            continue
        if not isinstance(getattr(p.body, 'arr', None), grammar.InArr):
            continue
        out.append(p)
    return out


def eval_depth(n, seed=779, depth=TRAIN_MAX_DEPTH + 1, train_depth=TRAIN_MAX_DEPTH):
    """Expressions deeper than anything in training."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        body = grammar.expr(rng, depth, 2)
        if depth_of(body) <= train_depth:
            continue                      # must actually be deeper
        out.append(Program(body, n_args=grammar._max_arg(body) + 1,
                           shape=grammar.shape_of(body),
                           phrase=rng.randrange(16)))
    return out
