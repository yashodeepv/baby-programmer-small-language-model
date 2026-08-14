"""
verify_ir.py - Property-based verification: every program, many inputs.

The previous verifier ran each program once, because every constant was baked
in. That could not tell a correct program from one that coincidentally lands on
the right number -- and it demonstrably did not: asked for the largest array
element, the model emitted a hardcoded `ldr w0, [sp, #4]`, which happened to
load the maximum, and scored PASS.

Here a program is a function, and it is called with K argument vectors chosen
to include the edges (empty range, single element, lo == hi, lo > hi, negatives,
large magnitudes). A program is correct only if EVERY vector matches the oracle.

The batching, hang detection and partial-output attribution are inherited from
gen_corpus: one binary per batch, one stdout byte per check, and a stalled
program is charged to itself rather than taking the batch down.
"""

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watchdog import _exec_watched, SAMPLE_TIMEOUT
from ir import evaluate
from lower import lower

BATCH = 60          # programs per binary; each contributes K checks


def emit_function(k, lines):
    """One IR program as a standalone function, labels namespaced per program."""
    import re
    labels = [l[:-1] for l in lines if l.endswith(':')]
    out = []
    for ln in lines:
        for lab in labels:
            ln = re.sub(rf'\b{re.escape(lab)}\b', f'p{k}_{lab}', ln)
        out.append(ln if ln.endswith(':') else '    ' + ln)
    return f'.p2align 2\n_p{k}:\n' + '\n'.join(out) + '\n    ret\n'


def _load_imm(reg, v):
    v &= 0xFFFFFFFF
    lo, hi = v & 0xFFFF, (v >> 16) & 0xFFFF
    s = f'    mov {reg}, #{lo}\n'
    if hi:
        s += f'    movk {reg}, #{hi}, lsl #16\n'
    return s


def emit_check(k, args, arr, expect):
    """Call program k with one input vector and write a pass/fail byte.

    When the program takes an array, the buffer is filled here and passed as
    pointer in x0 / length in w1, so the SAME program can be called at any
    length -- which is what makes "train short, test long" possible.
    """
    if arr is not None:
        s = '    adrp x6, _buf@PAGE\n    add x6, x6, _buf@PAGEOFF\n'
        for j, v in enumerate(arr):
            s += _load_imm('w7', v) + f'    str w7, [x6, #{j * 4}]\n'
        s += '    mov x0, x6\n' + _load_imm('w1', len(arr))
        s += ''.join(_load_imm(f'w{2 + i}', a) for i, a in enumerate(args))
    else:
        s = ''.join(_load_imm(f'w{i}', a) for i, a in enumerate(args))
    s += f'    bl _p{k}\n'
    s += _load_imm('w1', expect)
    return f"""{s}    cmp w0, w1
    mov w3, #46
    mov w4, #70
    csel w3, w3, w4, eq
    adrp x5, _mark@PAGE
    add x5, x5, _mark@PAGEOFF
    strb w3, [x5]
    mov x0, #1
    mov x1, x5
    mov x2, #1
    bl _write"""


HARNESS = """.section __DATA,__data
.p2align 3
_mark:
    .space 1
.p2align 3
_buf:
    .space 128

.section __TEXT,__text
{functions}
.global _main
.p2align 2
_main:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
{checks}
    mov w0, #0
    ldp x29, x30, [sp], #16
    ret
"""


ARRAY_SIZES = (0, 1, 2, 5, 8)      # training-range input lengths


def _arrays(rng, sizes, k):
    """Input arrays: the edges first, then random content.

    Empty and single-element inputs are mandatory -- they are where min/max
    over an empty selection, and any off-by-one in the loop bound, actually
    show up.
    """
    out = []
    for n in sizes:
        if n == 0:
            out.append([])
        else:
            out.append([rng.randint(-30, 60) for _ in range(n)])
            out.append([7] * n)                        # all equal
            out.append(sorted(rng.randint(-20, 60) for _ in range(n)))
    while len(out) < k:
        n = rng.choice([s for s in sizes if s]) if any(sizes) else 1
        out.append([rng.randint(-30, 60) for _ in range(n)])
    return out[:k]


def arg_vectors(prog, rng, k=12, sizes=ARRAY_SIZES):
    """(scalars, array) vectors for one program: edges first, then random."""
    from lower import uses_input_array
    n = prog.n_args
    arrays = _arrays(rng, sizes, k) if uses_input_array(prog) else [None] * k

    if n == 0:
        scal = [()] * len(arrays)
    else:
        edges = [tuple(0 for _ in range(n)),
                 tuple(1 for _ in range(n)),
                 tuple(-1 for _ in range(n))]
        if n >= 2:
            edges += [(5, 5), (7, 3)]      # lo == hi, and lo > hi (empty range)
        scal = [e[:n] for e in edges if len(e) >= n]
        while len(scal) < len(arrays):
            scal.append(tuple(rng.randint(-50, 200) for _ in range(n)))
    return list(zip(scal[:len(arrays)], arrays))


def build_cases(progs, rng, sizes=ARRAY_SIZES):
    """(program, [(args, arr, expected)]) with oracle-undefined inputs dropped."""
    cases = []
    for p in progs:
        vecs = []
        for a, arr in arg_vectors(p, rng, sizes=sizes):
            try:
                want = evaluate(p, a, arr)
            except Exception:
                continue                    # oracle undefined here; not a test
            if isinstance(want, list) or not (-2**31 <= want < 2**31):
                continue
            vecs.append((a, arr, want & 0xFFFFFFFF))
        cases.append((p, vecs))
    return cases


def _run_batch(cases, workdir, tag):
    """cases: [(asm_lines, [(args, expected), ...])].

    Takes assembly rather than IR so the very same harness scores MODEL output:
    a generated program is correct only if it matches the oracle on every
    argument vector, which is what "is the program right" actually means.
    """
    fns, checks, index = [], [], []
    for k, (lines, vecs) in enumerate(cases):
        fns.append(emit_function(k, lines))
        for args, arr, want in vecs:
            checks.append(emit_check(k, args, arr, want))
            index.append(k)

    src = HARNESS.format(functions='\n'.join(fns), checks='\n'.join(checks))
    s_path = os.path.join(workdir, f'v{tag}.s')
    b_path = os.path.join(workdir, f'v{tag}.out')
    with open(s_path, 'w') as f:
        f.write(src)

    asm = subprocess.run(['clang', s_path, '-o', b_path],
                         capture_output=True, text=True)
    if asm.returncode != 0:
        err = asm.stderr.strip().splitlines()[:2]
        return None, err
    marks, hung = _exec_watched(b_path)
    # _exec_watched hands back raw bytes; iterating those yields ints, so the
    # per-check comparison must be done on decoded text.
    return (marks.decode('ascii', 'replace'), index, hung), None


def _tally(got, batch, workdir, tag, offset=0):
    """Score one batch run, re-running past a hang instead of failing the rest.

    A program that does not halt stops the binary, so every check after it is
    missing. Charging those to their programs would fail up to BATCH-1 innocent
    programs for one bad neighbour -- so the hang is charged to its own program
    and the remainder is rebuilt and re-run.
    """
    marks, index, hung = got
    out = {k: [0, len(v)] for k, (_lines, v) in enumerate(batch)}
    done = min(len(marks), len(index))
    for j in range(done):
        out[index[j]][0] += (marks[j] == '.')

    if not hung or done >= len(index):
        return {k: tuple(v) for k, v in out.items()}

    culprit = index[done]                     # the check that never returned
    rest = batch[culprit + 1:]
    for k in range(culprit + 1, len(batch)):  # unknown until re-run
        out[k] = [0, len(batch[k][1])]
    if rest:
        got2, err2 = _run_batch(rest, workdir, f'{tag}x')
        if err2 is None:
            for k, r in _tally(got2, rest, workdir, f'{tag}y').items():
                out[culprit + 1 + k] = list(r)
    return {k: tuple(v) for k, v in out.items()}


def verify(progs, jobs=None, rng=None, sizes=ARRAY_SIZES):
    """Returns (results, errors). results[i] = (n_pass, n_total) per program."""
    import random
    rng = rng or random.Random(0)
    spec = build_cases(progs, rng, sizes=sizes)
    return verify_cases([(lower(p), v) for p, v in spec], jobs)


def verify_cases(cases, jobs=None):
    """Score pre-lowered (or model-written) assembly against expected values."""
    jobs = jobs or (os.cpu_count() or 4)
    batches = [cases[i:i + BATCH] for i in range(0, len(cases), BATCH)]
    results = {}
    errors = []

    with tempfile.TemporaryDirectory() as wd:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_run_batch, b, wd, i): (i, b)
                    for i, b in enumerate(batches)}
            for fut in as_completed(futs):
                i, batch = futs[fut]
                got, err = fut.result()
                base = i * BATCH
                if err is not None:
                    errors.append((base, err))
                    for k, (_lines, vecs) in enumerate(batch):
                        results[base + k] = (0, len(vecs))
                    continue
                for k, r in _tally(got, batch, wd, f'{i}r').items():
                    results[base + k] = r

    return [results.get(i, (0, 0)) for i in range(len(cases))], errors


def summarise(progs, results, label='verification'):
    full = sum(1 for (ok, n) in results if n and ok == n)
    anyp = sum(ok for ok, _ in results)
    tot  = sum(n for _, n in results)
    print(f'{label}: {full}/{len(progs)} programs correct on ALL inputs '
          f'({anyp}/{tot} individual checks)')
    bad = [(p, r) for p, r in zip(progs, results) if r[1] and r[0] != r[1]]
    for p, (ok, n) in bad[:8]:
        print(f'\n  FAIL [{p.shape}] {ok}/{n} inputs')
        print(f'    {__import__("ir").question(p)}')
        for l in lower(p):
            print(f'      {l}')
    return bad
