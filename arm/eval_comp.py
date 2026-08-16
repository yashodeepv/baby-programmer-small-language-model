"""
eval_comp.py - Score the model by running what it writes, on many inputs.

A generated program is correct only when it matches the oracle on EVERY
argument vector. That is the fix for the accident this project kept hitting:
asked for the largest array element, the old model emitted a hardcoded
`ldr w0, [sp, #4]`, which happened to load the maximum on the one input it was
checked against, and scored PASS.

Reported per eval set, each a superset of the next:

    parses     produced a non-empty body terminated by `ret`
    builds     clang accepts and links it
    correct    matches the oracle on every input          <- the score
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from ir import Program, question
from verify_ir import build_cases, verify_cases

PROMPT = "USER: {q}\nASSISTANT:"


def first_line(text):
    """The s-expression target is one line; take it, ignoring any invented
    next turn the model rolls into."""
    for raw in text.split('USER:')[0].split('\n'):
        line = raw.strip()
        if line:
            return line
    return ''


def evaluate_sexpr(model, tok, progs, seed=0, sizes=None, max_new_tokens=90):
    """Score the model as a PARSER rather than a generator.

    Three outcomes, deliberately separated:
      unparseable  the output is not valid IR at all -- the analogue of
                   "does not compile"
      exact        the parsed IR is structurally identical to the reference
      equivalent   it lowers and executes identically on every input, even if
                   expressed differently. This is the honest headline: a
                   differently-shaped but correct parse IS a correct parse.
    """
    import sexpr
    from lower import lower
    from verify_ir import ARRAY_SIZES

    texts = generate(model, tok, progs, max_new_tokens=max_new_tokens)
    spec = build_cases(progs, random.Random(seed), sizes=sizes or ARRAY_SIZES)

    st = {'n': len(progs), 'parsed': 0, 'exact': 0, 'equivalent': 0}
    cases, idx = [], []
    for i, ((p, vecs), t) in enumerate(zip(spec, texts)):
        line = first_line(t)
        try:
            got = sexpr.parse(line)
        except Exception:
            continue
        st['parsed'] += 1
        if got == p.body:
            st['exact'] += 1
        if not vecs:
            continue
        try:
            lines = lower(Program(got, n_args=sexpr.n_args_of(got)))
        except Exception:
            continue          # parses, but cannot be lowered (e.g. too deep)
        cases.append((lines, vecs))
        idx.append(i)

    if cases:
        res, _ = verify_cases(cases)
        st['equivalent'] = sum(1 for ok, n in res if n and ok == n)
    return st, texts


def fmt_sexpr(st):
    n = st['n'] or 1
    return (f"parsed {st['parsed']:>4}/{st['n']} ({100*st['parsed']/n:5.1f}%)  "
            f"exact {st['exact']:>4} ({100*st['exact']/n:5.1f}%)  "
            f"equivalent {st['equivalent']:>4} ({100*st['equivalent']/n:5.1f}%)")


def parse_answer(text):
    """Assembly body out of a continuation; [] if `ret` never arrived."""
    body, done = [], False
    for raw in text.split('\n'):
        line = raw.strip()
        if not line:
            continue
        if line == 'ret':
            done = True
            break
        body.append(line)
    return body if done and body else []


@torch.no_grad()
def generate(model, tok, progs, max_new_tokens=300, batch_size=32):
    """Greedy decode, batched over prompts of equal token length."""
    device = next(model.parameters()).device
    stop = tok.stoi.get('ret')
    out = [''] * len(progs)
    buckets = {}
    for i, p in enumerate(progs):
        ids = tok.encode(PROMPT.format(q=question(p)), allow_unk=True)
        buckets.setdefault(len(ids), []).append((i, ids))

    for _, items in sorted(buckets.items()):
        for k in range(0, len(items), batch_size):
            chunk = items[k:k + batch_size]
            idx = torch.tensor([ids for _, ids in chunk],
                               dtype=torch.long, device=device)
            plen = idx.shape[1]
            gen = model.generate(idx, max_new_tokens, greedy=True, stop_token=stop)
            for (i, _), row in zip(chunk, gen.tolist()):
                out[i] = tok.decode(row[plen:])
    return out


def evaluate(model, tok, progs, seed=0, max_new_tokens=300, sizes=None):
    texts = generate(model, tok, progs, max_new_tokens=max_new_tokens)
    from verify_ir import ARRAY_SIZES
    spec = build_cases(progs, random.Random(seed), sizes=sizes or ARRAY_SIZES)

    cases, kept = [], []
    for (p, vecs), t in zip(spec, texts):
        body = parse_answer(t)
        if body and vecs:
            cases.append((body, vecs))
            kept.append(p)

    stats = {'n': len(progs), 'parses': len(cases), 'builds': 0, 'correct': 0}
    if not cases:
        return stats, texts

    # Build-check first: model output routinely fails to assemble, and one bad
    # program takes its whole batch's build down. Filtering here keeps the
    # property run clean and gives the `builds` number directly.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        ok_build = list(pool.map(lambda c: _builds(c[0]), cases))
    stats['builds'] = sum(ok_build)

    good = [c for c, b in zip(cases, ok_build) if b]
    if good:
        results, _ = verify_cases(good)
        stats['correct'] = sum(1 for ok, n in results if n and ok == n)
    return stats, texts


def _builds(lines):
    """Does this body assemble and link at all?"""
    import subprocess, tempfile
    from verify_ir import emit_function
    with tempfile.TemporaryDirectory() as wd:
        src = ('.section __TEXT,__text\n' + emit_function(0, lines) +
               '\n.global _main\n.p2align 2\n_main:\n'
               '    stp x29, x30, [sp, #-16]!\n    bl _p0\n'
               '    ldp x29, x30, [sp], #16\n    ret\n')
        p = os.path.join(wd, 'b.s')
        with open(p, 'w') as f:
            f.write(src)
        r = subprocess.run(['clang', p, '-o', os.path.join(wd, 'b.out')],
                           capture_output=True)
        return r.returncode == 0


def fmt(st):
    n = st['n'] or 1
    return (f"parses {st['parses']:>4}/{st['n']}  "
            f"builds {st['builds']:>4} ({100*st['builds']/n:5.1f}%)  "
            f"correct {st['correct']:>4} ({100*st['correct']/n:5.1f}%)")
