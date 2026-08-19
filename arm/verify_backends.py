"""verify_backends.py - Check every target language against the oracle.

Same contract the assembly path has always had: generate, run, and compare
against `ir.evaluate` on many inputs. A backend is correct only if it matches
on EVERY input vector of EVERY program.

    .venv/bin/python arm/verify_backends.py --n 200
"""

import argparse
import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grammar
from backends import CTarget, JsTarget, PyTarget
from ir import Arg, ArrLoop, Index, InArr, InLen, evaluate


def _uses(node, kinds, seen=None):
    if isinstance(node, kinds):
        return True
    for f in getattr(node, '__dataclass_fields__', {}):
        v = getattr(node, f)
        if isinstance(v, tuple):
            if any(_uses(x, kinds) for x in v if hasattr(x, '__dataclass_fields__')):
                return True
        elif hasattr(v, '__dataclass_fields__') and _uses(v, kinds):
            return True
    return False


def vectors(prog, rng, k=8):
    """Input vectors: scalars for Arg, an array when InArr/InLen appear."""
    needs_arr = _uses(prog.body, (InArr, InLen))
    out = []
    for _ in range(k):
        args = tuple(rng.randint(-20, 60) for _ in range(max(prog.n_args, 3)))
        arr = ([rng.randint(1, 40) for _ in range(rng.randint(3, 8))]
               if needs_arr else [])
        out.append((args, arr))
    return out


def run_python(src, vecs):
    ns = {}
    exec(compile(src, '<gen>', 'exec'), ns)
    return [ns['f'](list(a), list(r)) for a, r in vecs]


C_MAIN = """
#include <stdio.h>
int main(void) {
%s
    return 0;
}
"""


def run_c(src, vecs, workdir):
    calls = []
    for i, (args, arr) in enumerate(vecs):
        a = ', '.join(str(x) for x in args) or '0'
        calls.append(f'    {{ const int32_t A[] = {{{a}}};')
        if arr:
            calls.append(f'      const int32_t R[] = {{{", ".join(str(x) for x in arr)}}};')
            calls.append(f'      printf("%d\\n", f(A, R, {len(arr)})); }}')
        else:
            calls.append('      printf("%d\\n", f(A, 0, 0)); }')
    path = os.path.join(workdir, 'p.c')
    with open(path, 'w') as fh:
        fh.write(src + C_MAIN % '\n'.join(calls))
    exe = os.path.join(workdir, 'p')
    r = subprocess.run(['clang', '-O0', '-w', path, '-o', exe],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError('clang: ' + r.stderr.strip().split('\n')[0])
    out = subprocess.run([exe], capture_output=True, text=True, timeout=20)
    return [int(x) for x in out.stdout.split()]


def run_js(src, vecs, workdir):
    calls = []
    for args, arr in vecs:
        calls.append(f'console.log(f({list(args)}, {list(arr)}));')
    path = os.path.join(workdir, 'p.js')
    with open(path, 'w') as fh:
        fh.write(src + '\n' + '\n'.join(calls) + '\n')
    out = subprocess.run(['node', path], capture_output=True, text=True, timeout=20)
    if out.returncode:
        raise RuntimeError('node: ' + out.stderr.strip().split('\n')[0])
    return [int(x) for x in out.stdout.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--show', type=int, default=0, help='print N generated programs')
    a = ap.parse_args()

    rng = random.Random(a.seed)
    progs = [grammar.sample_program(rng) for _ in range(a.n)]
    targets = [('python', PyTarget()), ('c', CTarget()), ('javascript', JsTarget())]
    stats = {n: {'ok': 0, 'mismatch': 0, 'error': 0} for n, _ in targets}
    firsts = {n: None for n, _ in targets}

    with tempfile.TemporaryDirectory() as wd:
        for p in progs:
            vecs = vectors(p, rng)
            want = [evaluate(p, args=args, arr=arr) for args, arr in vecs]
            for name, tgt in targets:
                try:
                    src = tgt.compile(p)
                    got = (run_python(src, vecs) if name == 'python' else
                           run_c(src, vecs, wd) if name == 'c' else
                           run_js(src, vecs, wd))
                except Exception as exc:
                    stats[name]['error'] += 1
                    if firsts[name] is None:
                        firsts[name] = (p, f'{type(exc).__name__}: {exc}')
                    continue
                if got == want:
                    stats[name]['ok'] += 1
                else:
                    stats[name]['mismatch'] += 1
                    if firsts[name] is None:
                        bad = next(i for i in range(len(want)) if got[i] != want[i])
                        firsts[name] = (p, f'input {vecs[bad]} -> got {got[bad]}, oracle {want[bad]}')

    print(f'{a.n} programs, 8 input vectors each, compared against ir.evaluate\n')
    for name, _ in targets:
        s = stats[name]
        print(f'  {name:8} correct {s["ok"]:>4}/{a.n}   mismatch {s["mismatch"]:>3}   error {s["error"]:>3}')
        if firsts[name]:
            prog, why = firsts[name]
            print(f'           first failure: {grammar.shape_of(prog.body)}')
            print(f'           {why}')

    if a.show:
        for p in progs[:a.show]:
            print('\n' + '=' * 70)
            from ir import question
            print(question(p))
            for name, tgt in targets:
                print(f'--- {name} ---')
                print(tgt.compile(p))
    return 0 if all(stats[n]['ok'] == a.n for n, _ in targets) else 1


if __name__ == '__main__':
    sys.exit(main())
