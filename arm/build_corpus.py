"""
build_corpus.py - Generate, verify and write the compositional corpus.

Nothing is written until every program has been executed against every one of
its argument vectors. A sample that fails is a generator bug, not data.

Alongside the corpus it writes a manifest recording which grammar cell each
example came from. Phase 3 splits on that: hold out whole cells so the eval
faces unseen COMBINATIONS of seen parts, rather than unseen constants.

    .venv/bin/python arm/build_corpus.py --n 20000 --out data/comp_corpus.txt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grammar
from ir import question
from lower import lower
from verify_ir import verify


def render(prog):
    body = '\n'.join(l if l.endswith(':') else '    ' + l for l in lower(prog))
    return f'USER: {question(prog)}\nASSISTANT:\n{body}\n    ret\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='data/comp_corpus.txt')
    ap.add_argument('--manifest', default='data/comp_shapes.json')
    args = ap.parse_args()

    print(f'sampling {args.n} programs...')
    progs = grammar.sample(args.n, seed=args.seed)

    print('verifying every program on every input...')
    results, errors = verify(progs)
    bad = [(p, r) for p, r in zip(progs, results) if not (r[1] and r[0] == r[1])]
    checks_ok = sum(o for o, _ in results)
    checks = sum(n for _, n in results)
    print(f'  {len(progs) - len(bad)}/{len(progs)} programs correct on all inputs '
          f'({checks_ok}/{checks} checks)')
    for p, r in bad[:5]:
        print(f'  FAIL [{p.shape}] {r}')
        print(f'    {question(p)}')
    for base, e in errors[:3]:
        print(f'  BUILD ERROR near {base}: {e}')
    if bad or errors:
        print('\nAborting: the generator produced programs it cannot justify.')
        return 1

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        for p in progs:
            f.write(render(p) + '\n')

    shapes = [p.shape for p in progs]
    cells  = [grammar.cell_of(p) for p in progs]
    with open(os.path.abspath(args.manifest), 'w') as f:
        json.dump({'shapes': shapes, 'cells': cells}, f)

    text = open(out).read()
    from tokenizer import ArmTokenizer
    tok = ArmTokenizer.build(text)
    lens = sorted(len(tok.encode(b)) for b in text.strip().split('\n\n'))
    uniq = len(set(shapes))
    ncell = len(set(cells))
    qs = len({b.split('\nASSISTANT:')[0] for b in text.strip().split('\n\n')})

    print(f'\nwrote {out}')
    print(f'  examples          {len(progs):,}')
    print(f'  distinct shapes   {uniq:,}   (fine structural signature)')
    print(f'  grammar cells     {ncell:,}   (split unit for Phase 3)')
    print(f'  distinct questions{qs:>8,}')
    print(f'  characters        {len(text):,}')
    print(f'  vocab             {tok.vocab_size}')
    print(f'  tokens/example    median {lens[len(lens)//2]}  '
          f'p99 {lens[int(.99*len(lens))]}  max {lens[-1]}')
    over = sum(1 for l in lens if l > 256)
    print(f'  over BLOCK_SIZE=256: {over} ({100*over/len(lens):.2f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
