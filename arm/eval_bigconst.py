"""Score a checkpoint on range loops whose bounds sit OUTSIDE the trained
distribution. grammar.py:179-180 samples hi = a+span with a<=40, span<=60, so
training never shows hi > 100. Everything else is the grammar's own wording."""
import argparse
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grammar
from ir import Program, Loop, Const, question
from grammar import shape_of
from model import load_checkpoint
from tokenizer import ArmTokenizer
from eval_comp import evaluate, fmt

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--lo-max', type=int, default=40)
ap.add_argument('--hi-min', type=int, default=101)
ap.add_argument('--hi-max', type=int, default=400)
ap.add_argument('--in-dist', action='store_true',
                help='control: sample hi in the TRAINED range instead')
a = ap.parse_args()

rng = random.Random(4242)
progs = []
while len(progs) < a.n:
    op = rng.choice(grammar.REDUCE)
    if op == 'product':
        continue                      # overflows instantly at these bounds
    lo = rng.randint(1, a.lo_max)
    hi = (lo + rng.randint(2, 60) if a.in_dist
          else rng.randint(max(a.hi_min, lo + 2), a.hi_max))
    body = Loop(op, Const(lo), Const(hi))
    progs.append(Program(body, n_args=0, shape=shape_of(body),
                         phrase=rng.randrange(16), style=0))

model, stoi, itos, _ = load_checkpoint(a.ckpt)
tok = ArmTokenizer([itos[i] for i in range(len(itos))])
st, texts = evaluate(model, tok, progs)
band = ('trained band hi<=100' if a.in_dist
        else f'hi in [{a.hi_min},{a.hi_max}] -- outside training')
print(f'{a.ckpt}  {band}')
print(f'   {fmt(st)}')
bad = [(question(p), t.strip().split(chr(10))[0:3])
       for p, t in zip(progs, texts)][:2]
for q, t in bad:
    print(f'   e.g. {q[:78]}')
    print(f'        emitted {t}')
