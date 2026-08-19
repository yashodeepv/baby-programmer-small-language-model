"""Score an existing checkpoint on the current eval sets. No training."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import load_checkpoint, device
from tokenizer import ArmTokenizer
from ir import question
import split as splitmod
from eval_comp import evaluate, evaluate_sexpr, fmt, fmt_sexpr

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--target', choices=['asm', 'sexpr'], default='asm')
ap.add_argument('--eval-n', type=int, default=200)
ap.add_argument('--n-train', type=int, default=24000)
ap.add_argument('--seed', type=int, default=1337)
ap.add_argument('--train-max-depth', type=int, default=3)
ap.add_argument('--eval-depth', type=int, default=4)
ap.add_argument('--cells', default='seen,combo,depth,size,para')
ap.add_argument('--json-out', default=None)
ap.add_argument('--force-style', type=int, default=None,
                help='rewrite every eval program to this wording style')
a = ap.parse_args()

model, stoi, itos, vocab_size = load_checkpoint(a.ckpt)
model.eval()
tok = ArmTokenizer([itos[i] if i in itos else itos[str(i)]
                    for i in range(vocab_size)])
print(f'{a.ckpt}: {sum(p.numel() for p in model.parameters()):,} params, '
      f'vocab {vocab_size}, block {model.block_size}, target {a.target}')

held = splitmod.choose_holdout(frac=0.15, seed=0)
train_progs = splitmod.train_programs(a.n_train, held, seed=a.seed,
                                      max_depth=a.train_max_depth)
train_qs = {question(p) for p in train_progs}

allsets = {
    'seen':  lambda: splitmod.eval_seen(a.eval_n, held, train_qs,
                                        max_depth=a.train_max_depth),
    'combo': lambda: splitmod.eval_combo(a.eval_n, held),
    'depth': lambda: splitmod.eval_depth(a.eval_n, depth=a.eval_depth,
                                         train_depth=a.train_max_depth),
    'size':  lambda: splitmod.eval_size(a.eval_n, held),
    'para':  lambda: splitmod.eval_paraphrase(a.eval_n, held, train_qs),
}
want = [c for c in a.cells.split(',') if c]
scorer = evaluate_sexpr if a.target == 'sexpr' else evaluate
show = fmt_sexpr if a.target == 'sexpr' else fmt

out = {}
for k in want:
    progs = allsets[k]()
    if a.force_style is not None:
        for _p in progs:
            _p.style = a.force_style
    t0 = time.time()
    st, _ = scorer(model, tok, progs,
                   sizes=splitmod.TEST_SIZES if k == 'size' else None)
    out[k] = st
    print(f'  {k:6} {show(st)}   [{time.time()-t0:.0f}s]', flush=True)

if a.json_out:
    with open(a.json_out, 'w') as f:
        json.dump({'ckpt': a.ckpt, 'target': a.target, 'eval_n': a.eval_n,
                   **out}, f, indent=2)
