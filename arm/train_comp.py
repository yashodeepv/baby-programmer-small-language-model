"""
train_comp.py - Train on the compositional corpus, score five ways.

Training excludes whole grammar cells and any expression deeper than
TRAIN_MAX_DEPTH, so the eval can ask two questions the old setup could not:

    seen    trained cells, new constants        -- comparable to the 86.5% before
    combo   held-out cells                      -- unseen mix of seen parts
    depth   deeper than any training example    -- structural extrapolation

A low combo or depth score is a finding, not a failure. It is the first honest
measurement of compositional generalisation in this project; the previous
corpus could not produce one at all.

    .venv/bin/python arm/train_comp.py --steps 4000
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch

from model import BabyProgrammer, save_checkpoint, device
from tokenizer import ArmTokenizer
from ir import question
from lower import lower
import split as splitmod
from eval_comp import evaluate, evaluate_sexpr, fmt, fmt_sexpr


def render(prog):
    body = '\n'.join(l if l.endswith(':') else '    ' + l for l in lower(prog))
    return f'USER: {question(prog)}\nASSISTANT:\n{body}\n    ret\n'


def render_sexpr(prog):
    """Target the meaning, not the code. lower() produces the assembly."""
    from sexpr import emit
    return f'USER: {question(prog)}\nASSISTANT:\n{emit(prog.body)}\n'


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, bs, bat, iters=30):
    model.eval()
    out = {}
    for name, data in splits.items():
        L = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, bs, bat)
            _, l = model(x, y)
            L[k] = l.item()
        out[name] = L.mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=24000)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--batch-size', type=int, default=32)
    # 384 suits the assembly target. For --target sexpr the training corpus
    # fits in 128, but the DEPTH eval reaches 204 tokens (depth-4 expressions
    # are long), so sizing on the training corpus alone would silently truncate
    # that eval and score it near zero for the wrong reason. 256 covers it.
    ap.add_argument('--block-size', type=int, default=384)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--eval-interval', type=int, default=500)
    ap.add_argument('--eval-n', type=int, default=80)
    # Checkpoints are named after the run. A shared fixed path means any later
    # run -- including a 30-second smoke test -- silently destroys the previous
    # model, which is exactly how the Phase 3 checkpoint was lost.
    ap.add_argument('--out', default=None,
                    help='defaults to checkpoints/<tag>.pth')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--n-embd', type=int, default=384)
    ap.add_argument('--n-head', type=int, default=6)
    ap.add_argument('--n-layer', type=int, default=6)
    ap.add_argument('--train-max-depth', type=int, default=3)
    ap.add_argument('--eval-depth', type=int, default=4)
    ap.add_argument('--tag', default='phase3')
    ap.add_argument('--target', choices=['asm', 'sexpr'], default='asm',
                    help='what the model learns to produce')
    ap.add_argument('--facts', type=int, default=2500,
                    help='factual ISA Q&A pairs to mix into training')
    args = ap.parse_args()
    if args.out is None:
        os.makedirs('checkpoints', exist_ok=True)
        args.out = os.path.join('checkpoints', f'{args.tag}.pth')
    if os.path.exists(args.out):
        raise SystemExit(f'{args.out} already exists -- pass a new --tag rather '
                         f'than overwriting a trained model')

    torch.manual_seed(args.seed)

    held = splitmod.choose_holdout(frac=0.15, seed=0)
    print(f'held-out grammar cells: {len(held)}')

    print(f'sampling {args.n_train} training programs (held-out cells excluded)...')
    train_progs = splitmod.train_programs(args.n_train, held, seed=args.seed,
                                         max_depth=args.train_max_depth)
    train_qs = {question(p) for p in train_progs}
    render_fn = render_sexpr if args.target == 'sexpr' else render
    blocks = [render_fn(p) for p in train_progs]

    # Factual Q&A is the other half of the goal: answer basic questions about
    # the ISA, not only write code for described computations.
    import facts as factmod
    if args.facts:
        blocks += factmod.sample_text(args.facts, seed=args.seed)
        print(f'  + {args.facts} factual Q&A pairs')

    text = '\n'.join(blocks)
    tok = ArmTokenizer.build(text)
    rng = random.Random(args.seed)
    rng.shuffle(blocks)
    cut = int(0.97 * len(blocks))
    enc = lambda bs: torch.tensor(tok.encode('\n'.join(bs)), dtype=torch.long)
    splits = {'train': enc(blocks[:cut]), 'val': enc(blocks[cut:])}

    print(f'  tokens  {len(splits["train"]):,} train / {len(splits["val"]):,} val')
    print(f'  vocab   {tok.vocab_size}')

    model = BabyProgrammer(tok.vocab_size, args.block_size,
                           n_embd=args.n_embd, n_head=args.n_head,
                           n_layer=args.n_layer).to(device)
    print(f'model     {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, '
          f'block_size={args.block_size}, on {device}')

    evals = {
        'seen':  splitmod.eval_seen(args.eval_n, held, train_qs,
                                    max_depth=args.train_max_depth),
        'combo': splitmod.eval_combo(args.eval_n, held),
        'depth': splitmod.eval_depth(args.eval_n, depth=args.eval_depth,
                                     train_depth=args.train_max_depth),
        # Same programs, longer arrays than training ever showed. The correct
        # code is identical at every length, so a gap here means the model
        # learned a length rather than an algorithm.
        'size':  splitmod.eval_size(args.eval_n, held),
        # trained cells, trained constants -- only the WORDING is new
        'para':  splitmod.eval_paraphrase(args.eval_n, held, train_qs),
    }
    for k, v in evals.items():
        print(f'  eval[{k}]  {len(v)} programs')
    print()

    fact_eval = factmod.eval_set(40) if args.facts else []
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, start = -1, time.time()
    last_path = args.out.replace('.pth', '.last.pth')

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0:
            L = estimate_loss(model, splits, args.block_size, args.batch_size)
            scorer = evaluate_sexpr if args.target == 'sexpr' else evaluate
            st = {k: scorer(model, tok, v,
                            sizes=splitmod.TEST_SIZES if k == 'size' else None)[0]
                  for k, v in evals.items()}
            mins = (time.time() - start) / 60
            fact_acc, _ = (factmod.score(model, tok, fact_eval)
                           if args.facts else (0.0, []))
            line = (f'step {step:>5} | val {L["val"]:.3f} | ' +
                    ' | '.join(
                        f'{k} {100*s.get("equivalent", s.get("correct", 0))/max(s["n"],1):5.1f}%'
                        for k, s in st.items()) +
                    (f' | facts {100*fact_acc:5.1f}%' if args.facts else '') +
                    f' | {mins:.1f}m')
            key = 'equivalent' if args.target == 'sexpr' else 'correct'
            if st['seen'][key] > best:
                best = st['seen'][key]
                save_checkpoint(args.out, model, tok.stoi, tok.itos, tok.vocab_size)
                line += '  <- best'
            save_checkpoint(last_path, model, tok.stoi, tok.itos, tok.vocab_size)
            print(line, flush=True)
            with open(f'data/{args.tag}_log.jsonl', 'a') as f:
                f.write(json.dumps({'step': step, 'tag': args.tag,
                                    'params': sum(p.numel() for p in model.parameters()),
                                    'val': L['val'],
                                    **{k: s for k, s in st.items()}}) + '\n')

        if step == args.steps:
            break
        x, y = get_batch(splits['train'], args.block_size, args.batch_size)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(f'\nbest seen-cell score saved to {args.out}')
    print('\n--- final, in detail ---')
    for k, v in evals.items():
        if args.target == 'sexpr':
            st, _ = evaluate_sexpr(model, tok, v,
                                   sizes=splitmod.TEST_SIZES if k == 'size' else None)
            print(f'  {k:6} {fmt_sexpr(st)}')
        else:
            st, _ = evaluate(model, tok, v,
                             sizes=splitmod.TEST_SIZES if k == 'size' else None)
            print(f'  {k:6} {fmt(st)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
