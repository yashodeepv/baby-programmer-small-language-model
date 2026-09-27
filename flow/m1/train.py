"""train.py - Train the flow encoder.

    .venv/bin/python -m flow.m1.train --data data/flow_book.jsonl --out checkpoints/m1_poc

Trains on the `train` split, reports on `val` every epoch, and at the end on
the held-out splits: `style` (wording never read) and `catalog` (a business
domain never seen). `--order chapter` walks the book in chapter order within
each epoch instead of shuffling.
"""

import argparse
import json
import os
import time

import torch

from flow.m1.data import batches, collate, load_records, make_examples
from flow.m1.encoder import BertEncoder, build_encoder
from flow.m1.evaluate import evaluate, report
from flow.m1.model import FlowEncoder, loss


def device_of(name):
    if name != 'auto':
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to(batch, device):
    return {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
            for k, v in batch.items()}


def save(model, path, meta):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    model.encoder.save(path)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump({'encoder': model.encoder.config(), **meta}, f, indent=2)


def load(path, device):
    with open(os.path.join(path, 'config.json')) as f:
        cfg = json.load(f)
    model = FlowEncoder(build_encoder(cfg['encoder']))
    model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location='cpu'))
    return model.to(device), cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='checkpoints/m1_poc')
    ap.add_argument('--encoder', default='bert-base-cased')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-5, help='encoder learning rate')
    ap.add_argument('--head-lr', type=float, default=1e-3)
    ap.add_argument('--order', choices=['shuffle', 'chapter'], default='shuffle')
    ap.add_argument('--max-train', type=int, default=0, help='cap training examples (0 = all)')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    device = device_of(a.device)
    enc = BertEncoder(a.encoder)
    model = FlowEncoder(enc).to(device)

    records = load_records(a.data)
    split = lambda s: [r for r in records if r['split'] == s]
    train, skipped = make_examples(split('train')[:a.max_train or None], enc)
    val, _ = make_examples(split('val'), enc)
    if a.order == 'chapter':
        train.sort(key=lambda ex: ex.record.get('chapter', 0))
    print(f'device {device}  encoder {a.encoder}  train {len(train)} (skipped {skipped} too long)  val {len(val)}')

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{'params': enc_params, 'lr': a.lr},
                             {'params': head_params, 'lr': a.head_lr}], weight_decay=0.01)
    steps_total = a.epochs * ((len(train) + a.batch_size - 1) // a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.head_lr], total_steps=steps_total,
                                                pct_start=0.1, anneal_strategy='linear')
    step, t0 = 0, time.time()
    for epoch in range(a.epochs):
        run_loss = {}
        for batch in batches(train, a.batch_size, shuffle=(a.order == 'shuffle'), seed=a.seed + epoch):
            t = to(collate(batch, enc.pad_id()), device)
            out = model(t['input_ids'], t['attention_mask'], t['slot_pos'])
            total, parts = loss(out, t)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in parts.items():
                run_loss[k] = 0.98 * run_loss.get(k, v) + 0.02 * v
            if step % 50 == 0:
                print(f'  step {step}/{steps_total}  {time.time() - t0:6.0f}s  '
                      + '  '.join(f'{k} {v:.3f}' for k, v in run_loss.items()), flush=True)
        res = evaluate(model, val, device, enc.pad_id(), e2e=True)
        print(f'epoch {epoch + 1}  val:\n' + report(res, ['all']), flush=True)

    save(model, a.out, {'epochs': a.epochs, 'train_examples': len(train), 'data': a.data,
                        'seconds': round(time.time() - t0)})
    held = {s: make_examples(split(s), enc)[0] for s in ('val', 'style', 'catalog')}
    final = {}
    for s, exs in held.items():
        r = evaluate(model, exs, device, enc.pad_id())
        final[f'split:{s}'] = r['all']
    print('\nheld-out:\n' + report(final))
    with open(os.path.join(a.out, 'results.json'), 'w') as f:
        json.dump(final, f, indent=2)
    print(f'saved {a.out}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
