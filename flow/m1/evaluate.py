"""evaluate.py - How well does the flow encoder read requests?

Structure, field by field (slots compared in canonical order):

    kinds       every slot's kind right, EMPTY included
    edges       the exact edge set
    settings    every setting of every step
    copies      every copied value, character for character
    spans       every expression's English span, exactly
    mode        sync / async / abstain
    structure   all of the above except mode

and end to end: the predicted flow, with oracle step logic (decode.py),
passes the checker and reproduces every stored scenario's outcome: status,
response and side effects. Step names and paths differ by construction, so
they aren't compared.
"""

import collections

import torch

from flow.check import check
from flow.m1.data import batches, collate
from flow.m1.decode import build_flow, gold, predicted
from flow.m1.vocab import VALUE_ROLES
from flow.sim import SimError, run, summary


def replays(flow, record):
    if not check(flow).ok:
        return False
    for sc in record['scenarios'].values():
        try:
            res = run(flow, sc['input'], sc.get('mocks', {}))
        except SimError:
            return False
        want = sc['expect']
        if res.status != want['status'] or res.response != want.get('response'):
            return False
        if [summary(e) for e in res.effects] != want['effects']:
            return False
    return True


def compare(p, gd):
    ps, gs = p['slots'], gd['slots']
    vals = lambda s: {r: v for r, v in s['spans'].items() if r in VALUE_ROLES}
    exprs = lambda s: {r: v for r, v in s['spans'].items() if r not in VALUE_ROLES}
    real = [i for i, s in enumerate(gs) if s['kind'] != 'empty']
    m = {
        'kinds': all(a['kind'] == b['kind'] for a, b in zip(ps, gs)),
        'edges': p['edges'] == gd['edges'],
        'settings': all(ps[i]['settings'] == gs[i]['settings'] for i in real),
        'copies': all(vals(ps[i]) == vals(gs[i]) for i in real),
        'spans': all(exprs(ps[i]) == exprs(gs[i]) for i in real),
        'mode': p['mode'] == gd['mode'],
    }
    m['structure'] = all(m[k] for k in ('kinds', 'edges', 'settings', 'copies', 'spans'))
    return m


@torch.no_grad()
def evaluate(model, examples, device, pad_id, batch_size=16, e2e=True):
    model.eval()
    by = collections.defaultdict(lambda: collections.Counter())
    for batch in batches(examples, batch_size, shuffle=False):
        t = collate(batch, pad_id)
        out = model(t['input_ids'].to(device), t['attention_mask'].to(device), t['slot_pos'].to(device))
        out = {k: ({kk: vv.cpu() for kk, vv in v.items()} if isinstance(v, dict) else v.cpu())
               for k, v in out.items()}
        for b, ex in enumerate(batch):
            p = predicted(out, b, ex)
            m = compare(p, gold(ex))
            if e2e:
                m['end_to_end'] = replays(build_flow(p, ex.record), ex.record)
            for key in ('all', f"split:{ex.record['split']}", f"chapter:{ex.record.get('chapter', 0)}"):
                by[key]['n'] += 1
                for k, v in m.items():
                    by[key][k] += bool(v)
    model.train()
    return {k: {m: c[m] / c['n'] for m in c if m != 'n'} | {'n': c['n']} for k, c in by.items()}


def report(results, keys=None):
    cols = ['n', 'kinds', 'edges', 'settings', 'copies', 'spans', 'mode', 'structure', 'end_to_end']
    lines = [f"{'':14}" + ''.join(f'{c:>11}' for c in cols)]
    for k in keys or sorted(results):
        r = results[k]
        cells = [f"{r['n']:>11}"] + [f'{100 * r[c]:>10.1f}%' if c in r else f"{'-':>11}" for c in cols[1:]]
        lines.append(f'{k:14}' + ''.join(cells))
    return '\n'.join(lines)
