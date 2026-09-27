"""build.py - Generate, verify and write a flow corpus.

    python3.11 -m flow.gen.build --n 5000 --seed 0 --out data/flow_corpus.jsonl

Per flow: sample -> check -> invent scenarios until every edge is taken ->
render in one style -> derive labels. A flow that the coverage search cannot
fully exercise is dropped, and the reason counted.

One JSON object per line:

    id, catalog, patterns, n_steps, style, split
    flow        the IR2 JSON, each step's `intent` set to its clause
    scenarios   inputs, mocks and expected outcomes -- the test suite
    m1          flow-encoder example (labels.py)
    m2          logic-model examples, one per expression

`split` is decided here, so held-out data never leaks into training:

    catalog     every flow from --hold-out-catalog (default logistics)
    style       rendered in the held-out wording style
    val         a random 5% of the rest
    train       everything else
"""

import argparse
import collections
import json
import os
import random
import sys

from flow.gen.catalog import all_catalogs
from flow.gen.coverage import synthesize
from flow.gen.labels import m1_example, m2_examples
from flow.gen.render import HELD_OUT_STYLE, N_STYLES, render
from flow.gen.sampler import sample_flows
from flow.ir2.io import dump_flow


def mode_label(flow, stated):
    """What the encoder should answer for sync/async.

    Stated in the text -> the mode. Not stated, but only an http flow can be
    sync and only a sync flow answers the caller -> inferable, so the mode.
    Not stated and nothing implies it -> abstain: the model should ask.
    """
    trig = flow.trigger()
    if stated or trig.source != 'http':
        return flow.mode
    responds = any(getattr(s, 'response', None) for s in flow.steps)
    return 'sync' if responds else 'abstain'


def build_one(rng, flow, meta, cat, hold_out_catalog, state_p=0.75):
    scenarios, cov = synthesize(rng, flow, cat)
    if cov < 1.0:
        return None, 'partial coverage'
    style = rng.randrange(N_STYLES)
    stated = flow.trigger().source == 'http' and rng.random() < state_p
    text, spans = render(flow, style, meta['noun'], state_mode=stated)
    for sp in spans:
        if sp['role'] == 'clause':
            flow.step(sp['step']).intent = text[sp['start']:sp['end']]
    flow.intent = text
    if meta['catalog'] == hold_out_catalog:
        split = 'catalog'
    elif style == HELD_OUT_STYLE:
        split = 'style'
    else:
        split = 'val' if rng.random() < 0.05 else 'train'
    return {
        'id': flow.name,
        'catalog': meta['catalog'],
        'patterns': meta['patterns'],
        'n_steps': meta['n_steps'],
        'style': style,
        'split': split,
        'flow': dump_flow(flow),
        'scenarios': scenarios,
        'm1': m1_example(flow, text, spans, mode_label(flow, stated)),
        'm2': m2_examples(flow, text, spans, style),
    }, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=1000, help='flows to sample (some are dropped)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/flow_corpus.jsonl')
    ap.add_argument('--hold-out-catalog', default='logistics')
    ap.add_argument('--max-steps', type=int, default=16)
    a = ap.parse_args(argv)

    cats = all_catalogs()
    if a.hold_out_catalog not in cats:
        sys.exit(f'no catalog {a.hold_out_catalog!r} (have: {", ".join(cats)})')
    rng = random.Random(a.seed + 1)
    kept, dropped = collections.Counter(), collections.Counter()
    n_m2 = 0
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as out:
        for flow, meta in sample_flows(a.n, cats, seed=a.seed, max_steps=a.max_steps):
            rec, why = build_one(rng, flow, meta, cats[meta['catalog']], a.hold_out_catalog)
            if rec is None:
                dropped[why] += 1
                continue
            kept[rec['split']] += 1
            n_m2 += len(rec['m2'])
            out.write(json.dumps(rec) + '\n')

    total = sum(kept.values())
    print(f'wrote {total} flows, {n_m2} logic examples to {a.out}')
    print('  splits: ' + ', '.join(f'{k} {v}' for k, v in sorted(kept.items())))
    if dropped:
        print('  dropped: ' + ', '.join(f'{k} {v}' for k, v in dropped.items()))


if __name__ == '__main__':
    main()
