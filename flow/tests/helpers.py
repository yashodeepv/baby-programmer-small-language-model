"""Shared test helpers: load the examples, and mutate them one fault at a time."""

import copy
import glob
import json
import os

from flow.ir2.io import load_flow

EXAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         'examples', '*.json')))


def raw(name='order_checkout'):
    path = next(p for p in EXAMPLES if os.path.basename(p) == f'{name}.json')
    with open(path) as f:
        return json.load(f)


def step(r, sid):
    return next(s for s in r['steps'] if s['id'] == sid)


def mutated(fn, name='order_checkout'):
    """A copy of an example with `fn(raw_dict)` applied, loaded as a Flow."""
    r = copy.deepcopy(raw(name))
    fn(r)
    return load_flow(r)
