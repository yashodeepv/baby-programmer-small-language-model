"""catalog.py - A system catalog: the triggers, services and entities a flow may use.

English does not say what a payload looks like or what a service returns, so
neither the sampler nor the flow encoder invents them. They come from a
catalog -- the stand-in for a real project's declared APIs -- and a flow
picks from it by name. `catalogs/*.json` holds one per business domain.

Besides the contracts, a catalog carries the value pools that make sampled
flows plausible: which strings a field takes (tier: gold/silver/standard),
what range an int field lives in (days_late: 0-120), and who can be notified.
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List

from flow.ir2.graph import Signature
from flow.ir2.types import RecordT, parse_type

CATALOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catalogs')

DEFAULT_INT_RANGE = (0, 1000)
DEFAULT_STRINGS = ['alpha', 'beta', 'gamma']


@dataclass
class TriggerSpec:
    source: str
    config: Dict[str, str]
    noun: str
    payload: RecordT


@dataclass
class Catalog:
    name: str
    triggers: List[TriggerSpec]
    services: Dict[str, Dict[str, Signature]]
    entities: Dict[str, RecordT]
    strings: Dict[str, List[str]]
    ints: Dict[str, tuple]
    recipients: Dict[str, List[str]]

    def int_range(self, field):
        return tuple(self.ints.get(field, DEFAULT_INT_RANGE))

    def string_pool(self, field):
        return self.strings.get(field, DEFAULT_STRINGS)


def load_catalog(obj):
    return Catalog(
        name=obj['name'],
        triggers=[TriggerSpec(t['source'], dict(t['config']), t['noun'], parse_type(t['payload']))
                  for t in obj['triggers']],
        services={s: {o: Signature(parse_type(sig['input']), parse_type(sig['output']))
                      for o, sig in ops.items()}
                  for s, ops in obj['services'].items()},
        entities={e: parse_type(t) for e, t in obj['entities'].items()},
        strings={k: list(v) for k, v in obj.get('strings', {}).items()},
        ints={k: tuple(v) for k, v in obj.get('ints', {}).items()},
        recipients={k: list(v) for k, v in obj.get('recipients', {}).items()},
    )


def all_catalogs():
    """Every catalog in catalogs/, keyed by name."""
    out = {}
    for path in sorted(glob.glob(os.path.join(CATALOG_DIR, '*.json'))):
        with open(path) as f:
            c = load_catalog(json.load(f))
        out[c.name] = c
    return out
