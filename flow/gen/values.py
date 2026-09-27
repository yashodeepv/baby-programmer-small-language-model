"""values.py - Random runtime values of a type, shaped by the catalog.

Used twice: by the sampler, to estimate what an expression typically
evaluates to (so a threshold lands where it splits real data), and by the
coverage search, to invent inputs and service replies.
"""

from flow.ir2.types import BOOL, INT, STR, ListT, RecordT


def random_value(rng, t, cat, field=None, list_len=(0, 4)):
    if t == INT:
        lo, hi = cat.int_range(field)
        return rng.randint(lo, hi)
    if t == BOOL:
        return rng.random() < 0.5
    if t == STR:
        if field in cat.strings:
            return rng.choice(cat.strings[field])
        if field and field.endswith('email'):
            return rng.choice(['ann', 'bo', 'cy', 'di']) + '@example.com'
        if field and field.endswith('_id'):
            return f'{field[:2].upper()}-{rng.randint(1, 999)}'
        return rng.choice(['alpha', 'beta', 'gamma'])
    if isinstance(t, ListT):
        return [random_value(rng, t.elem, cat, field) for _ in range(rng.randint(*list_len))]
    if isinstance(t, RecordT):
        return {n: random_value(rng, ft, cat, n) for n, ft in t.fields}
    raise TypeError(f'no random value for {t}')
