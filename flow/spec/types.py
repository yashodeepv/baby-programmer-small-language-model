"""types.py - Baby Spec type syntax -> IR2 types.

    int | bool | str | list of <type> | (<field> <type>, ...) | <entity name>
"""

import re

from flow.ir2.types import BOOL, INT, STR, ListT, RecordT
from flow.spec.text import split_top

_PRIM = {'int': INT, 'bool': BOOL, 'str': STR}
_NAME = re.compile(r'[a-z][a-z0-9_]*')


class TypeSyntaxError(ValueError):
    pass


def parse_type(text, entities):
    t = text.strip()
    if t in _PRIM:
        return _PRIM[t]
    if t.startswith('list of '):
        return ListT(parse_type(t[len('list of '):], entities))
    if t.startswith('(') and t.endswith(')'):
        return RecordT(parse_fields(t[1:-1], entities))
    if t in entities:
        return entities[t]
    raise TypeSyntaxError(f'unknown type "{t}" (int, bool, str, list of T, (field type, ...), or an entity)')


def parse_fields(text, entities):
    """'order_id str, items list of (sku str, amount int)' -> ((name, type), ...)"""
    if not text.strip():
        return ()
    out, seen = [], set()
    for part in split_top(text, ','):
        bits = part.split(None, 1)
        if len(bits) != 2 or not _NAME.fullmatch(bits[0]):
            raise TypeSyntaxError(f'expected "<field> <type>", got "{part}"')
        if bits[0] in seen:
            raise TypeSyntaxError(f'field "{bits[0]}" appears twice')
        seen.add(bits[0])
        out.append((bits[0], parse_type(bits[1], entities)))
    return tuple(out)
