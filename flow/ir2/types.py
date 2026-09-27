"""types.py - Value types that flow between steps.

Deliberately small: int, bool, str, list of T, and records. `int` keeps the
arm IR's meaning -- signed 32-bit, wrapping -- so an expression means the same
thing inside a flow as it does in the assembly corpus. Money is therefore held
in minor units (cents), and totals above ~21 million in major units wrap.

JSON form, as written in a flow file:

    "int" | "bool" | "str"
    ["list", <type>]
    {"field": <type>, ...}          a record
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Prim:
    name: str                       # int bool str

    def __str__(self):
        return self.name


INT, BOOL, STR = Prim('int'), Prim('bool'), Prim('str')

# The type of a value whose definition already has a reported error. Anything
# that reads it is not reported again, so one mistake shows up once, at its
# source, instead of at every step downstream.
UNKNOWN = Prim('unknown')
_PRIMS = {'int': INT, 'bool': BOOL, 'str': STR}


@dataclass(frozen=True)
class ListT:
    elem: object

    def __str__(self):
        return f'list[{self.elem}]'


@dataclass(frozen=True)
class RecordT:
    fields: Tuple                   # ((name, type), ...) in declaration order

    def field(self, name) -> Optional[object]:
        for n, t in self.fields:
            if n == name:
                return t
        return None

    def names(self):
        return [n for n, _ in self.fields]

    def __str__(self):
        return '{' + ', '.join(f'{n}: {t}' for n, t in self.fields) + '}'


def record(**fields):
    return RecordT(tuple(fields.items()))


def parse_type(obj):
    if isinstance(obj, str):
        if obj not in _PRIMS:
            raise ValueError(f'unknown type {obj!r}')
        return _PRIMS[obj]
    if isinstance(obj, list):
        if len(obj) != 2 or obj[0] != 'list':
            raise ValueError(f'a list type is ["list", T], got {obj!r}')
        return ListT(parse_type(obj[1]))
    if isinstance(obj, dict):
        return RecordT(tuple((k, parse_type(v)) for k, v in obj.items()))
    raise ValueError(f'not a type: {obj!r}')


def dump_type(t):
    if isinstance(t, Prim):
        return t.name
    if isinstance(t, ListT):
        return ['list', dump_type(t.elem)]
    if isinstance(t, RecordT):
        return {n: dump_type(ft) for n, ft in t.fields}
    raise TypeError(f'not a type: {t!r}')


def conforms(value, t):
    """Does a runtime value have type t? bool is not an int here."""
    if t == INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if t == BOOL:
        return isinstance(value, bool)
    if t == STR:
        return isinstance(value, str)
    if isinstance(t, ListT):
        return isinstance(value, list) and all(conforms(v, t.elem) for v in value)
    if isinstance(t, RecordT):
        return (isinstance(value, dict)
                and set(value) == set(t.names())
                and all(conforms(value[n], ft) for n, ft in t.fields))
    return False
