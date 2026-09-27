"""expr.py - Step logic: the arm IR, extended.

Every arm IR node is an IR2 expression and keeps its exact semantics:
Const, Bin, Un, Cmp, Sel, Loop, ArrLoop, Arr, Index. What a flow adds:

    Ref(('input', 'total'))     a named value: the trigger payload, or the
                                output of an earlier step, then field access
    Str('gold'), Bool(True)     literals beyond int
    Not / All / Any             boolean logic (arm's `and`/`orr` are bitwise)
    Len(xs)                     length of any list
    Pluck(xs, 'price')          list of records -> list of one field, so the
                                arm reductions (sum, max, count :gt k, ...)
                                apply to business data unchanged
    Fmt(parts)                  string built from literals and values

Two binary operators are new, `div` and `mod`, both truncating toward zero
(Java / C semantics). Integer division by zero is a runtime error in a flow.
The assembly model has never been trained on either.

The register-style leaves -- Arg (w0..), InArr, InLen -- remain parseable so
the whole arm corpus is valid IR2 text, but a flow rejects them: a step reads
named fields, not registers.
"""

from dataclasses import dataclass
from typing import Tuple

from flow.arm_bridge import arm_ir

# The arm IR, re-exported unchanged.
Const, Arg, Bin, Un, Cmp, Sel = (arm_ir.Const, arm_ir.Arg, arm_ir.Bin,
                                  arm_ir.Un, arm_ir.Cmp, arm_ir.Sel)
Loop, Arr, InArr, InLen, Index, ArrLoop = (arm_ir.Loop, arm_ir.Arr, arm_ir.InArr,
                                           arm_ir.InLen, arm_ir.Index, arm_ir.ArrLoop)

ARM_BINOPS = ('add', 'sub', 'mul', 'and', 'orr', 'eor', 'lsl', 'lsr')
IR2_BINOPS = ('div', 'mod')
BINOPS = ARM_BINOPS + IR2_BINOPS
UNOPS = ('neg', 'abs')
CMPOPS = ('eq', 'ne', 'lt', 'le', 'gt', 'ge')
REDUCE = ('sum', 'product', 'count', 'min', 'max')

LEGACY = (Arg, InArr, InLen)        # register leaves: parseable, not allowed in a flow


@dataclass(frozen=True)
class Ref:
    path: Tuple                     # ('input',) or ('pricing', 'total')

    def __str__(self):
        return '.'.join(self.path)


@dataclass(frozen=True)
class Str:
    v: str


@dataclass(frozen=True)
class Bool:
    v: bool


@dataclass(frozen=True)
class Not:
    a: object


@dataclass(frozen=True)
class All:
    items: Tuple


@dataclass(frozen=True)
class Any:
    items: Tuple


@dataclass(frozen=True)
class Len:
    a: object


@dataclass(frozen=True)
class Pluck:
    a: object
    field: str


@dataclass(frozen=True)
class Fmt:
    parts: Tuple


def children(node):
    """Direct sub-expressions, for walks that do not care about node kind."""
    out = []
    for f in getattr(node, '__dataclass_fields__', {}):
        v = getattr(node, f)
        if isinstance(v, tuple):
            out.extend(x for x in v if hasattr(x, '__dataclass_fields__'))
        elif hasattr(v, '__dataclass_fields__'):
            out.append(v)
    return out


def refs(node):
    """Every Ref in an expression."""
    found, stack = [], [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Ref):
            found.append(n)
        stack.extend(children(n))
    return found


@dataclass(frozen=True)
class Invalid:
    """Expression text that did not parse.

    The loader keeps it instead of refusing the whole flow, so an editor can
    still open a flow with one bad step and show the error on that step. The
    checker reports every Invalid; nothing runs or compiles while one remains.
    """
    text: str
    error: str
