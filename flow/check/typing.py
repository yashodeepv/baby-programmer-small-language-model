"""typing.py - The type of an expression, given the names in scope.

    const -> int      "s" -> str      true -> bool
    add sub mul div mod and orr eor lsl lsr : int, int -> int
    neg abs : int -> int
    lt le gt ge : int, int -> bool      eq ne : T, T -> bool (T int/str/bool)
    sel : bool, T, T -> T
    not all any : bool... -> bool
    len : list[T] -> int
    pluck : list[record], field -> list[field type]
    sum product min max over list[int] (or a range) -> int
    count over any list -> int; with a map or predicate, list[int] only
    lit : int... -> list[int]      idx : list[T], int -> T
    fmt : (int | str | bool)... -> str

The arm register leaves (a0, (arr), (len)) are rejected with a hint: a step
reads named fields.
"""

from flow.ir2.expr import (All, Any, Arg, Arr, ArrLoop, Bin, Bool, Cmp, Const,
                           Fmt, Index, InArr, InLen, Invalid, Len, Loop, Not,
                           Pluck, Ref, Sel, Str, Un)
from flow.ir2.sexpr import emit
from flow.ir2.types import BOOL, INT, STR, UNKNOWN, ListT, RecordT


class TypeErr(Exception):
    def __init__(self, message, code='type-mismatch'):
        super().__init__(message)
        self.code = code


def _show(node):
    try:
        return emit(node)
    except Exception:
        return repr(node)


def _want(t, want, node, what):
    if t != want:
        raise TypeErr(f'{what} must be {want}, but {_show(node)} is {t}')


def type_of(node, scope):
    """`scope` maps top-level names (input, step ids) to types."""
    T = lambda n: type_of(n, scope)

    if isinstance(node, Invalid):
        raise TypeErr(f'cannot parse {node.text!r}: {node.error}', 'parse-error')
    if isinstance(node, Const):
        return INT
    if isinstance(node, Str):
        return STR
    if isinstance(node, Bool):
        return BOOL
    if isinstance(node, (Arg, InArr, InLen)):
        raise TypeErr(f'{_show(node)} is a register from the assembly IR; '
                      'refer to a named field instead, e.g. input.total', 'register-leaf')

    if isinstance(node, Ref):
        head = node.path[0]
        if head not in scope:
            avail = ', '.join(sorted(scope)) or 'nothing'
            raise TypeErr(f'{head!r} is not available here (available: {avail})', 'unknown-ref')
        t = scope[head]
        for i, name in enumerate(node.path[1:], 1):
            if t == UNKNOWN:
                break
            if not isinstance(t, RecordT):
                raise TypeErr(f'{".".join(node.path[:i])} is {t}, it has no field {name!r}',
                              'unknown-field')
            ft = t.field(name)
            if ft is None:
                raise TypeErr(f'{".".join(node.path[:i])} has no field {name!r} '
                              f'(fields: {", ".join(t.names())})', 'unknown-field')
            t = ft
        if t == UNKNOWN:
            raise TypeErr(f'{node} depends on a step with an error', 'suppressed')
        return t

    if isinstance(node, Bin):
        _want(T(node.a), INT, node.a, f'each side of {node.op}')
        _want(T(node.b), INT, node.b, f'each side of {node.op}')
        return INT
    if isinstance(node, Un):
        _want(T(node.a), INT, node.a, f'the operand of {node.op}')
        return INT
    if isinstance(node, Cmp):
        a, b = T(node.a), T(node.b)
        if node.op in ('eq', 'ne'):
            if a != b or a not in (INT, STR, BOOL):
                raise TypeErr(f'{node.op} compares two ints, strs or bools of the same type; '
                              f'got {a} and {b}')
        else:
            _want(a, INT, node.a, f'each side of {node.op}')
            _want(b, INT, node.b, f'each side of {node.op}')
        return BOOL
    if isinstance(node, Sel):
        _want(T(node.c), BOOL, node.c, 'the condition of sel')
        a, b = T(node.a), T(node.b)
        if a != b:
            raise TypeErr(f'both outcomes of sel must have one type; got {a} and {b}')
        return a
    if isinstance(node, Not):
        _want(T(node.a), BOOL, node.a, 'the operand of not')
        return BOOL
    if isinstance(node, (All, Any)):
        for x in node.items:
            _want(T(x), BOOL, x, 'each operand of ' + ('all' if isinstance(node, All) else 'any'))
        return BOOL
    if isinstance(node, Len):
        if not isinstance(T(node.a), ListT):
            raise TypeErr(f'len needs a list, but {_show(node.a)} is {T(node.a)}')
        return INT
    if isinstance(node, Pluck):
        t = T(node.a)
        if not (isinstance(t, ListT) and isinstance(t.elem, RecordT)):
            raise TypeErr(f'pluck needs a list of records, but {_show(node.a)} is {t}')
        ft = t.elem.field(node.field)
        if ft is None:
            raise TypeErr(f'records in {_show(node.a)} have no field {node.field!r} '
                          f'(fields: {", ".join(t.elem.names())})', 'unknown-field')
        return ListT(ft)
    if isinstance(node, Loop):
        _want(T(node.lo), INT, node.lo, 'a range bound')
        _want(T(node.hi), INT, node.hi, 'a range bound')
        return INT
    if isinstance(node, ArrLoop):
        t = T(node.arr)
        if not isinstance(t, ListT):
            raise TypeErr(f'{node.op} needs a list, but {_show(node.arr)} is {t}')
        plain_count = node.op == 'count' and node.mapf is None and node.pred is None
        if not plain_count and t.elem != INT:
            raise TypeErr(f'{node.op} with a map or filter needs list[int], '
                          f'but {_show(node.arr)} is {t}; pluck a number field first')
        return INT
    if isinstance(node, Arr):
        for x in node.items:
            _want(T(x), INT, x, 'each element of lit')
        return ListT(INT)
    if isinstance(node, Index):
        t = T(node.arr)
        if not isinstance(t, ListT):
            raise TypeErr(f'idx needs a list, but {_show(node.arr)} is {t}')
        _want(T(node.idx), INT, node.idx, 'an index')
        return t.elem
    if isinstance(node, Fmt):
        for x in node.parts:
            if T(x) not in (INT, STR, BOOL):
                raise TypeErr(f'fmt joins ints, strs and bools; {_show(x)} is {T(x)}')
        return STR
    raise TypeErr(f'unknown expression {node!r}', 'unknown-node')
