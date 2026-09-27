"""english.py - IR2 expressions as English noun phrases and conditions.

The arm renderer (`ir.show`) extended to the IR2 nodes. The arm wordings are
reused verbatim from arm/ir.py -- operators, comparisons, reductions, maps,
predicates, three styles each -- so an arithmetic phrase reads the same in a
flow as it does in the assembly corpus.

Names and literals go through a `Namer`, never straight into the text:

    namer.ref(path)    'the order total', 'the status from payments.charge'
    namer.num(v)       '100000'
    namer.text(v)      '"gold"'
    namer.field(name)  'amount' -- a field named on its own, as in a pluck
    namer.is_bool(path)

The flow renderer's namer produces real English. The logic model's
anonymised examples use a namer that produces placeholders ('the f3', 'n1')
from the same walk, so the two texts can never drift apart.
"""

from flow.arm_bridge import arm_ir
from flow.ir2.types import BOOL
from flow.ir2.expr import (All, Any, Arr, ArrLoop, Bin, Bool, Cmp, Const, Fmt,
                           Index, Len, Loop, Not, Pluck, Ref, Sel, Str, Un)

_w = arm_ir._w

_IR2_OPWORD = {
    'div': ('divided by', 'divided evenly by', 'over'),
    'mod': ('modulo', 'mod', 'modulo'),
}

_NOT = ('not ({x})', 'it is not the case that {x}', 'the opposite of ({x})')
_ALL = ('{a} and {b}', 'both {a} and {b}', '{a}, and also {b}')
_ANY = ('{a} or {b}', 'either {a} or {b}', '{a}, or alternatively {b}')
_SEL = ('{a} if {c}, otherwise {b}', '{a} when {c}, else {b}', 'if {c} then {a}, else {b}')
_FMT = (('the text ', ' followed by '), ('the message ', ' then '), ('the words ', ' plus '))
_PLUCK = ('the {f}{field} of {src}', 'the {f}{field} values in {src}', 'each {f}{field} in {src}')
_LEN = ('the length of {x}', 'the number of entries in {x}', 'the size of {x}')
_BOOLREF = ('{x} is true', '{x} holds', '{x} is set')


def words(name):
    return name.replace('_', ' ')


def _opword(op, style):
    return _w(_IR2_OPWORD, op, style) if op in _IR2_OPWORD else _w(arm_ir._OPWORD, op, style)


def _mods(node, style, namer):
    """(prefix, filter adjective, tail) for a reduction's map and predicate."""
    pre = _w(arm_ir._MAPWORD, node.mapf, style) if node.mapf else ''
    filt = _w(arm_ir._PREDWORD, node.pred[0], style) if node.pred else ''
    tail = ''
    if node.pred and node.pred[0] in arm_ir._PREDTAIL:
        tail += _w(arm_ir._PREDTAIL, node.pred[0], style).format(k=namer.num(node.pred[1]))
    if node.mapf in arm_ir._MAPTAIL:
        tail += _w(arm_ir._MAPTAIL, node.mapf, style).format(k=namer.num(node.k))
    return pre, filt, tail


def cond_context(field, typ):
    """Is this expression read as a condition? Then a bare boolean field is
    said as "the flagged is true"; where it is a value being passed on or
    returned, it is just "the flagged"."""
    return field in ('rule', 'cond') or (field.startswith('assign.') and typ == BOOL)


def say(node, style, namer, cond=False):
    """An expression as English, in one of three styles.

    `cond` marks a position where a condition is expected: a rule, a branch,
    the test of a `sel`, the operands of not / all / any.
    """
    S = lambda n: say(n, style, namer)
    C = lambda n: say(n, style, namer, cond=True)

    def operand(n, c=False):
        text = say(n, style, namer, cond=c)
        return f'({text})' if isinstance(n, (Bin, Cmp, Sel, All, Any)) else text

    if isinstance(node, Ref):
        phrase = namer.ref(node.path)
        return _BOOLREF[style % 3].format(x=phrase) if cond and namer.is_bool(node.path) else phrase
    if isinstance(node, Const):
        return namer.num(node.v)
    if isinstance(node, Str):
        return namer.text(node.v)
    if isinstance(node, Bool):
        return 'true' if node.v else 'false'
    if isinstance(node, Bin):
        return f'{operand(node.a)} {_opword(node.op, style)} {operand(node.b)}'
    if isinstance(node, Un):
        return _w(arm_ir._UNARY, node.op, style).format(x=S(node.a))
    if isinstance(node, Cmp):
        return f'{S(node.a)} is {_w(arm_ir._CMPWORD, node.op, style)} {S(node.b)}'
    if isinstance(node, Sel):
        return _SEL[style % 3].format(a=operand(node.a), c=C(node.c), b=operand(node.b))
    if isinstance(node, Not):
        return _NOT[style % 3].format(x=C(node.a))
    if isinstance(node, (All, Any)):
        parts = [operand(x, c=True) for x in node.items]
        if len(parts) == 2:
            tmpl = (_ALL if isinstance(node, All) else _ANY)[style % 3]
            return tmpl.format(a=parts[0], b=parts[1])
        return (' and ' if isinstance(node, All) else ' or ').join(parts)
    if isinstance(node, Len):
        return _LEN[style % 3].format(x=S(node.a))
    if isinstance(node, Pluck):
        return _PLUCK[style % 3].format(f='', field=namer.field(node.field), src=S(node.a))
    if isinstance(node, Fmt):
        head, sep = _FMT[style % 3]
        return head + sep.join(S(p) for p in node.parts)
    if isinstance(node, ArrLoop):
        pre, filt, tail = _mods(node, style, namer)
        noun = _w(arm_ir._NOUN, node.op, style)
        if isinstance(node.arr, Pluck):
            src = _PLUCK[style % 3].format(f=filt, field=namer.field(node.arr.field), src=S(node.arr.a))
        else:
            src = S(node.arr)
            if filt:
                src = f'the {filt}{src[4:]}' if src.startswith('the ') else filt + src
        return f'{noun} {pre}{src}{tail}'
    if isinstance(node, Loop):
        if node.op == 'countdown':
            return f'zero after counting down from {S(node.hi)}'
        pre, filt, tail = _mods(node, style, namer)
        src = arm_ir._RANGE[style % len(arm_ir._RANGE)].format(f=filt, lo=S(node.lo), hi=S(node.hi))
        return f'{_w(arm_ir._NOUN, node.op, style)} {pre}{src}{tail}'
    if isinstance(node, Arr):
        return '[' + ', '.join(S(x) for x in node.items) + ']'
    if isinstance(node, Index):
        return f'element at index {S(node.idx)} of {S(node.arr)}'
    raise TypeError(f'cannot say {node!r}')
