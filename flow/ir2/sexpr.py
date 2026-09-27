"""sexpr.py - Text form of IR2 expressions: a strict superset of arm/sexpr.py.

Every string arm/sexpr.py accepts parses here to the same tree, so the
assembly model's s-expression output drops straight into a flow step. New:

    input.total  pricing.subtotal         Ref, dotted field access
    "gold"                                Str, with \\" and \\\\ escapes
    true  false                           Bool
    (div a b)  (mod a b)                  truncating integer division
    (not x)  (all x y ...)  (any x y ...) boolean logic
    (len xs)                              Len; (len) alone stays arm's InLen
    (pluck xs price)                      Pluck
    (fmt "Order " input.id " ok")         Fmt
    (sum xs :gt 30)                       a reduction over any list, not only
                                          (arr) / (lit ...) / (rng ...)

Same contract as arm: exact round trip, and a strict parser that raises on
anything it does not recognise rather than guessing.

Names: a bare identifier is a Ref, except `a0`..`a9` (arm's Arg), `true` and
`false`. Step ids that collide with those are rejected by the checker.
"""

import re

from flow.ir2.expr import (BINOPS, CMPOPS, REDUCE, UNOPS, All, Any,
                           Arg, Arr, ArrLoop, Bin, Bool, Cmp, Const, Fmt, Index,
                           InArr, InLen, Len, Loop, Not, Pluck, Ref, Sel, Str, Un)
from flow.arm_bridge import arm_sexpr

MAPKW, PREDKW = arm_sexpr.MAPKW, arm_sexpr.PREDKW
_MAP_OUT = {v[0]: k for k, v in MAPKW.items()}
_PRED_OUT = {v[0]: k for k, v in PREDKW.items()}

RESERVED = {'true', 'false'} | {f'a{i}' for i in range(10)}


class ParseError(ValueError):
    pass


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------

def _quote(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _mods(n):
    out = ''
    if n.mapf:
        kw = _MAP_OUT[n.mapf]
        out += f' :{kw}' + (f' {n.k}' if MAPKW[kw][1] else '')
    if n.pred:
        kind, k = n.pred
        kw = _PRED_OUT[kind]
        out += f' :{kw}' + (f' {k}' if PREDKW[kw][1] else '')
    return out


def emit(node):
    """IR2 expression -> text. Arm nodes emit exactly as arm/sexpr.py does."""
    if isinstance(node, Ref):
        return str(node)
    if isinstance(node, Str):
        return _quote(node.v)
    if isinstance(node, Bool):
        return 'true' if node.v else 'false'
    if isinstance(node, Not):
        return f'(not {emit(node.a)})'
    if isinstance(node, All):
        return '(all ' + ' '.join(emit(x) for x in node.items) + ')'
    if isinstance(node, Any):
        return '(any ' + ' '.join(emit(x) for x in node.items) + ')'
    if isinstance(node, Len):
        return f'(len {emit(node.a)})'
    if isinstance(node, Pluck):
        return f'(pluck {emit(node.a)} {node.field})'
    if isinstance(node, Fmt):
        return '(fmt ' + ' '.join(emit(x) for x in node.parts) + ')'
    if isinstance(node, Sel):
        return f'(sel {emit(node.c)} {emit(node.a)} {emit(node.b)})'
    if isinstance(node, (Bin, Cmp)):
        return f'({node.op} {emit(node.a)} {emit(node.b)})'
    if isinstance(node, Un):
        return f'({node.op} {emit(node.a)})'
    if isinstance(node, Loop):
        if node.op == 'countdown':
            return f'(down {emit(node.hi)})'
        return f'({node.op} (rng {emit(node.lo)} {emit(node.hi)}){_mods(node)})'
    if isinstance(node, ArrLoop):
        return f'({node.op} {emit(node.arr)}{_mods(node)})'
    if isinstance(node, Index):
        return f'(idx {emit(node.arr)} {emit(node.idx)})'
    if isinstance(node, Arr):
        return '(lit ' + ' '.join(emit(x) for x in node.items) + ')'
    # Leaves shared with arm: Const, Arg, InArr, InLen.
    return arm_sexpr.emit(node)


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------

_IDENT = r'[a-z_][a-z0-9_]*'
_TOK = re.compile(r'"(?:[^"\\]|\\.)*"|\(|\)|:[a-z]+|-?\d+|'
                  + _IDENT + r'(?:\.' + _IDENT + r')*')


def _lex(text):
    out, pos = [], 0
    for m in _TOK.finditer(text):
        if text[pos:m.start()].strip():
            raise ParseError(f'unexpected {text[pos:m.start()]!r} at {pos}')
        out.append(m.group(0))
        pos = m.end()
    if text[pos:].strip():
        raise ParseError(f'trailing {text[pos:]!r}')
    return out


def _unquote(tok):
    body, out, i = tok[1:-1], [], 0
    while i < len(body):
        if body[i] == '\\':
            if i + 1 >= len(body) or body[i + 1] not in '"\\':
                raise ParseError(f'bad escape in {tok}')
            out.append(body[i + 1])
            i += 2
        else:
            out.append(body[i])
            i += 1
    return ''.join(out)


class _P:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, want=None):
        if self.i >= len(self.t):
            raise ParseError('unexpected end of input')
        tok = self.t[self.i]
        if want is not None and tok != want:
            raise ParseError(f'expected {want!r}, got {tok!r}')
        self.i += 1
        return tok

    def int_arg(self):
        tok = self.take()
        if not re.fullmatch(r'-?\d+', tok):
            raise ParseError(f'expected an integer, got {tok!r}')
        return int(tok)

    def mods(self):
        mapf = pred = None
        k = 0
        while self.peek() and self.peek().startswith(':'):
            kw = self.take()[1:]
            # `add` / `mul` are maps only; `gt` / `lt` / `even` ... predicates.
            if kw in MAPKW and kw not in PREDKW:
                if mapf is not None:
                    raise ParseError('two maps')
                name, arg = MAPKW[kw]
                mapf, k = name, (self.int_arg() if arg else 0)
            elif kw in PREDKW:
                if pred is not None:
                    raise ParseError('two predicates')
                name, arg = PREDKW[kw]
                pred = (name, self.int_arg() if arg else 0)
            else:
                raise ParseError(f'unknown keyword :{kw}')
        return mapf, pred, k

    def rest(self):
        items = []
        while self.peek() != ')':
            if self.peek() is None:
                raise ParseError('unclosed (')
            items.append(self.expr())
        self.take(')')
        return tuple(items)

    def expr(self):
        tok = self.peek()
        if tok is None:
            raise ParseError('unexpected end of input')
        if tok.startswith('"'):
            return Str(_unquote(self.take()))
        if re.fullmatch(r'-?\d+', tok):
            return Const(int(self.take()))
        if re.fullmatch(r'a[0-9]', tok):
            return Arg(int(self.take()[1:]))
        if tok in ('true', 'false'):
            self.take()
            return Bool(tok == 'true')
        if tok.startswith(':'):
            raise ParseError(f'modifier {tok} outside a reduction')
        if tok != '(':
            if tok == ')':
                raise ParseError('unexpected )')
            return Ref(tuple(self.take().split('.')))

        self.take('(')
        head = self.take()

        if head == 'arr':
            self.take(')'); return InArr()
        if head == 'len':
            if self.peek() == ')':
                self.take(')'); return InLen()
            a = self.expr()
            self.take(')'); return Len(a)
        if head == 'lit':
            items = self.rest()
            if not items:
                raise ParseError('(lit) needs at least one element')
            return Arr(items)
        if head == 'rng':
            lo, hi = self.expr(), self.expr()
            self.take(')'); return ('rng', lo, hi)
        if head in BINOPS:
            a, b = self.expr(), self.expr()
            self.take(')'); return Bin(head, a, b)
        if head in UNOPS:
            a = self.expr()
            self.take(')'); return Un(head, a)
        if head in CMPOPS:
            a, b = self.expr(), self.expr()
            self.take(')'); return Cmp(head, a, b)
        if head == 'sel':
            c, a, b = self.expr(), self.expr(), self.expr()
            self.take(')'); return Sel(c, a, b)
        if head == 'idx':
            arr, i = self.expr(), self.expr()
            self.take(')'); return Index(arr, i)
        if head == 'down':
            hi = self.expr()
            self.take(')'); return Loop('countdown', Const(0), hi)
        if head == 'not':
            a = self.expr()
            self.take(')'); return Not(a)
        if head in ('all', 'any'):
            items = self.rest()
            if len(items) < 2:
                raise ParseError(f'({head} …) needs at least two operands')
            return All(items) if head == 'all' else Any(items)
        if head == 'pluck':
            a = self.expr()
            field = self.take()
            if not re.fullmatch(_IDENT, field):
                raise ParseError(f'pluck needs a field name, got {field!r}')
            self.take(')'); return Pluck(a, field)
        if head == 'fmt':
            parts = self.rest()
            if not parts:
                raise ParseError('(fmt) needs at least one part')
            return Fmt(parts)
        if head in REDUCE:
            src = self.expr()
            mapf, pred, k = self.mods()
            self.take(')')
            if isinstance(src, tuple):
                return Loop(head, src[1], src[2], mapf=mapf, pred=pred, k=k)
            return ArrLoop(head, src, mapf=mapf, pred=pred, k=k)
        raise ParseError(f'unknown form ({head} …)')


def parse(text):
    """Text -> IR2 expression. Raises ParseError on anything malformed."""
    p = _P(_lex(text))
    node = p.expr()
    if p.i != len(p.t):
        raise ParseError(f'trailing tokens from {p.t[p.i]!r}')
    if isinstance(node, tuple):
        raise ParseError('(rng …) is not an expression on its own')
    return node
