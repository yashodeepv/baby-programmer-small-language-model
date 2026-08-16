"""
sexpr.py - A compact textual surface for the IR, and a parser back from it.

The model's training target is moving from assembly to meaning. `lower()`
already turns an IR tree into AArch64 deterministically and exactly, so a model
that emits assembly is spending four fifths of every answer reproducing a
compiler pass. Measured on 3,000 programs: 90 tokens of assembly derived from
18 tokens of meaning.

Python's `repr()` would be the lazy choice and is a bad target -- verbose, with
redundant defaults like `mapf=None, k=0`, measuring 36 tokens. This designed
s-expression measures 18:

    (sum (rng 2 20) :even)
    (max (arr) :sq :even)
    (sel (ne 240 (lsl 278 2)) (add 374 110) (mul 469 444))

The binding requirement is EXACT ROUND-TRIP: parse(emit(ir)) == ir for every
program the grammar can produce. An ambiguous surface would let a correct model
output be read back as a different program -- a failure that looks like a model
error and is not one.

The parser is deliberately strict. Unknown atoms, wrong arity, unclosed
parentheses and trailing junk all raise. A parser that guesses would silently
turn malformed output into a valid-looking program.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir import (Arg, Arr, ArrLoop, Bin, Cmp, Const, InArr, InLen, Index, Loop,
                Program, Sel, Un)

BINOPS = ('add', 'sub', 'mul', 'and', 'orr', 'eor', 'lsl', 'lsr')
UNOPS  = ('neg', 'abs')
CMPOPS = ('eq', 'ne', 'lt', 'le', 'gt', 'ge')
REDUCE = ('sum', 'product', 'count', 'min', 'max')

# keyword → (kind, takes_argument)
MAPKW  = {'sq': ('square', False), 'dbl': ('double', False),
          'add': ('addk', True), 'mul': ('mulk', True), 'const': ('constk', True)}
PREDKW = {'even': ('even', False), 'odd': ('odd', False), 'gt': ('gt', True),
          'lt': ('lt', True), 'divk': ('divk', True)}

_MAP_OUT  = {v[0]: k for k, v in MAPKW.items()}
_PRED_OUT = {v[0]: k for k, v in PREDKW.items()}


class ParseError(ValueError):
    pass


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------

def _mods(n):
    """Map first, then predicate -- a fixed order, so emit is canonical."""
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
    """IR -> s-expression text."""
    if isinstance(node, Program):
        return emit(node.body)
    if isinstance(node, Const):
        return str(node.v)
    if isinstance(node, Arg):
        return f'a{node.i}'
    if isinstance(node, InArr):
        return '(arr)'
    if isinstance(node, InLen):
        return '(len)'
    if isinstance(node, Bin):
        return f'({node.op} {emit(node.a)} {emit(node.b)})'
    if isinstance(node, Un):
        return f'({node.op} {emit(node.a)})'
    if isinstance(node, Cmp):
        return f'({node.op} {emit(node.a)} {emit(node.b)})'
    if isinstance(node, Sel):
        return f'(sel {emit(node.c)} {emit(node.a)} {emit(node.b)})'
    if isinstance(node, Arr):
        return '(lit ' + ' '.join(emit(x) for x in node.items) + ')'
    if isinstance(node, Index):
        return f'(idx {emit(node.arr)} {emit(node.idx)})'
    if isinstance(node, Loop):
        if node.op == 'countdown':
            return f'(down {emit(node.hi)})'
        return f'({node.op} (rng {emit(node.lo)} {emit(node.hi)}){_mods(node)})'
    if isinstance(node, ArrLoop):
        return f'({node.op} {emit(node.arr)}{_mods(node)})'
    raise TypeError(f'cannot emit {node!r}')


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------

_TOK = re.compile(r'\(|\)|:[a-z]+|-?\d+|a[0-9]|[a-z]+')


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

    # -- modifiers ------------------------------------------------------
    def mods(self):
        mapf = pred = None
        k = 0
        while self.peek() and self.peek().startswith(':'):
            kw = self.take()[1:]
            if kw in MAPKW and mapf is None and kw not in PREDKW:
                name, arg = MAPKW[kw]
                mapf, k = name, (int(self.take()) if arg else 0)
            elif kw in PREDKW:
                name, arg = PREDKW[kw]
                if pred is not None:
                    raise ParseError('two predicates')
                pred = (name, int(self.take()) if arg else 0)
            elif kw in MAPKW:
                name, arg = MAPKW[kw]
                if mapf is not None:
                    raise ParseError('two maps')
                mapf, k = name, (int(self.take()) if arg else 0)
            else:
                raise ParseError(f'unknown keyword :{kw}')
        return mapf, pred, k

    # -- expressions ----------------------------------------------------
    def expr(self):
        tok = self.peek()
        if tok is None:
            raise ParseError('unexpected end of input')
        if re.fullmatch(r'-?\d+', tok):
            return Const(int(self.take()))
        if re.fullmatch(r'a[0-9]', tok):
            return Arg(int(self.take()[1:]))
        if tok != '(':
            raise ParseError(f'unexpected {tok!r}')

        self.take('(')
        head = self.take()

        if head == 'arr':
            self.take(')'); return InArr()
        if head == 'len':
            self.take(')'); return InLen()
        if head == 'lit':
            items = []
            while self.peek() != ')':
                items.append(self.expr())
            self.take(')')
            if not items:
                raise ParseError('(lit) needs at least one element')
            return Arr(tuple(items))
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
            c = self.expr()
            if not isinstance(c, Cmp):
                raise ParseError('sel needs a comparison first')
            a, b = self.expr(), self.expr()
            self.take(')'); return Sel(c, a, b)
        if head == 'idx':
            arr, i = self.expr(), self.expr()
            self.take(')'); return Index(arr, i)
        if head == 'down':
            hi = self.expr()
            self.take(')'); return Loop('countdown', Const(0), hi)
        if head in REDUCE:
            src = self.expr()
            mapf, pred, k = self.mods()
            self.take(')')
            if isinstance(src, tuple) and src[0] == 'rng':
                return Loop(head, src[1], src[2], mapf=mapf, pred=pred, k=k)
            if isinstance(src, (InArr, Arr)):
                return ArrLoop(head, src, mapf=mapf, pred=pred, k=k)
            raise ParseError(f'{head} needs (rng …), (arr) or (lit …)')
        raise ParseError(f'unknown form ({head} …)')


def parse(text):
    """s-expression text -> IR. Raises ParseError on anything malformed."""
    p = _P(_lex(text))
    node = p.expr()
    if p.i != len(p.t):
        raise ParseError(f'trailing tokens from {p.t[p.i]!r}')
    if isinstance(node, tuple):
        raise ParseError('(rng …) is not a program on its own')
    return node


def n_args_of(node):
    """Highest argument index referenced, +1 -- recovers Program.n_args."""
    best = -1
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, Arg):
            best = max(best, n.i)
        for f in getattr(n, '__dataclass_fields__', {}):
            v = getattr(n, f)
            if isinstance(v, tuple):
                stack.extend(x for x in v if hasattr(x, '__dataclass_fields__'))
            elif hasattr(v, '__dataclass_fields__'):
                stack.append(v)
    return best + 1
