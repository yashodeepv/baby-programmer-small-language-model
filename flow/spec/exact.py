"""exact.py - Exact expressions (LANGUAGE.md §8.1) -> IR2.

    `subtotal / 10 if tier == "gold" else 0`
    `count(invoices.days_late where > VERY_LATE_DAYS)`

A precedence-climbing parser over a small token set. Every construct maps
one-to-one onto an IR2 node, so an exact expression is compiled as written
and never passes through a model.
"""

import re

from flow.ir2.expr import All, Any, ArrLoop, Bin, Bool, Cmp, Const, Len, Loop, Not, Sel, Un
from flow.spec.document import SpecError
from flow.spec.values import template

_TOKEN = re.compile(r'\s*(?:(?P<str>"(?:[^"\\]|\\.)*")|(?P<int>\d+)|(?P<const>[A-Z][A-Z0-9_]*)'
                    r'|(?P<name>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)'
                    r'|(?P<op>==|!=|<=|>=|[<>+\-*/%(),]))')

CMP = {'==': 'eq', '!=': 'ne', '<': 'lt', '<=': 'le', '>': 'gt', '>=': 'ge'}
REDUCE = ('sum', 'product', 'min', 'max', 'count')
KEYWORDS = {'and', 'or', 'not', 'if', 'else', 'where', 'each', 'true', 'false',
            'even', 'odd', 'divisible', 'by', 'squared'}


def _lex(text, line):
    out, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise SpecError('syntax', line, f'unexpected "{text[pos:].strip()[:20]}" in `{text}`')
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
        pos = m.end()
    return out


class _P:
    def __init__(self, text, resolver, line):
        self.text, self.r, self.line = text, resolver, line
        self.t, self.i = _lex(text, line), 0

    def err(self, msg):
        raise SpecError('syntax', self.line, f'{msg} in `{self.text}`')

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def at(self, value):
        return self.peek()[1] == value

    def take(self, value=None):
        kind, v = self.peek()
        if kind is None:
            self.err('unexpected end')
        if value is not None and v != value:
            self.err(f'expected "{value}", found "{v}"')
        self.i += 1
        return kind, v

    def integer(self):
        neg = False
        if self.at('-'):
            self.take()
            neg = True
        kind, v = self.take()
        if kind == 'int':
            n = int(v)
        elif kind == 'const':
            lit, _ = self.r.constant(v, self.line)
            if not isinstance(lit, Const):
                self.err(f'{v} is not an integer constant')
            n = lit.v
        else:
            self.err(f'expected an integer, found "{v}"')
        return -n if neg else n

    # expr := or [if or else expr]
    def expr(self):
        a = self.or_()
        if self.at('if'):
            self.take('if')
            c = self.or_()
            self.take('else')
            return Sel(c, a, self.expr())
        return a

    def or_(self):
        items = [self.and_()]
        while self.at('or'):
            self.take()
            items.append(self.and_())
        return items[0] if len(items) == 1 else Any(tuple(items))

    def and_(self):
        items = [self.not_()]
        while self.at('and'):
            self.take()
            items.append(self.not_())
        return items[0] if len(items) == 1 else All(tuple(items))

    def not_(self):
        if self.at('not'):
            self.take()
            return Not(self.not_())
        return self.cmp()

    def cmp(self):
        a = self.sum_()
        if self.peek()[1] in CMP:
            op = CMP[self.take()[1]]
            return Cmp(op, a, self.sum_())
        return a

    def sum_(self):
        a = self.term()
        while self.peek()[1] in ('+', '-'):
            op = 'add' if self.take()[1] == '+' else 'sub'
            a = Bin(op, a, self.term())
        return a

    def term(self):
        a = self.unary()
        while self.peek()[1] in ('*', '/', '%'):
            op = {'*': 'mul', '/': 'div', '%': 'mod'}[self.take()[1]]
            a = Bin(op, a, self.unary())
        return a

    def unary(self):
        if self.at('-'):
            self.take()
            if self.peek()[0] == 'int':
                return Const(-int(self.take()[1]))
            return Un('neg', self.unary())
        return self.atom()

    def atom(self):
        kind, v = self.peek()
        if kind == 'int':
            self.take()
            return Const(int(v))
        if kind == 'str':
            self.take()
            return template(v, self.r, self.line)
        if kind == 'const':
            self.take()
            return self.r.constant(v, self.line)[0]
        if v == '(':
            self.take()
            e = self.expr()
            self.take(')')
            return e
        if kind == 'name':
            if v in ('true', 'false'):
                self.take()
                return Bool(v == 'true')
            if v in KEYWORDS:
                self.err(f'"{v}" is a keyword here')
            if self.peek(1)[1] == '(' and '.' not in v:
                return self.call()
            self.take()
            return self.r.resolve(v, self.line)[0]
        self.err(f'unexpected "{v}"' if v else 'unexpected end')

    def call(self):
        _, fn = self.take()
        self.take('(')
        if fn == 'len':
            e = Len(self.expr())
        elif fn == 'abs':
            e = Un('abs', self.expr())
        elif fn in REDUCE:
            e = self.reduction(fn)
        elif fn == 'range':
            self.err('range(...) only appears as the list of a reduction')
        else:
            self.err(f'unknown function "{fn}"')
        self.take(')')
        return e

    def reduction(self, fn):
        if self.at('range') and self.peek(1)[1] == '(':
            self.take('range')
            self.take('(')
            lo = self.expr()
            self.take(',')
            hi = self.expr()
            self.take(')')
            src = ('range', lo, hi)
        else:
            src = self.expr()
        mapf, k, pred = None, 0, None
        while self.peek()[1] in ('each', 'where'):
            if self.take()[1] == 'each':
                if mapf is not None:
                    self.err('only one "each"')
                if self.at('squared'):
                    self.take()
                    mapf = 'square'
                elif self.peek()[1] in ('*', '+'):
                    op = self.take()[1]
                    mapf, k = ('mulk' if op == '*' else 'addk'), self.integer()
                else:
                    self.err('"each" is followed by "* k", "+ k" or "squared"')
            else:
                if pred is not None:
                    self.err('only one "where"')
                w = self.peek()[1]
                if w in ('>', '<'):
                    self.take()
                    pred = ('gt' if w == '>' else 'lt', self.integer())
                elif w in ('even', 'odd'):
                    self.take()
                    pred = (w, 0)
                elif w == 'divisible':
                    self.take()
                    self.take('by')
                    pred = ('divk', self.integer())
                else:
                    self.err('"where" is followed by "> k", "< k", "even", "odd" or "divisible by k"')
        if isinstance(src, tuple):
            return Loop(fn, src[1], src[2], mapf=mapf, pred=pred, k=k)
        return ArrLoop(fn, src, mapf=mapf, pred=pred, k=k)


def parse_exact(text, resolver, line):
    p = _P(text, resolver, line)
    if not p.t:
        raise SpecError('syntax', line, 'an empty expression')
    e = p.expr()
    if p.i != len(p.t):
        p.err(f'unexpected "{p.peek()[1]}"')
    return e
