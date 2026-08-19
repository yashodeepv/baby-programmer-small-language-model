"""backends.py - Compile the IR to several target languages.

The point of the s-expression target (`train_comp.py --target sexpr`) is that
the model stops emitting AArch64 and emits the IR instead. Once it does, the
translation to any particular language is a DETERMINISTIC compiler pass, not
something the model has to know. This module is that pass, three times over,
to demonstrate the claim: one IR in, C / Python / JavaScript out, every one of
them agreeing with `ir.evaluate` on every input.

Nothing here is learned and nothing here is approximate. Adding a language is
a new Target subclass, not a corpus regeneration and a retrain.

Semantics that must be preserved exactly (see ir.py):
  * every arithmetic result wraps to signed 32-bit, like a w register
  * min / max over an empty selection are 0, not an error or a sentinel
  * product accumulates with a wrap at each step, not only at the end
  * countdown is 0 by construction
  * and / orr / eor / lsl / lsr operate on the unsigned 32-bit pattern
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ir import (Arg, Arr, ArrLoop, Bin, Cmp, Const, Index, InArr, InLen, Loop,
                Program, Sel, Un)


class Target:
    """One language. Subclasses supply syntax; the walk below is shared."""
    name = ext = None
    prologue = ''

    def __init__(self):
        self.n = 0

    def reset(self):
        pass

    def tmp(self, stem='t'):
        self.n += 1
        return f'{stem}{self.n}'

    # --- syntax hooks -----------------------------------------------------
    def s32(self, e):        raise NotImplementedError
    def mul(self, a, b):     raise NotImplementedError
    def arr_lit(self, vals): raise NotImplementedError
    def decl(self, name, e): raise NotImplementedError
    def for_range(self, v, lo, hi):  raise NotImplementedError
    def for_each(self, v, seq):      raise NotImplementedError
    def if_(self, cond):     raise NotImplementedError
    def end(self):           raise NotImplementedError
    def func(self, body, ret): raise NotImplementedError

    # --- shared lowering --------------------------------------------------
    def compile(self, prog):
        self.n = 0
        self.reset()
        out = []
        e = self.expr(prog.body if isinstance(prog, Program) else prog, out)
        return self.prologue + self.func(out, e)

    def expr(self, node, out):
        E = lambda x: self.expr(x, out)
        if isinstance(node, Const):  return self.const(node.v)
        if isinstance(node, Arg):    return self.arg(node.i)
        if isinstance(node, InLen):  return self.alen()
        if isinstance(node, Bin):    return self.binop(node.op, E(node.a), E(node.b))
        if isinstance(node, Un):
            return self.s32(f'-({E(node.a)})') if node.op == 'neg' else self.absv(E(node.a))
        if isinstance(node, Cmp):    return self.cmp(node.op, E(node.a), E(node.b))
        if isinstance(node, Sel):
            return self.select(E(node.c), E(node.a), E(node.b), out)
        if isinstance(node, Index):
            seq = self.seq(node.arr, out)
            return self.index(seq, E(node.idx))
        if isinstance(node, Loop):
            if node.op == 'countdown':
                return self.const(0)
            return self.reduce_range(node, out)
        if isinstance(node, ArrLoop):
            return self.reduce_seq(node, out)
        raise TypeError(f'{self.name}: cannot compile {node!r}')

    def seq(self, node, out):
        """An array-valued node becomes a named variable."""
        if isinstance(node, InArr):
            return self.ain()
        if isinstance(node, Arr):
            v = self.tmp('a')
            out.append(self.arr_decl(v, [self.expr(x, out) for x in node.items]))
            return v
        raise TypeError(f'{self.name}: not an array: {node!r}')

    def _accumulate(self, node, out, item, opener):
        """Shared body of both reductions: open a loop, filter, map, fold.

        `opener` is a THUNK, not a string: building it advances the indent, so
        it must not run until the accumulator declarations are already out.
        """
        op = node.op
        acc, have = self.tmp('acc'), self.tmp('have')
        out.append(self.decl(acc, self.const(1 if op == 'product' else 0)))
        if op in ('min', 'max'):
            out.append(self.decl(have, self.const(0)))
        out.append(opener())
        keep = self.pred(node.pred, item)
        if keep:
            out.append(self.if_(keep))
        val = self.mapf(node.mapf, item, node.k)
        if op == 'sum':
            out.append(self.assign(acc, self.s32(f'{acc} + ({val})')))
        elif op == 'product':
            out.append(self.assign(acc, self.mul(acc, val)))
        elif op == 'count':
            out.append(self.assign(acc, self.s32(f'{acc} + 1')))
        else:
            v = self.tmp('v')
            out.append(self.decl(v, val))
            cmp_ = '<' if op == 'min' else '>'
            out.append(self.if_(f'{have} == 0 || {v} {cmp_} {acc}'
                                if self.name != 'python'
                                else f'{have} == 0 or {v} {cmp_} {acc}'))
            out.append(self.assign(acc, v))
            out.append(self.end())
            out.append(self.assign(have, self.const(1)))
        if keep:
            out.append(self.end())
        out.append(self.end())
        return acc

    def reduce_range(self, node, out):
        i = self.tmp('i')
        lo, hi = self.expr(node.lo, out), self.expr(node.hi, out)
        return self._accumulate(node, out, i, lambda: self.for_range(i, lo, hi))

    def reduce_seq(self, node, out):
        v = self.tmp('x')
        seq = self.seq(node.arr, out)
        return self._accumulate(node, out, v, lambda: self.for_each(v, seq))

    def pred(self, pred, item):
        if pred is None:
            return None
        kind, k = pred
        if kind == 'even':  return f'({item}) % 2 == 0'
        if kind == 'odd':   return f'({item}) % 2 != 0'
        if kind == 'gt':    return f'({item}) > {k}'
        if kind == 'lt':    return f'({item}) < {k}'
        if kind == 'divk':  return f'({item}) % {k} == 0'
        raise ValueError(kind)

    def mapf(self, mapf, item, k):
        if mapf is None:     return item
        if mapf == 'square': return self.mul(item, item)
        if mapf == 'double': return self.s32(f'({item}) * 2')
        if mapf == 'addk':   return self.s32(f'({item}) + {k}')
        if mapf == 'mulk':   return self.mul(item, str(k))
        if mapf == 'constk': return str(k)
        raise ValueError(mapf)


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------

class PyTarget(Target):
    name, ext = 'python', 'py'
    prologue = ('def _s32(x):\n'
                '    x &= 0xFFFFFFFF\n'
                '    return x - (1 << 32) if x & 0x80000000 else x\n\n\n')

    def __init__(self):
        super().__init__()
        self.depth = 1

    def reset(self):
        self.depth = 1

    def const(self, v):  return str(v)
    def arg(self, i):    return f'args[{i}]'
    def ain(self):       return 'arr'
    def alen(self):      return 'len(arr)'
    def s32(self, e):    return f'_s32({e})'
    def mul(self, a, b): return f'_s32(({a}) * ({b}))'
    def absv(self, e):   return f'_s32(abs({e}))'
    def index(self, seq, i): return f'{seq}[{i}]'

    def binop(self, op, a, b):
        M = 0xFFFFFFFF
        if op in ('add', 'sub', 'mul'):
            sym = {'add': '+', 'sub': '-', 'mul': '*'}[op]
            return f'_s32(({a}) {sym} ({b}))'
        if op in ('and', 'orr', 'eor'):
            sym = {'and': '&', 'orr': '|', 'eor': '^'}[op]
            return f'_s32((({a}) & {M}) {sym} (({b}) & {M}))'
        if op == 'lsl':  return f'_s32((({a}) & {M}) << (({b}) & 31))'
        if op == 'lsr':  return f'_s32((({a}) & {M}) >> (({b}) & 31))'
        raise ValueError(op)

    def cmp(self, op, a, b):
        sym = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}[op]
        return f'(1 if ({a}) {sym} ({b}) else 0)'

    def select(self, c, a, b, out):
        v = self.tmp('s')
        out.append(self.decl(v, f'({a}) if ({c}) else ({b})'))
        return v

    def pad(self):            return '    ' * self.depth
    def decl(self, n, e):     return f'{self.pad()}{n} = {e}'
    def assign(self, n, e):   return f'{self.pad()}{n} = {e}'
    def arr_decl(self, n, vals): return f'{self.pad()}{n} = [{", ".join(vals)}]'

    def for_range(self, v, lo, hi):
        s = f'{self.pad()}for {v} in range({lo}, ({hi}) + 1):'
        self.depth += 1
        return s

    def for_each(self, v, seq):
        s = f'{self.pad()}for {v} in {seq}:'
        self.depth += 1
        return s

    def if_(self, cond):
        s = f'{self.pad()}if {cond}:'
        self.depth += 1
        return s

    def end(self):
        self.depth -= 1
        return None                      # Python closes blocks by dedent

    def func(self, body, ret):
        lines = [l for l in body if l is not None]
        return ('def f(args=(), arr=()):\n'
                '    arr = list(arr)\n'
                + ('\n'.join(lines) + '\n' if lines else '')
                + f'    return {ret}\n')


# --------------------------------------------------------------------------
# C -- braces, explicit types, arithmetic done on uint32_t so the wrap is
# defined rather than undefined behaviour
# --------------------------------------------------------------------------

class CTarget(Target):
    name, ext = 'c', 'c'
    prologue = ('#include <stdint.h>\n'
                '#include <stdlib.h>\n\n'
                'static int32_t s32(int64_t x) { return (int32_t)(uint32_t)(x & 0xFFFFFFFF); }\n\n')

    def reset(self):
        self.depth = 1

    def __init__(self):
        super().__init__()
        self.depth = 1

    def const(self, v):  return str(v)
    def arg(self, i):    return f'args[{i}]'
    def ain(self):       return 'arr'
    def alen(self):      return '(int32_t)alen'
    def s32(self, e):    return f's32((int64_t)({e}))'
    def mul(self, a, b): return f's32((int64_t)({a}) * (int64_t)({b}))'
    def absv(self, e):   return f's32((int64_t)llabs((int64_t)({e})))'
    def index(self, seq, i): return f'{seq}[{i}]'

    def binop(self, op, a, b):
        if op in ('add', 'sub', 'mul'):
            sym = {'add': '+', 'sub': '-', 'mul': '*'}[op]
            return f's32((int64_t)({a}) {sym} (int64_t)({b}))'
        if op in ('and', 'orr', 'eor'):
            sym = {'and': '&', 'orr': '|', 'eor': '^'}[op]
            return f'(int32_t)((uint32_t)({a}) {sym} (uint32_t)({b}))'
        if op == 'lsl':  return f'(int32_t)((uint32_t)({a}) << ((uint32_t)({b}) & 31))'
        if op == 'lsr':  return f'(int32_t)((uint32_t)({a}) >> ((uint32_t)({b}) & 31))'
        raise ValueError(op)

    def cmp(self, op, a, b):
        sym = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}[op]
        return f'(({a}) {sym} ({b}) ? 1 : 0)'

    def select(self, c, a, b, out):
        v = self.tmp('s')
        out.append(self.decl(v, f'({c}) ? ({a}) : ({b})'))
        return v

    def pad(self):          return '    ' * self.depth
    def decl(self, n, e):   return f'{self.pad()}int32_t {n} = {e};'
    def assign(self, n, e): return f'{self.pad()}{n} = {e};'

    def arr_decl(self, n, vals):
        return f'{self.pad()}int32_t {n}[] = {{{", ".join(vals)}}};'

    def for_range(self, v, lo, hi):
        s = (f'{self.pad()}for (int32_t {v} = {lo}; {v} <= ({hi}); {v}++) {{')
        self.depth += 1
        return s

    def for_each(self, v, seq):
        i = self.tmp('k')
        n = (f'(int32_t)(sizeof({seq}) / sizeof({seq}[0]))'
             if seq != 'arr' else '(int32_t)alen')
        s = (f'{self.pad()}for (int32_t {i} = 0; {i} < {n}; {i}++) {{\n'
             f'{self.pad()}    int32_t {v} = {seq}[{i}];')
        self.depth += 1
        return s

    def if_(self, cond):
        s = f'{self.pad()}if ({cond}) {{'
        self.depth += 1
        return s

    def end(self):
        self.depth -= 1
        return f'{self.pad()}}}'

    def func(self, body, ret):
        lines = [l for l in body if l is not None]
        return (self.prologue.replace(self.prologue, '') +
                'int32_t f(const int32_t *args, const int32_t *arr, size_t alen) {\n'
                + ('\n'.join(lines) + '\n' if lines else '')
                + f'    return {ret};\n}}\n')


# --------------------------------------------------------------------------
# JavaScript -- bitwise operators are already int32, so most wraps are free
# --------------------------------------------------------------------------

class JsTarget(Target):
    name, ext = 'javascript', 'js'
    prologue = 'const s32 = (x) => x | 0;\n\n'

    def __init__(self):
        super().__init__()
        self.depth = 1

    def reset(self):
        self.depth = 1

    def const(self, v):  return str(v)
    def arg(self, i):    return f'args[{i}]'
    def ain(self):       return 'arr'
    def alen(self):      return 'arr.length'
    def s32(self, e):    return f's32({e})'
    def mul(self, a, b): return f'Math.imul({a}, {b})'
    def absv(self, e):   return f's32(Math.abs({e}))'
    def index(self, seq, i): return f'{seq}[{i}]'

    def binop(self, op, a, b):
        if op == 'mul':  return f'Math.imul({a}, {b})'
        if op in ('add', 'sub'):
            sym = '+' if op == 'add' else '-'
            return f's32(({a}) {sym} ({b}))'
        if op in ('and', 'orr', 'eor'):
            sym = {'and': '&', 'orr': '|', 'eor': '^'}[op]
            return f'(({a}) {sym} ({b}))'
        if op == 'lsl':  return f'(({a}) << (({b}) & 31))'
        if op == 'lsr':  return f'((({a}) >>> (({b}) & 31)) | 0)'
        raise ValueError(op)

    def cmp(self, op, a, b):
        sym = {'eq': '===', 'ne': '!==', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}[op]
        return f'((({a}) {sym} ({b})) ? 1 : 0)'

    def select(self, c, a, b, out):
        v = self.tmp('s')
        out.append(self.decl(v, f'({c}) ? ({a}) : ({b})'))
        return v

    def pad(self):          return '  ' * self.depth
    def decl(self, n, e):   return f'{self.pad()}let {n} = {e};'
    def assign(self, n, e): return f'{self.pad()}{n} = {e};'

    def arr_decl(self, n, vals):
        return f'{self.pad()}const {n} = [{", ".join(vals)}];'

    def for_range(self, v, lo, hi):
        s = f'{self.pad()}for (let {v} = {lo}; {v} <= ({hi}); {v}++) {{'
        self.depth += 1
        return s

    def for_each(self, v, seq):
        s = f'{self.pad()}for (const {v} of {seq}) {{'
        self.depth += 1
        return s

    def if_(self, cond):
        s = f'{self.pad()}if ({cond}) {{'
        self.depth += 1
        return s

    def end(self):
        self.depth -= 1
        return f'{self.pad()}}}'

    def func(self, body, ret):
        lines = [l for l in body if l is not None]
        return ('function f(args, arr) {\n'
                + ('\n'.join(lines) + '\n' if lines else '')
                + f'  return {ret};\n}}\n')


# --------------------------------------------------------------------------
# Java -- int is 32-bit two's complement and wraps on overflow, which is
# exactly the oracle's s32. So the wrap helper is the identity here.
# --------------------------------------------------------------------------

class JavaTarget(Target):
    name, ext = 'java', 'java'
    prologue = ''

    def __init__(self):
        super().__init__()
        self.depth = 2

    def reset(self):
        self.depth = 2

    def const(self, v):  return str(v)
    def arg(self, i):    return f'args[{i}]'
    def ain(self):       return 'arr'
    def alen(self):      return 'arr.length'
    def s32(self, e):    return f'({e})'          # int already wraps
    def mul(self, a, b): return f'(({a}) * ({b}))'
    def absv(self, e):   return f'Math.abs({e})'
    def index(self, seq, i): return f'{seq}[{i}]'

    def binop(self, op, a, b):
        if op in ('add', 'sub', 'mul'):
            sym = {'add': '+', 'sub': '-', 'mul': '*'}[op]
            return f'(({a}) {sym} ({b}))'
        if op in ('and', 'orr', 'eor'):
            sym = {'and': '&', 'orr': '|', 'eor': '^'}[op]
            return f'(({a}) {sym} ({b}))'
        if op == 'lsl':  return f'(({a}) << (({b}) & 31))'
        if op == 'lsr':  return f'(({a}) >>> (({b}) & 31))'
        raise ValueError(op)

    def cmp(self, op, a, b):
        sym = {'eq': '==', 'ne': '!=', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}[op]
        return f'((({a}) {sym} ({b})) ? 1 : 0)'

    def select(self, c, a, b, out):
        v = self.tmp('s')
        out.append(self.decl(v, f'({c}) ? ({a}) : ({b})'))
        return v

    def pad(self):          return '    ' * self.depth
    def decl(self, n, e):   return f'{self.pad()}int {n} = {e};'
    def assign(self, n, e): return f'{self.pad()}{n} = {e};'

    def arr_decl(self, n, vals):
        return f'{self.pad()}int[] {n} = {{{", ".join(vals)}}};'

    def for_range(self, v, lo, hi):
        s = f'{self.pad()}for (int {v} = {lo}; {v} <= ({hi}); {v}++) {{'
        self.depth += 1
        return s

    def for_each(self, v, seq):
        s = f'{self.pad()}for (int {v} : {seq}) {{'
        self.depth += 1
        return s

    def if_(self, cond):
        s = f'{self.pad()}if ({cond}) {{'
        self.depth += 1
        return s

    def end(self):
        self.depth -= 1
        return f'{self.pad()}}}'

    def func(self, body, ret):
        lines = [l for l in body if l is not None]
        return ('public class Program {\n'
                '    static int f(int[] args, int[] arr) {\n'
                + ('\n'.join(lines) + '\n' if lines else '')
                + f'        return {ret};\n    }}\n}}\n')


TARGETS = {t.name: t for t in (PyTarget, CTarget, JsTarget, JavaTarget)}
