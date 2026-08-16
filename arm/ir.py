"""
ir.py - Program IR: one sampled object, three derived views.

The old generator hand-maintained three things per template -- the question
text, the assembly, and the expected value. They could drift, and writing a
thousand of them by hand is not possible. Here a program is a single IR tree
and everything else is derived from it:

    render(prog)   -> the English question          (this file)
    lower(prog)    -> AArch64 assembly              (lower.py)
    evaluate(prog) -> the reference semantics       (this file)

`lower` and `evaluate` are INDEPENDENT implementations of the same IR. A
codegen bug still shows up as an execution mismatch, which is the property
that made the previous verifier worth trusting.

What that does NOT catch: if the IR's *meaning* is misunderstood, both the
assembly and the oracle inherit the mistake and agree with each other. That is
what golden.py exists for -- hand-written cases that never touch this file's
notion of semantics.

All arithmetic is signed 32-bit, matching a `w` register.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

MASK = 0xFFFFFFFF


def s32(x):
    """Wrap to signed 32-bit, the way a w register does."""
    x &= MASK
    return x - (1 << 32) if x & 0x80000000 else x


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Const:
    v: int


@dataclass(frozen=True)
class Arg:
    i: int                      # incoming argument, w0..w2


@dataclass(frozen=True)
class Bin:
    op: str                     # add sub mul and orr eor lsl lsr
    a: object
    b: object


@dataclass(frozen=True)
class Un:
    op: str                     # neg abs
    a: object


@dataclass(frozen=True)
class Cmp:
    op: str                     # eq ne lt le gt ge
    a: object
    b: object                   # evaluates to 0 or 1


@dataclass(frozen=True)
class Sel:
    c: object                   # a Cmp
    a: object                   # taken when the comparison holds
    b: object


@dataclass(frozen=True)
class Loop:
    """reduce(op, map(mapf, filter(pred, [lo..hi])))

    mapf and pred are the compositional axes. Phase 1 only ever generates
    them as None (identity / no filter); Phase 2 opens them up without any
    change to the lowering or the oracle contract.
    """
    op:   str                          # sum product count countdown
    lo:   object
    hi:   object
    mapf: Optional[str] = None         # square double addk mulk
    pred: Optional[Tuple[str, int]] = None   # ('even',0) ('gt',k) ...
    k:    int = 0                      # parameter for mapf


@dataclass(frozen=True)
class Arr:
    """A fixed-size array materialised on the stack."""
    items: Tuple


@dataclass(frozen=True)
class InArr:
    """The caller-supplied array: pointer in x0, length in w1.

    This is the whole point of the array step. ArrLoop already reduces over an
    array with a map and a filter; making that array an ARGUMENT rather than a
    constant baked into the program turns each of those grammar cells into a
    real algorithm over variable-length input -- which is what makes it
    possible to ask the only question that settles "did it learn an algorithm
    or a length": train on short inputs, test on longer ones.
    """


@dataclass(frozen=True)
class InLen:
    """Length of the caller-supplied array."""


@dataclass(frozen=True)
class Index:
    arr: object
    idx: object


@dataclass(frozen=True)
class ArrLoop:
    op:   str
    arr:  object
    mapf: Optional[str] = None
    pred: Optional[Tuple[str, int]] = None
    k:    int = 0


@dataclass
class Program:
    body:   object
    n_args: int = 0
    shape:  str = ""            # grammar cell id, used for held-out splits
    phrase: int = 0             # which wrapper sentence
    style:  int = 0             # which wording of the operation itself


# --------------------------------------------------------------------------
# View 1: the oracle
# --------------------------------------------------------------------------

_BIN = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'and': lambda a, b: (a & MASK) & (b & MASK),
    'orr': lambda a, b: (a & MASK) | (b & MASK),
    'eor': lambda a, b: (a & MASK) ^ (b & MASK),
    'lsl': lambda a, b: (a & MASK) << (b & 31),
    'lsr': lambda a, b: (a & MASK) >> (b & 31),
}

_CMP = {
    'eq': lambda a, b: a == b, 'ne': lambda a, b: a != b,
    'lt': lambda a, b: a <  b, 'le': lambda a, b: a <= b,
    'gt': lambda a, b: a >  b, 'ge': lambda a, b: a >= b,
}

_MAP = {
    None:     lambda i, k: i,
    'square': lambda i, k: i * i,
    'double': lambda i, k: i * 2,
    'addk':   lambda i, k: i + k,
    'mulk':   lambda i, k: i * k,
    'constk': lambda i, k: k,
}


def _keep(pred, i):
    if pred is None:
        return True
    kind, k = pred
    if kind == 'even':  return i % 2 == 0
    if kind == 'odd':   return i % 2 != 0
    if kind == 'gt':    return i > k
    if kind == 'lt':    return i < k
    if kind == 'divk':  return k != 0 and i % k == 0
    raise ValueError(f'unknown predicate {kind}')


def _reduce(op, values):
    if op == 'sum':     return sum(values)
    if op == 'product':
        acc = 1
        for v in values:
            acc = s32(acc * v)
        return acc
    if op == 'count':   return len(values)
    # min/max over an empty selection are defined as 0. The lowering carries a
    # have-a-value flag so it agrees on that case, rather than one side
    # inventing a sentinel the other does not know about.
    if op == 'min':     return min(values) if values else 0
    if op == 'max':     return max(values) if values else 0
    raise ValueError(f'unknown reduce op {op}')


def evaluate(node, args=(), arr=None):
    """Reference semantics. Independent of lower.py by construction.

    `arr` is the caller-supplied input array behind InArr / InLen.
    """
    e = lambda n: evaluate(n, args, arr)
    if isinstance(node, Program):
        return e(node.body)
    if isinstance(node, InArr):
        return list(arr or [])
    if isinstance(node, InLen):
        return len(arr or [])
    if isinstance(node, Const):
        return s32(node.v)
    if isinstance(node, Arg):
        return s32(args[node.i])
    if isinstance(node, Bin):
        return s32(_BIN[node.op](e(node.a), e(node.b)))
    if isinstance(node, Un):
        v = e(node.a)
        return s32(-v) if node.op == 'neg' else s32(abs(v))
    if isinstance(node, Cmp):
        return 1 if _CMP[node.op](e(node.a), e(node.b)) else 0
    if isinstance(node, Sel):
        return e(node.a) if e(node.c) else e(node.b)
    if isinstance(node, Loop):
        lo, hi = e(node.lo), e(node.hi)
        if node.op == 'countdown':
            return 0                     # decrements to zero by construction
        vals = [_MAP[node.mapf](i, node.k)
                for i in range(lo, hi + 1) if _keep(node.pred, i)]
        return s32(_reduce(node.op, vals))
    if isinstance(node, Arr):
        return [e(x) for x in node.items]
    if isinstance(node, Index):
        return e(node.arr)[e(node.idx)]
    if isinstance(node, ArrLoop):
        vals = [_MAP[node.mapf](v, node.k)
                for v in e(node.arr) if _keep(node.pred, v)]
        return s32(_reduce(node.op, vals))
    raise TypeError(f'cannot evaluate {node!r}')


# --------------------------------------------------------------------------
# View 2: the question
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Surface wordings. THREE styles per construct.
#
# The corpus previously had exactly one wording per operation, so the model
# never saw the same meaning expressed two ways and never learned that meaning
# survives rephrasing -- "Sum the array" produced a confident, valid, WRONG
# program. The 16 wrappers did not help: they vary only the packaging, which
# carries no meaning, and the model learned to discard them entirely (all 16
# produce byte-identical output).
#
# Style 0 reproduces the original strings exactly, so the golden set stays
# valid. split.py holds one style out of training entirely, which makes
# paraphrase robustness measurable for the first time.
# ---------------------------------------------------------------------------

N_STYLES = 3

_OPWORD = {
    'add': ('plus', 'added to', 'summed with'),
    'sub': ('minus', 'less', 'reduced by'),
    'mul': ('times', 'multiplied by', 'scaled by'),
    'and': ('bitwise AND', 'AND-ed with', 'masked with'),
    'orr': ('bitwise OR', 'OR-ed with', 'bitwise-or-ed with'),
    'eor': ('bitwise XOR', 'XOR-ed with', 'exclusive-OR-ed with'),
    'lsl': ('shifted left by', 'left-shifted by', 'moved left by'),
    'lsr': ('shifted right by', 'right-shifted by', 'moved right by'),
}

_CMPWORD = {
    'eq': ('equal to', 'the same as', 'exactly'),
    'ne': ('not equal to', 'different from', 'anything other than'),
    'lt': ('less than', 'below', 'smaller than'),
    'le': ('at most', 'no more than', 'not above'),
    'gt': ('greater than', 'above', 'larger than'),
    'ge': ('at least', 'no less than', 'not below'),
}

_NOUN = {
    'sum':     ('the sum of', 'the total of', 'everything added up from'),
    'product': ('the product of', 'the result of multiplying',
                'the product formed from'),
    'count':   ('how many of', 'the number of', 'a count of'),
    'min':     ('the smallest of', 'the minimum of', 'the lowest of'),
    'max':     ('the largest of', 'the maximum of', 'the highest of'),
}

_MAPWORD = {
    None:     ('', '', ''),
    'square': ('the squares of ', 'the squared values of ', 'the second powers of '),
    'double': ('double ', 'twice ', 'two times '),
    'addk':   ('', '', ''),
    'mulk':   ('', '', ''),
    'constk': ('', '', ''),
}

_MAPTAIL = {
    'addk': (', each increased by {k}', ', with {k} added to each',
             ', after adding {k} to each'),
    'mulk': (', each multiplied by {k}', ', with each scaled by {k}',
             ', after multiplying each by {k}'),
}

_PREDWORD = {
    'even': ('even ', 'even-numbered ', 'even-valued '),
    'odd':  ('odd ', 'odd-numbered ', 'odd-valued '),
    'gt':   ('', '', ''),
    'lt':   ('', '', ''),
    'divk': ('', '', ''),
}

_PREDTAIL = {
    'gt':   (' that are greater than {k}', ' above {k}', ' larger than {k}'),
    'lt':   (' that are less than {k}', ' below {k}', ' smaller than {k}'),
    'divk': (' that are divisible by {k}', ' that are multiples of {k}',
             ' divisible evenly by {k}'),
}

_RANGE = ('the {f}integers from {lo} to {hi}',
          'the {f}integers between {lo} and {hi}',
          'every {f}integer from {lo} up to {hi}')

_ARRAY = ('the {f}elements of the array',
          'the {f}values in the array',
          "the array's {f}elements")

_LITERAL = ('the {f}elements of {a}', 'the {f}values in {a}',
            'the {f}entries of {a}')

_UNARY = {
    'neg': ('the negation of {x}', 'minus {x}', 'the negative of {x}'),
    'abs': ('the absolute value of {x}', 'the magnitude of {x}',
            'the size of {x} ignoring sign'),
}


def _w(table, key, style):
    """One wording from a style tuple, wrapping if a table is short."""
    opts = table[key]
    return opts[style % len(opts)]


def _operand(node, style):
    """Render a sub-expression, bracketed when its grouping is not obvious."""
    text = show(node, style)
    return f'({text})' if isinstance(node, (Bin, Cmp, Sel)) else text


def _mods(node, style):
    """(prefix-before-noun, suffix-after-source) for a loop's map and filter."""
    pre = _w(_MAPWORD, node.mapf, style) if node.mapf else ''
    filt = _w(_PREDWORD, node.pred[0], style) if node.pred else ''
    tail = ''
    if node.pred and node.pred[0] in _PREDTAIL:
        tail += _w(_PREDTAIL, node.pred[0], style).format(k=node.pred[1])
    if node.mapf in _MAPTAIL:
        tail += _w(_MAPTAIL, node.mapf, style).format(k=node.k)
    return pre, filt, tail


def show(node, style=0):
    """Render a node as an English noun phrase, in one of N_STYLES wordings."""
    sh = lambda n: show(n, style)

    if isinstance(node, Const):
        return str(node.v)
    if isinstance(node, Arg):
        return f'w{node.i}'
    if isinstance(node, InArr):
        return 'the array'
    if isinstance(node, InLen):
        return 'the length of the array'
    if isinstance(node, Bin):
        return (f'{_operand(node.a, style)} {_w(_OPWORD, node.op, style)} '
                f'{_operand(node.b, style)}')
    if isinstance(node, Un):
        return _w(_UNARY, node.op, style).format(x=sh(node.a))
    if isinstance(node, Cmp):
        return f'{sh(node.a)} is {_w(_CMPWORD, node.op, style)} {sh(node.b)}'
    if isinstance(node, Sel):
        return f'{sh(node.a)} if {sh(node.c)}, otherwise {sh(node.b)}'

    if isinstance(node, Loop):
        if node.op == 'countdown':
            return f'zero after counting down from {sh(node.hi)}'
        if node.mapf == 'constk':
            return (f'the result of adding {node.k} once for every integer '
                    f'from {sh(node.lo)} to {sh(node.hi)}')
        pre, filt, tail = _mods(node, style)
        src = _RANGE[style % len(_RANGE)].format(f=filt, lo=sh(node.lo), hi=sh(node.hi))
        return f'{_w(_NOUN, node.op, style)} {pre}{src}{tail}'

    if isinstance(node, ArrLoop):
        pre, filt, tail = _mods(node, style)
        if isinstance(node.arr, InArr):
            src = _ARRAY[style % len(_ARRAY)].format(f=filt)
        else:
            src = _LITERAL[style % len(_LITERAL)].format(f=filt, a=sh(node.arr))
        return f'{_w(_NOUN, node.op, style)} {pre}{src}{tail}'

    if isinstance(node, Arr):
        return '[' + ', '.join(sh(x) for x in node.items) + ']'
    if isinstance(node, Index):
        return f'element at index {sh(node.idx)} of {sh(node.arr)}'
    raise TypeError(f'cannot render {node!r}')


_PHRASINGS = [
    'Write a function that returns {body} in w0.',
    'Compute {body}, leaving the result in w0.',
    'Return {body} in w0.',
    'Give me assembly that computes {body}. Put the answer in w0.',
    'I need {body}. Leave it in w0.',
    'Calculate {body} and place the result in w0.',
    'Produce {body} in register w0.',
    'How do I compute {body}? The result goes in w0.',
    'Write AArch64 that works out {body}, answer in w0.',
    'The result in w0 should be {body}.',
    'In w0, put {body}.',
    'Emit code for {body}. w0 holds the result.',
    'What assembly computes {body}? Return it in w0.',
    'Store {body} in w0.',
    'Work out {body}; the answer belongs in w0.',
    'Assembly please: {body}, result in w0.',
]


def _has_input_array(n):
    stack = [n]
    while stack:
        x = stack.pop()
        if isinstance(x, (InArr, InLen)):
            return True
        for f in getattr(x, '__dataclass_fields__', {}):
            v = getattr(x, f)
            if isinstance(v, tuple):
                stack.extend(y for y in v if hasattr(y, '__dataclass_fields__'))
            elif hasattr(v, '__dataclass_fields__'):
                stack.append(v)
    return False


def question(prog):
    args = ''
    if _has_input_array(prog.body):
        # State the ABI: without it the task is underspecified and the model
        # would be guessing where the data lives.
        args = ' The array pointer is in x0 and its length in w1.'
    if prog.n_args:
        base = 2 if _has_input_array(prog.body) else 0
        names = ', '.join(f'w{base + i}' for i in range(prog.n_args))
        args += f' The inputs are in {names}.'
    return _PHRASINGS[prog.phrase % len(_PHRASINGS)].format(
        body=show(prog.body, prog.style)) + args
