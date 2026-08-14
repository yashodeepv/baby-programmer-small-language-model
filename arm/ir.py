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
    phrase: int = 0             # which surface phrasing to render


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

_OPWORD = {'add': 'plus', 'sub': 'minus', 'mul': 'times',
           'and': 'bitwise AND', 'orr': 'bitwise OR', 'eor': 'bitwise XOR',
           'lsl': 'shifted left by', 'lsr': 'shifted right by'}

_CMPWORD = {'eq': 'equal to', 'ne': 'not equal to', 'lt': 'less than',
            'le': 'at most', 'gt': 'greater than', 'ge': 'at least'}

_MAPWORD = {None: '', 'square': 'the squares of ', 'double': 'double ',
            'addk': '', 'mulk': '', 'constk': ''}

_PREDWORD = {'even': 'even ', 'odd': 'odd ', 'gt': '', 'lt': '', 'divk': ''}


def _operand(node):
    """Render a sub-expression, bracketed when its grouping is not obvious."""
    text = show(node)
    return f'({text})' if isinstance(node, (Bin, Cmp, Sel)) else text


def show(node):
    """Render a node as an English noun phrase."""
    if isinstance(node, Const):
        return str(node.v)
    if isinstance(node, Arg):
        return f'w{node.i}'
    if isinstance(node, Bin):
        # Nested operands MUST be bracketed. "21 plus 95 times 42" reads by
        # ordinary precedence as 21 + (95*42), but the tree means (21+95)*42 --
        # the oracle and the assembly would agree with each other while the
        # QUESTION says something else, training the model on a mislabelled
        # example that no amount of execution testing can catch.
        return f'{_operand(node.a)} {_OPWORD[node.op]} {_operand(node.b)}'
    if isinstance(node, Un):
        return (f'the negation of {show(node.a)}' if node.op == 'neg'
                else f'the absolute value of {show(node.a)}')
    if isinstance(node, Cmp):
        return f'{show(node.a)} is {_CMPWORD[node.op]} {show(node.b)}'
    if isinstance(node, Sel):
        return f'{show(node.a)} if {show(node.c)}, otherwise {show(node.b)}'
    if isinstance(node, Loop):
        if node.op == 'countdown':
            return f'zero after counting down from {show(node.hi)}'
        if node.mapf == 'constk':
            return (f'the result of adding {node.k} once for every integer '
                    f'from {show(node.lo)} to {show(node.hi)}')
        body = _MAPWORD[node.mapf]
        filt = _PREDWORD[node.pred[0]] if node.pred else ''
        noun = {'sum': 'the sum of', 'product': 'the product of',
                'count': 'how many of', 'min': 'the smallest of',
                'max': 'the largest of'}[node.op]
        tail = f'the {filt}integers from {show(node.lo)} to {show(node.hi)}'
        extra = ''
        if node.pred and node.pred[0] in ('gt', 'lt', 'divk'):
            w = {'gt': 'greater than', 'lt': 'less than',
                 'divk': 'divisible by'}[node.pred[0]]
            extra = f' that are {w} {node.pred[1]}'
        if node.mapf == 'addk':
            extra += f', each increased by {node.k}'
        if node.mapf == 'mulk':
            extra += f', each multiplied by {node.k}'
        return f'{noun} {body}{tail}{extra}'
    if isinstance(node, InArr):
        return 'the array'
    if isinstance(node, InLen):
        return 'the length of the array'
    if isinstance(node, Arr):
        return '[' + ', '.join(show(x) for x in node.items) + ']'
    if isinstance(node, Index):
        return f'element at index {show(node.idx)} of {show(node.arr)}'
    if isinstance(node, ArrLoop):
        # The predicate MUST reach the question. Without it a filtered loop
        # reads exactly like an unfiltered one, and the label is wrong while
        # the oracle and the assembly still agree -- the same failure as the
        # unbracketed nested expression.
        noun = {'sum': 'the sum of', 'product': 'the product of',
                'count': 'how many of', 'min': 'the smallest of',
                'max': 'the largest of'}[node.op]
        body = _MAPWORD[node.mapf]
        filt, extra = '', ''
        if node.pred:
            kind, k = node.pred
            filt = _PREDWORD[kind]
            if kind in ('gt', 'lt', 'divk'):
                w = {'gt': 'greater than', 'lt': 'less than',
                     'divk': 'divisible by'}[kind]
                extra = f' that are {w} {k}'
        if node.mapf == 'addk':
            extra += f', each increased by {node.k}'
        if node.mapf == 'mulk':
            extra += f', each multiplied by {node.k}'
        where = 'the array' if isinstance(node.arr, InArr) else show(node.arr)
        return f'{noun} {body}the {filt}elements of {where}{extra}'
    raise TypeError(f'cannot render {node!r}')


# Surface variety. With only three fixed phrasings the model was barely tested
# on "based on what is described" -- and the novel-task probe showed the
# failure mode, where it matched a familiar sentence shape and silently dropped
# the qualifier that changed the code. These vary the verb, the framing, the
# register mention and the word order, so the DESCRIPTION has to be read rather
# than pattern-matched.
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
        body=show(prog.body)) + args
