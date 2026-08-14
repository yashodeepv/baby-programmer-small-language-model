"""
facts.py - Factual Q&A about the AArch64 subset this project uses.

The goal is a model that answers basic questions about ARM assembly AND writes
basic code. Everything before this served only the second half: the corpus was
entirely "describe a computation, get assembly", with no way to ask what an
instruction does.

These are hand-written facts rather than generated ones, because there is no
oracle for "what does cset mean" -- you cannot execute an English sentence. So
unlike the program corpus, this part is trusted rather than verified, and it is
deliberately kept small, factual and checkable by eye.

Scope is the instruction subset the generator actually emits. Claiming coverage
of instructions the model never sees would be teaching it to bluff.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (instruction, one-line meaning, worked example, what the example leaves behind)
INSTRUCTIONS = [
    ('mov',  'copies a value into a register',
     'mov w0, #42', 'w0 holds 42'),
    ('movk', 'writes 16 bits into part of a register, keeping the rest',
     'movk w0, #999, lsl #16', 'the high half of w0 becomes 999'),
    ('add',  'adds two values',
     'add w0, w1, w2', 'w0 holds w1 + w2'),
    ('sub',  'subtracts the second value from the first',
     'sub w0, w1, w2', 'w0 holds w1 - w2'),
    ('mul',  'multiplies two values',
     'mul w0, w1, w2', 'w0 holds w1 * w2'),
    ('neg',  'negates a value',
     'neg w0, w1', 'w0 holds -w1'),
    ('and',  'computes a bitwise AND',
     'and w0, w1, w2', 'w0 holds w1 & w2'),
    ('orr',  'computes a bitwise OR',
     'orr w0, w1, w2', 'w0 holds w1 | w2'),
    ('eor',  'computes a bitwise exclusive OR',
     'eor w0, w1, w2', 'w0 holds w1 ^ w2'),
    ('lsl',  'shifts a value left, filling with zeros',
     'lsl w0, w1, #3', 'w0 holds w1 multiplied by 8'),
    ('lsr',  'shifts a value right, filling with zeros',
     'lsr w0, w1, #3', 'w0 holds w1 divided by 8, rounded down'),
    ('cmp',  'compares two values and sets the condition flags',
     'cmp w0, w1', 'the flags describe w0 - w1, and no register changes'),
    ('cset', 'sets a register to 1 when a condition holds, otherwise 0',
     'cset w0, gt', 'w0 holds 1 if the last comparison was greater than'),
    ('csel', 'selects between two registers based on a condition',
     'csel w0, w1, w2, eq', 'w0 holds w1 when equal, otherwise w2'),
    ('ldr',  'loads a value from memory into a register',
     'ldr w0, [x1, #4]', 'w0 holds the word four bytes past the address in x1'),
    ('str',  'stores a register into memory',
     'str w0, [x1, #0]', 'the word at the address in x1 becomes w0'),
    ('b',    'jumps to a label unconditionally',
     'b loop', 'execution continues at loop'),
    ('bl',   'calls a function, saving the return address in x30',
     'bl _write', 'execution continues at _write and returns afterwards'),
    ('ret',  'returns from a function, jumping to the address in x30',
     'ret', 'control goes back to the caller'),
]

CONDITIONS = [
    ('eq', 'equal'), ('ne', 'not equal'),
    ('lt', 'less than'), ('le', 'less than or equal'),
    ('gt', 'greater than'), ('ge', 'greater than or equal'),
]

REGISTERS = [
    ('w0', 'the 32-bit result register, and the first argument'),
    ('x0', 'the 64-bit form of w0, used here for pointers'),
    ('w1', 'the second argument; in this project the array length'),
    ('sp', 'the stack pointer'),
    ('x30', 'the link register, holding the return address'),
    ('wzr', 'the zero register, which always reads as 0'),
]

CONCEPTS = [
    ('the difference between b and bl',
     'b jumps and does not come back. bl saves the return address in x30 first, '
     'so ret can return to the caller.'),
    ('why a label ends with a colon',
     'A colon marks a branch target. b loop jumps to the line written loop:.'),
    ('the difference between w0 and x0',
     'They are the same register. w0 is the low 32 bits, x0 is all 64.'),
    ('how a comparison works',
     'cmp sets flags describing the subtraction, then a conditional branch or '
     'cset reads those flags. cmp itself changes no register.'),
    ('why the stack pointer must stay 16-byte aligned',
     'AArch64 requires sp to be a multiple of 16, so reserve space in multiples '
     'of 16 with sub sp, sp, #16 and release it with add sp, sp, #16.'),
    ('how a loop is written',
     'Put a label at the top, compare, branch past the body when done, and '
     'branch back to the label at the bottom.'),
    ('where a function returns its result',
     'In w0 for a 32-bit value, or x0 for 64 bits.'),
    ('what uxtw means in ldr w0, [x1, w2, uxtw #2]',
     'It widens the 32-bit index w2 to 64 bits and multiplies it by 4, so w2 '
     'indexes words rather than bytes.'),
]

_Q_INSTR = [
    'What does the {m} instruction do?',
    'Explain {m}.',
    'What is {m} used for?',
    'What happens when I write {m}?',
]

_Q_COND = [
    'What does the condition {c} mean?',
    'When does a {c} branch or cset fire?',
]

_Q_REG = [
    'What is {r} used for?',
    'What does the register {r} hold?',
]


def qa_pairs(rng, n):
    """Generate n factual question/answer pairs."""
    out = []
    while len(out) < n:
        kind = rng.random()
        if kind < 0.55:
            m, meaning, ex, eff = rng.choice(INSTRUCTIONS)
            q = rng.choice(_Q_INSTR).format(m=m)
            a = f'{m} {meaning}. For example, {ex} means {eff}.'
        elif kind < 0.7:
            c, meaning = rng.choice(CONDITIONS)
            q = rng.choice(_Q_COND).format(c=c)
            a = f'{c} means {meaning}. It tests the flags set by the last cmp.'
        elif kind < 0.85:
            r, meaning = rng.choice(REGISTERS)
            q = rng.choice(_Q_REG).format(r=r)
            a = f'{r} is {meaning}.'
        else:
            topic, answer = rng.choice(CONCEPTS)
            q = f'Explain {topic}.'
            a = answer
        out.append((q, a))
    return out


def render(q, a):
    return f'USER: {q}\nASSISTANT:\n{a}\n'


def sample_text(n, seed=0):
    rng = random.Random(seed)
    return [render(q, a) for q, a in qa_pairs(rng, n)]


def eval_set(n, seed=999):
    """Held-out factual questions, paired with their reference answers."""
    rng = random.Random(seed)
    return qa_pairs(rng, n)


def score(model, tok, pairs, max_new_tokens=90):
    """Exact-match accuracy on held-out factual questions.

    Exact match is strict but fair here: the answers come from a fixed table,
    so the correct string is well defined and there is no oracle to run.
    """
    import torch
    device = next(model.parameters()).device
    ok = 0
    got_examples = []
    for q, want in pairs:
        ids = tok.encode(f'USER: {q}\nASSISTANT:', allow_unk=True)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens, greedy=True)
        text = tok.decode(out[0].tolist()[len(ids):])
        line = next((l.strip() for l in text.split('\n') if l.strip()), '')
        if line == want:
            ok += 1
        elif len(got_examples) < 3:
            got_examples.append((q, want, line))
    return ok / max(len(pairs), 1), got_examples


if __name__ == '__main__':
    for line in sample_text(6, seed=1):
        print(line)
