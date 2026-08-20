"""
ask.py - Talk to the model by hand.

    # one question
    .venv/bin/python arm/ask.py "Sum the even integers from 2 to 20 in w0."

    # a factual question
    .venv/bin/python arm/ask.py "What does the cset instruction do?"

    # generate, then assemble and RUN it, printing what w0 actually holds
    .venv/bin/python arm/ask.py --run "Return the largest of 17 and 42 in w0."

    # array questions need input data
    .venv/bin/python arm/ask.py --run --array 4,9,2,7 \\
        "Sum the elements of the array in w0. The array pointer is in x0 and its length in w1."

    # interactive
    .venv/bin/python arm/ask.py

`--run` executes what the model wrote and reports the value it returns. It
cannot tell you whether that value is *correct* -- there is no oracle for a
question you typed yourself -- so it reports what the code does and leaves the
judgement to you.
"""

import argparse
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import load_checkpoint
from tokenizer import ArmTokenizer

MNEMONICS = ('mov', 'add', 'sub', 'mul', 'and', 'orr', 'eor', 'lsl', 'lsr',
             'cmp', 'cset', 'csel', 'ldr', 'str', 'neg', 'movk', 'ret', 'b')


def looks_like_code(lines):
    return any(l.split()[0].rstrip(',') in MNEMONICS for l in lines if l.split())


def answer(model, tok, question, max_new_tokens=300):
    device = next(model.parameters()).device
    ids = tok.encode(f'USER: {question}\nASSISTANT:', allow_unk=True)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, greedy=True,
                         stop_token=tok.stoi.get('ret'))
    text = tok.decode(out[0].tolist()[len(ids):])
    # Nothing stops the model rolling straight into an invented next turn, so
    # cut at the first one before deciding what kind of answer this is.
    text = text.split('USER:')[0]

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines and looks_like_code(lines):
        body = []
        for l in lines:
            body.append(l)
            if l == 'ret':
                break
        return 'code', body
    return 'text', lines[:1]


def run_code(body, array=None, scalars=()):
    """Assemble the generated function, call it, return what w0 holds."""
    from verify_ir import emit_function, _load_imm
    setup = ''
    if array is not None:
        setup += '    adrp x6, _buf@PAGE\n    add x6, x6, _buf@PAGEOFF\n'
        for j, v in enumerate(array):
            setup += _load_imm('w7', v) + f'    str w7, [x6, #{j * 4}]\n'
        setup += '    mov x0, x6\n' + _load_imm('w1', len(array))
        setup += ''.join(_load_imm(f'w{2 + i}', a) for i, a in enumerate(scalars))
    else:
        setup += ''.join(_load_imm(f'w{i}', a) for i, a in enumerate(scalars))

    # The exit status is only the low byte, which silently truncates any answer
    # above 255 -- a product of odd numbers came back as 115 instead of
    # 654,729,075. Write the full 32-bit result to stdout instead and decode it.
    src = f""".section __DATA,__data
.p2align 3
_buf:
    .space 256
.p2align 3
_out:
    .space 4

.section __TEXT,__text
{emit_function(0, [l for l in body if l != 'ret'])}
.global _main
.p2align 2
_main:
    stp x29, x30, [sp, #-16]!
{setup}    bl _p0
    adrp x1, _out@PAGE
    add x1, x1, _out@PAGEOFF
    str w0, [x1]
    mov x0, #1
    mov x2, #4
    bl _write
    mov w0, #0
    ldp x29, x30, [sp], #16
    ret
"""
    with tempfile.TemporaryDirectory() as wd:
        s = os.path.join(wd, 'a.s')
        b = os.path.join(wd, 'a.out')
        with open(s, 'w') as f:
            f.write(src)
        r = subprocess.run(['clang', s, '-o', b], capture_output=True, text=True)
        if r.returncode != 0:
            return None, r.stderr.strip().splitlines()[:3]
        try:
            # exit status is the low byte only, so read the full value via echo
            p = subprocess.run([b], capture_output=True, timeout=5)
            if len(p.stdout) < 4:
                return None, ['produced no result']
            return int.from_bytes(p.stdout[:4], 'little', signed=True), None
        except subprocess.TimeoutExpired:
            return None, ['did not terminate']


def translate(line, langs):
    """An s-expression answer is an IR, so it compiles to any target we have.

    This is the whole point of --target sexpr: the model states the algorithm
    and the translation is a deterministic pass, not something it learned.
    """
    import sexpr
    from backends import TARGETS
    try:
        node = sexpr.parse(line)
    except Exception:
        return None
    out = []
    for name in langs:
        tgt = TARGETS.get(name)
        if tgt is None:
            out.append((name, f'(no backend named {name!r}; '
                              f'have {", ".join(sorted(TARGETS))})'))
            continue
        try:
            out.append((name, tgt().compile(node)))
        except Exception as exc:
            out.append((name, f'({type(exc).__name__}: {exc})'))
    return out


def show(model, tok, q, do_run, array, scalars, expect=None, langs=()):
    kind, body = answer(model, tok, q)
    print()
    if kind == 'text':
        line = body[0] if body else ''
        print(line or '(no answer)')
        if langs and line:
            tr = translate(line, langs)
            if tr is None:
                print('\n   not an s-expression, so there is no IR to compile')
            else:
                for name, src in tr:
                    print(f'\n--- {name} ---')
                    print(src.rstrip())
        return
    for l in body:
        print(('' if l.endswith(':') else '    ') + l)
    if langs:
        print('\n   this checkpoint emits assembly, not IR, so there is nothing to '
              'translate.\n   use an s-expression checkpoint: '
              '--ckpt checkpoints/sexpr_v2.pth --lang c,java,python')
    if do_run:
        val, err = run_code(body, array, scalars)
        print()
        if err:
            print('   did not build/run:', err[0])
        elif expect is None:
            print(f'   ran -> w0 = {val}')
        else:
            print(f'   ran -> w0 = {val}   you expected {expect}   '
                  f'{"PASS" if val == expect else "FAIL"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('question', nargs='*', help='ask one question and exit')
    ap.add_argument('--ckpt', default='checkpoints/arm_25M_v1.1.pth')
    ap.add_argument('--run', action='store_true', help='assemble and execute the answer')
    ap.add_argument('--array', default=None, help='comma-separated input array')
    ap.add_argument('--args', default=None, help='comma-separated scalar inputs')
    ap.add_argument('--lang', default='',
                    help='comma-separated target languages to compile the IR to '
                         '(c, java, python, javascript, or all). Needs a '
                         'checkpoint trained with --target sexpr.')
    ap.add_argument('--expect', type=int, default=None,
                    help='what you believe the answer is; prints PASS/FAIL')
    a = ap.parse_args()

    model, stoi, itos, _ = load_checkpoint(a.ckpt)
    tok = ArmTokenizer([itos[i] for i in range(len(itos))])
    arr = [int(x) for x in a.array.split(',')] if a.array else None
    sca = tuple(int(x) for x in a.args.split(',')) if a.args else ()

    from backends import TARGETS
    langs = ([] if not a.lang else
             sorted(TARGETS) if a.lang == 'all' else
             [x.strip() for x in a.lang.split(',') if x.strip()])

    if a.question:
        show(model, tok, ' '.join(a.question), a.run, arr, sca, a.expect, langs)
        return 0

    print(f'{a.ckpt} loaded. Ask a question, or Ctrl-D to quit.')
    print('Prefix with "!" to also assemble and run the answer.\n')
    while True:
        try:
            q = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        do_run = q.startswith('!')
        show(model, tok, q.lstrip('!').strip(), do_run, arr, sca, a.expect, langs)
        print()


if __name__ == '__main__':
    sys.exit(main())
