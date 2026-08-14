"""
watchdog.py - Run a built binary and kill it only when it stops making progress.

Extracted from the first-iteration generator because it is current,
load-bearing code: every verification path depends on it, including the
compositional one.

A fixed deadline cannot tell "hung" from "slow to start". The first execution
of a freshly built binary pays an OS code-signing check that is serialised
machine-wide, so under parallel workers a healthy batch can sit for seconds
before its first instruction runs -- and a short timeout charges those as hangs
and retries forever. Since each verified sample writes one byte as it finishes,
progress is the honest signal.
"""

import os
import selectors
import subprocess
import time

# Hang detection is based on PROGRESS, not total elapsed time. A fixed deadline
# cannot tell "hung" from "slow to start": the first exec of a fresh binary pays
# an OS code-signing check (~0.45s) that is serialized machine-wide, so under
# parallel workers a perfectly healthy batch can sit for seconds before its
# first instruction runs. A short fixed timeout charges those as hangs and
# retries forever. Each sample writes exactly one byte as it finishes, so:
START_GRACE    = 30   # seconds to wait for the FIRST byte (code-signing queue)
IDLE_TIMEOUT   = 5    # seconds without a new byte once running == hung
SAMPLE_TIMEOUT = 10   # seconds for one isolated single-sample binary

BATCH_HARNESS = """.section __DATA,__data
.p2align 3
_mark:
    .space 1

.section __TEXT,__text
{functions}
.global _main
.p2align 2
_main:
    stp x29, x30, [sp, #-16]!
    mov x29, sp
{checks}
    mov w0, #0
    ldp x29, x30, [sp], #16
    ret
"""


def emit_function(k, sample):
    """Render one sample as a standalone function, namespacing its labels.

    The corpus keeps canonical `loop:`/`done:` names; only this verification
    copy rewrites them, since all functions share one translation unit.
    """
    labels = [ln[:-1] for ln in sample.body if ln.endswith(':')]
    lines  = []
    for ln in sample.body:
        for lab in labels:
            ln = re.sub(rf'\b{re.escape(lab)}\b', f's{k}_{lab}', ln)
        lines.append(ln if ln.endswith(':') else '    ' + ln)
    body = '\n'.join(lines)
    return f'.p2align 2\n_s{k}:\n{body}\n    ret\n'


def emit_check(k, expect):
    """Call sample k and immediately write '.' (pass) or 'F' (fail) to stdout.

    Writing per sample rather than buffering to the end is what makes a hang
    attributable: `_write` is an unbuffered syscall, so if the binary is killed
    on timeout the bytes already received name exactly how many samples
    finished -- the next one is the one that never returned.
    """
    if not 0 <= expect <= 0xFFFFFFFF:
        raise ValueError(f'expect {expect} does not fit a 32-bit register')
    # movz gives 16 bits; anything larger needs movk for the high half. Without
    # this the verifier caps every template's result at 65535, which is what
    # squeezed t_product_range down to 297 distinct questions.
    lo, hi = expect & 0xFFFF, (expect >> 16) & 0xFFFF
    load = f'mov w1, #{lo}'
    if hi:
        load += f'\n    movk w1, #{hi}, lsl #16'
    return f"""    bl _s{k}
    {load}
    cmp w0, w1
    mov w3, #46
    mov w4, #70
    csel w3, w3, w4, eq
    adrp x5, _mark@PAGE
    add x5, x5, _mark@PAGEOFF
    strb w3, [x5]
    mov x0, #1
    mov x1, x5
    mov x2, #1
    bl _write"""


def _exec_watched(bin_path):
    """Run a batch binary, killing it only when it stops making progress.

    Returns (bytes_received, hung). Waits START_GRACE for the first byte --
    that window is dominated by the machine-wide code-signing queue, not by the
    program -- then requires a new byte every IDLE_TIMEOUT seconds.
    """
    p = subprocess.Popen([bin_path], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    buf  = bytearray()
    last = time.monotonic()
    os.set_blocking(p.stdout.fileno(), False)
    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ)
    try:
        while True:
            if sel.select(timeout=0.2):
                chunk = p.stdout.read()
                if chunk:
                    buf += chunk
                    last = time.monotonic()
                    continue
                if chunk == b'':          # EOF: process finished writing
                    break
            if p.poll() is not None:
                trailing = p.stdout.read()
                if trailing:
                    buf += trailing
                break
            limit = START_GRACE if not buf else IDLE_TIMEOUT
            if time.monotonic() - last > limit:
                p.kill()
                p.wait()
                return bytes(buf), True
    finally:
        sel.close()
        p.stdout.close()
        if p.poll() is None:
            p.kill()
        p.wait()
    return bytes(buf), False


