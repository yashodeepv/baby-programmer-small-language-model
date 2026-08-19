# baby-programmer

Writes AArch64 assembly from a plain-English description, and answers basic
questions about the instruction set. **25.8M parameters.**

Trained on a corpus where every example was **compiled and executed** before it
became training data — 26,000 programs, ~312,000 executions, zero unverified
samples.

```
USER: Compute the sum of the even integers from 2 to 20, leaving the result in w0.

    mov w1, #2
    mov w2, #20
    mov w0, #0
loop1:
    cmp w1, w2
    b.gt done2
    mov w3, w1
    and w4, w3, #1        ← the "even" filter
    cmp w4, #0
    b.ne skip3
    add w0, w0, w3
skip3:
    add w1, w1, #1
    b loop1
done2:
    ret

   ran -> w0 = 110   PASS
```

> [!IMPORTANT]
> **This is a very small model** — 25.8M parameters, trained for about 100
> minutes on a single generated corpus. It is not a general assistant and does
> not understand free English. It answers questions built from a **fixed
> vocabulary** (below), and anything outside it produces confident, valid
> assembly that computes the wrong thing, with no error.
>
> An [interactive query builder](https://yashodeepv.github.io/baby-programmer-small-language-model/#builder)
> constructs valid questions for you.

---

## Setup

```bash
git clone https://github.com/yashodeepv/baby-programmer-small-language-model
cd baby-programmer-small-language-model
python3.11 -m venv .venv && .venv/bin/pip install torch numpy

mkdir -p checkpoints
curl -L -o checkpoints/arm_25M_v1.1.pth \
  https://github.com/yashodeepv/baby-programmer-small-language-model/releases/download/v1.1-arm/arm_25M_v1.1.pth
```

Needs `clang` — the tool assembles and runs what the model writes. On macOS:
`xcode-select --install`.

The checkpoint is not in the repo (103 MB, over GitHub's file limit); it ships
as a release asset. `-L` matters on that `curl` — without it you get a redirect
stub instead of the model.

---

## Use it

```bash
.venv/bin/python arm/ask.py --run --expect 110 \
  "Compute the sum of the even integers from 2 to 20, leaving the result in w0."
#    ran -> w0 = 110   you expected 110   PASS
```

`--run` assembles and executes the answer. `--expect` checks the result.

```bash
# array input — passed as pointer in x0, length in w1
.venv/bin/python arm/ask.py --run --array 4,9,2,7,30 --expect 52 \
  "Compute the sum of the elements of the array, leaving the result in w0. \
   The array pointer is in x0 and its length in w1."

# scalar inputs
.venv/bin/python arm/ask.py --run --args 12,188 --expect 188 \
  "Compute w0 if w0 is at least w1, otherwise w1, leaving the result in w0. \
   The inputs are in w0, w1."

# factual question
.venv/bin/python arm/ask.py "What does the cset instruction do?"

# interactive; prefix with ! to also run the answer
.venv/bin/python arm/ask.py
```

| Flag | Effect |
|---|---|
| `--run` | assemble and execute the generated code |
| `--expect N` | compare the result to N, print PASS / FAIL |
| `--array 4,9,2` | input array (pointer in `x0`, length in `w1`) |
| `--args 12,188` | scalar inputs in `w0, w1` — or `w2, w3` when an array is present |
| `--ckpt PATH` | use a different checkpoint |

> [!WARNING]
> `--expect N` compares the executed result against `N` and nothing else. It has
> no idea what you asked for, so a stale `--expect` left over from a previous
> command will happily print **PASS** for a wrong answer. Trust it only when you
> just typed the expected value for the question you just typed.

---

## Phrasing

**The model knows two wordings per operation.** A third exists in the generator
but is held out of training, so it can be used to measure how well the model
survives rephrasing — it scores 18.5% there, against 90.0% on wordings it knows.
Stray outside the two and you get confident, valid assembly that computes
something else — no error, no warning.

A question is built from two parts. The **wrapper** is flexible; the **operation
phrase** is nearly fixed.

### The 16 wrappers

Defined in `arm/ir.py` as `_PHRASINGS`. `{body}` is where the operation phrase
goes. Any of these work; nothing else does.

| # | Wrapper |
|---|---|
| 1 | `Write a function that returns {body} in w0.` |
| 2 | `Compute {body}, leaving the result in w0.` |
| 3 | `Return {body} in w0.` |
| 4 | `Give me assembly that computes {body}. Put the answer in w0.` |
| 5 | `I need {body}. Leave it in w0.` |
| 6 | `Calculate {body} and place the result in w0.` |
| 7 | `Produce {body} in register w0.` |
| 8 | `How do I compute {body}? The result goes in w0.` |
| 9 | `Write AArch64 that works out {body}, answer in w0.` |
| 10 | `The result in w0 should be {body}.` |
| 11 | `In w0, put {body}.` |
| 12 | `Emit code for {body}. w0 holds the result.` |
| 13 | `What assembly computes {body}? Return it in w0.` |
| 14 | `Store {body} in w0.` |
| 15 | `Work out {body}; the answer belongs in w0.` |
| 16 | `Assembly please: {body}, result in w0.` |

### Operation phrases — a fixed vocabulary

`X` above is assembled from these slots, and **only** these:

Each slot has two trained wordings — either works.

| Slot | Wording 1 | Wording 2 |
|---|---|---|
| reduce | `the sum of` · `the product of` · `how many of` · `the smallest of` · `the largest of` | `the total of` · `the result of multiplying` · `the number of` · `the minimum of` · `the maximum of` |
| source | `the integers from A to B` · `the elements of the array` · `the elements of [1, 2, 3]` | `the integers between A and B` · `the values in the array` · `the values in [1, 2, 3]` |
| filter | `even ` · `odd ` (before the noun) · `that are greater than K` · `that are less than K` · `that are divisible by K` (after) | `even-numbered ` · `odd-numbered ` (before) · `above K` · `below K` · `that are multiples of K` (after) |
| map | `the squares of ` · `double ` (before) · `, each increased by K` · `, each multiplied by K` (after) | `the squared values of ` · `twice ` (before) · `, with K added to each` · `, with each scaled by K` (after) |

Composed, that reads: `the sum of the squares of the even integers from 2 to 20`
— or `the total of the squared values of the even-numbered integers between 2
and 20`, which the model handles equally well.

The full tables are `_NOUN`, `_RANGE`, `_ARRAY`, `_PREDWORD`, `_PREDTAIL`,
`_MAPWORD` and `_MAPTAIL` in `arm/ir.py`; each is a 3-tuple whose third entry is
the held-out wording.

Other forms it knows: `element at index N of [...]`, `the length of the array`,
`the negation of N`, `the absolute value of N`, `A plus B`, `A times B`,
`A bitwise AND B`, `A shifted left by N`, `A if A is at least B, otherwise B`,
`1 if A is greater than B, otherwise 0`, `zero after counting down from N`.

### Consequence

| Works | Fails silently |
|---|---|
| `the sum of the elements of the array` | "Sum the array" |
| `the largest of the elements of the array` | "Find the max" |
| `the sum of the even integers from 2 to 20` | "Add up the evens" |
| `how many of the elements of the array that are greater than 10` | "Count elements over 10" |

All four of those still fail in v1.1 — verified, not assumed. Two wordings per
operation buys tolerance for rephrasings that keep the sentence shape (`Total
the elements of the array, leaving the result in w0.` works), but not for terse
reformulations that drop the grammar's structure entirely.

This remains a **corpus limitation, not a model one**. `ir.show()` now renders
three wordings per node where it once had one, and that measurably helped —
see [Accuracy](#accuracy) — but the held-out-wording score of 18.5% says the
generator still has to teach the wording before the model can read it.

List real examples:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'arm')
import grammar; from ir import question
for p in grammar.sample(12, seed=7): print(question(p))"
```

---

## Scope

**Handles:** sum, product, count, smallest, largest — over an integer range, a
literal array, or a caller-supplied array — with an optional filter (even, odd,
greater/less than *k*, divisible by *k*) and an optional map (squares, doubles,
each increased or multiplied by *k*). Plus arithmetic, bitwise operations,
comparisons, if/else, indexing, absolute value. Plus factual questions about the
instructions it emits.

**Does not handle:** sorting, searching, strings, division, floating point, or
expressions nested deeper than it was trained on.

---

## Accuracy

Each answer compiled and run against a reference oracle on ~12 inputs. Correct
means **every** input matched.

Measured at n=200 per cell on `arm_25M_v1.1`.

| Test | What it measures | Correct |
|---|---|---:|
| seen | trained shapes, new numbers | **90.0%** |
| combo | operation combinations never trained together | **85.0%** |
| size | arrays longer than any seen in training | **88.5%** |
| facts | held-out questions about the instruction set | **95.0%** |
| constants | values inside the trained range | **95.0%** |
| constants | values *outside* the trained range | 80.0% |
| paraphrase | a wording of the operation never trained on | 18.5% |
| depth | expressions nested deeper than training | 4.0% |

`combo` says it composes operations it was never shown together. `size` says it
learned an algorithm rather than a per-length template.

`constants` is the newest cell and the one worth understanding. Reproduce it
with `.venv/bin/python arm/eval_bigconst.py --ckpt checkpoints/arm_25M_v1.1.pth`;
the other rows come from `arm/eval_ckpt.py --eval-n 200`. Range bounds are
sampled at `hi ≤ 100` during training (`arm/grammar.py:179-180`), so anything
larger is extrapolation. The previous release truncated silently there — asked
for the sum of the evens from 2 to 200 it emitted `mov w2, #20` and returned
110, a wrong answer that compiles, runs, and looks right. v1.1 returns 10100.

Plain constants are a separate slot with its own ceiling: they are sampled
`0–500` (`arm/grammar.py:134`) and both releases truncate above it. `Compute
750` returns `mov w0, #75`. Nothing in the model handles four-digit constants,
so treat any value over 500 as unsupported and check the emitted `mov` against
what you asked for.

`depth` is a real wall — it survived a 13× parameter sweep, deeper training
data, and step-by-step decomposition. The model learned a maximum nesting
depth, not a recursion.

---

## How it works

The unit of training data is not a string — it is a sampled **IR tree**, from
which three views are derived:

```
Loop(op='sum', lo=2, hi=20, pred=('even',0))
    ├── render()   → "the sum of the even integers from 2 to 20"
    ├── lower()    → AArch64 with real register allocation
    └── evaluate() → 110
```

`lower()` and `evaluate()` are independent implementations of the same
semantics, so a codegen bug shows up as an execution mismatch. Every generated
program is compiled and run against the oracle on ~12 inputs before it enters
the corpus; anything that disagrees stops the build.

📐 **[Full architecture, diagrams and design notes →](https://yashodeepv.github.io/baby-programmer-small-language-model/)**

---

## Retrain

```bash
.venv/bin/python arm/golden.py                  # check the generator against hand-written truth
.venv/bin/python arm/build_corpus.py --n 20000  # generate + verify a corpus
.venv/bin/python arm/train_comp.py --steps 3000 \
    --n-embd 512 --n-head 8 --n-layer 8 --facts 2500 --tag mymodel
```

About 100 minutes on an Apple M-series GPU.

---

## Repository layout

| Path | Role |
|---|---|
| `arm/ir.py` | IR node types, the oracle, the question renderer |
| `arm/lower.py` | IR → AArch64, deterministic register allocation |
| `arm/grammar.py` | samples IR structures; defines grammar cells |
| `arm/verify_ir.py` | batch compile + execute + compare against the oracle |
| `arm/watchdog.py` | progress-based hang detection |
| `arm/golden.py` | hand-written ground truth for all three views |
| `arm/split.py` | held-out cells, input sizes, depths |
| `arm/facts.py` | factual instruction-set question/answer pairs |
| `arm/tokenizer.py` | closed-vocabulary reversible tokenizer |
| `arm/build_corpus.py` | generate + verify + write a corpus |
| `arm/train_comp.py` | training loop with five evaluation metrics |
| `arm/eval_comp.py` | generate → build → execute → compare |
| `arm/eval_ckpt.py` | score a saved checkpoint on every cell, no retraining |
| `arm/eval_bigconst.py` | score constants outside the trained range |
| `arm/ask.py` | ask the model questions by hand |
| `model.py` | the transformer |
