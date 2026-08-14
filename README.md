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

---

## Setup

```bash
git clone https://github.com/yashodeepv/baby-programmer-small-language-model
cd baby-programmer-small-language-model
python3.11 -m venv .venv && .venv/bin/pip install torch numpy

mkdir -p checkpoints
curl -L -o checkpoints/goal_25M.pth \
  https://github.com/yashodeepv/baby-programmer-small-language-model/releases/download/v1.0-arm/goal_25M.pth
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

---

## Phrasing

**The model knows one wording per operation.** Reword it and you get confident,
valid assembly that computes something else — no error, no warning.

| Works | Fails silently |
|---|---|
| `the sum of the elements of the array` | "Sum the array" |
| `the largest of the elements of the array` | "Find the max" |
| `the sum of the even integers from 2 to 20` | "Add up the evens" |
| `how many of the elements of the array that are greater than 10` | "Count elements over 10" |

The sentence *around* the operation is flexible — "Compute X…", "I need X…",
"What assembly computes X?" all work. Only the phrase naming the operation is
fixed.

List what it knows:

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

| Test | What it measures | Correct |
|---|---|---:|
| seen | trained shapes, new numbers | **90.5%** |
| combo | operation combinations never trained together | **83.0%** |
| size | arrays longer than any seen in training | **86.0%** |
| facts | held-out questions about the instruction set | **95.0%** |
| depth | expressions nested deeper than training | 5.5% |

`combo` says it composes operations it was never shown together. `size` says it
learned an algorithm rather than a per-length template. `depth` is a real wall —
it survived a 13× parameter sweep, deeper training data, and step-by-step
decomposition. The model learned a maximum nesting depth, not a recursion.

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

📐 **[Full architecture, diagrams and design notes →](https://claude.ai/code/artifact/d315f876-103f-45d5-aa5d-bc29d04a46fe)**

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
| `arm/ask.py` | ask the model questions by hand |
| `model.py` | the transformer |
