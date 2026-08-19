# v1.1-arm — paraphrase-aware corpus, constant extrapolation fixed

A **25.8M-parameter** language model that writes AArch64 assembly from
plain-English descriptions and answers basic questions about the instruction
set — trained on a corpus where **every example was compiled and executed
before it became training data**. 26,000 programs, ~312,000 executions, zero
unverified samples.

## What changed since v1.0

**Three wordings per operation** instead of one. Two are trained; the third is
held out so paraphrase robustness can be measured rather than assumed.

**Constants outside the trained range no longer truncate.** This is the
user-visible fix. Range bounds are sampled at `hi ≤ 100` in training, and v1.0
silently dropped digits beyond that — asked for the sum of the evens from 2 to
200 it emitted `mov w2, #20` and returned 110. It compiled, it ran, it looked
right, and it was wrong. v1.1 returns 10100.

| | v1.0 | v1.1 |
|---|---:|---:|
| constants inside the trained range | 100% | 95% |
| constants **outside** the trained range | **22%** | **80%** |

`countdown` is also fixed: v1.0 returned the loop bound instead of zero, and at
some bounds emitted code that would not link.

Worth being straight about the cost: v1.1 is a point or two behind v1.0 on
in-distribution cells (`seen` 91.0 vs 92.0 on identical wordings, `facts` 95.0
vs 96.5, constants in-band 95 vs 100). The out-of-range fix is worth it, but it
is a trade rather than a clean win.

## Results

Measured by compiling and running the model's output against a reference oracle
on ~12 inputs each. A program counts as correct only if **every** input matches.
n=200 per cell.

| Evaluation | What it tests | Correct |
|---|---|---:|
| **seen** | trained program shapes, new constants | **90.0%** |
| **combo** | *unseen combinations* of trained operations | **85.0%** |
| **size** | trained on arrays ≤ 8, tested at 12 and 16 | **88.5%** |
| **facts** | held-out questions about the ISA | **95.0%** |
| **constants** | values outside the trained range | **80.0%** |
| paraphrase | a wording of the operation never trained on | 18.5% |
| depth | expressions nested deeper than any in training | 4.0% |

`combo` says it composes operations it was never shown together. `size` says it
learned an *algorithm*, not a per-length template — the correct program is
identical at every input length, and it holds up at lengths never trained on.
`paraphrase` and `depth` are the two open walls; see [Limits](#limits).

## What's in this release

| Asset | Size | Contents |
|---|---:|---|
| `arm_25M_v1.1.pth` | 103 MB | model weights, tokenizer vocabulary, and architecture dims |

The checkpoint is self-describing — `load_checkpoint()` reads the dimensions and
vocabulary out of the file, so you do not need to pass any config.

## Downloading the weights

The checkpoint is **not in the repository** — at 103 MB it is over GitHub's
100 MB file limit, so it ships as the release asset below. It must land in
`checkpoints/`, which is where the tooling looks by default.

```bash
git clone https://github.com/yashodeepv/baby-programmer-small-language-model
cd baby-programmer-small-language-model
mkdir -p checkpoints
```

Then fetch it, whichever you prefer:

```bash
# curl (note -L: the download is a redirect)
curl -L -o checkpoints/arm_25M_v1.1.pth \
  https://github.com/yashodeepv/baby-programmer-small-language-model/releases/download/v1.1-arm/arm_25M_v1.1.pth

# or wget
wget -O checkpoints/arm_25M_v1.1.pth \
  https://github.com/yashodeepv/baby-programmer-small-language-model/releases/download/v1.1-arm/arm_25M_v1.1.pth

# or the GitHub CLI
gh release download v1.1-arm --repo yashodeepv/baby-programmer-small-language-model \
  --pattern arm_25M_v1.1.pth --dir checkpoints
```

Or click `arm_25M_v1.1.pth` in the **Assets** list at the bottom of this page and
move it into `checkpoints/`.

Check it arrived intact — a truncated download is the usual cause of a
confusing load error:

```bash
ls -l checkpoints/arm_25M_v1.1.pth      # expect 103,338,437 bytes
```

## Using it

```bash
python3.11 -m venv .venv && .venv/bin/pip install torch numpy

# ask it something, then assemble and run what it writes
.venv/bin/python arm/ask.py --run --expect 110 \
  "Compute the sum of the even integers from 2 to 20, leaving the result in w0."
#     ran -> w0 = 110   you expected 110   PASS
```

Requires **clang** (Xcode command line tools) — `--run` genuinely assembles and
executes the generated code. Runs on Apple Silicon (MPS), CUDA, or CPU.

### More examples

```bash
# array input
.venv/bin/python arm/ask.py --run --array 4,9,2,7,30 --expect 52 \
  "Compute the sum of the elements of the array, leaving the result in w0. \
   The array pointer is in x0 and its length in w1."

# nine-digit answer, composed from product x odd x a range
.venv/bin/python arm/ask.py --run --expect 654729075 \
  "Compute the product of the odd integers from 2 to 20, leaving the result in w0."

# factual question
.venv/bin/python arm/ask.py "What does the cset instruction do?"

# interactive
.venv/bin/python arm/ask.py
```

## Phrase questions the way the corpus does

**Read this before concluding the model is broken.** The 16 surface phrasings
vary only the *wrapper* — "Compute X…", "I need X…", "Emit code for X…". The
**body** naming the operation has two trained renderings as of v1.1, not one.
Stray from both and the model matches the nearest familiar shape and silently
substitutes a different operation, with no sign of confusion.

Rephrasings that keep the sentence shape usually survive (`Total the elements of
the array, leaving the result in w0.` works). Terse reformulations still do not
— all four "not this" examples below were re-tested against v1.1 and all four
still fail.

| use this | not this |
|---|---|
| `the sum of the elements of the array` | "Sum the array" |
| `the largest of the elements of the array` | "Find the max" |
| `the sum of the even integers from 2 to 20` | "Add up the evens up to 20" |
| `how many of the elements of the array that are greater than 10` | "Count elements over 10" |

To print the exact forms it knows:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'arm')
import grammar; from ir import question
for p in grammar.sample(12, seed=7): print(question(p))"
```

## Limits

**It composes, but it does not recurse.** Asked for an expression nested one
level deeper than anything in training, it emits well-formed, compiling code
that is systematically *too short* — a depth-3 program for a depth-4 question.
Move the training ceiling from 3 to 4 and the failure moves to 5. It learned a
maximum depth, not a recursion. This survived every lever tried: a 13×
parameter sweep (0.5% → 3.0% → 2.5%), deeper training data (0.5%), and
step-by-step decomposition (0.7%).

**Rephrasing is still a wall.** On the held-out wording the model scores 18.5%,
against 90.0% on wordings it was trained on. Three wordings per operation is an
improvement over one, but it did not generalize to a fourth.

**Plain constants above 500 truncate, in both releases.** They are sampled
`0–500` during training; `Compute 750` comes back as `mov w0, #75`. This is not
a regression in v1.1 — v1.0 does it too — but it was previously undocumented,
and the architecture page claimed the opposite.

**Range bounds far outside the trained range still degrade.** Range bounds train at
`hi ≤ 100`; v1.1 holds 80% out to 400, which is a large improvement on v1.0's
22% but is not the 95% it manages in-band. If you need a specific large bound,
check the emitted `mov` against what you asked for.

Also worth knowing:

- Learned absolute position embeddings cannot extrapolate past `block_size=384`.
- The factual Q&A is *trusted, not verified* — you cannot execute an English
  sentence. It is hand-written and scoped to the instruction subset the
  generator emits.
- No division; `udiv`/`sdiv` are outside the instruction subset.

## Reproducing it

The weights are a convenience — everything here regenerates from source:

```bash
.venv/bin/python arm/golden.py                      # verify the generator itself
.venv/bin/python arm/build_corpus.py --n 20000      # generate + verify a corpus
.venv/bin/python arm/train_comp.py --steps 3000 --n-embd 512 --n-head 8 \
    --n-layer 8 --facts 2500 --tag mymodel          # ~100 min on an M-series GPU
```

`golden.py` is the trust anchor: 68 hand-written cases pinning the question
text, the generated assembly and the oracle independently. If it passes, the
corpus generator agrees with ground truth that was never derived from it.

Full architecture and design notes are in the
[README](https://github.com/yashodeepv/baby-programmer-small-language-model#readme).
