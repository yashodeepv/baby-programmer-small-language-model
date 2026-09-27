# flow — IR2

IR2 extends the baby-programmer IR from **one expression** to a **whole flow**:
a trigger, a flowchart of typed steps, and the data each step passes on. The
logic inside a step is still the arm IR, with named fields, strings and
booleans added. Every arm s-expression is valid IR2 and means the same thing
(`tests/test_sexpr.py` checks this on 400 grammar samples).

```
English request ─► flow encoder ─► flow graph ─┐        (planned: n-way classifier)
                                               ├─► IR2 ─► check ─► simulate ─► backends
step English ───► logic model ──► expressions ─┘         (planned: arm decoder, retrained)
                        ▲
         user edits in the flowchart UI write IR2 directly — no model in between
```

This directory holds the deterministic core (the definition, the verifier and
the reference interpreter) and the training-data generator. No models, UI or
code generation yet.

## Layout

| Path | What |
|---|---|
| `ir2/types.py` | value types: `int` (signed 32-bit, as in arm), `bool`, `str`, lists, records |
| `ir2/expr.py` | step logic: arm IR nodes + `Ref`, `Str`, `Bool`, `Not/All/Any`, `Len`, `Pluck`, `Fmt`, `div`/`mod` |
| `ir2/sexpr.py` | expression text form — a strict superset of `arm/sexpr.py` |
| `ir2/graph.py` | the flow: step catalog, edges, services, entities |
| `ir2/io.py` | JSON load/dump, the format an editor reads and writes |
| `check/structure.py` | graph rules: ids, one trigger, edge labels, reachability, no cycles, parallel/join |
| `check/dataflow.py` | what each step can see; service, entity and field contracts |
| `check/typing.py` | expression types |
| `check/diagnostics.py` | findings, each tied to a step and field, with a stable code |
| `sim/evaluate.py` | expression semantics, sharing arm's oracle tables |
| `sim/runner.py` | runs a flow against mocked services |
| `sim/scenarios.py` | stored test runs with expected outcomes |
| `gen/catalogs/*.json` | four system catalogs: ecommerce, billing, hr, logistics |
| `gen/catalog.py` | loads a catalog: triggers, services, entities, value pools |
| `gen/exprgen.py` | type-directed expression sampler |
| `gen/sampler.py` | samples flows by chaining patterns |
| `gen/values.py` | random values of a type, shaped by the catalog |
| `gen/coverage.py` | invents scenarios until every edge has been taken |
| `gen/order.py` | canonical step order: one step per encoder slot |
| `gen/english.py` | expressions → English, reusing arm's wordings |
| `gen/render.py` | flow → English request, recording which words came from which step |
| `gen/labels.py` | flow-encoder labels; logic-model examples, anonymised |
| `gen/curriculum.py` | the book: seven chapters, one new idea each |
| `gen/build.py` | builds the corpus: sample → check → cover → render → label |
| `spec/LANGUAGE.md` | **Baby Spec**: the system spec language an analyst writes and baby-programmer reads |
| `spec/examples/` | example system specs |
| `examples/` | hand-written flows, each with scenarios |
| `tests/` | unit tests |

## Use it

```bash
python3.11 -m flow check flow/examples/order_checkout.json
python3.11 -m flow show  flow/examples/order_checkout.json
python3.11 -m flow run   flow/examples/order_checkout.json --scenario declined
python3.11 -m flow run   flow/examples/order_checkout.json --all

python3.11 -m unittest discover -s flow/tests -t .
```

## The book

baby-programmer's premise is that a programmer isn't made by web-scale
compute. A person learns from one class, one good book, and then the job.
The arm corpus is the class (control flow, conditions, loops). The IR2
corpus is the book: seven chapters, each adding one idea on top of the
earlier ones. Every flow in a chapter must use that chapter's idea, and flows
grow longer and expressions deeper as the book goes on.

```bash
python3.11 -m flow.gen.curriculum                                   # contents, one sample per chapter
python3.11 -m flow.gen.build --curriculum 800 --out data/flow_book.jsonl
```

| # | Chapter | New idea | Steps (median) | Words (median) |
|---|---|---|---:|---:|
| 1 | Hello, flow | a trigger, one action, an end; plain fields only | 3 | 29 |
| 2 | Working things out | compute steps name new values | 4 | 33 |
| 3 | Rules and rejections | validate; stop with an error or fail the flow | 5 | 59 |
| 4 | Talking to other systems | call a service, retry, handle failure | 5 | 64 |
| 5 | Decisions | branch: one path, the other, or a way out | 7 | 94 |
| 6 | Doing things at once | parallel branches and a join | 11 | 133 |
| 7 | Putting it together | everything, with deeper expressions | 11 | 144 |

Measured on 400 flows per chapter. Each exercise has a checked answer key:
the flow passes the verifier and has been run until every edge was taken.
The mock exams are the held-out splits, per chapter.

## Training data

```bash
python3.11 -m flow.gen.build --n 5000 --seed 0 --out data/flow_corpus.jsonl
```

Same principle as the assembly corpus: **sample the structure, derive
everything else, and execute everything.** Per flow:

1. **Sample** from a catalog by chaining patterns (validate, compute, call,
   branch, parallel, persist, notify). Expressions are generated by type, and
   thresholds are placed inside the range the expression actually takes, so
   both sides of a comparison can happen.
2. **Check.** Every sample must pass; a failure is a sampler bug and stops the build.
3. **Cover.** Invent inputs and service behaviours (succeed, fail, flaky)
   until every edge has been taken in the simulator. Flows that can't be
   fully covered are contradictions nobody would write, and are dropped
   (about 11%).
4. **Render** in one of three wording styles, recording spans for each
   step's clause, the values to copy, and the English of each expression.
5. **Label.** Encoder: step kinds in canonical order, settings, edges, copy
   spans, and a sync/async/abstain mode. Logic model: (intent, scope) →
   s-expression, plus an anonymised copy (names → `v1`/`f1`, numbers →
   `n1`, strings → `s1`) so it learns structure rather than business names.

Held-out splits are decided at build time: the `logistics` catalog entirely,
style 2 entirely, and 5% validation.

Measured on 5,000 sampled flows (4,446 kept, 18 s):

| | |
|---|---|
| request length | median 105 words, p95 191, max 310 |
| steps per flow | median 8, max 16 |
| logic examples | 47,285, but a median target of 1 token |
| mode labels | sync 1,627 · async 2,679 · abstain 140 |

Known limits, for the trainer to handle:

- **Most logic examples are trivial.** Two thirds are a bare name or literal
  (call arguments, saved fields). They're flagged `trivial` so they can be
  down-weighted. Expression variety after anonymisation is modest (about
  2,900 unique non-trivial examples per 2,000 flows); co-training with the
  arm corpus, or deeper expressions in `exprgen.py`, would widen it.
- **Requests don't fit today's block size.** At p95 ≈ 190 words the encoder
  needs 512 tokens, not `model.py`'s 256.
- **Field names aren't in the closed vocabulary.** The encoder must copy them
  by pointer (spans are provided); the logic model avoids the problem by
  anonymisation.
- **The English is templated.** It will hit the same paraphrase wall as the
  assembly model until a paraphrase-and-filter pass is added; the labels
  won't change when it is.

## The step catalog

| kind | does | edges out |
|---|---|---|
| `trigger` | starts the flow: `http`, `event` or `schedule`; payload bound as `input` | next |
| `call` | calls a declared service operation, with `retries` | next, error? |
| `compute` | names values — the step's business logic | next |
| `validate` | stops unless a rule holds | next, error? |
| `branch` | picks a path | true, false |
| `parallel` | runs every branch, then its `join` | branch ×2+ |
| `join` | where parallel branches meet | next |
| `persist` | create / update / delete a declared entity | next |
| `notify` | email / sms / slack / webhook | next |
| `end` | finishes `ok` or `error`, optionally with a response | — |

A value is usable at a step only if it is produced on **every** path into it.
The exception is a join, where all branches have run. A call's error edge
carries no output, because the call never returned.

## Decisions worth knowing

- **Repetition lives in expressions, not the graph.** The graph is acyclic;
  loops are the arm reductions (`sum`, `count :gt 30`, ...) over lists.
- **Money is in minor units**, because `int` keeps arm's 32-bit wrap. That
  limit is visible and should be revisited before real use.
- **`div`/`mod` are new** and truncate toward zero. The assembly model has
  never seen them.
- **A bad expression doesn't block loading.** It loads as `Invalid`, so an
  editor can open the flow and show the error on that step. Nothing runs
  until it's fixed.
- **Errors are reported once**, at their source. Steps that depend on a
  broken value are not re-reported.

## Next

1. The logic model: the arm decoder retrained on the anonymised IR2 examples
2. The flow encoder: English → step kinds (n-way per slot) + edges + copied spans
3. End-to-end evaluation: English → both models → IR2 → replay the stored scenarios
4. Code generation: IR2 → a Java project, checked against `sim` on every scenario
5. A flowchart editor over the JSON form

## M1 proof of concept (temporary BERT encoder)

`flow/m1/` is the flow encoder. Its text encoder sits behind one interface
(`encoder.py`); the POC fine-tunes pretrained `bert-base-cased` there as a
temporary stand-in, to be replaced by our own encoder trained from scratch.

```bash
.venv/bin/python -m flow.m1.train --data data/flow_book.jsonl --out checkpoints/m1_poc_bert --epochs 5
```

First run: 3,745 training flows from the book, 5 epochs, 58 minutes on an
Apple M5. "End to end" means the predicted flowchart, given correct step
logic, passes the checker and reproduces every stored scenario.

| Split | kinds | edges | copies | spans | end to end |
|---|---:|---:|---:|---:|---:|
| val (seen wording and domains) | 86.2% | 85.7% | 82.0% | 67.7% | **61.4%** |
| catalog (logistics, never seen) | 59.1% | 65.9% | 51.0% | 40.3% | 34.6% |
| style (wording never seen) | 7.0% | 36.0% | 20.2% | 9.9% | **1.6%** |

On val, chapters 1–2 are at 100% end to end and chapter 3 at 90%, but the
long flows of chapters 6–7 are near 15%.

Unseen wording collapses because the model learned **template shortcuts**,
not the language: in training, `x = y` always meant a saved field or a call
argument, so style 2's `compute total = ...` is read as a save. "stop there"
resembles a branch's "stop", so an extra end step is predicted. BERT's
English doesn't help when the templates offer an easier cue. Each flow is
also rendered in a single style, so the model can recognise the style from
the first sentence. The fix belongs in the data: mix wordings within a flow,
add more wordings per construct, and hold out individual wordings rather
than a whole style.
