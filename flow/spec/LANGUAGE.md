# Baby Spec 0.1 — language reference

Baby Spec is the language an analyst uses to describe a whole system to
baby-programmer in one text document. The analyst may be a person or a
frontier model working through a skill. baby-programmer reads the spec,
builds the flows and their logic in IR2, verifies them, and the compilers
port the result to a target language.

```
analyst ──► system spec (Baby Spec) ──► baby-programmer ──► IR2 ──► check + simulate ──► compilers
                                                              │
                                                              └──► flowchart ◄── business people review here
```

**Who reads what.** A spec is a machine-to-machine contract: an analyst
writes it, and baby-programmer reads it. Business people are not its
audience. They review and edit the **flowchart** built from it, so the only
business-facing text in a spec is the step titles that label the flowchart's
boxes (§7.3). The language is designed for precision, for a deterministic
parser, and for an analyst (above all, a frontier model) to write
correctly the first time. It is not designed to be pleasant to read.

This document is normative: a spec that follows it means exactly one thing.
Version 0.1 covers what IR2 can express today (see [Scope](#10-scope-of-01)).

---

## 1. Principles

1. **Structure is formal; logic may be intent.** Sections, names, types,
   signatures, triggers, step order, branches and parallel blocks follow a
   fixed grammar and are read by a deterministic parser. Only the logic
   inside a step may be written as intent: controlled English that says
   *what* a value is and leaves *how* to baby-programmer. That is the one
   part its models read.
2. **The analyst decides where the programming happens.** Logic written
   exactly (`subtotal - discount`) is compiled as written, with no model
   involved. Logic written as intent is programmed by baby-programmer. The
   more a spec states intent rather than code, the more of the work, and the
   cost, moves from the analyst to baby-programmer.
3. **Everything is named, and every name resolves.** No pronouns, no "it",
   no "the result". A value is referred to by its name.
4. **One meaning per construct.** Each step keyword maps to exactly one IR2
   construct. Nothing is inferred from tone or context.
5. **Order is document order.** Steps run in the order written, except where
   a step says otherwise (`If`, `In parallel`, a stop).

---

## 2. Document layout

A spec is a Markdown document. The layout is:

```
Spec: baby-spec/0.1                 ← required first line

# System: <system_name>             ← exactly one
<description>                       ← optional free text, never parsed

## Constants                        ← optional, at most once
## Entities                         ← optional, at most once
## Services                         ← optional, at most once
## Flow: <flow_name>                ← one or more
```

- Section headings are exactly as shown: `#` and `##`, a keyword, and for
  `System` and `Flow`, a colon and a name.
- Sections may appear in any order after `# System`, but `Constants`,
  `Entities` and `Services` must come before any `Flow` that uses them.
- Blank lines are ignored. A line whose first non-blank characters are `>`
  is a comment for engineers debugging a spec: never parsed.
- Free text is allowed only in the description under `# System`, the
  `Intent:` line of a flow (§7.1), and step titles (§7.3).
- Markdown is used because a frontier model writes it reliably, and because
  it diffs and reviews well in version control, not for business readers.

---

## 3. Lexical conventions

| Token | Form | Examples |
|---|---|---|
| name | lower-case letters, digits, `_`; starts with a letter | `order_id`, `pricing`, `days_late` |
| constant name | upper-case letters, digits, `_` | `LARGE_ORDER`, `MAX_RETRY_DAYS` |
| path | names joined by `.` | `charge.charge_id`, `items.amount` |
| integer | optional `-`, digits | `0`, `100000`, `-5` |
| string | double-quoted; `\"` and `\\` escape | `"gold"`, `"order has no items"` |
| template | a string containing `{path}` placeholders; `{{` and `}}` are literal braces | `"Order {order_id} confirmed"` |
| exact expression | backquoted, see §8 | `` `subtotal - discount` `` |

Reserved words, which can't be used as names: `input`, `true`, `false`,
`and`, `or`, `not`, `if`, `else`, `where`, `each`, and `a0`–`a9`.

---

## 4. Types

| Type | Written | Values |
|---|---|---|
| integer | `int` | signed 32-bit, wraps on overflow. **Money is in minor units** (cents) |
| boolean | `bool` | `true`, `false` |
| string | `str` | text |
| list | `list of <type>` | `list of int` |
| record | `(<field> <type>, ...)` | `(sku str, amount int)` |
| entity | the entity's name | `order` stands for that entity's record |

Records nest: `list of (sku str, amount int)` is a list of records.

---

## 5. Constants

```
## Constants
- LARGE_ORDER = 100000
- SALES_CHANNEL = "#sales"
```

Grammar: `- CONSTANT_NAME = integer | string | true | false`

A constant can appear anywhere a value of its type can. It is substituted
before anything else happens, so it never reaches IR2 as a name.

---

## 6. Entities and services

These declare the system's data and the outside systems it talks to. They
are the catalog: baby-programmer never invents a field or an operation.

### 6.1 Entities

```
## Entities
- order: order_id str, total int, status str, charge_id str
- refund: order_id str, amount int, reason str
```

Grammar: `- <entity>: <field> <type>, <field> <type>, ...`

The first field is the entity's key. `Delete` matches on it (§7.8).

### 6.2 Services

```
## Services
- payments.charge(order_id str, amount int) -> (charge_id str, status str)
- billing.list_overdue(as_of str) -> (invoices list of (invoice_id str, customer str, amount int, days_late int))
```

Grammar: `- <service>.<operation>(<param> <type>, ...) -> (<field> <type>, ...)`

Every operation can fail. What happens then is decided by the calling step (§7.4).

---

## 7. Flows

### 7.1 Flow header

```
## Flow: order_checkout
Intent: Take an order, charge it and confirm it.
Trigger: POST /orders with order_id str, customer_email str, tier str, items list of (sku str, amount int)
Mode: sync
Steps:
1. ...
```

| Line | Required | Meaning |
|---|---|---|
| `Intent:` | no | free text: why the flow exists. Kept for people, never parsed |
| `Trigger:` | yes | what starts the flow, and the payload it receives (§7.2) |
| `Mode:` | yes | `sync`: the caller waits and receives the `Respond`. `async`: nobody waits |
| `Steps:` | yes | the numbered steps, from the next line on |

### 7.2 Triggers

| Form | Example |
|---|---|
| `<METHOD> <path> with <fields>` | `POST /orders with order_id str, items list of (sku str, amount int)` |
| `event <event_name> with <fields>` | `event order_returned with order_id str, refund_amount int` |
| `schedule "<cron>" with <fields>` | `schedule "0 6 * * *" with run_date str` |

- `METHOD` is one of `GET`, `POST`, `PUT`, `DELETE`.
- `<fields>` is a comma-separated `<field> <type>` list: the payload.
- Payload fields are in scope from the first step, by bare name
  (`order_id`) or qualified (`input.order_id`).
- Only an HTTP trigger can be `Mode: sync`.

### 7.3 Steps

Steps are a numbered list: `1.`, `2.`, `3.`, …. Each step starts with one
**step keyword** that determines its kind:

| Keyword | Kind | Section |
|---|---|---|
| `Validate:` | validate | §7.5 |
| `Compute <name>:` | compute | §7.6 |
| `Call` | call | §7.4 |
| `If` … / `Otherwise:` | branch | §7.7 |
| `In parallel:` | parallel + join | §7.9 |
| `Save`, `Update`, `Delete` | persist | §7.8 |
| `Email`, `Slack`, `SMS`, `Webhook` | notify | §7.10 |
| `Respond with`, `Stop` | end | §7.11 |

Nested steps (inside `If`, `Otherwise` and `In parallel`) are lettered
`a.`, `b.`, `c.`, … and indented by three spaces under their parent. Nesting
may go deeper, lettering again at each level.

A sentence inside a step ends with `.`. Optional clauses of a step, such as
`Retry …` and `On failure …`, are separate sentences in the order shown.

**Step titles.** Any step may begin with a title in square brackets:

```
3. [Charge the card] Call payments.charge with order_id, amount = total. Retry 1 time.
4. [$1000 or more?] If `total >= LARGE_ORDER`:
```

A title is the label on the step's box in the flowchart, the text a
business person reviews. Write it in the business's own words. It is never
parsed for meaning. A step without a title is labelled from its kind and
names (`payments.charge`).

### 7.4 Call

```
3. Call payments.charge with order_id, amount = total. Retry 1 time. On failure stop with error "payment failed".
```

Grammar:

```
Call <service>.<operation> [as <name>] with <arg>, <arg>, ... .
     [Retry <n> time|times.]
     [On failure stop with error "<message>" [, responding with <arg>, ...].]
<arg> := <param> = <value>
       | <param>                  ← shorthand for <param> = <param>
```

- Every parameter of the operation must be given exactly once.
- `Retry n times` means up to *n* more attempts after the first failure (0–3).
- `On failure stop with error "…"` ends the flow with that error. Without
  it, a failed call **fails the whole flow**.
- The call's outputs are in scope after the step, as `<name>.<field>`. The
  name defaults to the operation name (`charge.charge_id`). A bare field
  name (`charge_id`) works when no other value in scope has that name. The
  outputs are not in scope on the failure path.

### 7.5 Validate

```
1. Validate: `len(items) > 0`. Otherwise stop with error "order has no items".
1. Validate: the order has at least one item. Otherwise fail with "order has no items".
```

Grammar:

```
Validate: <condition>. Otherwise stop with error "<message>" [, responding with <arg>, ...].
Validate: <condition>. Otherwise fail with "<message>".
```

`<condition>` is a boolean value (§8). `stop with error` ends the flow with
an error; `fail with` fails the whole flow, with no end step. For the
error response, see §7.11.

### 7.6 Compute

```
2. Compute pricing:
   - subtotal: `sum(items.amount)`
   - discount: 10% of subtotal when tier is "gold", otherwise 0
   - total: `subtotal - discount`
```

Grammar: `Compute <name>:` followed by indented lines `- <var>: <value>`.

- Each `<var>` is a new name, in scope after its line: later lines of the
  same step may use it. Refer to it as `<var>` or `<name>.<var>`.
- A var's type is its value's type.

### 7.7 If / Otherwise

```
4. If `total >= LARGE_ORDER`:
   a. Slack SALES_CHANNEL: "Large order {order_id}: {total} cents".
   Otherwise:
   a. Stop.
```

Grammar:

```
If <condition>:
   <lettered steps>
[Otherwise:
   <lettered steps>]
```

- `Otherwise:` sits at the same indentation as the `If` step's lettered steps.
- Both sides rejoin at the next numbered step, unless a side ends with a stop.
- A value computed on only one side is **not** in scope after they rejoin.

### 7.8 Save, Update, Delete

```
5. Save order: order_id, total, status = "paid", charge_id.
6. Update order: order_id, total, status = "refunded", charge_id.
7. Delete order where order_id.
```

Grammar:

```
Save <entity>: <arg>, ... .              ← create; every field given
Update <entity>: <arg>, ... .            ← update; every field given, matched on the key
Delete <entity> where <arg> .            ← delete, matched on the key field
```

`<arg>` is as in §7.4: `field = value`, or `field` alone for a same-named value.

### 7.9 In parallel

```
5. In parallel:
   a. Email customer_email: "Order {order_id} confirmed. Total: {total}".
   b. Save order: order_id, total, status = "paid", charge_id.
```

- Each lettered step is one branch. All of them run, in letter order when
  simulated. The flow continues at the next numbered step once every branch
  has finished.
- Values produced in any branch (by a `Call`) are in scope afterwards.
- A branch can't stop the flow. In 0.1, each branch is a single step.

### 7.10 Notify

```
Email <to>: <message>.
Slack <to>: <message>.
SMS <to>: <message>.
Webhook <to>: <message>.
```

`<to>` and `<message>` are string values: a name, a string, a template, or
an exact expression.

### 7.11 Ending a flow

```
Respond with order_id, charge_id, total = total.            ← ok, with a response (sync flows)
Stop.                                                       ← ok, no response
Stop with error "payment declined".                         ← error
Stop with error "payment declined", responding with order_id.
```

- `Respond with` takes `key = value` pairs, or `key` alone for a same-named value.
- In a `sync` flow, a stop with an error responds `reason = "<message>"`,
  plus any `responding with` pairs. In an `async` flow it responds nothing,
  and `responding with` is an error.
- The same error form ends `On failure` (§7.4) and `Otherwise stop with error` (§7.5).
- A flow whose last numbered step isn't one of these ends with an implied `Stop.`.
- `Respond with` in an `async` flow is an error: nobody receives it.

---

## 8. Values

Wherever a step takes a value (a condition, an argument, a field, a
message), it may be written in one of five forms:

| Form | Example | Read by |
|---|---|---|
| **name** or path | `total`, `charge.charge_id`, `LARGE_ORDER` | parser |
| **literal** | `0`, `"gold"`, `true` | parser |
| **template** | `"Order {order_id} confirmed"` | parser |
| **exact expression** | `` `subtotal - discount` `` | parser |
| **prose** | `10% of subtotal when tier is "gold", otherwise 0` | **baby-programmer's logic model** |

Anything that isn't one of the first four forms is prose.

### 8.1 Exact expressions

Written between backquotes.

```
expr    := cond
cond    := or [ "if" or "else" cond ]
or      := and { "or" and }
and     := not { "and" not }
not     := "not" not | cmp
cmp     := sum [ ("==" | "!=" | "<" | "<=" | ">" | ">=") sum ]
sum     := term { ("+" | "-") term }
term    := unary { ("*" | "/" | "%") unary }
unary   := "-" unary | atom
atom    := integer | string | template | "true" | "false" | path | CONSTANT
         | func "(" args ")" | "(" expr ")"
```

| Function | Meaning |
|---|---|
| `len(xs)` | number of elements |
| `sum(xs)`, `product(xs)`, `min(xs)`, `max(xs)` | reductions; `min`/`max` of an empty list is `0` |
| `count(xs)` | number of elements, like `len`, but accepts a filter |
| `abs(x)` | absolute value |
| `range(lo, hi)` | the integers `lo` to `hi` inclusive; only as a reduction's list |

Function names are not reserved. A name followed by `(` is a function call;
otherwise it is a name. So a var called `count` and the call `count(invoices)`
don't collide.

A reduction's list may be followed by an element map and a filter:

```
sum(items.amount)                      the amounts, summed
sum(items.amount where > 25000)        only amounts over 25000
count(invoices.days_late where > 30)   how many are over 30 days late
sum(items.amount each * 2)             each amount doubled, then summed
max(range(1, n) where even)            the largest even number from 1 to n
```

- `xs.field` on a list of records is the list of that field.
- Filters: `where > k`, `where < k`, `where even`, `where odd`, `where divisible by k`.
- Maps: `each * k`, `each + k`, `each squared`.
- The filter applies to the original element, then the map, as written.
- `/` and `%` truncate toward zero. Dividing by zero fails the flow.
- `==` and `!=` compare two values of the same type. The other comparisons are for `int` only.

### 8.2 Prose

Prose states a value's **intent** and leaves its implementation to
baby-programmer: `10% of subtotal when tier is "gold", otherwise 0`. It
isn't there for readability; business people never see it. It is how an
analyst delegates programming. Prose must:

- refer to values **only by their names** (`subtotal`, `tier`), never by description;
- write literals as literals: numbers as digits, strings in quotes;
- describe a single value: one condition, one number, one text. Never a
  sequence of actions, which belongs in steps.

Prose is compiled by baby-programmer's logic model into an exact
expression. A spec with no prose at all is compiled with no model involved.

### 8.3 Scope

At each step, the names in scope are:

- the trigger payload's fields;
- the vars of every `Compute` step, and the outputs of every `Call` step,
  that **run on every path** to this step. Values from both sides of an
  `If` are in scope only if both sides produce them; values from all
  branches of an `In parallel` are in scope after it;
- constants.

A bare name must refer to exactly one value in scope. If two values share a
name, qualify it: `input.status`, `charge.status`.

---

## 9. Validity rules

A spec is valid when every rule below holds. baby-programmer rejects an
invalid spec with the rule's name and line number, so an analyst can fix it.

| Rule | Meaning |
|---|---|
| `layout` | the first line is `Spec: baby-spec/0.1`; one `# System`; section headings as in §2 |
| `unique-names` | system, flow, entity, service operation, compute and var names are unique where they're declared |
| `known-type` | every type is from §4, or a declared entity |
| `known-service` | every `Call` names a declared service operation |
| `call-args` | every parameter is given exactly once, with the right type |
| `known-entity` | every `Save`/`Update`/`Delete` names a declared entity |
| `entity-fields` | `Save`/`Update` give every field, with the right type; `Delete` gives the key |
| `resolves` | every name in a value refers to exactly one thing in scope (§8.3) |
| `types` | every exact value has the type its position needs |
| `sync-http` | `Mode: sync` only with an HTTP trigger |
| `respond-sync` | `Respond with` and `responding with` only in `sync` flows |
| `retry-range` | `Retry` between 0 and 3 |
| `parallel-branch` | an `In parallel` branch is a single step, and not a stop |
| `reachable` | no step after an unconditional stop at the same level |

Prose can't be type-checked until it's compiled. baby-programmer compiles
it, then applies the same rules to the result.

---

## 10. Scope of 0.1

Baby Spec 0.1 expresses exactly what IR2 expresses today. On the roadmap,
together with the matching IR2 extensions:

- **functions**: reusable logic called from several flows;
- **queries**: reading and finding entities, not only writing them;
- **timeouts, error types** and richer failure handling;
- **multi-step parallel branches** and loops over lists with steps inside;
- **bitwise operators**, which IR2 has but business specs rarely need.

---

## 11. Mapping to IR2

| Baby Spec | IR2 |
|---|---|
| `Trigger:` | `trigger` step; payload fields are `input.<field>` |
| `Mode:` | `flow.mode` |
| `Validate: … Otherwise stop with error "m".` | `validate` + `error` edge to an `end` (status error, response `reason`) |
| `Validate: … Otherwise fail with "m".` | `validate` with no `error` edge, message `m` |
| `Compute <name>:` | `compute` step with id `<name>`; vars are `<name>.<var>` |
| `Call s.op [as n]` | `call` step with id `n` (default `op`); `Retry` sets `retries`; `On failure` adds an `error` edge to an `end` |
| `If` / `Otherwise:` | `branch` with `true` / `false` edges; the sides rejoin at the next step |
| `In parallel:` | `parallel` with `branch` edges, and a `join` before the next step |
| `Save` / `Update` / `Delete` | `persist` with op `create` / `update` / `delete` |
| `Email` / `Slack` / `SMS` / `Webhook` | `notify` with that channel |
| `Respond with` / `Stop` / `Stop with error` | `end` |
| exact expression | an IR2 expression, one-to-one (§8.1 ↔ `flow/ir2/sexpr.py`) |
| prose | an IR2 expression written by the logic model |
| template | `fmt` |
| constant | inlined as a literal |
| `[step title]` | the step's `title`: its label in the flowchart |
| `Intent:` | `flow.intent` |

Two example specs, [`examples/order_checkout.spec.md`](examples/order_checkout.spec.md)
and [`examples/overdue_reminders.spec.md`](examples/overdue_reminders.spec.md),
describe the same systems as the hand-written IR2 flows in `flow/examples/`,
so the two can be compared line by line.
