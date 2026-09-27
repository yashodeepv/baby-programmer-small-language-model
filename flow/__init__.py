"""flow - IR2: flowchart orchestration built on the arm IR.

The arm IR describes one expression. IR2 describes a whole flow: a trigger,
a graph of typed steps (call a service, compute, validate, branch, run in
parallel, persist, notify, end), and the data each step hands to the next.
The logic inside a step is still an arm IR expression, extended with named
fields, strings and booleans -- the existing IR is enhanced, not replaced.

    flow/ir2/     the IR2 definition: types, expressions, graph, JSON form
    flow/check/   the verifier: structure, dataflow and type rules
    flow/sim/     the reference interpreter, run against mocked services
    flow/examples hand-written flows, each with scenarios and expected results
    flow/tests/   unit tests

    python -m flow check flow/examples/order_checkout.json
    python -m flow run   flow/examples/order_checkout.json --scenario happy
"""
