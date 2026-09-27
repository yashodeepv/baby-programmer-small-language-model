"""codegen - IR2 flows -> source code in real languages.

    python.py      a Python module:        run(input, services, store, notify) -> result
    javascript.py  an ES module:     async run(input, services, store, notify) -> result
    verify.py      runs generated code on scenarios and compares it with the simulator

Generated code follows the simulator exactly (flow/sim): 32-bit integer
wrap-around, division truncating toward zero, retries, error edges, the
same order of side effects. Parallel branches run one after another in
edge order, as in the simulator. Nothing is trusted until verify.py has
replayed it against the simulator.

The shape of the generated code is a small state machine: one function per
step, each returning the id of the next step (or a result at an end). It
handles any flow the checker accepts, merges and parallels included, at the
cost of reading a little less like hand-written code.

The runtime contract, the same in every language:

    services(service, op, args, step)   returns the output record, or raises/throws ServiceError
    store(entity, op, data)             create / update / delete
    notify(channel, to, message)
    result = {status: 'ok' | 'error', response: record | null, error: text | null}
"""

from flow.codegen.javascript import to_javascript
from flow.codegen.python import to_python

TARGETS = {'python': to_python, 'javascript': to_javascript}

__all__ = ['to_python', 'to_javascript', 'TARGETS']
