"""coverage.py - Invent scenarios until every edge of a flow has been taken.

Random inputs from the catalog's value pools, and for each call a random
behaviour: succeed, fail every attempt (to reach the error edge), or fail
then succeed (to exercise retries). Each run goes through the simulator; a
run is kept as a scenario only if it takes an edge no earlier run took.

A flow whose edges cannot all be reached -- `name == "Ana"` followed by
`if name == "Raj"` -- is a flow nobody would write. `synthesize` reports the
coverage it managed, and the corpus builder drops anything short of full.

The scenarios double as the test suite for generated code: a backend must
reproduce every one of them.
"""

from flow.ir2 import graph as g
from flow.gen.values import random_value
from flow.sim import run, summary


def _mocks(rng, flow, cat):
    mocks = {}
    for s in flow.steps:
        if not isinstance(s, g.Call):
            continue
        out = flow.services[s.service][s.op].output
        ok = {'ok': random_value(rng, out, cat)}
        err = {'error': rng.choice(['timeout', 'unavailable', 'rejected'])}
        mode = rng.choices(['ok', 'fail', 'flaky'], [6, 3, 2 if s.retries else 0])[0]
        if mode == 'ok':
            mocks[s.id] = ok
        elif mode == 'fail':
            mocks[s.id] = err
        else:
            mocks[s.id] = [err] * rng.randint(1, s.retries) + [ok]
    return mocks


def expectation(res):
    exp = {'status': res.status, 'path': res.path(),
           'effects': [summary(e) for e in res.effects]}
    if res.response is not None:
        exp['response'] = res.response
    if res.error:
        exp['error'] = res.error
    return exp


def synthesize(rng, flow, cat, tries=300):
    """Returns (scenarios, fraction of edges covered)."""
    trig = flow.trigger()
    want = {(e.src, e.dst, e.on) for e in flow.edges}
    covered, scenarios = set(), {}
    for _ in range(tries):
        payload = random_value(rng, trig.payload, cat)
        mocks = _mocks(rng, flow, cat)
        res = run(flow, payload, mocks)
        new = set(res.edges) - covered
        if not new:
            continue
        covered |= new
        used = {s for s in mocks if s in res.path()}
        scenarios[f's{len(scenarios) + 1}'] = {
            'input': payload,
            'mocks': {s: m for s, m in mocks.items() if s in used},
            'expect': expectation(res),
        }
        if covered >= want:
            break
    return scenarios, len(covered & want) / len(want)
