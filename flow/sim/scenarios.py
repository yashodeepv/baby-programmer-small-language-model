"""scenarios.py - Named test runs stored alongside a flow.

A flow file may carry scenarios: an input, mocks, and what should happen.

    "scenarios": {
      "declined": {
        "input": {...},
        "mocks": {"charge": {"error": "card declined"}},
        "expect": {
          "status": "error",
          "response": {"reason": "payment failed"},
          "path": ["start", "has_items", "pricing", "charge", "payment_failed"],
          "effects": ["call payments.charge #1", "call payments.charge #2"]
        }
      }
    }

Every key under `expect` is optional; only the ones given are compared.
Effects are compared in their one-line form (see `summary`), in order.
"""

from flow.sim.runner import run


def summary(effect):
    d = effect.data
    if effect.kind == 'call':
        return f'call {d["service"]}.{d["op"]} #{d["attempt"]}'
    if effect.kind == 'persist':
        return f'persist {d["entity"]}.{d["op"]}'
    if effect.kind == 'notify':
        return f'notify {d["channel"]} {d["to"]}: {d["message"]}'
    return f'{effect.kind} {effect.step}'


def run_scenario(flow, scenario):
    """Returns (result, [mismatch messages]); an empty list means it passed."""
    res = run(flow, scenario['input'], scenario.get('mocks', {}))
    want = scenario.get('expect', {})
    got = {
        'status': res.status,
        'response': res.response,
        'error': res.error,
        'path': res.path(),
        'effects': [summary(e) for e in res.effects],
    }
    bad = []
    for key, w in want.items():
        if key not in got:
            bad.append(f'unknown expectation {key!r}')
        elif got[key] != w:
            bad.append(f'{key}: expected {w!r}, got {got[key]!r}')
    return res, bad
