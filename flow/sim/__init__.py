"""sim - The IR2 reference interpreter.

    evaluate.py   expression semantics, sharing arm/ir.py's oracle tables
    runner.py     runs a flow against mocked services
    trace.py      Result / Step / Effect: what a run records
    scenarios.py  named runs with expected outcomes, stored in a flow file

This is to IR2 what `ir.evaluate` is to the arm IR: the independent
definition every generated backend is compared against.
"""

from flow.sim.runner import SimError, run
from flow.sim.scenarios import run_scenario, summary

__all__ = ['run', 'run_scenario', 'summary', 'SimError']
