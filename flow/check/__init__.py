"""check - The IR2 verifier.

    structure.py    the graph: ids, trigger, edges, reachability, cycles, parallel/join
    dataflow.py     scope at each step; service, entity and field contracts
    typing.py       the type of an expression
    diagnostics.py  Diagnostic / Report

A flow that passes has no errors; only then may it run or compile. Warnings
(e.g. a response nobody receives) do not block.
"""

from flow.check.dataflow import check_dataflow
from flow.check.diagnostics import Diagnostic, Report
from flow.check.structure import check_structure


def check(flow):
    """Verify a flow. Returns a Report; `report.ok` is True when there are no errors."""
    rep = Report()
    order = check_structure(flow, rep)
    if order is not None:
        check_dataflow(flow, order, rep)
    return rep


__all__ = ['check', 'Diagnostic', 'Report']
