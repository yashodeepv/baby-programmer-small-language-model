"""trace.py - What a run records.

`effects` is the contract a generated backend must reproduce: the ordered
list of calls, writes and messages the flow causes. `steps` is for people --
the path the run took and the values it saw, which is what an editor
animates along the flowchart.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Step:
    id: str
    kind: str
    detail: Dict = field(default_factory=dict)


@dataclass
class Effect:
    kind: str                   # call persist notify
    step: str
    data: Dict

    def as_dict(self):
        return {'kind': self.kind, 'step': self.step, **self.data}


@dataclass
class Result:
    status: str                 # ok error
    response: Optional[Dict] = None
    error: Optional[str] = None
    steps: List[Step] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
    edges: List[Tuple[str, str, str]] = field(default_factory=list)   # (src, dst, on) taken

    def path(self):
        return [s.id for s in self.steps]
