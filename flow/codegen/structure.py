"""structure.py - Recover nested code structure from an IR2 flow.

A flow is a graph; people write code as nested blocks. Every flow the spec
language produces is structured (sequences, if/otherwise that rejoin, stops
inside a branch, parallel blocks), and this module recovers that nesting:

    Action      compute, persist, notify: one statement
    Check       validate: `if not rule: <error path or fail>`
    CallStmt    call with retries; on failure `<error path>` or fail
    IfStmt      branch: `if cond: ... else: ...`, rejoining afterwards;
                or, when one side never rejoins, a guard: `if cond: ... return`
    Par         parallel branches, then the join
    Ret         end

Where two paths rejoin is the branch's immediate post-dominator. The result
is checked, not trusted: every step must appear exactly once. A flow that
doesn't structure cleanly -- possible only for graphs edited by hand --
raises Unstructured, and the caller falls back to the state-machine form.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from flow.ir2 import graph as g

EXIT = '__exit__'


class Unstructured(Exception):
    pass


@dataclass
class Action:
    step: object


@dataclass
class Check:
    step: object
    on_fail: Optional[list]          # statements for the error path, or None: fail the flow


@dataclass
class CallStmt:
    step: object
    on_fail: Optional[list]


@dataclass
class IfStmt:
    step: object
    then: list
    orelse: list = field(default_factory=list)
    negate: bool = False             # `if not cond:` -- a guard on the false side


@dataclass
class Par:
    step: object
    branches: List[list]


@dataclass
class Ret:
    step: object


class _Structurer:
    def __init__(self, flow):
        self.flow = flow
        self.succ = {s.id: [e.dst for e in flow.out_edges(s.id)] for s in flow.steps}
        self.order = self._topo()
        self.rank = {n: i for i, n in enumerate(self.order)}
        self.rank[EXIT] = len(self.order)
        self.reach, self.pdom = {}, {}
        for n in reversed(self.order):
            succ = self.succ[n]
            self.reach[n] = {n}.union(*(self.reach[s] for s in succ)) if succ else {n}
            if not succ:
                self.pdom[n] = {n, EXIT}
            else:
                common = set.intersection(*(self.pdom[s] for s in succ))
                self.pdom[n] = {n} | common
        self.seen = set()

    def _topo(self):
        indeg = {s.id: 0 for s in self.flow.steps}
        for e in self.flow.edges:
            indeg[e.dst] += 1
        ready, out = [s.id for s in self.flow.steps if indeg[s.id] == 0], []
        while ready:
            n = ready.pop(0)
            out.append(n)
            for d in self.succ[n]:
                indeg[d] -= 1
                if indeg[d] == 0:
                    ready.append(d)
        return out

    def ipdom(self, n):
        return min(self.pdom[n] - {n}, key=lambda x: self.rank[x])

    def disjoint(self, a, b):
        return not (self.reach[a] & self.reach[b])

    def mark(self, sid):
        if sid in self.seen:
            raise Unstructured(f'step {sid} would be written twice')
        self.seen.add(sid)

    def build(self, sid, stop):
        out = []
        f = self.flow
        while sid is not None and sid != stop:
            s = f.step(sid)
            self.mark(sid)
            nxt = f.next(sid)
            if isinstance(s, (g.Trigger, g.Join)):
                sid = nxt
            elif isinstance(s, (g.Compute, g.Persist, g.Notify)):
                out.append(Action(s))
                sid = nxt
            elif isinstance(s, g.End):
                out.append(Ret(s))
                return out
            elif isinstance(s, (g.Validate, g.Call)):
                err = f.next(sid, 'error')
                on_fail = None
                if err is not None:
                    if not self.disjoint(err, nxt):
                        raise Unstructured(f'the error path of {sid} rejoins the main path')
                    on_fail = self.build(err, None)
                out.append((Check if isinstance(s, g.Validate) else CallStmt)(s, on_fail))
                sid = nxt
            elif isinstance(s, g.Branch):
                t, e = f.next(sid, 'true'), f.next(sid, 'false')
                m = self.ipdom(sid)
                if m != EXIT:
                    out.append(IfStmt(s, self.build(t, m), self.build(e, m)))
                    sid = m
                elif self.disjoint(t, e):
                    # One side never comes back: write it as a guard, and carry
                    # on with the other side without nesting it.
                    if len(self.reach[t]) <= len(self.reach[e]):
                        out.append(IfStmt(s, self.build(t, None)))
                        sid = e
                    else:
                        out.append(IfStmt(s, self.build(e, None), negate=True))
                        sid = t
                else:
                    raise Unstructured(f'the sides of {sid} meet only partly')
            elif isinstance(s, g.Parallel):
                branches = [x.dst for x in f.out_edges(sid) if x.on == 'branch']
                out.append(Par(s, [self.build(b, s.join) for b in branches]))
                sid = s.join
            else:
                raise Unstructured(f'unknown step kind {s.kind}')
        return out


def structure(flow):
    """IR2 flow -> nested statements. Raises Unstructured if it can't be done cleanly."""
    st = _Structurer(flow)
    body = st.build(flow.trigger().id, None)
    missing = {s.id for s in flow.steps} - st.seen
    if missing:
        raise Unstructured(f'steps never reached in structure: {", ".join(sorted(missing))}')
    return body
