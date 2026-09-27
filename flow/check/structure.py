"""structure.py - Is the graph well formed?

Checked here, before any expression is looked at:

  * ids are unique identifiers and do not collide with reserved names
  * exactly one trigger, with nothing flowing into it
  * each step's settings are from its catalog (source, channel, op, status)
  * every edge joins existing steps, with labels and counts the kind allows
  * every step is reachable from the trigger, and the graph has no cycles
    (repetition lives inside expressions -- the arm reductions -- not in the
    graph)
  * each parallel names a join, and every path out of every branch reaches
    that join before it can end; each join belongs to exactly one parallel

Returns the topological order when the graph is a DAG, for the dataflow pass.
"""

import re

from flow.ir2 import graph as g
from flow.ir2.sexpr import RESERVED

_ID = re.compile(r'[a-z_][a-z0-9_]*')


def _settings(s, rep):
    def bad(field, value, allowed):
        rep.error('bad-setting', f'{field} {value!r} is not one of {", ".join(allowed)}',
                  s.id, field)
    if isinstance(s, g.Trigger):
        if s.source not in g.SOURCES:
            bad('source', s.source, g.SOURCES)
        need = {'http': ('method', 'path'), 'event': ('name',), 'schedule': ('cron',)}
        for k in need.get(s.source, ()):
            if not s.config.get(k):
                rep.error('missing-setting', f'a {s.source} trigger needs config.{k}', s.id, 'config')
    elif isinstance(s, g.Notify) and s.channel not in g.CHANNELS:
        bad('channel', s.channel, g.CHANNELS)
    elif isinstance(s, g.Persist) and s.op not in g.PERSIST_OPS:
        bad('op', s.op, g.PERSIST_OPS)
    elif isinstance(s, g.End) and s.status not in g.END_STATUS:
        bad('status', s.status, g.END_STATUS)
    elif isinstance(s, g.Call) and s.retries < 0:
        rep.error('bad-setting', 'retries cannot be negative', s.id, 'retries')


def _topo(flow):
    """Kahn's algorithm. Returns the order, or None if there is a cycle."""
    indeg = {s.id: 0 for s in flow.steps}
    for e in flow.edges:
        indeg[e.dst] += 1
    ready = [s.id for s in flow.steps if indeg[s.id] == 0]
    order = []
    while ready:
        n = ready.pop(0)
        order.append(n)
        for e in flow.out_edges(n):
            indeg[e.dst] -= 1
            if indeg[e.dst] == 0:
                ready.append(e.dst)
    return order if len(order) == len(flow.steps) else None


def _reach(flow, start):
    seen, stack = set(), [start]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(e.dst for e in flow.out_edges(n))
    return seen


def _parallels(flow, rep):
    owner = {}
    for p in flow.steps:
        if not isinstance(p, g.Parallel):
            continue
        j = flow.step(p.join)
        if not isinstance(j, g.Join):
            rep.error('bad-join', f'join {p.join!r} is not a join step', p.id, 'join')
            continue
        if p.join in owner:
            rep.error('shared-join', f'join {p.join!r} already belongs to {owner[p.join]!r}',
                      p.id, 'join')
        owner[p.join] = p.id
        # Every path from each branch must hit the join before an end.
        for e in flow.out_edges(p.id):
            stack, seen = [e.dst], set()
            while stack:
                n = stack.pop()
                if n == p.join or n in seen:
                    continue
                seen.add(n)
                if isinstance(flow.step(n), g.End):
                    rep.error('branch-escapes', f'branch reaching {n!r} ends before join '
                              f'{p.join!r}; a parallel branch must reach its join', p.id)
                    break
                stack.extend(x.dst for x in flow.out_edges(n))
        region = _reach(flow, p.id)
        for e in flow.in_edges(p.join):
            if e.src not in region:
                rep.error('join-entered-from-outside',
                          f'{e.src!r} flows into join {p.join!r} from outside its parallel',
                          p.join)
    for j in flow.steps:
        if isinstance(j, g.Join) and j.id not in owner:
            rep.error('orphan-join', 'no parallel names this join', j.id)


def check_structure(flow, rep):
    """Adds findings to `rep`. Returns the topological order, or None."""
    if flow.mode not in g.MODES:
        rep.error('bad-setting', f'mode {flow.mode!r} is not one of {", ".join(g.MODES)}',
                  field='mode')

    ids = [s.id for s in flow.steps]
    for sid in set(ids):
        if ids.count(sid) > 1:
            rep.error('duplicate-id', f'{ids.count(sid)} steps share this id', sid)
    for sid in ids:
        if not _ID.fullmatch(sid):
            rep.error('bad-id', 'ids are lower-case letters, digits and _', sid)
        elif sid in RESERVED or sid == 'input':
            rep.error('bad-id', f'{sid!r} is reserved', sid)
    if len(set(ids)) != len(ids):
        return None

    triggers = [s for s in flow.steps if isinstance(s, g.Trigger)]
    if len(triggers) != 1:
        rep.error('trigger-count', f'a flow has exactly one trigger, found {len(triggers)}')

    for s in flow.steps:
        _settings(s, rep)

    known = set(ids)
    edges_ok = True
    for e in flow.edges:
        for end in (e.src, e.dst):
            if end not in known:
                rep.error('unknown-step', f'edge {e.src} -> {e.dst} names unknown step {end!r}')
                edges_ok = False
    if not edges_ok:
        return None

    for s in flow.steps:
        allowed = g.EDGES[s.kind]
        outs = flow.out_edges(s.id)
        for e in outs:
            if e.on not in allowed:
                labels = ', '.join(allowed) or 'none'
                rep.error('bad-edge', f'a {s.kind} cannot have an {e.on!r} edge '
                          f'(allowed: {labels})', s.id)
        for label, (lo, hi) in allowed.items():
            n = sum(1 for e in outs if e.on == label)
            if n < lo:
                rep.error('missing-edge', f'a {s.kind} needs {lo} {label!r} edge'
                          + ('s' if lo > 1 else ''), s.id)
            if hi is not None and n > hi:
                rep.error('extra-edge', f'a {s.kind} has at most {hi} {label!r} edge, found {n}', s.id)
        if isinstance(s, g.Trigger) and flow.in_edges(s.id):
            rep.error('into-trigger', 'nothing can flow into the trigger', s.id)

    order = _topo(flow)
    if order is None:
        rep.error('cycle', 'the flow loops back on itself; repeat inside an expression instead')
        return None

    if len(triggers) == 1:
        seen = _reach(flow, triggers[0].id)
        for sid in ids:
            if sid not in seen:
                rep.error('unreachable', 'no path from the trigger reaches this step', sid)

    _parallels(flow, rep)
    return order
