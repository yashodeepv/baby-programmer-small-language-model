"""order.py - The canonical order of a flow's steps.

The flow encoder predicts one step per output slot, so the slot a step lands
in must be a function of the graph alone -- not of the order the sampler
happened to append steps, and not of what a user did in the editor. The
renderer walks the same order, so slot i roughly lines up with the i-th
clause of the request.

Definition: a depth-first walk from the trigger ranks steps by discovery,
following edges by label in the order

    error, true, false, branch, next

(an error exit is said right after the step that can fail; a branch's true
arm before its false arm; parallel branches in edge order; then the rest).
The canonical order is the topological order that always picks, among the
steps whose predecessors are all placed, the one discovered first.
"""

LABEL_ORDER = {'error': 0, 'true': 1, 'false': 2, 'branch': 3, 'next': 4}


def canonical_order(flow):
    trig = flow.trigger()
    rank, stack = {}, [trig.id]
    while stack:
        n = stack.pop()
        if n in rank:
            continue
        rank[n] = len(rank)
        outs = sorted(flow.out_edges(n), key=lambda e: LABEL_ORDER[e.on])
        stack.extend(e.dst for e in reversed(outs))       # first label popped first

    indeg = {s.id: 0 for s in flow.steps}
    for e in flow.edges:
        indeg[e.dst] += 1
    ready, order = [trig.id], []
    while ready:
        ready.sort(key=lambda n: rank.get(n, len(rank)))
        n = ready.pop(0)
        order.append(n)
        for e in flow.out_edges(n):
            indeg[e.dst] -= 1
            if indeg[e.dst] == 0:
                ready.append(e.dst)
    return order
