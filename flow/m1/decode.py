"""decode.py - Model output -> a structured prediction -> an IR2 flow.

A prediction is compared with the gold structure field by field (evaluate.py),
and also rebuilt into a real flow, which the checker and simulator then judge.

The logic model (M2) isn't trained yet, so each step's expressions are
filled by an oracle: the predicted English span is looked up among the true
spans of the request, and if it matches one exactly, the true expression is
used. A wrong or missing span leaves the expression unparsed, and the flow
fails the checker, as it should. What this measures is M1 alone: given
perfect step logic, does the flowchart it read from the English behave
correctly?
"""

from flow.gen.catalog import all_catalogs
from flow.ir2 import graph as g
from flow.ir2.expr import Invalid
from flow.ir2.sexpr import ParseError, parse
from flow.m1.vocab import CLAUSE_TAGS, EDGE_LABELS, EMPTY, KINDS, MODES, N_SLOTS, SETTINGS, TAGS

_CATALOGS = None


def catalogs():
    global _CATALOGS
    if _CATALOGS is None:
        _CATALOGS = all_catalogs()
    return _CATALOGS


# --------------------------------------------------------------------------
# Structure: the same shape for gold and predicted
# --------------------------------------------------------------------------

def _spans_from_tags(tag_ids, offsets, n_text, text):
    """B/I/O tags over tokens -> {role: [text, ...]} in reading order."""
    out, cur = {}, None
    for t in range(1, n_text + 1):
        tag = TAGS[tag_ids[t]]
        if tag == 'O':
            cur = None
            continue
        bi, role = tag.split('-', 1)
        if bi == 'B' or cur is None or cur[0] != role:
            cur = [role, offsets[t][0], offsets[t][1]]
            out.setdefault(role, []).append(cur)
        else:
            cur[2] = offsets[t][1]
    return {role: [text[a:b] for _, a, b in spans] for role, spans in out.items()}


def predicted(out, b, ex):
    """One example's model output -> structure."""
    text = ex.record['m1']['text']
    kinds = out['kind'][b].argmax(-1).tolist()
    sets = {k: v[b].argmax(-1).tolist() for k, v in out['settings'].items()}
    tags = out['tags'][b].argmax(-1).tolist()
    edges_arg = out['edges'][b].argmax(-1).tolist()
    slots = []
    for i in range(N_SLOTS):
        kind = KINDS[kinds[i]]
        spans = _spans_from_tags(tags[i], ex.offsets, ex.n_text, text) if kind != EMPTY else {}
        slots.append({
            'kind': kind,
            'settings': {k: SETTINGS[k][sets[k][i]] for k in SETTINGS},
            'spans': spans,
        })
    edges = sorted((i, j, EDGE_LABELS[edges_arg[i][j]]) for i in range(N_SLOTS) for j in range(N_SLOTS)
                   if edges_arg[i][j] != 0 and slots[i]['kind'] != EMPTY and slots[j]['kind'] != EMPTY)
    return {'slots': slots, 'edges': edges, 'mode': MODES[out['mode'][b].argmax(-1).item()]}


def gold(ex):
    """The same structure, read from the example's labels."""
    text = ex.record['m1']['text']
    slots = []
    for i in range(N_SLOTS):
        kind = KINDS[ex.kind[i]]
        spans = _spans_from_tags(ex.tags[i], ex.offsets, ex.n_text, text) if kind != EMPTY else {}
        slots.append({'kind': kind,
                      'settings': {k: SETTINGS[k][ex.settings[k][i]] for k in SETTINGS},
                      'spans': spans})
    edges = sorted((i, j, EDGE_LABELS[ex.edges[i][j]]) for i in range(N_SLOTS) for j in range(N_SLOTS)
                   if ex.edges[i][j] != 0)
    return {'slots': slots, 'edges': edges, 'mode': MODES[ex.mode]}


# --------------------------------------------------------------------------
# Rebuilding a flow
# --------------------------------------------------------------------------

def _oracle(record):
    """Span text -> the true expression, for every expression in the request."""
    text = record['m1']['text']
    steps = {s['id']: s for s in record['flow']['steps']}
    table = {}
    for slot in record['m1']['slots']:
        raw = steps[slot['id']]
        for field, (a, b) in slot['exprs'].items():
            head, _, key = field.partition('.')
            src = {'assign': 'assign', 'args': 'args', 'data': 'data', 'response': 'response'}.get(head)
            value = raw[src][key] if src else raw[field]
            table.setdefault(text[a:b], value)
    return table


def _expr(table, span_text):
    if span_text is None:
        return Invalid('', 'no span predicted')
    if span_text not in table:
        return Invalid(span_text, 'no logic model yet, and not a true span')
    try:
        return parse(table[span_text])
    except ParseError as e:
        return Invalid(span_text, str(e))


def _nth(spans, role, n):
    xs = spans.get(role, [])
    return xs[n] if n < len(xs) else None


def build_flow(pred, record):
    """Structure -> IR2 Flow. Steps take the true step's id where the kinds
    agree at that slot, so oracle expressions referring to earlier steps resolve."""
    cat = catalogs()[record['catalog']]
    services = {s: dict(ops) for s, ops in cat.services.items()}
    table = _oracle(record)
    gold_ids = [s['id'] for s in record['m1']['slots']]
    gold_kinds = [s['kind'] for s in record['m1']['slots']]
    ids, steps = {}, []
    for i, s in enumerate(pred['slots']):
        if s['kind'] == EMPTY:
            continue
        ids[i] = gold_ids[i] if i < len(gold_ids) and gold_kinds[i] == s['kind'] else f'x{i}'

    for i, sid in ids.items():
        s = pred['slots'][i]
        sp, st, kind = s['spans'], s['settings'], s['kind']
        E = lambda role, n=0: _expr(table, _nth(sp, role, n))
        if kind == 'trigger':
            source = st['source'] if st['source'] != 'none' else 'http'
            config = {'http': {'method': st['method'], 'path': _nth(sp, 'v_path', 0) or ''},
                      'event': {'name': _nth(sp, 'v_event', 0) or ''},
                      'schedule': {'cron': _nth(sp, 'v_cron', 0) or ''}}[source]
            spec = next((t for t in cat.triggers if t.source == source and t.config == config), None)
            payload = spec.payload if spec else g.RecordT(())
            steps.append(g.Trigger(id=sid, source=source, config=config, payload=payload))
        elif kind == 'call':
            service, op = _nth(sp, 'v_service', 0) or '', _nth(sp, 'v_op', 0) or ''
            sig = services.get(service, {}).get(op)
            params = [n for n, _ in sig.input.fields] if sig else []
            retries = int(st['retries']) if st['retries'] != 'none' else 0
            steps.append(g.Call(id=sid, service=service, op=op, retries=retries,
                                args={p: E('e_arg', n) for n, p in enumerate(params)}))
        elif kind == 'compute':
            names = sp.get('v_var', [])
            steps.append(g.Compute(id=sid, assign=[(nm, E('e_assign', n)) for n, nm in enumerate(names)]))
        elif kind == 'validate':
            steps.append(g.Validate(id=sid, rule=E('e_rule'), message=_nth(sp, 'v_message', 0) or ''))
        elif kind == 'branch':
            steps.append(g.Branch(id=sid, cond=E('e_cond')))
        elif kind == 'parallel':
            steps.append(g.Parallel(id=sid, join=''))
        elif kind == 'join':
            steps.append(g.Join(id=sid))
        elif kind == 'persist':
            entity = _nth(sp, 'v_entity', 0) or ''
            rec = cat.entities.get(entity)
            op = st['op'] if st['op'] != 'none' else 'create'
            fields = [] if rec is None else [n for n, _ in (rec.fields[:1] if op == 'delete' else rec.fields)]
            steps.append(g.Persist(id=sid, entity=entity, op=op,
                                   data={f: E('e_data', n) for n, f in enumerate(fields)}))
        elif kind == 'notify':
            channel = st['channel'] if st['channel'] != 'none' else 'email'
            steps.append(g.Notify(id=sid, channel=channel, to=E('e_to'), message=E('e_msg')))
        elif kind == 'end':
            status = st['status'] if st['status'] != 'none' else 'ok'
            resp = None
            if st['responds'] == 'yes':
                keys = sp.get('v_key', []) or (['reason'] if status == 'error' else [])
                resp = {k: E('e_resp', n) for n, k in enumerate(keys)}
            steps.append(g.End(id=sid, status=status, response=resp))

    edges = [g.Edge(ids[a], ids[b], on) for a, b, on in pred['edges'] if a in ids and b in ids]
    flow = g.Flow(name=record['id'], mode='sync', steps=steps, edges=edges, services={},
                  entities=dict(cat.entities))
    for p in steps:                       # a parallel joins at the first join after it
        if isinstance(p, g.Parallel):
            later = [s.id for s in steps[steps.index(p) + 1:] if isinstance(s, g.Join)]
            p.join = later[0] if later else ''
    flow.services = {s.service: services[s.service] for s in steps
                     if isinstance(s, g.Call) and s.service in services}
    mode = pred['mode']
    if mode == 'abstain':                 # the model would ask; for replay, infer from responses
        mode = 'sync' if any(isinstance(s, g.End) and s.response for s in steps) else 'async'
    flow.mode = mode
    return flow
