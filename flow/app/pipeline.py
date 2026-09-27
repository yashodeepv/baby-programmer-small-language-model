"""pipeline.py - What the Build button does: spec text -> everything the UI shows.

    spec text (as pasted, code fences and chatter allowed)
      -> compile (flow/spec)          errors with lines, or IR2 flows
      -> test runs (flow/gen/coverage) invented inputs until every edge is taken
      -> code (flow/codegen)          Python and JavaScript
      -> verify                       the code replays every test run like the simulator
"""

import random
import re

from flow.codegen import to_javascript, to_python
from flow.codegen.project import System, generate, verify_project
from flow.codegen.verify import verify
from flow.gen.coverage import synthesize
from flow.gen.order import canonical_order
from flow.ir2 import graph as g
from flow.ir2.expr import ArrLoop, Cmp, Const, Invalid, Pluck, Ref, Str, children
from flow.ir2.io import dump_flow
from flow.ir2.sexpr import emit
from flow.sim import run, summary
from flow.spec import compile_spec


def strip_fences(text):
    """Keep only the spec from an LLM reply: the fenced block if there is one,
    otherwise everything from the "Spec:" line on."""
    m = re.search(r'```[a-zA-Z]*\n(.*?)```', text, re.S)
    if m and 'Spec:' in m.group(1):
        text = m.group(1)
    i = text.find('Spec:')
    return text[i:] if i > 0 and text[:i].strip() else text


class FlowHints:
    """Value pools for inventing test inputs, read off the flow itself: a field
    compared with "gold" gets "gold" as a candidate, a field compared with
    100000 gets a range around it, so both sides of each test happen."""

    name = 'spec'

    def __init__(self, flow):
        self.ints, self.strings = {}, {}
        for e in _expressions(flow):
            stack = [e]
            while stack:
                n = stack.pop()
                stack.extend(children(n))
                if isinstance(n, Cmp):
                    for x, y in ((n.a, n.b), (n.b, n.a)):
                        f = _field(x)
                        if f and isinstance(y, Const):
                            self.ints.setdefault(f, set()).add(y.v)
                        if f and isinstance(y, Str):
                            self.strings.setdefault(f, set()).add(y.v)
                if isinstance(n, ArrLoop) and n.pred and isinstance(n.arr, Pluck) and n.pred[0] in ('gt', 'lt'):
                    self.ints.setdefault(n.arr.field, set()).add(n.pred[1])
        self.strings = {k: sorted(v) + ['other'] for k, v in self.strings.items()}

    def int_range(self, field):
        ks = self.ints.get(field)
        if not ks:
            return (0, 1000)
        return (min(0, min(ks) - 10), max(ks) * 2 + 10)

    def string_pool(self, field):
        return self.strings.get(field, ['alpha', 'beta'])


def _field(e):
    if isinstance(e, Ref):
        return e.path[-1]
    if isinstance(e, ArrLoop) and isinstance(e.arr, Pluck):
        return e.arr.field
    return None


def _expressions(flow):
    for s in flow.steps:
        if isinstance(s, g.Compute):
            yield from (e for _, e in s.assign)
        for attr in ('rule', 'cond', 'to', 'message'):
            if hasattr(s, attr):
                yield getattr(s, attr)
        for attr in ('args', 'data', 'response'):
            yield from (getattr(s, attr, None) or {}).values()


def _label(s):
    if s.title:
        return s.title
    if isinstance(s, g.Trigger):
        return {'http': f"{s.config.get('method')} {s.config.get('path')}", 'event': s.config.get('name'),
                'schedule': s.config.get('cron')}[s.source]
    if isinstance(s, g.Call):
        return f'{s.service}.{s.op}'
    if isinstance(s, g.Compute):
        return ', '.join(n for n, _ in s.assign)
    if isinstance(s, g.Persist):
        return f'{s.op} {s.entity}'
    if isinstance(s, g.Notify):
        return s.channel
    if isinstance(s, g.End):
        return 'stop' if s.status == 'ok' else 'error'
    return s.kind


def _sub(s):
    if isinstance(s, g.Call):
        return f'call · {s.service}.{s.op}' + (f' · retries {s.retries}' if s.retries else '')
    if isinstance(s, g.Trigger):
        return f'trigger · {s.source}'
    if isinstance(s, g.End):
        return f'end · {s.status}'
    if isinstance(s, g.Persist):
        return f'{s.op} · {s.entity}'
    if isinstance(s, g.Notify):
        return f'notify · {s.channel}'
    return {'branch': 'if'}.get(s.kind, s.kind)


def _logic(s):
    pairs = []
    if isinstance(s, g.Validate):
        pairs = [('rule', s.rule)]
    elif isinstance(s, g.Branch):
        pairs = [('cond', s.cond)]
    elif isinstance(s, g.Compute):
        pairs = list(s.assign)
    elif isinstance(s, g.Call):
        pairs = list(s.args.items())
    elif isinstance(s, g.Persist):
        pairs = list(s.data.items())
    elif isinstance(s, g.Notify):
        pairs = [('to', s.to), ('message', s.message)]
    elif isinstance(s, g.End) and s.response:
        pairs = list(s.response.items())
    out = []
    for k, e in pairs:
        out.append([k, f'(needs the logic model) {e.text}' if isinstance(e, Invalid) else emit(e)])
    return out


def build(text, seed=0):
    spec = strip_fences(text)
    res = compile_spec(spec)
    out = {
        'ok': res.ok, 'system': res.system, 'description': res.description, 'spec': spec,
        'errors': [{'line': e.line, 'rule': e.rule, 'message': e.message} for e in res.errors],
        'warnings': [{'line': e.line, 'rule': e.rule, 'message': e.message} for e in res.warnings],
        'intents': [{'line': i.line, 'flow': i.flow, 'step': i.step, 'field': i.field, 'text': i.text}
                    for i in res.intents],
        'flows': [],
    }
    rng = random.Random(seed)
    all_scenarios = {}
    for flow in res.flows:
        item = {
            'name': flow.name, 'mode': flow.mode, 'intent': flow.intent,
            'order': canonical_order(flow),
            'steps': [{'id': s.id, 'kind': s.kind, 'label': _label(s), 'sub': _sub(s),
                       'status': getattr(s, 'status', None), 'line': res.lines.get((flow.name, s.id), 0),
                       'logic': _logic(s)} for s in flow.steps],
            'edges': [{'src': e.src, 'dst': e.dst, 'on': e.on} for e in flow.edges],
            'ir2': dump_flow(flow),
            'runs': [], 'coverage': None, 'code': {}, 'verification': {},
        }
        pending = [i for i in res.intents if i.flow == flow.name]
        if res.errors:
            item['blocked'] = 'Fix the errors to generate test runs and code.'
        elif pending:
            item['blocked'] = (f'{len(pending)} value(s) are written as intent (prose), which needs the '
                               'logic model. Write them as exact expressions in backquotes to build code now.')
        else:
            scenarios, cov = synthesize(rng, flow, FlowHints(flow))
            all_scenarios[flow.name] = scenarios
            item['coverage'] = cov
            for name, sc in scenarios.items():
                r = run(flow, sc['input'], sc.get('mocks', {}))
                item['runs'].append({'name': name, 'input': sc['input'], 'mocks': sc.get('mocks', {}),
                                     'status': r.status, 'response': r.response, 'error': r.error,
                                     'path': r.path(), 'edges': [list(e) for e in r.edges],
                                     'effects': [summary(e) for e in r.effects]})
            item['code'] = {'python': to_python(flow), 'javascript': to_javascript(flow)}
            item['verification'] = verify(flow, scenarios.values())
        out['flows'].append(item)
    out['project'] = {}
    if res.flows and len(all_scenarios) == len(res.flows):
        system = System(res.system, res.description, res.flows, all_scenarios)
        for lang in ('python', 'javascript'):
            files = generate(system, lang)
            out['project'][lang] = {'files': files, 'verification': verify_project(files, lang)}
    return out


def project_zip(text, language, seed=0):
    """The generated project as a zip, or None if the spec doesn't build cleanly."""
    from flow.codegen.project import zip_bytes
    result = build(text, seed)
    project = result.get('project', {}).get(language)
    if not project:
        return None, result['system']
    return zip_bytes(project['files']), result['system']
