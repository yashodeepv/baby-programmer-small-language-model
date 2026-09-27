"""The corpus generator: samples check, scenarios reproduce, spans and labels are exact."""

import json
import random
import unittest

from flow.check import check
from flow.gen.build import build_one
from flow.gen.catalog import all_catalogs
from flow.gen.coverage import synthesize
from flow.gen.labels import deanonymize, m1_example, m2_examples
from flow.gen.order import canonical_order
from flow.gen.render import N_STYLES, render
from flow.gen.sampler import sample_flows
from flow.ir2 import graph as g
from flow.ir2.io import dump_flow, load_flow
from flow.ir2.sexpr import parse
from flow.sim import run_scenario

CATS = all_catalogs()


def flows(n, seed=0):
    return list(sample_flows(n, CATS, seed=seed))


class Sampler(unittest.TestCase):
    def test_every_sample_checks(self):
        for f, meta in flows(300, seed=4):
            with self.subTest(flow=f.name):
                self.assertTrue(check(f).ok)
                self.assertLessEqual(len(f.steps), 16)

    def test_deterministic(self):
        a = [dump_flow(f) for f, _ in flows(20, seed=9)]
        b = [dump_flow(f) for f, _ in flows(20, seed=9)]
        self.assertEqual(a, b)

    def test_uses_every_pattern_and_catalog(self):
        seen_p, seen_c = set(), set()
        for _, meta in flows(300, seed=1):
            seen_p |= set(meta['patterns'])
            seen_c.add(meta['catalog'])
        self.assertEqual(seen_p, {'validate', 'compute', 'call', 'branch', 'parallel',
                                  'persist', 'notify'})
        self.assertEqual(seen_c, set(CATS))


class Coverage(unittest.TestCase):
    def test_scenarios_replay_and_cover(self):
        rng = random.Random(0)
        full = 0
        for f, meta in flows(80, seed=2):
            scenarios, cov = synthesize(rng, f, CATS[meta['catalog']])
            full += cov == 1.0
            for name, sc in scenarios.items():
                with self.subTest(flow=f.name, scenario=name):
                    _, bad = run_scenario(f, json.loads(json.dumps(sc)))
                    self.assertEqual(bad, [])
        self.assertGreater(full, 60)          # contradictory flows are the rest


class Order(unittest.TestCase):
    def test_canonical_order_is_topological_and_ignores_list_order(self):
        for f, _ in flows(100, seed=3):
            order = canonical_order(f)
            pos = {s: i for i, s in enumerate(order)}
            self.assertEqual(sorted(order), sorted(s.id for s in f.steps))
            for e in f.edges:
                self.assertLess(pos[e.src], pos[e.dst])
            # Parallel branch edges keep their relative order: branches run in
            # edge order, so that order is part of the flow's meaning.
            edges = ([e for e in reversed(f.edges) if e.on != 'branch']
                     + [e for e in f.edges if e.on == 'branch'])
            shuffled = g.Flow(name=f.name, mode=f.mode, steps=list(reversed(f.steps)),
                              edges=edges, services=f.services, entities=f.entities)
            self.assertEqual(canonical_order(shuffled), order)


def _value(step, key):
    """The true value behind a `value:` span."""
    if key.startswith('name.'):
        return key[5:]
    if key.startswith('key.'):
        return key[4:]
    if isinstance(step, g.Trigger):
        return step.config[key]
    return getattr(step, key)


class Render(unittest.TestCase):
    def test_spans_are_exact(self):
        for f, meta in flows(150, seed=5):
            for style in range(N_STYLES):
                text, spans = render(f, style, meta['noun'])
                for sp in spans:
                    with self.subTest(flow=f.name, style=style, span=sp):
                        self.assertTrue(0 <= sp['start'] <= sp['end'] <= len(text))
                        if sp['role'].startswith('value:'):
                            self.assertEqual(text[sp['start']:sp['end']],
                                             _value(f.step(sp['step']), sp['role'][6:]))
                clauses = {sp['step'] for sp in spans if sp['role'] == 'clause'}
                for s in f.steps:
                    if not isinstance(s, g.Join):
                        self.assertIn(s.id, clauses, f'{f.name}: {s.id} has no clause')

    def test_everything_the_encoder_must_copy_is_in_the_text(self):
        for f, meta in flows(150, seed=6):
            text, spans = render(f, 0, meta['noun'])
            m1 = m1_example(f, text, spans, f.mode)
            for slot in m1['slots']:
                s = f.step(slot['id'])
                need = set()
                if isinstance(s, g.Trigger):
                    need = set(s.config)
                elif isinstance(s, g.Call):
                    need = {'service', 'op'}
                elif isinstance(s, g.Persist):
                    need = {'entity'}
                elif isinstance(s, g.Validate):
                    need = {'message'}
                elif isinstance(s, g.Compute):
                    need = {f'name.{n}' for n, _ in s.assign}
                elif isinstance(s, g.End) and s.response:
                    # An error's `reason` is implied by the wording ("stop with an
                    # error saying ..."), not copied.
                    need = {f'key.{k}' for k in s.response
                            if not (s.status == 'error' and k == 'reason')}
                with self.subTest(flow=f.name, step=s.id):
                    self.assertTrue(need <= set(slot['copy']), f'missing {need - set(slot["copy"])}')

    def test_hand_written_examples_render(self):
        from flow.tests.helpers import EXAMPLES
        for path in EXAMPLES:
            with open(path) as fh:
                f = load_flow(json.load(fh))
            for style in range(N_STYLES):
                text, spans = render(f, style)
                m2_examples(f, text, spans, style)       # raises if an expression was not said


class Labels(unittest.TestCase):
    def test_m1_edges_are_the_flow_edges(self):
        for f, meta in flows(100, seed=7):
            text, spans = render(f, 1, meta['noun'])
            m1 = m1_example(f, text, spans, f.mode)
            ids = [s['id'] for s in m1['slots']]
            got = sorted((ids[a], ids[b], on) for a, b, on in m1['edges'])
            self.assertEqual(got, sorted((e.src, e.dst, e.on) for e in f.edges))

    def test_m2_anonymisation_round_trips(self):
        for f, meta in flows(150, seed=8):
            style = f.steps.__len__() % N_STYLES
            text, spans = render(f, style, meta['noun'])
            for ex in m2_examples(f, text, spans, style):
                a = ex['anon']
                with self.subTest(flow=f.name, target=ex['target']):
                    self.assertEqual(parse(deanonymize(a['target'], a['mapping'])),
                                     parse(ex['target']))
                    self.assertEqual(ex['intent'], text[text.index(ex['intent']):][:len(ex['intent'])])
                    for name in a['mapping']['names'].values():
                        if '_' in name:
                            self.assertNotRegex(a['intent'], rf'\b{name.replace("_", " ")}\b')
                            self.assertNotIn(name, a['scope'])


class Curriculum(unittest.TestCase):
    def test_each_chapter_teaches_its_idea_and_only_what_came_before(self):
        from flow.gen.curriculum import CHAPTERS, chapter_flows
        from flow.ir2.expr import ArrLoop, children

        def nodes(e):
            out, stack = [], [e]
            while stack:
                n = stack.pop()
                out.append(n)
                stack.extend(children(n))
            return out

        for ch in CHAPTERS:
            for f, meta in chapter_flows(ch, 40, CATS, seed=5):
                with self.subTest(chapter=ch.number, flow=f.name):
                    self.assertTrue(check(f).ok)
                    self.assertEqual(meta['chapter'], ch.number)
                    self.assertTrue(set(meta['patterns']) <= set(ch.patterns))
                    if ch.require:
                        self.assertIn(ch.require, meta['patterns'])
                    if ch.depth == 0:
                        exprs = [e for s in f.steps for e in _step_exprs(s)]
                        self.assertFalse(any(isinstance(n, ArrLoop) for e in exprs for n in nodes(e)),
                                         'a level-0 chapter must not reduce over lists')

    def test_bool_fields_say_is_true_only_in_conditions(self):
        for f, meta in flows(120, seed=12):
            text, spans = render(f, 0, meta['noun'])
            for sp in spans:
                role = sp['role']
                if role.startswith('expr:response.') or role.startswith('expr:data.') \
                        or role.startswith('expr:args.'):
                    self.assertNotIn(' is true', text[sp['start']:sp['end']])


def _step_exprs(s):
    if isinstance(s, g.Compute):
        return [e for _, e in s.assign]
    out = []
    for attr in ('rule', 'cond', 'to', 'message'):
        if hasattr(s, attr):
            out.append(getattr(s, attr))
    for attr in ('args', 'data', 'response'):
        out.extend((getattr(s, attr, None) or {}).values())
    return out


class Build(unittest.TestCase):
    def test_records_are_json_and_deterministic(self):
        def build(seed):
            rng, out = random.Random(seed), []
            for f, meta in flows(40, seed=seed):
                rec, _ = build_one(rng, f, meta, CATS[meta['catalog']], 'logistics')
                if rec:
                    out.append(json.dumps(rec, sort_keys=True))
            return out
        a, b = build(11), build(11)
        self.assertEqual(a, b)
        self.assertGreater(len(a), 25)
        rec = json.loads(a[0])
        self.assertTrue(check(load_flow(rec['flow'])).ok)
        self.assertIn(rec['split'], {'train', 'val', 'style', 'catalog'})


if __name__ == '__main__':
    unittest.main()
