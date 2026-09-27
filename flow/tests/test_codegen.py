"""Generated Python and JavaScript behave exactly like the simulator."""

import json
import random
import unittest

from flow.codegen.verify import node_available, verify
from flow.gen.catalog import all_catalogs
from flow.gen.coverage import synthesize
from flow.gen.sampler import sample_flows
from flow.ir2.io import load_flow
from flow.tests.helpers import EXAMPLES

CATS = all_catalogs()

# Arithmetic edge cases: wrap-around, truncating division, big squares, shifts.
EDGES = {
    'name': 'edges', 'mode': 'sync',
    'steps': [
        {'id': 'start', 'kind': 'trigger', 'source': 'http', 'config': {'method': 'POST', 'path': '/x'},
         'payload': {'a': 'int', 'b': 'int', 'xs': ['list', {'v': 'int'}]}},
        {'id': 'calc', 'kind': 'compute', 'assign': {
            'wrap': '(add 2147483647 input.a)',
            'mulwrap': '(mul 65536 (add 65536 input.a))',
            'q': '(div (neg 7) (add 2 input.b))',
            'r': '(mod (neg 7) (add 2 input.b))',
            'sq': '(sum (pluck input.xs v) :sq)',
            'prod': '(product (pluck input.xs v))',
            'shr': '(lsr (neg 1) 28)',
            'shl': '(lsl input.a 31)',
            'bits': '(eor (and input.a 12) (orr input.b 3))',
            'lo': '(min (pluck input.xs v) :odd)',
            'hi': '(max (pluck input.xs v) :gt 100)',
            'cnt': '(count (pluck input.xs v) :divk 3)',
            'rng': '(sum (rng input.b 20) :mul 3 :even)',
            'text': '(fmt "a=" input.a " neg=" (lt input.a 0))',
            'pick': '(sel (all (gt input.a 0) (not (eq input.b 0))) (abs input.b) (neg input.a))',
        }},
        {'id': 'done', 'kind': 'end', 'status': 'ok', 'response': {
            'wrap': 'calc.wrap', 'mulwrap': 'calc.mulwrap', 'q': 'calc.q', 'r': 'calc.r', 'sq': 'calc.sq',
            'prod': 'calc.prod', 'shr': 'calc.shr', 'shl': 'calc.shl', 'bits': 'calc.bits', 'lo': 'calc.lo',
            'hi': 'calc.hi', 'cnt': 'calc.cnt', 'rng': 'calc.rng', 'text': 'calc.text', 'pick': 'calc.pick'}},
    ],
    'edges': [{'from': 'start', 'to': 'calc'}, {'from': 'calc', 'to': 'done'}],
}


class Codegen(unittest.TestCase):
    def check_flow(self, flow, scenarios):
        rep = verify(flow, scenarios)
        self.assertEqual(rep['python']['status'], 'verified', rep['python']['detail'])
        if node_available():
            self.assertEqual(rep['javascript']['status'], 'verified', rep['javascript']['detail'])

    def test_hand_written_examples(self):
        for path in EXAMPLES:
            with open(path) as f:
                raw = json.load(f)
            with self.subTest(path=path):
                self.check_flow(load_flow(raw), raw['scenarios'].values())

    def test_sampled_flows(self):
        rng = random.Random(0)
        flows = []
        for f, meta in sample_flows(60, CATS, seed=21):
            scenarios, _ = synthesize(rng, f, CATS[meta['catalog']])
            flows.append((f, list(scenarios.values())))
        for f, scenarios in flows:
            with self.subTest(flow=f.name):
                self.check_flow(f, scenarios)

    def test_every_sampled_flow_structures_and_the_fallback_still_verifies(self):
        from flow.codegen.structure import structure
        rng = random.Random(7)
        for f, meta in sample_flows(120, CATS, seed=8):
            structure(f)                                  # raises if it can't
        for f, meta in list(sample_flows(20, CATS, seed=9)):
            scenarios, _ = synthesize(rng, f, CATS[meta['catalog']])
            with self.subTest(flow=f.name):
                rep = verify(f, scenarios.values(), style='machine')
                self.assertEqual(rep['python']['status'], 'verified', rep['python']['detail'])
                if node_available():
                    self.assertEqual(rep['javascript']['status'], 'verified', rep['javascript']['detail'])

    def test_arithmetic_edge_cases(self):
        rng = random.Random(4)
        big = [-2147483648, 2147483647, -1, 0, 1, 46341, -46341, 99999, 3, 102]
        scenarios = []
        for _ in range(60):
            scenarios.append({'input': {
                'a': rng.choice(big + [rng.randint(-10 ** 6, 10 ** 6)]),
                'b': rng.choice([-2, -3, 0, 1, 5, rng.randint(-50, 50)]),
                'xs': [{'v': rng.choice(big)} for _ in range(rng.randint(0, 6))]}})
        self.check_flow(load_flow(EDGES), scenarios)


if __name__ == '__main__':
    unittest.main()
