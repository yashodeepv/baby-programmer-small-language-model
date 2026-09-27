"""The simulator: stored scenarios, runtime semantics, and the JSON round trip."""

import json
import unittest

from flow.ir2.io import dump_flow, load_flow
from flow.ir2.sexpr import parse
from flow.sim import SimError, run, run_scenario
from flow.sim.evaluate import FlowFault, evaluate
from flow.tests.helpers import EXAMPLES, mutated, raw, step


class Scenarios(unittest.TestCase):
    def test_every_stored_scenario(self):
        for path in EXAMPLES:
            with open(path) as f:
                r = json.load(f)
            flow = load_flow(r)
            for name, sc in r.get('scenarios', {}).items():
                with self.subTest(flow=flow.name, scenario=name):
                    _, bad = run_scenario(flow, sc)
                    self.assertEqual(bad, [])


class Semantics(unittest.TestCase):
    def ev(self, text, **env):
        return evaluate(parse(text), env)

    def test_div_mod_truncate_toward_zero(self):
        for a, b in [(7, 2), (-7, 2), (7, -2), (-7, -2), (0, 5)]:
            q, m = self.ev(f'(div {a} {b})'), self.ev(f'(mod {a} {b})')
            self.assertEqual(q, int(a / b))
            self.assertEqual(q * b + m, a)

    def test_div_by_zero_faults(self):
        with self.assertRaises(FlowFault):
            self.ev('(div 1 0)')

    def test_int_wraps_like_a_w_register(self):
        self.assertEqual(self.ev('(add 2147483647 1)'), -2147483648)

    def test_empty_reductions_match_arm(self):
        self.assertEqual(self.ev('(max (pluck input.xs v))', input={'xs': []}), 0)
        self.assertEqual(self.ev('(count input.xs)', input={'xs': []}), 0)

    def test_fmt(self):
        self.assertEqual(self.ev('(fmt "n=" 3 " ok=" true)'), 'n=3 ok=true')

    def test_div_by_zero_fails_the_flow(self):
        f = mutated(lambda r: step(r, 'pricing')['assign'].update(
            discount='(div pricing.subtotal 0)'))
        sc = raw()['scenarios']['happy']
        res = run(f, sc['input'], sc['mocks'])
        self.assertEqual(res.status, 'error')
        self.assertIn('division by zero', res.error)
        self.assertEqual(res.effects, [])


class Guards(unittest.TestCase):
    def test_refuses_a_flow_that_does_not_check(self):
        f = mutated(lambda r: step(r, 'big_order').update(cond='nope'))
        with self.assertRaises(SimError):
            run(f, raw()['scenarios']['happy']['input'])

    def test_refuses_ill_typed_input(self):
        f = load_flow(raw())
        with self.assertRaises(SimError):
            run(f, {'order_id': 1})

    def test_refuses_ill_typed_mock(self):
        f = load_flow(raw())
        sc = raw()['scenarios']['happy']
        with self.assertRaises(SimError):
            run(f, sc['input'], {'charge': {'ok': {'charge_id': 5}}})

    def test_refuses_missing_mock(self):
        f = load_flow(raw())
        with self.assertRaises(SimError):
            run(f, raw()['scenarios']['happy']['input'], {})


class JsonRoundTrip(unittest.TestCase):
    def test_load_dump_load(self):
        for path in EXAMPLES:
            with open(path) as f:
                r = json.load(f)
            with self.subTest(path=path):
                once = load_flow(r)
                twice = load_flow(json.loads(json.dumps(dump_flow(once))))
                self.assertEqual(once, twice)

    def test_invalid_expression_survives_round_trip(self):
        f = mutated(lambda r: step(r, 'big_order').update(cond='(ge pricing.total'))
        self.assertEqual(step(dump_flow(f), 'big_order')['cond'], '(ge pricing.total')


if __name__ == '__main__':
    unittest.main()
