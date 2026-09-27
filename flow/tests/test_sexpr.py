"""IR2 text form: exact round trip, strict parsing, and arm compatibility."""

import random
import unittest

from flow.arm_bridge import arm_ir, arm_sexpr
from flow.ir2.expr import All, Bin, Bool, Const, Fmt, Len, Not, Pluck, Ref, Str
from flow.ir2.sexpr import ParseError, emit, parse
from flow.sim.evaluate import evaluate

import grammar      # arm/grammar.py, on the path via arm_bridge

IR2_TEXTS = [
    'input.total',
    '"gold"',
    '"say \\"hi\\" \\\\ bye"',
    'true',
    '(div pricing.subtotal 10)',
    '(mod input.n 7)',
    '(not (eq input.tier "gold"))',
    '(all (gt input.a 1) (lt input.b 2) true)',
    '(any false (ne input.s ""))',
    '(len input.items)',
    '(sum (pluck input.items amount))',
    '(count (pluck fetch.invoices days_late) :gt 30)',
    '(max (pluck input.items amount) :dbl :even)',
    '(fmt "Order " input.id ": " pricing.total)',
    '(sel (all (gt input.a 0) true) input.a (neg input.a))',
    '(sum (rng 1 input.n) :sq)',
]


class RoundTrip(unittest.TestCase):
    def test_ir2_forms(self):
        for text in IR2_TEXTS:
            with self.subTest(text=text):
                node = parse(text)
                self.assertEqual(emit(node), text)
                self.assertEqual(parse(emit(node)), node)

    def test_trees(self):
        self.assertEqual(parse('input.items'), Ref(('input', 'items')))
        self.assertEqual(parse('(div a.b 10)'), Bin('div', Ref(('a', 'b')), Const(10)))
        self.assertEqual(parse('(pluck xs price)'), Pluck(Ref(('xs',)), 'price'))
        self.assertEqual(parse('(len xs)'), Len(Ref(('xs',))))
        self.assertEqual(parse('(not true)'), Not(Bool(True)))
        self.assertEqual(parse('(all true false)'), All((Bool(True), Bool(False))))
        self.assertEqual(parse('(fmt "a" 1)'), Fmt((Str('a'), Const(1))))

    def test_rejects(self):
        for bad in ['(', '(add 1)', '(add 1 2 3)', '(frob 1)', '"open', '(sum xs :bogus)',
                    '(pluck xs)', '(all true)', '(fmt)', '1 2', ')', ':gt', '(rng 1 2)',
                    '(sum xs :even :odd)', '"bad \\n escape"']:
            with self.subTest(text=bad):
                with self.assertRaises(ParseError):
                    parse(bad)


class ArmSuperset(unittest.TestCase):
    """Every program the arm grammar produces is valid IR2 with the same meaning."""

    def setUp(self):
        self.progs = grammar.sample(400, seed=11)

    def test_same_text_same_tree(self):
        for p in self.progs:
            text = arm_sexpr.emit(p.body)
            with self.subTest(text=text):
                node = parse(text)
                self.assertEqual(node, arm_sexpr.parse(text))
                self.assertEqual(emit(node), text)

    def test_same_value(self):
        rng = random.Random(3)
        for p in self.progs:
            node = parse(arm_sexpr.emit(p.body))
            for _ in range(4):
                args = [rng.randint(-50, 50) for _ in range(3)]
                arr = [rng.randint(-20, 20) for _ in range(rng.randint(0, 8))]
                want = arm_ir.evaluate(p.body, args, arr)
                got = evaluate(node, {}, legacy=(args, arr))
                self.assertEqual(got, want, arm_sexpr.emit(p.body))


if __name__ == '__main__':
    unittest.main()
