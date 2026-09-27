"""Baby Spec: the examples compile, match the hand-written IR2, and each rule fires."""

import glob
import json
import os
import unittest

from flow.check import check
from flow.ir2.expr import Invalid
from flow.ir2.sexpr import emit
from flow.sim import run, summary
from flow.spec import compile_spec
from flow.spec.exact import parse_exact
from flow.spec.values import Resolver
from flow.ir2.types import INT, STR, ListT, RecordT

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECS = os.path.join(HERE, 'spec', 'examples')

# What the examples' prose values mean -- standing in for the logic model.
INTENTS = {
    'the sum of the item amounts': 'sum(items.amount)',
    'the total amount owed across all invoices': 'sum(invoices.amount)',
    'days is at most balance, or kind is "sick"': 'days <= balance or kind == "sick"',
}


def spec(name):
    with open(os.path.join(SPECS, f'{name}.spec.md')) as f:
        return f.read()


def compiled(name, intent=INTENTS.get):
    res = compile_spec(spec(name), intent=intent)
    assert res.ok, '\n'.join(str(e) for e in res.errors)
    return res


class Examples(unittest.TestCase):
    def test_every_example_checks(self):
        for path in glob.glob(os.path.join(SPECS, '*.spec.md')):
            with self.subTest(path=path):
                with open(path) as f:
                    res = compile_spec(f.read())
                self.assertEqual([str(e) for e in res.errors], [])
                self.assertEqual(len(res.intents), 1)          # one prose value each, on purpose
                self.assertTrue(all(isinstance(_expr_of(res, i), Invalid) for i in res.intents))

    def test_same_system_as_the_hand_written_ir2(self):
        """The compiled spec replays the hand-written flow's scenarios with identical outcomes."""
        for name in ('order_checkout', 'overdue_reminders'):
            flow = compiled(name).flows[0]
            self.assertTrue(check(flow).ok)
            with open(os.path.join(HERE, 'examples', f'{name}.json')) as f:
                scenarios = json.load(f)['scenarios']
            for sname, sc in scenarios.items():
                with self.subTest(flow=name, scenario=sname):
                    res = run(flow, sc['input'], sc.get('mocks', {}))
                    want = sc['expect']
                    self.assertEqual(res.status, want['status'])
                    self.assertEqual(res.response, want.get('response'))
                    self.assertEqual([summary(e) for e in res.effects], want['effects'])

    def test_leave_requests_behaves(self):
        flow = compiled('leave_requests').flows[0]
        base = {'employee_id': 'E-1', 'employee_email': 'e@x', 'manager_email': 'm@x',
                'days': 3, 'balance': 10, 'kind': 'vacation'}
        ok = {'employee': {'ok': {'department': 'sales', 'level': 6, 'active': True}}}
        r = run(flow, base, ok)
        self.assertEqual(r.status, 'ok')
        self.assertEqual(r.response, {'employee_id': 'E-1', 'days': 3, 'needs_review': False})
        self.assertEqual([summary(e) for e in r.effects][1:],
                         ['persist leave_request.create', 'notify email m@x: E-1 requested 3 days of vacation leave'])
        r = run(flow, {**base, 'days': 12, 'balance': 30},
                {'employee': {'ok': {'department': 'sales', 'level': 2, 'active': True}}})
        self.assertTrue(r.response['needs_review'])
        self.assertIn('notify email hr@corp.example: Please review 12 days of vacation leave for E-1',
                      [summary(e) for e in r.effects])
        r = run(flow, {**base, 'days': 20}, ok)
        self.assertEqual((r.status, r.response), ('error', {'reason': 'not enough leave balance',
                                                            'employee_id': 'E-1', 'balance': 10}))
        r = run(flow, {**base, 'days': 20, 'kind': 'sick'}, ok)
        self.assertEqual(r.status, 'ok')                        # sick leave ignores the balance
        r = run(flow, base, {'employee': {'ok': {'department': 'x', 'level': 6, 'active': False}}})
        self.assertEqual((r.status, r.error), ('error', 'validate2: inactive employee'))

    def test_titles_become_flowchart_labels(self):
        flow = compiled('order_checkout').flows[0]
        titles = {s.title for s in flow.steps}
        self.assertTrue({'Has items?', 'Charge the card', '$1000 or more?', 'Order placed'} <= titles)


def _expr_of(res, intent):
    flow = next(f for f in res.flows if f.name == intent.flow)
    step = flow.step(intent.step)
    head, _, key = intent.field.partition('.')
    return dict(step.assign)[key] if head == 'assign' else getattr(step, head)


class Exact(unittest.TestCase):
    def setUp(self):
        self.r = Resolver({'LIMIT': (parse_exact('30', Resolver({}), 0), INT)})
        self.r.add_record('input', RecordT((('tier', STR), ('total', INT),
                                            ('items', ListT(RecordT((('amount', INT),)))))))

    def ir(self, text):
        return emit(parse_exact(text, self.r, 1))

    def test_translations(self):
        cases = {
            'total - 5': '(sub input.total 5)',
            'total / 10 if tier == "gold" else 0': '(sel (eq input.tier "gold") (div input.total 10) 0)',
            'sum(items.amount)': '(sum (pluck input.items amount))',
            'count(items.amount where > LIMIT)': '(count (pluck input.items amount) :gt 30)',
            'sum(items.amount each * 2 where even)': '(sum (pluck input.items amount) :mul 2 :even)',
            'max(range(1, total) where odd)': '(max (rng 1 input.total) :odd)',
            'len(items) > 0 and not (tier != "gold")': '(all (gt (len input.items) 0) (not (ne input.tier "gold")))',
            '-total': '(neg input.total)',
            '"Total: {total}"': '(fmt "Total: " input.total)',
            'total > 1 or total < 0': '(any (gt input.total 1) (lt input.total 0))',
        }
        for text, want in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.ir(text), want)

    def test_rejects(self):
        from flow.spec.document import SpecError
        for bad in ['total +', 'foo(total)', 'sum(items.amount where >)', 'range(1, 2)', 'unknown_name',
                    'items.nope', '(total', 'total total']:
            with self.subTest(text=bad):
                with self.assertRaises(SpecError):
                    parse_exact(bad, self.r, 1)


class Rules(unittest.TestCase):
    """Each validity rule fires on a minimal fault, at the right line."""

    def fault(self, old, new, name='order_checkout'):
        text = spec(name)
        self.assertIn(old, text)
        return compile_spec(text.replace(old, new, 1), intent=INTENTS.get)

    def assertRule(self, res, rule):
        rules = [e.rule for e in res.errors]
        self.assertIn(rule, rules, '\n'.join(str(e) for e in res.errors))
        self.assertTrue(all(e.line > 0 for e in res.errors), 'every error points at a line')

    def test_layout(self):
        self.assertRule(self.fault('Spec: baby-spec/0.1', 'Spec: baby-spec/9'), 'layout')
        self.assertRule(self.fault('2. [Price the order]', '3. [Price the order]'), 'layout')

    def test_unique_names(self):
        self.assertRule(self.fault('Compute pricing:', 'Compute charge:'), 'unique-names')

    def test_known_service_and_args(self):
        self.assertRule(self.fault('Call payments.charge', 'Call payments.refund'), 'known-service')
        self.assertRule(self.fault('with order_id, amount = total', 'with order_id'), 'call-args')
        self.assertRule(self.fault('with order_id, amount = total', 'with order_id, amount = total, x = 1'),
                        'call-args')

    def test_entities(self):
        self.assertRule(self.fault('Save order:', 'Save orders:'), 'known-entity')
        self.assertRule(self.fault('Save order: order_id, total, charge_id', 'Save order: order_id, total'),
                        'entity-fields')

    def test_resolves(self):
        self.assertRule(self.fault('`subtotal - discount`', '`subtotal - discont`'), 'resolves')
        self.assertRule(self.fault('Respond with order_id, charge_id, total',
                                   'Respond with order_id, charge_id, total, colour'), 'resolves')

    def test_value_from_one_side_of_an_if_is_not_available_after(self):
        text = spec('leave_requests').replace(
            '   a. [Approve] Save leave_request: employee_id, days, status = "approved".',
            '   a. [Approve] Compute approval:\n      - code: 1')
        res = compile_spec(text.replace('Respond with employee_id, days, needs_review',
                                        'Respond with employee_id, days, code'), intent=INTENTS.get)
        self.assertRule(res, 'resolves')

    def test_types(self):
        self.assertRule(self.fault('amount = total', 'amount = order_id'), 'types')

    def test_sync_and_respond(self):
        self.assertRule(self.fault('Mode: async', 'Mode: sync', 'overdue_reminders'), 'sync-http')
        self.assertRule(self.fault('Mode: sync', 'Mode: async'), 'respond-sync')

    def test_retry_range(self):
        self.assertRule(self.fault('Retry 1 time.', 'Retry 7 times.'), 'retry-range')

    def test_parallel_branch(self):
        self.assertRule(self.fault('   b. [Save the order] Save order: order_id, total, charge_id.',
                                   '   b. Stop.'), 'parallel-branch')

    def test_reachable(self):
        self.assertRule(self.fault('4. [$1000 or more?]', '4. Stop.\n5. [$1000 or more?]'), 'layout')
        res = self.fault('6. [Order placed] Respond with order_id, charge_id, total.',
                         '6. [Order placed] Respond with order_id, charge_id, total.\n7. Stop.')
        self.assertRule(res, 'reachable')

    def test_every_error_reported_at_once_without_cascades(self):
        text = (spec('order_checkout')
                .replace('amount = total', 'amount = order_id')
                .replace('`subtotal - discount`', '`subtotal - discont`')
                .replace('Retry 1 time', 'Retry 9 times')
                .replace('Save order: order_id, total, charge_id', 'Save order: order_id, total'))
        res = compile_spec(text, intent=INTENTS.get)
        self.assertEqual(sorted(e.rule for e in res.errors),
                         ['entity-fields', 'resolves', 'retry-range', 'types'])

    def test_syntax(self):
        self.assertRule(self.fault('`subtotal - discount`', '`subtotal - `'), 'syntax')
        self.assertRule(self.fault('1. [Has items?] Validate:', '1. [Has items?] Check:'), 'syntax')


if __name__ == '__main__':
    unittest.main()
