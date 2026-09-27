"""The verifier: examples pass, and each single fault is reported with its code."""

import unittest

from flow.check import check
from flow.ir2.io import FormatError, load_flow
from flow.tests.helpers import EXAMPLES, mutated, raw, step


def codes(flow):
    return check(flow).codes()


class Examples(unittest.TestCase):
    def test_examples_pass(self):
        import json
        for path in EXAMPLES:
            with self.subTest(path=path):
                with open(path) as f:
                    rep = check(load_flow(json.load(f)))
                self.assertTrue(rep.ok, str(rep))


class Structure(unittest.TestCase):
    def test_missing_required_edge(self):
        f = mutated(lambda r: r['edges'].remove({'from': 'big_order', 'to': 'fulfil', 'on': 'false'}))
        self.assertIn('missing-edge', codes(f))

    def test_edge_label_not_allowed(self):
        f = mutated(lambda r: r['edges'].append({'from': 'pricing', 'to': 'reject', 'on': 'error'}))
        self.assertIn('bad-edge', codes(f))

    def test_unknown_step_in_edge(self):
        f = mutated(lambda r: r['edges'].append({'from': 'reply', 'to': 'nowhere'}))
        self.assertIn('unknown-step', codes(f))

    def test_cycle(self):
        def fn(r):
            r['edges'].remove({'from': 'done', 'to': 'reply'})
            r['edges'].append({'from': 'done', 'to': 'pricing'})
        self.assertIn('cycle', codes(mutated(fn)))

    def test_unreachable(self):
        def fn(r):
            r['steps'].append({'id': 'stray', 'kind': 'end', 'status': 'ok'})
        self.assertIn('unreachable', codes(mutated(fn)))

    def test_two_triggers(self):
        def fn(r):
            t = dict(step(r, 'start'), id='start2')
            r['steps'].append(t)
            r['edges'].append({'from': 'start2', 'to': 'has_items'})
        self.assertIn('trigger-count', codes(mutated(fn)))

    def test_duplicate_id(self):
        f = mutated(lambda r: r['steps'].append({'id': 'reply', 'kind': 'join'}))
        self.assertIn('duplicate-id', codes(f))

    def test_reserved_id(self):
        def fn(r):
            step(r, 'done')['id'] = 'a1'
            step(r, 'fulfil')['join'] = 'a1'
            for e in r['edges']:
                e['from'] = 'a1' if e['from'] == 'done' else e['from']
                e['to'] = 'a1' if e['to'] == 'done' else e['to']
        self.assertIn('bad-id', codes(mutated(fn)))

    def test_branch_escapes_parallel(self):
        def fn(r):
            r['edges'].remove({'from': 'save_order', 'to': 'done'})
            r['edges'].append({'from': 'save_order', 'to': 'reject'})
        self.assertIn('branch-escapes', codes(mutated(fn)))

    def test_bad_settings(self):
        f = mutated(lambda r: step(r, 'confirm_email').update(channel='pigeon'))
        self.assertIn('bad-setting', codes(f))
        f = mutated(lambda r: step(r, 'start')['config'].pop('path'))
        self.assertIn('missing-setting', codes(f))


class Dataflow(unittest.TestCase):
    def test_unknown_ref_names_what_is_available(self):
        f = mutated(lambda r: step(r, 'big_order').update(cond='(ge pricng.total 100000)'))
        rep = check(f)
        self.assertEqual(rep.codes(), ['unknown-ref'])
        self.assertIn('pricing', rep.errors[0].message)
        self.assertEqual((rep.errors[0].step, rep.errors[0].field), ('big_order', 'cond'))

    def test_unknown_field_reported_once_at_its_source(self):
        # pricing.subtotal is read by five later steps; only the source is reported.
        f = mutated(lambda r: step(r, 'pricing')['assign'].update(
            subtotal='(sum (pluck input.items price))'))
        rep = check(f)
        self.assertEqual(rep.codes(), ['unknown-field'])
        self.assertEqual((rep.errors[0].step, rep.errors[0].field), ('pricing', 'assign.subtotal'))

    def test_unknown_op_not_cascaded(self):
        f = mutated(lambda r: step(r, 'charge').update(op='refund'))
        self.assertEqual(codes(f), ['unknown-op'])

    def test_type_mismatch(self):
        f = mutated(lambda r: step(r, 'big_order').update(cond='pricing.total'))
        self.assertIn('type-mismatch', codes(f))
        f = mutated(lambda r: step(r, 'charge')['args'].update(amount='input.order_id'))
        self.assertIn('type-mismatch', codes(f))

    def test_value_from_one_branch_is_not_available_after_merge(self):
        # alert_sales only runs on the true side, so after the paths merge at
        # `fulfil`, a value it produced cannot be relied on.
        def fn(r):
            s = step(r, 'alert_sales')
            s.clear()
            s.update(id='alert_sales', kind='compute', assign={'flag': 'true'})
            step(r, 'reply')['response']['flag'] = 'alert_sales.flag'
        self.assertIn('unknown-ref', codes(mutated(fn)))

    def test_call_output_not_available_on_error_edge(self):
        f = mutated(lambda r: step(r, 'payment_failed')['response'].update(ch='charge.charge_id'))
        self.assertIn('unknown-ref', codes(f))

    def test_parallel_outputs_combine_at_join(self):
        def fn(r):
            s = step(r, 'save_order')
            s.clear()
            s.update(id='save_order', kind='compute', assign={'saved': 'true'})
            step(r, 'reply')['response']['saved'] = 'save_order.saved'
        self.assertTrue(check(mutated(fn)).ok)

    def test_call_contract(self):
        f = mutated(lambda r: step(r, 'charge')['args'].pop('amount'))
        self.assertIn('missing-field', codes(f))
        f = mutated(lambda r: step(r, 'charge').update(op='refund'))
        self.assertIn('unknown-op', codes(f))
        f = mutated(lambda r: step(r, 'charge').update(service='bank'))
        self.assertIn('unknown-service', codes(f))

    def test_persist_contract(self):
        f = mutated(lambda r: step(r, 'save_order')['data'].update(color='"red"'))
        self.assertIn('unknown-field', codes(f))

    def test_register_leaf_rejected(self):
        f = mutated(lambda r: step(r, 'big_order').update(cond='(ge a0 100000)'))
        self.assertIn('register-leaf', codes(f))

    def test_parse_error_is_reported_not_raised(self):
        f = mutated(lambda r: step(r, 'big_order').update(cond='(ge pricing.total'))
        self.assertIn('parse-error', codes(f))

    def test_async_response_warns(self):
        f = mutated(lambda r: step(r, 'finished').update(response={'n': 'summarize.count'}),
                    'overdue_reminders')
        rep = check(f)
        self.assertTrue(rep.ok)
        self.assertIn('response-ignored', rep.codes())


class Format(unittest.TestCase):
    def test_unknown_kind(self):
        r = raw()
        r['steps'][0]['kind'] = 'teleport'
        with self.assertRaises(FormatError):
            load_flow(r)

    def test_missing_field(self):
        r = raw()
        del step(r, 'charge')['service']
        with self.assertRaises(FormatError):
            load_flow(r)


if __name__ == '__main__':
    unittest.main()
