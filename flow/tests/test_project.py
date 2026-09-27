"""Generated projects: complete layout, and their own test suites pass."""

import random
import unittest

from flow.app.pipeline import FlowHints
from flow.app.prompt import example_spec
from flow.codegen.project import System, generate, verify_project, zip_bytes
from flow.codegen.verify import node_available
from flow.gen.coverage import synthesize
from flow.spec import compile_spec


def system_from(text):
    res = compile_spec(text)
    assert res.ok, res.errors
    rng = random.Random(0)
    return System(res.system, res.description, res.flows,
                  {f.name: synthesize(rng, f, FlowHints(f))[0] for f in res.flows})


def checkout():
    with open('flow/spec/examples/order_checkout.spec.md') as f:
        return f.read().replace('the sum of the item amounts', '`sum(items.amount)`')


class Projects(unittest.TestCase):
    def test_python_layout(self):
        files = generate(system_from(checkout()), 'python')
        for path in ['pyproject.toml', 'README.md', 'src/shop_orders/runtime.py', 'src/shop_orders/models.py',
                     'src/shop_orders/ports.py', 'src/shop_orders/flows/order_checkout.py',
                     'src/shop_orders/triggers/http.py', 'src/shop_orders/adapters/memory.py',
                     'tests/test_order_checkout.py']:
            self.assertIn(f'shop-orders/{path}', files)
        flow = files['shop-orders/src/shop_orders/flows/order_checkout.py']
        self.assertIn('def order_checkout(request, io):', flow)
        self.assertIn('Form: structured', flow)
        self.assertIn('class Payments(Protocol):', files['shop-orders/src/shop_orders/ports.py'])

    def test_javascript_layout(self):
        files = generate(system_from(checkout()), 'javascript')
        for path in ['package.json', 'README.md', 'src/runtime.js', 'src/flows/order_checkout.js',
                     'src/triggers/http.js', 'test/order_checkout.test.js']:
            self.assertIn(f'shop-orders/{path}', files)
        self.assertIn('export async function order_checkout(request, io)',
                      files['shop-orders/src/flows/order_checkout.js'])

    def test_generated_test_suites_pass(self):
        for text in (checkout(), example_spec()):
            system = system_from(text)
            for lang in ('python', 'javascript'):
                if lang == 'javascript' and not node_available():
                    continue
                with self.subTest(system=system.name, lang=lang):
                    rep = verify_project(generate(system, lang), lang)
                    self.assertEqual(rep['status'], 'verified', rep['detail'])
                    self.assertGreater(rep['total'], 2)

    def test_a_system_with_several_flows_is_one_project(self):
        extra = ('\n## Flow: cancel_leave\nTrigger: event leave_cancelled with employee_id str, '
                 'manager_email str, days int\nMode: async\nSteps:\n'
                 '1. [Remove the request] Delete leave_request where employee_id.\n'
                 '2. [Tell the manager] Email manager_email: "{employee_id} cancelled {days} days".\n')
        system = system_from(example_spec() + extra)
        for lang in ('python', 'javascript'):
            if lang == 'javascript' and not node_available():
                continue
            files = generate(system, lang)
            with self.subTest(lang=lang):
                self.assertEqual(len([p for p in files if '/flows/' in p and ('cancel' in p or 'request_leave' in p)]), 2)
                rep = verify_project(files, lang)
                self.assertEqual(rep['status'], 'verified', rep['detail'])

    def test_generated_tests_catch_a_bug(self):
        files = generate(system_from(checkout()), 'python')
        path = 'shop-orders/src/shop_orders/flows/order_checkout.py'
        files[path] = files[path].replace("pricing['subtotal'] - pricing['discount']",
                                          "pricing['subtotal'] + pricing['discount']")
        self.assertEqual(verify_project(files, 'python')['status'], 'failed')

    def test_zip(self):
        import io
        import zipfile
        files = generate(system_from(checkout()), 'python')
        with zipfile.ZipFile(io.BytesIO(zip_bytes(files))) as z:
            self.assertEqual(sorted(z.namelist()), sorted(files))


if __name__ == '__main__':
    unittest.main()
