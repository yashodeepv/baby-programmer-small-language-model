"""python_project.py - A system -> a complete Python project (standard library only).

    <system>/
      pyproject.toml   README.md
      src/<pkg>/  runtime.py models.py ports.py app.py __main__.py
                  flows/  one module per flow, and a registry
                  triggers/  http.py events.py schedule.py
                  adapters/  memory.py  (in-memory store, printing notifier, fake services)
      tests/      one test module per flow, one test per scenario
"""

import json
import re

from flow.codegen.project.common import (camel, collect_types, outline, py_type, snake,
                                         trigger_line)
from flow.codegen.python import RUNTIME, flow_function, _machine, var
from flow.codegen.structure import Unstructured
from flow.ir2.types import BOOL, INT, STR, ListT, RecordT

HELPERS = ['CallFailed', 'Failed', 'call', 'failed', 'ok', '_div', '_mod', '_fmt', '_idx', '_reduce', '_s32', '_M']

EXECUTE = '''

def execute(flow, request, io):
    """Run one flow function and turn failures into an error result."""
    try:
        return flow(request, io)
    except Failed as e:
        return failed(str(e))
    except _Fault as e:
        return failed(f'{io.at}: {e}')
'''

CRON = '''

def cron_matches(cron, when):
    """Does a 5-field cron expression (minute hour day month weekday) match `when`?
    Supports *, */n, a-b, a,b and plain numbers; weekday 0 is Sunday."""
    fields = cron.split()
    if len(fields) != 5:
        raise ValueError(f'expected 5 cron fields, got {cron!r}')
    values = [when.minute, when.hour, when.day, when.month, (when.weekday() + 1) % 7]
    return all(_field_matches(f, v) for f, v in zip(fields, values))


def _field_matches(field, value):
    for part in field.split(','):
        step = 1
        if '/' in part:
            part, step = part.split('/')
            step = int(step)
        if part == '*':
            if value % step == 0:
                return True
            continue
        lo, _, hi = part.partition('-')
        lo, hi = int(lo), int(hi or lo)
        if lo <= value <= hi and (value - lo) % step == 0:
            return True
    return False
'''


def _default(t):
    if t == INT:
        return '0'
    if t == BOOL:
        return 'False'
    if t == STR:
        return "''"
    if isinstance(t, ListT):
        return '[]'
    if isinstance(t, RecordT):
        return '{' + ', '.join(f'{n!r}: {_default(ft)}' for n, ft in t.fields) + '}'
    return 'None'


def _models(system, namer):
    out = ['"""Data shapes: entities, flow inputs and service contracts (plain dicts at runtime)."""',
           '', 'from typing import List, TypedDict', '']
    for name, rec in namer.defs:
        out += ['', f'class {name}(TypedDict):']
        out += [f'    {n}: {py_type(t, namer)}' for n, t in rec.fields] or ['    pass']
    return '\n'.join(out) + '\n'


def _ports(system, namer):
    out = ['"""The outside world, as interfaces. Implement these to connect real systems."""', '',
           'from typing import Protocol', '', 'from .models import *  # noqa: F401,F403',
           'from .runtime import ServiceError', '']
    for s, ops in sorted(system.services.items()):
        out += ['', f'class {camel(s)}(Protocol):', f'    """The {s} service. Raise ServiceError when a call fails."""']
        for op, sig in ops.items():
            params = ', '.join(f'{n}: {py_type(t, namer)}' for n, t in sig.input.fields)
            out.append(f'    def {op}(self{", " if params else ""}{params}) -> {namer.names[sig.output]}: ...')
    out += ['''

class Store(Protocol):
    def __call__(self, entity: str, op: str, data: dict) -> None:
        """op is 'create', 'update' or 'delete'; update and delete match the entity's key field."""


class Notifier(Protocol):
    def __call__(self, channel: str, to: str, message: str) -> None:
        """channel is 'email', 'slack', 'sms' or 'webhook'."""


class Services:
    """Routes the flows' calls to your service objects:

        Services(payments=MyPayments(), inventory=MyInventory())
    """

    def __init__(self, **impls):
        self.impls = impls

    def __call__(self, service, op, args, step=None):
        impl = self.impls.get(service)
        if impl is None:
            raise ServiceError(f'no implementation for service {service}')
        return getattr(impl, op)(**args)
''']
    return '\n'.join(out)


def _memory(system, namer):
    keys = {e: rec.fields[0][0] for e, rec in system.entities.items()}
    out = ['"""In-memory adapters: run and try the flows before real systems are connected."""', '',
           'from ..ports import Services', '', '',
           f'KEYS = {keys!r}', '', '',
           '''class MemoryStore:
    """Keeps records in a dict of lists, one list per entity."""

    def __init__(self):
        self.records = {}

    def __call__(self, entity, op, data):
        rows = self.records.setdefault(entity, [])
        key = KEYS.get(entity)
        if op == 'create':
            rows.append(dict(data))
            return
        matching = [r for r in rows if r.get(key) == data.get(key)]
        for r in matching:
            rows.remove(r)
        if op == 'update':
            rows.append(dict(data))


class PrintNotifier:
    """Prints messages and keeps them in .sent."""

    def __init__(self, quiet=False):
        self.sent, self.quiet = [], quiet

    def __call__(self, channel, to, message):
        self.sent.append((channel, to, message))
        if not self.quiet:
            print(f'[{channel} -> {to}] {message}')
''']
    for s, ops in sorted(system.services.items()):
        out += ['', '', f'class Fake{camel(s)}:', f'    """Always succeeds with placeholder values."""']
        for op, sig in ops.items():
            params = ', '.join(n for n, _ in sig.input.fields)
            out += ['', f'    def {op}(self{", " if params else ""}{params}):',
                    f'        return {_default(sig.output)}']
    fakes = ', '.join(f'{s}=Fake{camel(s)}()' for s in sorted(system.services))
    out += ['', '', 'def fake_services():', f'    return Services({fakes})', '']
    return '\n'.join(out)


def _flow_module(flow, pkg):
    try:
        code = flow_function(flow, var(flow.name))
        form = 'structured'
    except Unstructured:
        code = _machine(flow)
        form = 'state machine'
    used = [h for h in HELPERS if re.search(rf'(?<![\w.]){re.escape(h)}\b', code)]
    head = [f'"""{flow.name}: {flow.intent or "generated flow"}',
            '', f'Trigger: {trigger_line(flow)}   Mode: {flow.mode}   Form: {form}',
            'Generated by baby-programmer from IR2. Regenerate rather than edit by hand."""', '']
    if used:
        head.append(f'from ..runtime import {", ".join(used)}')
    return '\n'.join(head) + '\n\n\n' + code + '\n'


def _triggers_meta(system):
    meta = {}
    for f in system.flows:
        t = f.trigger()
        meta[f.name] = {'source': t.source, 'mode': f.mode, **t.config}
    return meta


def _flows_init(system):
    lines = ['"""Every flow, by name, and what triggers it."""', '']
    for f in system.flows:
        lines.append(f'from .{snake(f.name)} import {var(f.name)}')
    lines += ['', 'FLOWS = {'] + [f'    {f.name!r}: {var(f.name)},' for f in system.flows] + ['}', '']
    lines.append('TRIGGERS = ' + json.dumps(_triggers_meta(system), indent=4))
    return '\n'.join(lines) + '\n'


APP = '''"""Wiring: build an IO from adapters, and run a flow by name."""

from .adapters.memory import MemoryStore, PrintNotifier, fake_services
from .flows import FLOWS
from .runtime import IO, execute


def make_io(services=None, store=None, notify=None):
    """Real systems go here: make_io(services=Services(payments=...), store=..., notify=...)."""
    return IO(services or fake_services(), store or MemoryStore(), notify or PrintNotifier())


def run_flow(name, request, io=None):
    return execute(FLOWS[name], request, io or make_io())
'''

HTTP = '''"""HTTP entry points for the flows with an HTTP trigger (standard library server)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..app import make_io, run_flow
from ..flows import TRIGGERS

ROUTES = {(t['method'], t['path']): name for name, t in TRIGGERS.items() if t['source'] == 'http'}


def make_handler(io_factory=make_io):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self):
            name = ROUTES.get((self.command, self.path.split('?')[0]))
            if name is None:
                return self._reply(404, {'error': 'no flow here'})
            try:
                n = int(self.headers.get('Content-Length', 0))
                request = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                return self._reply(400, {'error': 'the body must be JSON'})
            if TRIGGERS[name]['mode'] == 'async':
                threading.Thread(target=run_flow, args=(name, request, io_factory()), daemon=True).start()
                return self._reply(202, {'accepted': True})
            try:
                result = run_flow(name, request, io_factory())
            except (KeyError, TypeError) as e:
                return self._reply(400, {'error': f'missing or malformed field: {e}'})
            if result['status'] == 'ok':
                return self._reply(200, result['response'] or {})
            return self._reply(422, {'error': result['error'], **(result['response'] or {})})

        do_GET = do_POST = do_PUT = do_DELETE = _handle

    return Handler


def serve(port=8000, io_factory=make_io):
    httpd = ThreadingHTTPServer(('127.0.0.1', port), make_handler(io_factory))
    for (method, path), name in sorted(ROUTES.items()):
        print(f'  {method:6} {path}  ->  {name}')
    print(f'listening on http://127.0.0.1:{port}/')
    httpd.serve_forever()
'''

EVENTS = '''"""Event entry points: publish(event_name, payload) runs the flow listening for it."""

from ..app import run_flow
from ..flows import TRIGGERS

EVENTS = {t['name']: name for name, t in TRIGGERS.items() if t['source'] == 'event'}


def publish(event, payload, io=None):
    name = EVENTS.get(event)
    if name is None:
        raise KeyError(f'no flow listens for event {event!r}')
    return run_flow(name, payload, io)
'''

SCHEDULE = '''"""Scheduled flows: run_due() runs every flow whose cron matches the current minute."""

import datetime
import time

from ..app import run_flow
from ..flows import FLOWS, TRIGGERS
from ..runtime import cron_matches

SCHEDULES = [(t['cron'], name) for name, t in TRIGGERS.items() if t['source'] == 'schedule']


def payload_for(name, when):
    """The payload a scheduled run receives. Edit to suit: by default every str
    field gets today's date and every int field 0."""
    return PAYLOADS[name](when)


PAYLOADS = {PAYLOAD_BUILDERS}


def run_due(when=None, io=None):
    when = when or datetime.datetime.now()
    return {name: run_flow(name, payload_for(name, when), io)
            for cron, name in SCHEDULES if cron_matches(cron, when)}


def loop(io_factory=None):
    last = None
    while True:
        now = datetime.datetime.now().replace(second=0, microsecond=0)
        if now != last:
            last = now
            for name, result in run_due(now, io_factory() if io_factory else None).items():
                print(f'{now:%Y-%m-%d %H:%M} {name}: {result["status"]}')
        time.sleep(5)
'''

MAIN = '''"""python -m {pkg} <command>

    serve [--port 8000]      HTTP triggers, plus the scheduler in the background
    run FLOW 'JSON'          run one flow with a JSON payload
    event NAME 'JSON'        publish an event
    tick                     run the scheduled flows due this minute
"""

import argparse
import json
import sys
import threading

from .app import run_flow
from .flows import FLOWS
from .triggers import events, http, schedule


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m {pkg}', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['serve', 'run', 'event', 'tick'])
    ap.add_argument('name', nargs='?')
    ap.add_argument('payload', nargs='?', default='{{}}')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args(argv)
    if a.command == 'serve':
        if schedule.SCHEDULES:
            threading.Thread(target=schedule.loop, daemon=True).start()
        http.serve(a.port)
    elif a.command == 'run':
        if a.name not in FLOWS:
            sys.exit(f'unknown flow {{a.name!r}} (flows: {{", ".join(FLOWS)}})')
        print(json.dumps(run_flow(a.name, json.loads(a.payload)), indent=2))
    elif a.command == 'event':
        print(json.dumps(events.publish(a.name, json.loads(a.payload)), indent=2))
    else:
        print(json.dumps(schedule.run_due(), indent=2))


if __name__ == '__main__':
    main()
'''

TEST = '''"""{flow}: every generated scenario, replayed. Generated from the reference simulator."""

import json
import unittest

from {pkg}.flows.{module} import {fn}
from {pkg}.runtime import IO, ServiceError, execute

SCENARIOS = json.loads(r\'\'\'{scenarios}\'\'\')


def replay(scenario):
    mocks, effects, attempts = scenario.get('mocks', {{}}), [], {{}}

    def services(service, op, args, step=None):
        n = attempts[step] = attempts.get(step, 0) + 1
        outcomes = mocks[step] if isinstance(mocks[step], list) else [mocks[step]]
        outcome = outcomes[min(n, len(outcomes)) - 1]
        effects.append(f'call {{service}}.{{op}} #{{n}}')
        if 'ok' in outcome:
            return json.loads(json.dumps(outcome['ok']))
        raise ServiceError(outcome['error'])

    def store(entity, op, data):
        effects.append(f'persist {{entity}}.{{op}}')

    def notify(channel, to, message):
        effects.append(f'notify {{channel}} {{to}}: {{message}}')

    result = execute({fn}, json.loads(json.dumps(scenario['input'])), IO(services, store, notify))
    return result, effects


class Test{cls}(unittest.TestCase):
    pass


def _make(name):
    def test(self):
        scenario = SCENARIOS[name]
        result, effects = replay(scenario)
        want = scenario['expect']
        self.assertEqual(result['status'], want['status'])
        self.assertEqual(result['response'], want.get('response'))
        self.assertEqual(effects, want['effects'])
    return test


for _name in SCENARIOS:
    setattr(Test{cls}, f'test_{{_name}}', _make(_name))


if __name__ == '__main__':
    unittest.main()
'''


def _payload_builders(system):
    items = []
    for f in system.flows:
        t = f.trigger()
        if t.source != 'schedule':
            continue
        fields = []
        for n, ft in t.payload.fields:
            v = "when.strftime('%Y-%m-%d')" if ft == STR else _default(ft)
            fields.append(f'{n!r}: {v}')
        items.append(f"\n    {f.name!r}: lambda when: {{{', '.join(fields)}}},")
    return '{' + ''.join(items) + ('\n}' if items else '}')


def _readme(system, pkg, verified=None):
    lines = [f'# {system.name}', '']
    if system.description:
        lines += [system.description, '']
    lines += ['Generated by baby-programmer from a Baby Spec. Every flow was verified against the',
              'reference simulator; `tests/` replays those scenarios, so keep them passing when you',
              'edit.', '', '## Flows', '']
    for f in system.flows:
        lines += [f'### {f.name}', '', f'`{trigger_line(f)}`, {f.mode}.', '']
        if f.intent:
            lines += [f.intent, '']
        lines += outline(f) + ['']
    lines += ['## Run', '', 'From this folder:', '', '```bash',
              f'PYTHONPATH=src python -m unittest discover -s tests', f'PYTHONPATH=src python -m {pkg} serve',
              f"PYTHONPATH=src python -m {pkg} run {system.flows[0].name} '{{...}}'", '```', '',
              '## Connect real systems', '',
              f'Implement the interfaces in `src/{pkg}/ports.py` and pass them to `make_io`:', '', '```python',
              f'from {pkg}.app import make_io, run_flow', f'from {pkg}.ports import Services', '',
              'io = make_io(services=Services(' + ', '.join(f'{s}=My{camel(s)}()' for s in sorted(system.services))
              + '),', '             store=my_store, notify=my_notifier)',
              f'run_flow({system.flows[0].name!r}, payload, io)', '```', '',
              'Parallel steps run one after another, in the order written. Integers are signed 32-bit',
              '(money in cents) and wrap on overflow, exactly as in the spec.', '']
    return '\n'.join(lines)


def generate(system):
    """System -> {relative path: file content}."""
    pkg = snake(system.name)
    namer = collect_types(system)
    root = system.name
    files = {
        f'{root}/pyproject.toml': f'''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "{system.name}"
version = "0.1.0"
description = "{(system.description or 'Generated by baby-programmer').replace('"', "'")[:200]}"
requires-python = ">=3.9"

[tool.setuptools.packages.find]
where = ["src"]
''',
        f'{root}/README.md': _readme(system, pkg),
        f'{root}/src/{pkg}/__init__.py': f'"""{system.name} - generated by baby-programmer."""\n\n__version__ = "0.1.0"\n',
        f'{root}/src/{pkg}/runtime.py': '"""Shared runtime for the generated flows: results, service calls, 32-bit arithmetic."""\n\n'
                                         + RUNTIME.strip() + '\n' + EXECUTE + CRON,
        f'{root}/src/{pkg}/models.py': _models(system, namer),
        f'{root}/src/{pkg}/ports.py': _ports(system, namer),
        f'{root}/src/{pkg}/app.py': APP,
        f'{root}/src/{pkg}/__main__.py': MAIN.replace('{pkg}', pkg).replace('{{', '{').replace('}}', '}'),
        f'{root}/src/{pkg}/flows/__init__.py': _flows_init(system),
        f'{root}/src/{pkg}/triggers/__init__.py': '"""What starts a flow: HTTP requests, events, schedules."""\n',
        f'{root}/src/{pkg}/triggers/http.py': HTTP,
        f'{root}/src/{pkg}/triggers/events.py': EVENTS,
        f'{root}/src/{pkg}/triggers/schedule.py': SCHEDULE.replace('{PAYLOAD_BUILDERS}', _payload_builders(system)),
        f'{root}/src/{pkg}/adapters/__init__.py': '"""Adapters: stand-ins for real systems."""\n',
        f'{root}/src/{pkg}/adapters/memory.py': _memory(system, namer),
        f'{root}/tests/__init__.py': '',
    }
    for f in system.flows:
        mod = snake(f.name)
        files[f'{root}/src/{pkg}/flows/{mod}.py'] = _flow_module(f, pkg)
        scenarios = system.scenarios.get(f.name, {})
        files[f'{root}/tests/test_{mod}.py'] = TEST.format(
            flow=f.name, pkg=pkg, module=mod, fn=var(f.name), cls=camel(f.name),
            scenarios=json.dumps(scenarios, indent=1))
    return files
