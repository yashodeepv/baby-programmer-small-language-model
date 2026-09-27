"""javascript_project.py - A system -> a complete Node.js project (no dependencies).

    <system>/
      package.json   README.md
      src/  runtime.js models.js ports.js app.js main.js
            flows/  one module per flow, and index.js
            triggers/  http.js events.js schedule.js
            adapters/  memory.js
      test/ one test file per flow (node:test), one test per scenario
"""

import json
import re

from flow.codegen.javascript import RUNTIME, flow_code, var
from flow.codegen.project.common import (camel, collect_types, js_type, outline, snake,
                                         trigger_line)
from flow.ir2.types import BOOL, INT, STR, ListT, RecordT

HELPERS = ['CallFailed', 'Failed', 'call', 'failed', 'ok', 'div', 'mod', 'fmt', 'idx', 'reduce', 'range', 's32']

EXECUTE = '''

/** Run one flow function and turn failures into an error result. */
export async function execute(flow, request, io) {
  try {
    return await flow(request, io);
  } catch (e) {
    if (e instanceof Failed) return failed(e.message);
    if (e instanceof Fault) return failed(`${io.at}: ${e.message}`);
    throw e;
  }
}

/** Does a 5-field cron expression (minute hour day month weekday) match `when`?
 *  Supports *, *\\/n, a-b, a,b and plain numbers; weekday 0 is Sunday. */
export function cronMatches(cron, when) {
  const fields = cron.trim().split(/\\s+/);
  if (fields.length !== 5) throw new Error(`expected 5 cron fields, got ${cron}`);
  const values = [when.getMinutes(), when.getHours(), when.getDate(), when.getMonth() + 1, when.getDay()];
  return fields.every((f, i) => fieldMatches(f, values[i]));
}

function fieldMatches(field, value) {
  return field.split(',').some((part) => {
    let step = 1;
    if (part.includes('/')) { [part, step] = part.split('/'); step = Number(step); }
    if (part === '*') return value % step === 0;
    const [lo, hi] = part.split('-').map(Number);
    const top = Number.isNaN(hi) || hi === undefined ? lo : hi;
    return value >= lo && value <= top && (value - lo) % step === 0;
  });
}
'''


def _default(t):
    if t == INT:
        return '0'
    if t == BOOL:
        return 'false'
    if t == STR:
        return "''"
    if isinstance(t, ListT):
        return '[]'
    if isinstance(t, RecordT):
        return '{ ' + ', '.join(f'{n}: {_default(ft)}' for n, ft in t.fields) + ' }'
    return 'null'


def _models(namer):
    out = ['// Data shapes: entities, flow inputs and service contracts (plain objects at runtime).', '']
    for name, rec in namer.defs:
        out.append('/**')
        out.append(f' * @typedef {{Object}} {name}')
        out += [f' * @property {{{js_type(t, namer)}}} {n}' for n, t in rec.fields]
        out.append(' */')
        out.append('')
    out.append('export {};')
    return '\n'.join(out) + '\n'


def _ports(system, namer):
    out = ["// The outside world, as interfaces. Implement these to connect real systems.",
           "import { ServiceError } from './runtime.js';", '']
    for s, ops in sorted(system.services.items()):
        out += ['/**', f' * The {s} service. Throw ServiceError when a call fails.', f' * @typedef {{Object}} {camel(s)}']
        for op, sig in ops.items():
            params = ', '.join(f'{n}: {js_type(t, namer)}' for n, t in sig.input.fields)
            out.append(f' * @property {{(args: {{{params}}}) => Promise<{namer.names[sig.output]}>}} {op}')
        out += [' */', '']
    out.append('''/**
 * Routes the flows' calls to your service objects:
 *   new Services({ payments: myPayments, inventory: myInventory })
 * Each method receives one object of named arguments.
 */
export class Services {
  constructor(impls) { this.impls = impls; }

  async call(service, op, args) {
    const impl = this.impls[service];
    if (!impl) throw new ServiceError(`no implementation for service ${service}`);
    return impl[op](args);
  }

  asFunction() { return (service, op, args) => this.call(service, op, args); }
}
''')
    return '\n'.join(out)


def _memory(system):
    keys = {e: rec.fields[0][0] for e, rec in system.entities.items()}
    out = ["// In-memory adapters: run and try the flows before real systems are connected.",
           "import { Services } from '../ports.js';", '',
           f'const KEYS = {json.dumps(keys)};', '',
           '''export class MemoryStore {
  constructor() { this.records = {}; }

  async store(entity, op, data) {
    const rows = (this.records[entity] ||= []);
    const key = KEYS[entity];
    if (op === 'create') { rows.push({ ...data }); return; }
    this.records[entity] = rows.filter((r) => r[key] !== data[key]);
    if (op === 'update') this.records[entity].push({ ...data });
  }

  asFunction() { return (entity, op, data) => this.store(entity, op, data); }
}

export class PrintNotifier {
  constructor(quiet = false) { this.sent = []; this.quiet = quiet; }

  async notify(channel, to, message) {
    this.sent.push([channel, to, message]);
    if (!this.quiet) console.log(`[${channel} -> ${to}] ${message}`);
  }

  asFunction() { return (channel, to, message) => this.notify(channel, to, message); }
}
''']
    for s, ops in sorted(system.services.items()):
        out.append(f'/** Always succeeds with placeholder values. */\nexport const fake{camel(s)} = {{')
        for op, sig in ops.items():
            out.append(f'  async {op}() {{ return {_default(sig.output)}; }},')
        out.append('};\n')
    fakes = ', '.join(f'{s}: fake{camel(s)}' for s in sorted(system.services))
    out.append(f'export const fakeServices = () => new Services({{ {fakes} }});')
    return '\n'.join(out) + '\n'


def _flow_module(flow):
    code = flow_code(flow, var(flow.name), export=True)
    form = 'structured' if 'state-machine form' not in code else 'state machine'
    used = [h for h in HELPERS if re.search(rf'(?<![\w.]){re.escape(h)}\b', code)]
    head = [f'// {flow.name}: {flow.intent or "generated flow"}',
            f'// Trigger: {trigger_line(flow)}   Mode: {flow.mode}   Form: {form}',
            '// Generated by baby-programmer from IR2. Regenerate rather than edit by hand.']
    if used:
        head.append(f"import {{ {', '.join(used)} }} from '../runtime.js';")
    return '\n'.join(head) + '\n\n' + code + '\n'


def _flows_index(system):
    lines = ['// Every flow, by name, and what triggers it.']
    for f in system.flows:
        lines.append(f"import {{ {var(f.name)} }} from './{snake(f.name)}.js';")
    lines += ['', 'export const FLOWS = {'] + [f'  {json.dumps(f.name)}: {var(f.name)},' for f in system.flows] + ['};', '']
    meta = {f.name: {'source': f.trigger().source, 'mode': f.mode, **f.trigger().config} for f in system.flows}
    lines.append(f'export const TRIGGERS = {json.dumps(meta, indent=2)};')
    return '\n'.join(lines) + '\n'


APP = '''// Wiring: build an IO from adapters, and run a flow by name.
import { MemoryStore, PrintNotifier, fakeServices } from './adapters/memory.js';
import { FLOWS } from './flows/index.js';
import { execute } from './runtime.js';

/** Real systems go here: makeIO({ services: new Services({...}).asFunction(), store, notify }). */
export function makeIO({ services, store, notify } = {}) {
  return {
    services: services || fakeServices().asFunction(),
    store: store || new MemoryStore().asFunction(),
    notify: notify || new PrintNotifier().asFunction(),
    at: null,
  };
}

export function runFlow(name, request, io = makeIO()) {
  return execute(FLOWS[name], request, io);
}
'''

HTTP = '''// HTTP entry points for the flows with an HTTP trigger (node:http).
import http from 'node:http';
import { makeIO, runFlow } from '../app.js';
import { TRIGGERS } from '../flows/index.js';

const ROUTES = Object.fromEntries(Object.entries(TRIGGERS)
  .filter(([, t]) => t.source === 'http').map(([name, t]) => [`${t.method} ${t.path}`, name]));

export function serve(port = 8000, ioFactory = makeIO) {
  const server = http.createServer(async (req, res) => {
    const reply = (code, body) => { res.writeHead(code, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(body)); };
    const name = ROUTES[`${req.method} ${req.url.split('?')[0]}`];
    if (!name) return reply(404, { error: 'no flow here' });
    let body = '';
    for await (const chunk of req) body += chunk;
    let request;
    try { request = body ? JSON.parse(body) : {}; } catch { return reply(400, { error: 'the body must be JSON' }); }
    if (TRIGGERS[name].mode === 'async') {
      runFlow(name, request, ioFactory()).catch((e) => console.error(name, e));
      return reply(202, { accepted: true });
    }
    try {
      const result = await runFlow(name, request, ioFactory());
      if (result.status === 'ok') return reply(200, result.response || {});
      return reply(422, { error: result.error, ...(result.response || {}) });
    } catch (e) {
      return reply(400, { error: `missing or malformed field: ${e.message}` });
    }
  });
  server.listen(port, '127.0.0.1', () => {
    for (const [route, name] of Object.entries(ROUTES)) console.log(`  ${route}  ->  ${name}`);
    console.log(`listening on http://127.0.0.1:${port}/`);
  });
  return server;
}
'''

EVENTS = '''// Event entry points: publish(eventName, payload) runs the flow listening for it.
import { runFlow } from '../app.js';
import { TRIGGERS } from '../flows/index.js';

export const EVENTS = Object.fromEntries(Object.entries(TRIGGERS)
  .filter(([, t]) => t.source === 'event').map(([name, t]) => [t.name, name]));

export function publish(event, payload, io) {
  const name = EVENTS[event];
  if (!name) throw new Error(`no flow listens for event ${event}`);
  return runFlow(name, payload, io);
}
'''

SCHEDULE = '''// Scheduled flows: runDue() runs every flow whose cron matches the current minute.
import { runFlow } from '../app.js';
import { TRIGGERS } from '../flows/index.js';
import { cronMatches } from '../runtime.js';

export const SCHEDULES = Object.entries(TRIGGERS)
  .filter(([, t]) => t.source === 'schedule').map(([name, t]) => [t.cron, name]);

const today = (when) => when.toISOString().slice(0, 10);

/** The payload a scheduled run receives. Edit to suit. */
export const PAYLOADS = {PAYLOAD_BUILDERS};

export async function runDue(when = new Date(), io) {
  const out = {};
  for (const [cron, name] of SCHEDULES) {
    if (cronMatches(cron, when)) out[name] = await runFlow(name, PAYLOADS[name](when), io);
  }
  return out;
}

export function loop(ioFactory) {
  let last = null;
  return setInterval(async () => {
    const now = new Date(); now.setSeconds(0, 0);
    if (last && now.getTime() === last) return;
    last = now.getTime();
    const results = await runDue(now, ioFactory ? ioFactory() : undefined);
    for (const [name, r] of Object.entries(results)) console.log(`${now.toISOString()} ${name}: ${r.status}`);
  }, 5000);
}
'''

MAIN = '''// node src/main.js <command>
//   serve [port]          HTTP triggers, plus the scheduler
//   run FLOW 'JSON'       run one flow with a JSON payload
//   event NAME 'JSON'     publish an event
//   tick                  run the scheduled flows due this minute
import { runFlow } from './app.js';
import { FLOWS } from './flows/index.js';
import { publish } from './triggers/events.js';
import { serve } from './triggers/http.js';
import { SCHEDULES, loop, runDue } from './triggers/schedule.js';

const [command, name, payload = '{}'] = process.argv.slice(2);
if (command === 'serve') {
  if (SCHEDULES.length) loop();
  serve(Number(name) || 8000);
} else if (command === 'run') {
  if (!FLOWS[name]) { console.error(`unknown flow ${name} (flows: ${Object.keys(FLOWS).join(', ')})`); process.exit(1); }
  console.log(JSON.stringify(await runFlow(name, JSON.parse(payload)), null, 2));
} else if (command === 'event') {
  console.log(JSON.stringify(await publish(name, JSON.parse(payload)), null, 2));
} else if (command === 'tick') {
  console.log(JSON.stringify(await runDue(), null, 2));
} else {
  console.error('usage: node src/main.js serve [port] | run FLOW JSON | event NAME JSON | tick');
  process.exit(1);
}
'''

TEST = '''// {flow}: every generated scenario, replayed. Generated from the reference simulator.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { {fn} } from '../src/flows/{module}.js';
import { ServiceError, execute } from '../src/runtime.js';

const SCENARIOS = {scenarios};

async function replay(scenario) {
  const mocks = scenario.mocks || {}, effects = [], attempts = {};
  const io = {
    at: null,
    services: async (service, op, args, step) => {
      const n = attempts[step] = (attempts[step] || 0) + 1;
      const outcomes = Array.isArray(mocks[step]) ? mocks[step] : [mocks[step]];
      const outcome = outcomes[Math.min(n, outcomes.length) - 1];
      effects.push(`call ${service}.${op} #${n}`);
      if ('ok' in outcome) return structuredClone(outcome.ok);
      throw new ServiceError(outcome.error);
    },
    store: async (entity, op) => { effects.push(`persist ${entity}.${op}`); },
    notify: async (channel, to, message) => { effects.push(`notify ${channel} ${to}: ${message}`); },
  };
  const result = await execute({fn}, structuredClone(scenario.input), io);
  return { result, effects };
}

for (const [name, scenario] of Object.entries(SCENARIOS)) {
  test(name, async () => {
    const { result, effects } = await replay(scenario);
    assert.equal(result.status, scenario.expect.status);
    assert.deepEqual(result.response, scenario.expect.response ?? null);
    assert.deepEqual(effects, scenario.expect.effects);
  });
}
'''


def _payload_builders(system):
    items = []
    for f in system.flows:
        t = f.trigger()
        if t.source != 'schedule':
            continue
        fields = ', '.join(f'{n}: {"today(when)" if ft == STR else _default(ft)}' for n, ft in t.payload.fields)
        items.append(f'\n  {json.dumps(f.name)}: (when) => ({{ {fields} }}),')
    return '{' + ''.join(items) + ('\n}' if items else '}')


def _readme(system):
    lines = [f'# {system.name}', '']
    if system.description:
        lines += [system.description, '']
    lines += ['Generated by baby-programmer from a Baby Spec. Every flow was verified against the',
              'reference simulator; `test/` replays those scenarios, so keep them passing when you edit.',
              'No dependencies: Node.js 18 or newer.', '', '## Flows', '']
    for f in system.flows:
        lines += [f'### {f.name}', '', f'`{trigger_line(f)}`, {f.mode}.', '']
        if f.intent:
            lines += [f.intent, '']
        lines += outline(f) + ['']
    first = system.flows[0].name
    lines += ['## Run', '', '```bash', 'npm test', 'npm start                      # serve', f"node src/main.js run {first} '{{...}}'",
              '```', '', '## Connect real systems', '',
              'Implement the service interfaces documented in `src/ports.js` and pass them to `makeIO`:', '',
              '```js', "import { makeIO, runFlow } from './src/app.js';", "import { Services } from './src/ports.js';", '',
              'const io = makeIO({ services: new Services({ '
              + ', '.join(f'{s}: my{camel(s)}' for s in sorted(system.services)) + ' }).asFunction(),',
              '                    store: myStore, notify: myNotifier });', f"await runFlow('{first}', payload, io);",
              '```', '', 'Parallel steps run one after another, in the order written. Integers are signed 32-bit',
              '(money in cents) and wrap on overflow, exactly as in the spec.', '']
    return '\n'.join(lines)


def generate(system):
    root = system.name
    namer = collect_types(system)
    files = {
        f'{root}/package.json': json.dumps({
            'name': system.name, 'version': '0.1.0', 'private': True, 'type': 'module',
            'description': (system.description or 'Generated by baby-programmer')[:200],
            'engines': {'node': '>=18'},
            'scripts': {'start': 'node src/main.js serve', 'test': 'node --test test/*.test.js'},
        }, indent=2) + '\n',
        f'{root}/README.md': _readme(system),
        f'{root}/src/runtime.js': '// Shared runtime for the generated flows: results, service calls, 32-bit arithmetic.\n'
                                  + RUNTIME.strip() + '\n' + EXECUTE,
        f'{root}/src/models.js': _models(namer),
        f'{root}/src/ports.js': _ports(system, namer),
        f'{root}/src/app.js': APP,
        f'{root}/src/main.js': MAIN,
        f'{root}/src/flows/index.js': _flows_index(system),
        f'{root}/src/triggers/http.js': HTTP,
        f'{root}/src/triggers/events.js': EVENTS,
        f'{root}/src/triggers/schedule.js': SCHEDULE.replace('{PAYLOAD_BUILDERS}', _payload_builders(system)),
        f'{root}/src/adapters/memory.js': _memory(system),
    }
    for f in system.flows:
        mod = snake(f.name)
        files[f'{root}/src/flows/{mod}.js'] = _flow_module(f)
        files[f'{root}/test/{mod}.test.js'] = (TEST.replace('{flow}', f.name).replace('{fn}', var(f.name))
                                               .replace('{module}', mod)
                                               .replace('{scenarios}', json.dumps(system.scenarios.get(f.name, {}), indent=1)))
    return files
