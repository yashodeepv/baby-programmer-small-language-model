"""verify.py - Run generated code on scenarios and compare it with the simulator.

The simulator (flow/sim) is the reference. Generated code passes a scenario
only if its status, response, error text and ordered side effects all equal
the simulator's. Mocked services behave the same way in every language: one
outcome per attempt, keyed by step id.

JavaScript runs under Node; if Node isn't installed, that language is
reported as not verified rather than passed.
"""

import copy
import json
import os
import shutil
import subprocess
import tempfile

from flow.codegen.java import class_name, to_java
from flow.codegen.javascript import to_javascript
from flow.codegen.python import to_python
from flow.sim import run, summary


def reference(flow, scenarios):
    out = []
    for sc in scenarios:
        r = run(flow, sc['input'], sc.get('mocks', {}))
        out.append({'status': r.status, 'response': r.response, 'error': r.error,
                    'effects': [summary(e) for e in r.effects]})
    return out


def run_python(code, scenarios):
    ns = {}
    exec(compile(code, '<generated>', 'exec'), ns)
    results = []
    for sc in scenarios:
        mocks, effects, attempts = sc.get('mocks', {}), [], {}

        def services(service, op, args, step):
            n = attempts[step] = attempts.get(step, 0) + 1
            outcomes = mocks[step] if isinstance(mocks[step], list) else [mocks[step]]
            o = outcomes[min(n, len(outcomes)) - 1]
            effects.append(f'call {service}.{op} #{n}')
            if 'ok' in o:
                return copy.deepcopy(o['ok'])
            raise ns['ServiceError'](o['error'])

        def store(entity, op, data):
            effects.append(f'persist {entity}.{op}')

        def notify(channel, to, message):
            effects.append(f'notify {channel} {to}: {message}')

        r = ns['run'](copy.deepcopy(sc['input']), services, store, notify)
        results.append({**r, 'effects': effects})
    return results


HARNESS = r'''
import { run, ServiceError } from './flow.mjs';
import { readFileSync } from 'node:fs';
const scenarios = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const sc of scenarios) {
  const mocks = sc.mocks || {}, effects = [], attempts = {};
  const services = async (service, op, args, step) => {
    const n = attempts[step] = (attempts[step] || 0) + 1;
    const outcomes = Array.isArray(mocks[step]) ? mocks[step] : [mocks[step]];
    const o = outcomes[Math.min(n, outcomes.length) - 1];
    effects.push(`call ${service}.${op} #${n}`);
    if ('ok' in o) return structuredClone(o.ok);
    throw new ServiceError(o.error);
  };
  const store = async (entity, op, data) => { effects.push(`persist ${entity}.${op}`); };
  const notify = async (channel, to, message) => { effects.push(`notify ${channel} ${to}: ${message}`); };
  const r = await run(structuredClone(sc.input), services, store, notify);
  out.push({ ...r, effects });
}
console.log(JSON.stringify(out));
'''


def node_available():
    return shutil.which('node') is not None


def run_javascript(code, scenarios):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, 'flow.mjs'), 'w') as f:
            f.write(code)
        with open(os.path.join(d, 'harness.mjs'), 'w') as f:
            f.write(HARNESS)
        with open(os.path.join(d, 'scenarios.json'), 'w') as f:
            json.dump(scenarios, f)
        p = subprocess.run(['node', 'harness.mjs', 'scenarios.json'], cwd=d,
                           capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip()[-2000:])
        return json.loads(p.stdout)


JAVA_HARNESS = r'''
import java.nio.file.*;
import java.util.*;

public final class Harness {
    @SuppressWarnings("unchecked")
    public static void main(String[] args) throws Exception {
        List<Object> scenarios = (List<Object>) R.parseJson(Files.readString(Path.of(args[0])));
        List<Object> out = new ArrayList<>();
        for (Object o : scenarios) {
            Map<String, Object> sc = (Map<String, Object>) o;
            Map<String, Object> mocks = sc.containsKey("mocks") ? (Map<String, Object>) sc.get("mocks") : new HashMap<>();
            List<Object> effects = new ArrayList<>();
            Map<String, Integer> attempts = new HashMap<>();
            R.Services services = (service, op, a, step) -> {
                int n = attempts.merge(step, 1, Integer::sum);
                Object m = mocks.get(step);
                List<Object> outcomes = m instanceof List ? (List<Object>) m : List.of(m);
                Map<String, Object> outcome = (Map<String, Object>) outcomes.get(Math.min(n, outcomes.size()) - 1);
                effects.add("call " + service + "." + op + " #" + n);
                if (outcome.containsKey("ok")) return (Map<String, Object>) R.copy(outcome.get("ok"));
                throw new R.ServiceError((String) outcome.get("error"));
            };
            R.Store store = (entity, op, data) -> effects.add("persist " + entity + "." + op);
            R.Notifier notifier = (channel, to, message) -> effects.add("notify " + channel + " " + to + ": " + message);
            Map<String, Object> r = FLOWCLASS.run((Map<String, Object>) R.copy(sc.get("input")), services, store, notifier);
            r.put("effects", effects);
            out.add(r);
        }
        System.out.println(R.toJson(out));
    }
}
'''


def java_tools():
    """(javac, java) paths, or None. Checks JAVA_HOME, Homebrew's OpenJDK, then PATH;
    the macOS /usr/bin stubs are skipped because they run only when a JDK is installed."""
    candidates = []
    if os.environ.get('JAVA_HOME'):
        candidates.append(os.path.join(os.environ['JAVA_HOME'], 'bin'))
    candidates += ['/opt/homebrew/opt/openjdk/bin', '/usr/local/opt/openjdk/bin']
    found = shutil.which('javac')
    if found:
        candidates.append(os.path.dirname(found))
    for d in candidates:
        javac, java = os.path.join(d, 'javac'), os.path.join(d, 'java')
        if os.path.exists(javac) and os.path.exists(java):
            try:
                if subprocess.run([javac, '-version'], capture_output=True, timeout=30).returncode == 0:
                    return javac, java
            except (OSError, subprocess.TimeoutExpired):
                continue
    return None


def run_java(code, cls, scenarios):
    tools = java_tools()
    javac, java = tools
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, f'{cls}.java'), 'w') as f:
            f.write(code)
        with open(os.path.join(d, 'Harness.java'), 'w') as f:
            f.write(JAVA_HARNESS.replace('FLOWCLASS', cls))
        with open(os.path.join(d, 'scenarios.json'), 'w') as f:
            json.dump(scenarios, f)
        p = subprocess.run([javac, '--release', '17', '-nowarn', '-d', 'out', f'{cls}.java', 'Harness.java'],
                           cwd=d, capture_output=True, text=True, timeout=180)
        if p.returncode != 0:
            raise RuntimeError('javac: ' + p.stderr.strip()[-3000:])
        p = subprocess.run([java, '-cp', 'out', 'Harness', 'scenarios.json'], cwd=d,
                           capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip()[-3000:])
        return json.loads(p.stdout)


def verify(flow, scenarios, languages=('python', 'javascript', 'java'), style='auto'):
    """{language: {'status': 'verified' | 'mismatch' | 'not verified', 'passed', 'total', 'detail'}}"""
    scenarios = list(scenarios)
    want = reference(flow, scenarios)
    report = {}
    for lang in languages:
        if lang == 'javascript' and not node_available():
            report[lang] = {'status': 'not verified', 'passed': 0, 'total': len(scenarios),
                            'detail': 'Node.js is not installed'}
            continue
        if lang == 'java' and java_tools() is None:
            report[lang] = {'status': 'not verified', 'passed': 0, 'total': len(scenarios),
                            'detail': 'no JDK found (javac)'}
            continue
        try:
            if lang == 'python':
                got = run_python(to_python(flow, style), scenarios)
            elif lang == 'javascript':
                got = run_javascript(to_javascript(flow, style), scenarios)
            else:
                got = run_java(to_java(flow), class_name(flow), scenarios)
        except Exception as e:
            report[lang] = {'status': 'mismatch', 'passed': 0, 'total': len(scenarios), 'detail': str(e)}
            continue
        bad = [(i, w, gt) for i, (w, gt) in enumerate(zip(want, got)) if w != gt]
        report[lang] = {
            'status': 'verified' if not bad else 'mismatch',
            'passed': len(scenarios) - len(bad), 'total': len(scenarios),
            'detail': '' if not bad else f'scenario {bad[0][0]}: expected {bad[0][1]}, got {bad[0][2]}',
        }
    return report
