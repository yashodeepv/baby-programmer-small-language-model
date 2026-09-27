"""project - A whole system -> a complete project, verified by its own tests.

    common.py              the system model, names, types
    python_project.py      Python package: flows, models, ports, triggers, adapters, tests
    javascript_project.py  Node.js project: the same, as ES modules with node:test

    files = generate(system, 'python')           # {relative path: content}
    report = verify_project(files, 'python')     # runs the generated test suite

A generated project is verified the way a person would check it: by
running its own test suite, in its own language, with its own test runner.
The tests are the simulator's scenarios, so passing them means the project
behaves exactly like the verified flows.
"""

import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from flow.codegen.project import javascript_project, python_project
from flow.codegen.project.common import System

GENERATORS = {'python': python_project.generate, 'javascript': javascript_project.generate}


def generate(system, language):
    return GENERATORS[language](system)


def write(files, root):
    for path, content in files.items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, 'w') as f:
            f.write(content)


def zip_bytes(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for path, content in sorted(files.items()):
            z.writestr(path, content)
    return buf.getvalue()


def verify_project(files, language, timeout=120):
    """{'status': 'verified' | 'failed' | 'not verified', 'passed', 'total', 'detail'}"""
    top = sorted({p.split('/')[0] for p in files})
    if len(top) != 1:
        raise ValueError('a project has exactly one top-level folder')
    with tempfile.TemporaryDirectory() as d:
        write(files, d)
        root = os.path.join(d, top[0])
        if language == 'python':
            env = {**os.environ, 'PYTHONPATH': os.path.join(root, 'src')}
            p = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-t', '.'],
                               cwd=root, env=env, capture_output=True, text=True, timeout=timeout)
            text = p.stderr + p.stdout
            m = re.search(r'Ran (\d+) tests?', text)
            total = int(m.group(1)) if m else 0
            fails = sum(int(x) for x in re.findall(r'(?:failures|errors)=(\d+)', text))
            passed = total - fails if p.returncode == 0 or fails else 0
        else:
            if shutil.which('node') is None:
                return {'status': 'not verified', 'passed': 0, 'total': 0, 'detail': 'Node.js is not installed'}
            tests = sorted(os.path.join('test', f) for f in os.listdir(os.path.join(root, 'test'))
                           if f.endswith('.test.js'))
            p = subprocess.run(['node', '--test', '--test-reporter=tap', *tests], cwd=root,
                               capture_output=True, text=True, timeout=timeout)
            text = p.stdout + p.stderr
            pm, fm = re.search(r'# pass (\d+)', text), re.search(r'# fail (\d+)', text)
            passed, fails = int(pm.group(1)) if pm else 0, int(fm.group(1)) if fm else 0
            total = passed + fails
        ok = p.returncode == 0 and total > 0 and passed == total
        return {'status': 'verified' if ok else 'failed', 'passed': passed, 'total': total,
                'detail': '' if ok else text.strip()[-3000:]}


__all__ = ['System', 'generate', 'verify_project', 'write', 'zip_bytes']
