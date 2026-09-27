"""format.py - Run each language's standard formatter over a generated project.

    python      ruff format          (black-compatible)
    javascript  prettier
    java        google-java-format

Formatting happens before verification, so the files a user downloads are
the files that were tested. A formatter that isn't installed leaves the
project as generated and says so; it never blocks a build.
"""

import os
import shutil
import subprocess
import tempfile

from flow.codegen.project import write

TOOLS = {
    'python': 'ruff',
    'javascript': 'prettier',
    'java': 'google-java-format',
}
EXTENSIONS = {'python': ('.py',), 'javascript': ('.js', '.mjs', '.json'), 'java': ('.java',)}


def _command(language, root, paths):
    if language == 'python':
        return ['ruff', 'format', '--quiet', '--line-length', '100', '.']
    if language == 'javascript':
        return ['prettier', '--write', '--log-level', 'warn', '--print-width', '100', '--single-quote', *paths]
    return ['google-java-format', '--replace', *paths]


def format_project(files, language):
    """Returns (files, {'status': 'formatted' | 'not formatted', 'tool', 'detail'})."""
    tool = TOOLS[language]
    if shutil.which(tool) is None:
        return files, {'status': 'not formatted', 'tool': tool, 'detail': f'{tool} is not installed'}
    targets = [p for p in files if p.endswith(EXTENSIONS[language])]
    if not targets:
        return files, {'status': 'formatted', 'tool': tool, 'detail': ''}
    with tempfile.TemporaryDirectory() as d:
        write(files, d)
        p = subprocess.run(_command(language, d, targets), cwd=d, capture_output=True, text=True, timeout=180)
        if p.returncode != 0:
            return files, {'status': 'not formatted', 'tool': tool,
                           'detail': (p.stderr or p.stdout).strip()[-2000:]}
        out = {}
        for path in files:
            with open(os.path.join(d, path)) as f:
                out[path] = f.read()
    return out, {'status': 'formatted', 'tool': tool, 'detail': ''}
