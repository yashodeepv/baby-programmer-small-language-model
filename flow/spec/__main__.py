"""python3.11 -m flow.spec <command> FILE

    check FILE              validate a spec; exit 1 on errors. Lists the intent
                            (prose) values the logic model will have to program
    show FILE               print each compiled flow as an outline
    compile FILE --out DIR  write each flow as IR2 JSON, DIR/<flow>.json
"""

import argparse
import json
import os
import sys

from flow.ir2.io import dump_flow
from flow.spec.compile import compile_spec


def _report(res, path):
    for e in sorted(res.errors, key=lambda e: e.line):
        print(f'{path}:{e.line}: error [{e.rule}] {e.message}')
    for w in sorted(res.warnings, key=lambda e: e.line):
        print(f'{path}:{w.line}: warning [{w.rule}] {w.message}')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m flow.spec', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['check', 'show', 'compile'])
    ap.add_argument('file')
    ap.add_argument('--out', default='build')
    a = ap.parse_args(argv)

    try:
        with open(a.file) as f:
            text = f.read()
    except OSError as e:
        sys.exit(str(e))
    res = compile_spec(text)
    _report(res, a.file)
    if res.errors:
        print(f'{len(res.errors)} error(s)')
        return 1

    if a.command == 'check':
        steps = sum(len(f.steps) for f in res.flows)
        print(f'ok: system {res.system}, {len(res.flows)} flow(s), {steps} steps')
        if res.intents:
            print(f'{len(res.intents)} intent value(s) for the logic model:')
            for i in res.intents:
                print(f'  line {i.line}  {i.flow}.{i.step}.{i.field}: {i.text}')
        return 0
    if a.command == 'show':
        from flow.__main__ import cmd_show
        for flow in res.flows:
            cmd_show(flow)
            print()
        return 0
    os.makedirs(a.out, exist_ok=True)
    for flow in res.flows:
        path = os.path.join(a.out, f'{flow.name}.json')
        with open(path, 'w') as f:
            json.dump(dump_flow(flow), f, indent=2)
        print(f'wrote {path}')
    if res.intents:
        print(f'note: {len(res.intents)} intent value(s) are placeholders until the logic model fills them')
    return 0


if __name__ == '__main__':
    sys.exit(main())
