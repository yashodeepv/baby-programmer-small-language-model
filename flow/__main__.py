"""python -m flow <command> FILE

    check FILE                   verify the flow; exit 1 on errors
    show  FILE                   print the flow as an indented outline
    run   FILE --scenario NAME   run one stored scenario and show the trace
    run   FILE --all             run every stored scenario; exit 1 on a mismatch
"""

import argparse
import json
import sys

from flow.check import check
from flow.ir2 import graph as g
from flow.ir2.io import FormatError, load
from flow.ir2.sexpr import emit
from flow.sim import SimError, run_scenario, summary


def _load(path):
    try:
        return load(path)
    except (FormatError, json.JSONDecodeError, OSError) as e:
        sys.exit(f'{path}: {e}')


def _text(e):
    return getattr(e, 'text', None) or emit(e)


def _describe(s):
    if isinstance(s, g.Trigger):
        cfg = ' '.join(s.config.values())
        return f'{s.source} {cfg}'
    if isinstance(s, g.Call):
        args = ', '.join(f'{k}={_text(v)}' for k, v in s.args.items())
        retry = f'  retries={s.retries}' if s.retries else ''
        return f'{s.service}.{s.op}({args}){retry}'
    if isinstance(s, g.Compute):
        return '; '.join(f'{k} = {_text(v)}' for k, v in s.assign)
    if isinstance(s, g.Validate):
        return _text(s.rule)
    if isinstance(s, g.Branch):
        return _text(s.cond)
    if isinstance(s, g.Parallel):
        return f'join at {s.join}'
    if isinstance(s, g.Persist):
        return f'{s.op} {s.entity}'
    if isinstance(s, g.Notify):
        return f'{s.channel} to {_text(s.to)}'
    if isinstance(s, g.End):
        resp = ', '.join(s.response or {})
        return s.status + (f' -> {{{resp}}}' if resp else '')
    return ''


def cmd_show(flow):
    print(f'{flow.title or flow.name}  ({flow.mode})')
    seen = set()

    def walk(sid, depth, label):
        s = flow.step(sid)
        pad = '  ' * depth
        tag = f'[{label}] ' if label not in ('next', None) else ''
        if sid in seen:
            print(f'{pad}{tag}-> {sid}')
            return
        seen.add(sid)
        print(f'{pad}{tag}{sid:<16} {s.kind:<9} {_describe(s)}')
        outs = flow.out_edges(sid)
        if len(outs) == 1 and outs[0].on == 'next':
            walk(outs[0].dst, depth, None)
        else:
            for e in outs:
                walk(e.dst, depth + 1, e.on)

    trig = flow.trigger()
    if trig:
        walk(trig.id, 1, None)


def cmd_run(flow, raw, names):
    scenarios = raw.get('scenarios', {})
    failed = 0
    for name in names:
        if name not in scenarios:
            sys.exit(f'no scenario {name!r} (have: {", ".join(scenarios) or "none"})')
        try:
            res, bad = run_scenario(flow, scenarios[name])
        except SimError as e:
            print(f'{name}: cannot run: {e}')
            failed += 1
            continue
        if len(names) == 1:
            for st in res.steps:
                print(f'  {st.id:<16} {st.kind:<9} {json.dumps(st.detail)}')
            print('effects:')
            for e in res.effects:
                print(f'  {summary(e)}')
            print(f'status: {res.status}' + (f'  response: {json.dumps(res.response)}'
                                            if res.response is not None else '')
                  + (f'  error: {res.error}' if res.error else ''))
        print(f'{name}: ' + ('PASS' if not bad else 'FAIL'))
        for b in bad:
            print(f'  {b}')
        failed += bool(bad)
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m flow', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['check', 'show', 'run'])
    ap.add_argument('file')
    ap.add_argument('--scenario')
    ap.add_argument('--all', action='store_true')
    a = ap.parse_args(argv)

    flow, raw = _load(a.file)
    if a.command == 'show':
        cmd_show(flow)
        return 0

    rep = check(flow)
    if a.command == 'check' or not rep.ok:
        print(rep)
        if not rep.ok:
            return 1
        if a.command == 'check':
            return 0

    if a.all:
        names = list(raw.get('scenarios', {}))
    elif a.scenario:
        names = [a.scenario]
    else:
        sys.exit('run needs --scenario NAME or --all')
    return cmd_run(flow, raw, names)


if __name__ == '__main__':
    sys.exit(main())
