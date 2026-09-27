"""runner.py - Run a flow against mocked services. The IR2 oracle.

Deterministic by construction: parallel branches run one after another in
edge order, and every service call is answered by a mock, keyed by step id.
A mock is either one outcome or a list of outcomes, one per attempt:

    "charge": {"ok": {"charge_id": "ch_1", "status": "paid"}}
    "charge": [{"error": "timeout"}, {"ok": {...}}]      fails once, then succeeds

A call with `retries: n` makes up to n+1 attempts; every attempt is an effect.
A failed call or validate follows its error edge, or fails the flow if it has
none. An expression fault (division by zero) fails the flow.

Mocks and inputs are checked against the declared types, so a scenario cannot
quietly feed a flow data its contract says is impossible.
"""

from flow.check import check
from flow.ir2 import graph as g
from flow.ir2.types import conforms
from flow.sim.evaluate import FlowFault, evaluate
from flow.sim.trace import Effect, Result, Step


class SimError(Exception):
    """The run could not be set up: an invalid flow, a missing or ill-typed mock."""


class _Stop(Exception):
    def __init__(self, result):
        self.result = result


class _Run:
    def __init__(self, flow, mocks):
        self.flow = flow
        self.mocks = mocks
        self.env = {}
        self.res = Result(status='ok')

    def ev(self, e):
        try:
            return evaluate(e, self.env)
        except FlowFault as f:
            self.fail(f'{self.current}: {f}')

    def fail(self, message):
        self.res.status, self.res.error = 'error', message
        raise _Stop(self.res)

    def record(self, step, **detail):
        self.res.steps.append(Step(step.id, step.kind, detail))

    def attempts(self, step):
        m = self.mocks.get(step.id)
        if m is None:
            raise SimError(f'no mock for call step {step.id!r}')
        return m if isinstance(m, list) else [m]

    def call(self, step):
        sig = self.flow.services[step.service][step.op]
        args = {k: self.ev(e) for k, e in step.args.items()}
        outcomes = self.attempts(step)
        for attempt in range(step.retries + 1):
            o = outcomes[min(attempt, len(outcomes) - 1)]
            if not isinstance(o, dict) or len(o) != 1 or next(iter(o)) not in ('ok', 'error'):
                raise SimError(f'mock for {step.id!r} must be {{"ok": ...}} or {{"error": ...}}, got {o!r}')
            self.res.effects.append(Effect('call', step.id, {
                'service': step.service, 'op': step.op, 'args': args, 'attempt': attempt + 1}))
            if 'ok' in o:
                if not conforms(o['ok'], sig.output):
                    raise SimError(f'mock for {step.id!r} returns {o["ok"]!r}, '
                                   f'which is not {sig.output}')
                self.env[step.id] = o['ok']
                self.record(step, args=args, attempts=attempt + 1, output=o['ok'])
                return 'next'
            last = o['error']
        self.record(step, args=args, attempts=step.retries + 1, error=last)
        if self.flow.next(step.id, 'error'):
            return 'error'
        self.fail(f'{step.id}: {step.service}.{step.op} failed: {last}')

    def exec(self, step):
        """Run one step. Returns the label of the edge to follow, or None at an end."""
        self.current = step.id
        if isinstance(step, g.Trigger):
            self.record(step, input=self.env['input'])
            return 'next'
        if isinstance(step, g.Call):
            return self.call(step)
        if isinstance(step, g.Compute):
            out = {}
            self.env[step.id] = out
            for name, e in step.assign:
                out[name] = self.ev(e)
            self.record(step, values=dict(out))
            return 'next'
        if isinstance(step, g.Validate):
            held = self.ev(step.rule)
            self.record(step, held=held)
            if held:
                return 'next'
            if self.flow.next(step.id, 'error'):
                return 'error'
            self.fail(f'{step.id}: {step.message or "validation failed"}')
        if isinstance(step, g.Branch):
            took = bool(self.ev(step.cond))
            self.record(step, took='true' if took else 'false')
            return 'true' if took else 'false'
        if isinstance(step, g.Persist):
            data = {k: self.ev(e) for k, e in step.data.items()}
            self.res.effects.append(Effect('persist', step.id,
                                           {'entity': step.entity, 'op': step.op, 'data': data}))
            self.record(step, data=data)
            return 'next'
        if isinstance(step, g.Notify):
            to, msg = self.ev(step.to), self.ev(step.message)
            self.res.effects.append(Effect('notify', step.id,
                                           {'channel': step.channel, 'to': to, 'message': msg}))
            self.record(step, to=to, message=msg)
            return 'next'
        if isinstance(step, g.Join):
            self.record(step)
            return 'next'
        if isinstance(step, g.End):
            resp = None
            if step.response is not None:
                resp = {k: self.ev(e) for k, e in step.response.items()}
            self.record(step, status=step.status, response=resp)
            self.res.status, self.res.response = step.status, resp
            if step.status == 'error':
                self.res.error = f'{step.id}: {step.title or "ended with error"}'
            return None
        raise SimError(f'cannot run step kind {step.kind!r}')

    def walk(self, sid, stop=None):
        """Follow edges from `sid` until reaching `stop` (not run) or an end."""
        while sid is not None and sid != stop:
            step = self.flow.step(sid)
            if isinstance(step, g.Parallel):
                self.current = step.id
                branches = [e.dst for e in self.flow.out_edges(sid) if e.on == 'branch']
                self.record(step, branches=branches)
                for b in branches:
                    self.res.edges.append((sid, b, 'branch'))
                    self.walk(b, stop=step.join)
                sid = step.join
                continue
            label = self.exec(step)
            if label is None:
                return
            nxt = self.flow.next(sid, label)
            self.res.edges.append((sid, nxt, label))
            sid = nxt


def run(flow, payload, mocks=None):
    """Run `flow` on a trigger payload. Returns a Result.

    Raises SimError if the flow does not pass the checker, or a mock or the
    payload does not match its declared type.
    """
    rep = check(flow)
    if not rep.ok:
        raise SimError('flow does not check:\n' + str(rep))
    trig = flow.trigger()
    if not conforms(payload, trig.payload):
        raise SimError(f'input {payload!r} is not {trig.payload}')
    r = _Run(flow, mocks or {})
    r.env['input'] = payload
    try:
        r.walk(trig.id)
    except _Stop as s:
        return s.result
    return r.res
