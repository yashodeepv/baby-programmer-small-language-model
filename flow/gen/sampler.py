"""sampler.py - Sample IR2 flows from a catalog, one pattern at a time.

A flow is a trigger, a chain of patterns, and a final end. Each pattern is a
familiar piece of business logic:

    validate    check a rule; on failure stop with an error (or fail the flow)
    compute     name 1-3 values -- the step logic the logic model writes
    call        call a catalog service, with retries, optionally an error exit
    branch      if a condition holds do one thing, otherwise another (or nothing)
    parallel    do 2-3 things at once, then join
    persist     create / update / delete a catalog entity
    notify      email / slack / sms

The chain is built left to right. Whatever is left dangling after a pattern
(`pending` edges) flows into the first step of the next, which is how branch
arms merge back.

Scope grows monotonically: branch arms only hold steps that produce nothing
(notify, persist, end), so a merge never loses a value, and a call inside a
parallel branch is visible after the join -- matching the checker's rules.
The checker still runs on every sample; the sampler is meant never to fail it.
"""

import random
from collections import Counter

from flow.check import check
from flow.gen.exprgen import ExprGen, refs_of
from flow.gen.values import random_value
from flow.sim.evaluate import evaluate
from flow.ir2 import graph as g
from flow.ir2.expr import Ref, Str
from flow.ir2.types import BOOL, INT, STR, RecordT

PATTERNS = {
    'validate': 2.0, 'compute': 3.0, 'call': 3.0, 'branch': 3.0,
    'parallel': 1.5, 'persist': 1.5, 'notify': 1.5,
}

NAMES = {
    INT:  ['total', 'fee', 'subtotal', 'discount', 'score', 'net', 'bonus', 'penalty',
           'margin', 'budget', 'limit', 'weight'],
    BOOL: ['eligible', 'is_large', 'needs_review', 'is_late', 'approved', 'urgent'],
    STR:  ['label', 'summary', 'note', 'headline'],
}

N_ENVS = 12        # sample environments used to place thresholds

MESSAGES = ['not allowed', 'invalid request', 'limit exceeded', 'missing data',
            'rejected by policy', 'out of range']


class SampleError(Exception):
    pass


class _Builder:
    def __init__(self, rng, cat, trig, mode, allowed=tuple(PATTERNS), depth=1):
        self.rng, self.cat, self.mode = rng, cat, mode
        self.allowed, self.depth = set(allowed), depth
        self.envs = [{'input': random_value(rng, trig.payload, cat)} for _ in range(N_ENVS)]
        self.gen = ExprGen(rng, cat, self.envs, level=depth)
        self.trig = trig
        self.steps, self.edges, self.pending = [], [], []
        self.scope = {'input': trig.payload}
        self.counts = Counter()
        self.used_names = set()
        self.used_ops = set()
        self.patterns = []
        self.sync_http = mode == 'sync' and trig.source == 'http'

    # -- plumbing ---------------------------------------------------------
    def new_id(self, stem):
        self.counts[stem] += 1
        return f'{stem}{self.counts[stem]}'

    def add(self, step):
        """Append `step` and route every dangling edge into it."""
        self.steps.append(step)
        for src, on in self.pending:
            self.edges.append(g.Edge(src, step.id, on))
        self.pending = []
        return step

    def side(self, step, src, on):
        """Append `step` hanging off `src` by an `on` edge; pending is untouched."""
        self.steps.append(step)
        self.edges.append(g.Edge(src, step.id, on))
        return step

    def fresh_name(self, t):
        pool = [n for n in NAMES[t] if n not in self.used_names]
        name = self.rng.choice(pool or [f'value{len(self.used_names)}'])
        self.used_names.add(name)
        return name

    def sample_output(self, sid, t):
        for env in self.envs:
            env[sid] = random_value(self.rng, t, self.cat)

    # -- single steps -----------------------------------------------------
    def error_end(self, reason):
        resp = {'reason': Str(reason)} if self.sync_http else None
        return g.End(id=self.new_id('fail'), status='error', response=resp)

    def notify_step(self):
        channel = self.rng.choices(['email', 'slack', 'sms'], [3, 3, 1])[0]
        if channel == 'email':
            emails = [p for p in refs_of(self.scope, STR) if p[-1].endswith('email')]
            to = (Ref(self.rng.choice(emails)) if emails and self.rng.random() < 0.8
                  else Str(self.rng.choice(self.cat.recipients['email'])))
        else:
            to = Str(self.rng.choice(self.cat.recipients[channel]))
        return g.Notify(id=self.new_id('notify'), channel=channel, to=to,
                        message=self.gen.message(self.scope))

    def persist_step(self):
        entity = self.rng.choice(sorted(self.cat.entities))
        rec = self.cat.entities[entity]
        op = self.rng.choices(['create', 'update', 'delete'], [15, 4, 1])[0]
        fields = rec.fields[:1] if op == 'delete' else rec.fields
        data = {n: self.gen.expr(self.scope, t, field=n, depth=self.depth) for n, t in fields}
        return g.Persist(id=self.new_id('save'), entity=entity, op=op, data=data)

    def call_step(self, allow_list_args=True):
        ops = [(s, o, sig) for s, os_ in sorted(self.cat.services.items())
               for o, sig in sorted(os_.items())]
        self.rng.shuffle(ops)
        ops.sort(key=lambda x: (x[0], x[1]) in self.used_ops)      # unused first, stable
        for service, op, sig in ops:
            args = {n: self.gen.expr(self.scope, t, field=n, depth=self.depth) for n, t in sig.input.fields}
            if any(v is None for v in args.values()):
                continue
            sid = op if op not in self.counts else self.new_id(op)
            self.counts[op] += 1
            self.used_ops.add((service, op))
            return g.Call(id=sid, service=service, op=op, args=args,
                          retries=self.rng.choices([0, 1, 2, 3], [5, 3, 2, 1])[0]), sig
        return None, None

    # -- patterns ---------------------------------------------------------
    def p_validate(self):
        v = self.add(g.Validate(id=self.new_id('check'), rule=self.gen.bool_(self.scope, self.depth),
                                message=self.rng.choice(MESSAGES)))
        if self.rng.random() < 0.7:
            self.side(self.error_end(v.message), v.id, 'error')
        self.pending = [(v.id, 'next')]

    def p_compute(self):
        sid = self.new_id('calc')
        fields, inner = [], dict(self.scope)
        assign = []
        for env in self.envs:
            env[sid] = {}
        for _ in range(self.rng.choices([1, 2, 3], [4, 3, 1])[0]):
            t = self.rng.choices([INT, BOOL, STR], [6, 3, 1])[0]
            name = self.fresh_name(t)
            e = self.gen.expr(inner, t, depth=self.depth)
            for _ in range(4):              # two names for one value teach nothing
                if all(e != prev for _, prev in assign):
                    break
                e = self.gen.expr(inner, t, depth=self.depth)
            for env in self.envs:
                env[sid][name] = evaluate(e, env)
            assign.append((name, e))
            fields.append((name, t))
            inner[sid] = RecordT(tuple(fields))
        self.add(g.Compute(id=sid, assign=assign))
        self.scope[sid] = RecordT(tuple(fields))
        self.pending = [(sid, 'next')]

    def p_call(self):
        step, sig = self.call_step()
        if step is None:
            return False
        self.add(step)
        if self.rng.random() < 0.5:
            self.side(self.error_end(f'{step.op} failed'), step.id, 'error')
        self.scope[step.id] = sig.output
        self.sample_output(step.id, sig.output)
        self.pending = [(step.id, 'next')]

    def arm(self, kinds, weights):
        kind = self.rng.choices(kinds, weights)[0]
        if kind == 'notify':
            return self.notify_step()
        if kind == 'persist':
            return self.persist_step()
        if kind == 'stop':
            resp = None
            return g.End(id=self.new_id('stop'), status='ok', response=resp)
        if kind == 'fail':
            return self.error_end(self.rng.choice(MESSAGES))
        return None

    def p_branch(self):
        b = self.add(g.Branch(id=self.new_id('if'), cond=self.gen.bool_(self.scope, self.depth)))
        t = self.side(self.arm(['notify', 'persist', 'stop', 'fail'], [4, 3, 1, 1]), b.id, 'true')
        pending = [] if isinstance(t, g.End) else [(t.id, 'next')]
        f = self.arm(['none', 'notify', 'persist'], [5, 2, 1])
        if f is None:
            pending.append((b.id, 'false'))
        else:
            self.side(f, b.id, 'false')
            pending.append((f.id, 'next'))
        self.pending = pending

    def p_parallel(self):
        p = self.add(g.Parallel(id=self.new_id('fork'), join=''))
        join = g.Join(id=self.new_id('join'))
        p.join = join.id
        outputs = {}
        for _ in range(self.rng.choices([2, 3], [3, 1])[0]):
            kind = self.rng.choices(['notify', 'persist', 'call'],
                                    [3, 3, 1 if 'call' in self.allowed else 0])[0]
            s = None
            if kind == 'call':
                s, sig = self.call_step()
                if s is not None:
                    outputs[s.id] = sig.output
                    self.sample_output(s.id, sig.output)
            if s is None:
                s = self.notify_step() if kind != 'persist' else self.persist_step()
            self.side(s, p.id, 'branch')
            self.edges.append(g.Edge(s.id, join.id, 'next'))
        self.steps.append(join)
        self.scope.update(outputs)
        self.pending = [(join.id, 'next')]

    def p_persist(self):
        s = self.add(self.persist_step())
        self.pending = [(s.id, 'next')]

    def p_notify(self):
        s = self.add(self.notify_step())
        self.pending = [(s.id, 'next')]

    def finish(self):
        resp = None
        if self.sync_http:
            # Prefer values the flow worked out over echoing the request.
            made = [p for t in (INT, STR, BOOL) for p in refs_of(self.scope, t) if p[0] != 'input']
            given = [p for t in (INT, STR) for p in refs_of(self.scope, t) if p[0] == 'input']
            picks, keys = [], set()
            for p in self.rng.sample(made, min(len(made), 2)) + self.rng.sample(given, 1):
                if p[-1] not in keys:
                    keys.add(p[-1])
                    picks.append(p)
            resp = {p[-1]: Ref(p) for p in picks[:self.rng.randint(1, 3)]}
        self.add(g.End(id=self.new_id('done'), status='ok', response=resp))


def sample_flow(rng, cat, n_patterns=None, max_steps=16, name='flow',
                patterns=None, require=None, depth=1):
    """One flow from `cat`. Returns (flow, meta).

    n_patterns  how many patterns to chain: an int, a (lo, hi) range, or
                None for 2-6
    patterns    the patterns allowed (default: all)
    require     a pattern that must appear at least once -- a chapter's
                new idea
    depth       expression nesting depth
    """
    allowed = list(patterns or PATTERNS)
    weights = [PATTERNS[k] for k in allowed]
    for _ in range(50):
        trig_spec = rng.choice(cat.triggers)
        mode = ('sync' if rng.random() < 0.7 else 'async') if trig_spec.source == 'http' else 'async'
        b = _Builder(rng, cat, trig_spec, mode, allowed, depth)
        trig = b.add(g.Trigger(id='start', source=trig_spec.source,
                               config=dict(trig_spec.config), payload=trig_spec.payload))
        b.pending = [(trig.id, 'next')]
        if n_patterns is None:
            want = rng.randint(2, 6)
        elif isinstance(n_patterns, tuple):
            want = rng.randint(*n_patterns)
        else:
            want = n_patterns
        forced = rng.randrange(want) if require else -1
        tries = 0
        while len(b.patterns) < want and tries < 100:
            tries += 1
            kind = require if len(b.patterns) == forced else rng.choices(allowed, weights)[0]
            if getattr(b, f'p_{kind}')() is False:
                continue
            b.patterns.append(kind)
        if len(b.patterns) < want:
            continue
        b.finish()
        if len(b.steps) > max_steps:
            continue
        flow = g.Flow(name=name, mode=mode, steps=b.steps, edges=b.edges,
                      services={s: dict(ops) for s, ops in cat.services.items()
                                if any(isinstance(x, g.Call) and x.service == s for x in b.steps)},
                      entities={e: t for e, t in cat.entities.items()
                                if any(isinstance(x, g.Persist) and x.entity == e for x in b.steps)})
        rep = check(flow)
        if not rep.ok:
            raise SampleError(f'sampler produced a flow that does not check:\n{rep}')
        return flow, {'catalog': cat.name, 'patterns': b.patterns, 'n_steps': len(b.steps),
                      'noun': trig_spec.noun}
    raise SampleError(f'could not fit {n_patterns} patterns in {max_steps} steps')


def sample_flows(n, catalogs, seed=0, **kw):
    rng = random.Random(seed)
    names = sorted(catalogs)
    for i in range(n):
        cat = catalogs[rng.choice(names)]
        yield sample_flow(rng, cat, name=f'{cat.name}_{i}', **kw)
