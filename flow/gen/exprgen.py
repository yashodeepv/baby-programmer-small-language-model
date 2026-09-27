"""exprgen.py - Sample well-typed IR2 expressions from the values in scope.

Type-directed: the caller asks for a type (int, bool, str, or an exact list
or record type), and only productions that yield that type from the names
actually visible are considered. So every sample type-checks by construction;
the checker stays in the pipeline as a backstop, not a filter.

Business-shaped, not uniform: comparisons set a field against a threshold
drawn from that field's catalog range; string tests use the field's own value
pool; divisions are by a non-zero constant, so no sample can fault.

`Len` is never produced: `(count xs)` already says the same thing, and two
targets for one English sentence would make the logic model's job ambiguous.
"""

from flow.ir2.expr import (All, Any, ArrLoop, Bin, Bool, Cmp, Const, Fmt, Not,
                           Pluck, Ref, Sel, Str)
from flow.ir2.types import BOOL, INT, STR, ListT, RecordT
from flow.sim.evaluate import evaluate

DEFAULT_POOL = ['n/a', 'unknown']


# --------------------------------------------------------------------------
# What is in scope
# --------------------------------------------------------------------------

def paths(scope):
    """Every (path, type) reachable through records, not into lists."""
    out = []

    def walk(path, t):
        out.append((path, t))
        if isinstance(t, RecordT):
            for n, ft in t.fields:
                walk(path + (n,), ft)

    for name, t in scope.items():
        walk((name,), t)
    return out


def refs_of(scope, t):
    return [p for p, pt in paths(scope) if pt == t and len(p) > 1]


def record_lists(scope):
    """(path, element record) for every list of records in scope."""
    return [(p, t.elem) for p, t in paths(scope)
            if isinstance(t, ListT) and isinstance(t.elem, RecordT)]


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

def round_number(rng, lo, hi):
    """A value in [lo, hi] a person would write: 30, 250, 100000 -- not 97413."""
    v = rng.randint(lo, hi)
    if v < 10:
        return v
    mag = 10 ** (len(str(v)) - 1)
    step = mag if rng.random() < 0.6 else max(1, mag // 2)
    return max(lo, min(hi, (v // step) * step))


def field_of(expr):
    """The field an int expression is 'about', to pick a realistic threshold."""
    if isinstance(expr, Ref):
        return expr.path[-1]
    if isinstance(expr, ArrLoop) and isinstance(expr.arr, Pluck):
        return expr.arr.field
    return None


class ExprGen:
    """`envs` are sample runtime environments kept in step with the scope by
    the sampler. Thresholds are placed inside the range an expression actually
    takes over them, so both sides of a comparison happen in practice."""

    def __init__(self, rng, catalog, envs=None, level=1):
        self.rng = rng
        self.cat = catalog
        self.envs = envs if envs is not None else []
        # 0: plain fields and constants only -- no reductions over lists,
        # no filters. The first chapters of the book stay first-chapter simple.
        self.level = level

    def pick(self, options):
        """options: [(weight, thunk)] with thunk possibly None (unavailable)."""
        live = [(w, f) for w, f in options if f is not None and w > 0]
        total = sum(w for w, _ in live)
        r = self.rng.random() * total
        for w, f in live:
            r -= w
            if r <= 0:
                return f()
        return live[-1][1]()

    def threshold(self, expr, op='ge'):
        """A constant that splits the values `expr` takes, for `expr <op> k`.

        x > k and x <= k both need k below the maximum; x >= k and x < k need
        it above the minimum. Otherwise one side of the test can never happen.
        """
        seen = []
        for env in self.envs:
            try:
                seen.append(evaluate(expr, env))
            except Exception:
                pass
        if seen:
            lo, hi = min(seen), max(seen)
            if lo < hi:
                lo, hi = (lo, hi - 1) if op in ('gt', 'le') else (lo + 1, hi)
                return Const(round_number(self.rng, lo, hi))
            return Const(lo)
        name = field_of(expr)
        lo, hi = self.cat.int_range(name) if name else (0, 1000)
        if isinstance(expr, ArrLoop) and expr.op == 'count':
            lo, hi = 0, 5
        return Const(round_number(self.rng, lo, hi))

    # ------------------------------------------------------------------ int
    def reduction(self, scope):
        if self.level < 1:
            return None
        lists = [(p, rec, n) for p, rec in record_lists(scope)
                 for n, ft in rec.fields if ft == INT]
        if not lists:
            return None

        def make():
            p, rec, fname = self.rng.choice(lists)
            op = self.rng.choice(['sum', 'sum', 'count', 'max', 'min'])
            if op == 'count' and self.rng.random() < 0.4:
                return ArrLoop('count', Ref(p))
            src = Pluck(Ref(p), fname)
            pred = None
            if self.rng.random() < 0.35:
                lo, hi = self.cat.int_range(fname)
                kind = self.rng.choice(['gt', 'lt', 'even'])
                pred = (kind, round_number(self.rng, lo, hi) if kind != 'even' else 0)
            mapf, k = None, 0
            if op == 'sum' and self.rng.random() < 0.1:
                mapf, k = 'double', 0
            return ArrLoop(op, src, mapf=mapf, pred=pred, k=k)
        return make

    def int_leaf(self, scope, allow_const=True):
        ints = refs_of(scope, INT)
        return self.pick([
            (5, (lambda: Ref(self.rng.choice(ints))) if ints else None),
            (3, self.reduction(scope)),
            (1 if allow_const and not ints else 0, lambda: Const(round_number(self.rng, 0, 1000))),
        ])

    def has_int_source(self, scope):
        return bool(refs_of(scope, INT)) or self.reduction(scope) is not None

    def int_(self, scope, depth):
        if depth <= 0:
            return self.int_leaf(scope)
        d = depth - 1
        return self.pick([
            (4, lambda: self.int_leaf(scope)),
            (2, lambda: self.bin(self.rng.choice(['add', 'sub']), scope, d)),
            (1, lambda: Bin('mul', self.int_leaf(scope), Const(self.rng.randint(2, 12)))),
            (1, lambda: Bin('div', self.int_leaf(scope), Const(self.rng.choice([2, 4, 10, 100])))),
            (1, lambda: Bin('add', self.int_leaf(scope), Const(round_number(self.rng, 1, 500)))),
            (1, lambda: self.sel(scope, d)),
        ])

    def bin(self, op, scope, d):
        a = self.int_(scope, d)
        for _ in range(4):
            b = self.int_(scope, d)
            if b != a:
                return Bin(op, a, b)
        return Bin(op, a, Const(round_number(self.rng, 1, 500)))

    def sel(self, scope, d):
        a = self.int_(scope, d)
        for _ in range(4):
            b = self.int_(scope, d)
            if b != a:
                break
        else:
            b = Const(0) if a != Const(0) else Const(1)
        return Sel(self.bool_(scope, d), a, b)

    # ----------------------------------------------------------------- bool
    def compare(self, scope):
        x = self.int_leaf(scope, allow_const=False)      # never constant vs constant
        op = self.rng.choice(['gt', 'ge', 'lt', 'le', 'gt', 'ge'])
        return Cmp(op, x, self.threshold(x, op))

    def str_test(self, scope):
        # Only fields with a value pool; free text (ids, emails) is never
        # tested against a literal.
        strs = [p for p in refs_of(scope, STR) if p[-1] in self.cat.strings]
        if not strs:
            return None

        def make():
            p = self.rng.choice(strs)
            pool = [s for s in self.cat.string_pool(p[-1]) if s]
            return Cmp(self.rng.choice(['eq', 'eq', 'ne']), Ref(p), Str(self.rng.choice(pool)))
        return make

    def bool_leaf(self, scope):
        bools = refs_of(scope, BOOL)
        return self.pick([
            (5, (lambda: self.compare(scope)) if self.has_int_source(scope) else None),
            (3, self.str_test(scope)),
            (2, (lambda: Ref(self.rng.choice(bools))) if bools else None),
            # Last resort, when the scope has nothing to compare: any str field
            # against a value it plausibly takes.
            (0.01, lambda: Cmp('ne', Ref(self.rng.choice(refs_of(scope, STR))),
                               Str(self.rng.choice(DEFAULT_POOL)))),
        ])

    def bool_(self, scope, depth):
        if depth <= 0:
            return self.bool_leaf(scope)
        d = depth - 1
        return self.pick([
            (6, lambda: self.bool_leaf(scope)),
            (1, lambda: self.two(All, scope)),
            (1, lambda: self.two(Any, scope)),
            (0.5, lambda: Not(self.bool_leaf(scope))),
        ])

    def two(self, cls, scope):
        a = self.bool_leaf(scope)
        for _ in range(4):
            b = self.bool_leaf(scope)
            if b != a:
                return cls((a, b))
        return a

    # ------------------------------------------------------------------ str
    def str_(self, scope, depth, field=None):
        strs = refs_of(scope, STR)
        named = [p for p in strs if field and p[-1] == field]

        def fmt():
            parts = [Str(self.rng.choice(['Order ', 'Update: ', 'Alert: ', 'Done: ', 'Note: ']))]
            vals = strs + refs_of(scope, INT)
            for _ in range(self.rng.randint(1, 2)):
                parts.append(Ref(self.rng.choice(vals)))
                parts.append(Str(self.rng.choice([' ok', ' pending', ' needs review', ' total ', ' / '])))
            if parts[-1].v.endswith(' ') and self.rng.random() < 0.5:
                parts.append(Ref(self.rng.choice(vals)))
            return Fmt(tuple(parts))

        if field is None:
            # A free-standing string (a computed label): a field, a literal,
            # or text composed from values.
            return self.pick([
                (3, (lambda: Ref(self.rng.choice(strs))) if strs else None),
                (2 if depth > 0 and strs else 0, fmt),
                (0.5, lambda: Str(self.rng.choice(['n/a', 'manual']))),
            ])
        # Filling a named slot (a call argument, a saved field). Anything
        # type-correct would pass the checker, but `status = the invoice id`
        # reads as nonsense, so:
        #   same name in scope     -> that field
        #   the slot has a pool    -> one of its values (status, tier, region)
        #   same kind of name      -> a field sharing its last word (_id, email, date)
        #   otherwise              -> any string field, or a placeholder literal
        if named:
            return Ref(self.rng.choice(named))
        pool = [v for v in self.cat.strings.get(field, []) if v]
        if pool:
            return Str(self.rng.choice(pool))
        kin = [p for p in strs if p[-1].split('_')[-1] == field.split('_')[-1]]
        if kin:
            return Ref(self.rng.choice(kin))
        return self.pick([
            (3, (lambda: Ref(self.rng.choice(strs))) if strs else None),
            (1, lambda: Str(self.rng.choice(['n/a', 'manual']))),
        ])

    def message(self, scope):
        """A notification's text: a lead-in, what it is about, and one value.

        "Update on " + order_id + ": " + status -- never an email address or a
        random string field on its own, which reads as nonsense.
        """
        strs = [p for p in refs_of(scope, STR) if not p[-1].endswith('email')]
        ids = [p for p in strs if p[-1].endswith('_id') or p[-1] in ('name', 'customer')]
        vals = refs_of(scope, INT) + [p for p in strs if p[-1] in self.cat.strings]
        parts = [Str(self.rng.choice(['Update on ', 'New ', 'Alert for ', 'Done: ', 'Reminder: ']))]
        about = ids or strs
        if about:
            parts.append(Ref(self.rng.choice(about)))
        else:
            parts[0] = Str(parts[0].v + 'your request')
        if vals and self.rng.random() < 0.7:
            parts.append(Str(self.rng.choice([': ', ' - ', ', now '])))
            parts.append(Ref(self.rng.choice(vals)))
        return Fmt(tuple(parts)) if len(parts) > 1 else parts[0]

    # ------------------------------------------------------------ dispatch
    def expr(self, scope, t, field=None, depth=1):
        """An expression of type `t`. `field` names the slot being filled, if any,
        so a value with the same name in scope is preferred (amount <- amount)."""
        same = [p for p in refs_of(scope, t) if field and p[-1] == field]
        if same and self.rng.random() < 0.95:
            return Ref(self.rng.choice(same))
        if t == INT:
            return self.int_(scope, depth)
        if t == BOOL:
            return self.bool_(scope, depth)
        if t == STR:
            return self.str_(scope, depth, field)
        exact = refs_of(scope, t)
        if exact:
            return Ref(self.rng.choice(exact))
        return None
