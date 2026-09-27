"""values.py - Value slots, names and templates -> IR2 expressions.

A value in a spec is one of five forms (LANGUAGE.md §8): a name or path, a
literal, a template, an exact expression in backquotes, or prose (intent).
The first four compile here, deterministically. Prose is handed to an
intent hook -- the logic model, eventually -- and without one it stays an
`Invalid` placeholder that the rest of the pipeline reports as intent.

Names resolve through a `Resolver`, which knows every value declared so far
in document order: payload fields, compute vars and call outputs. Whether a
value is actually available on every path to a step is the IR2 checker's
job; the resolver only maps a spelling to a declaration.
"""

import re

from flow.ir2.expr import Bool, Const, Fmt, Invalid, Pluck, Ref, Str
from flow.ir2.types import BOOL, INT, STR, UNKNOWN, ListT, RecordT
from flow.spec.document import SpecError
from flow.spec.text import QUOTED, unquote

PATH = re.compile(r'[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*')
CONST = re.compile(r'[A-Z][A-Z0-9_]*')
INTEGER = re.compile(r'-?\d+')
INTENT = 'intent'           # the Invalid.error marking a prose value awaiting the logic model


class Resolver:
    def __init__(self, constants):
        self.constants = constants       # NAME -> (IR2 literal, type)
        self.roots = {}                  # 'input' and step names -> type
        self.decls = []                  # (bare name, path, type)

    def add_root(self, name, t):
        self.roots[name] = t

    def add_field(self, root, name, t):
        self.decls.append((name, (root, name), t))
        rec = self.roots.get(root)
        fields = tuple(rec.fields) if isinstance(rec, RecordT) else ()
        self.roots[root] = RecordT(fields + ((name, t),))

    def add_record(self, root, record):
        self.roots[root] = RecordT(())
        for n, t in record.fields:
            self.add_field(root, n, t)

    def constant(self, name, line):
        if name not in self.constants:
            raise SpecError('resolves', line, f'unknown constant {name}')
        return self.constants[name]

    def resolve(self, path, line):
        """A spelled name or path -> (IR2 expression, type)."""
        segs = path.split('.')
        if segs[0] in self.roots:
            where, t = (segs[0],), self.roots[segs[0]]
        else:
            hits = [(p, t) for n, p, t in self.decls if n == segs[0]]
            if not hits:
                known = sorted({n for n, _, _ in self.decls} | set(self.roots))
                raise SpecError('resolves', line, f'"{segs[0]}" is not a declared value '
                                f'(known: {", ".join(known) or "none"})')
            if len(hits) > 1:
                opts = ' or '.join('.'.join(p) for p, _ in hits)
                raise SpecError('resolves', line, f'"{segs[0]}" is ambiguous: write {opts}')
            where, t = hits[0]
        for i, seg in enumerate(segs[1:], 1):
            if t == UNKNOWN:
                where = where + (seg,)
            elif isinstance(t, RecordT):
                ft = t.field(seg)
                if ft is None:
                    raise SpecError('resolves', line, f'{".".join(segs[:i])} has no field "{seg}" '
                                    f'(fields: {", ".join(t.names())})')
                where, t = where + (seg,), ft
            elif isinstance(t, ListT) and isinstance(t.elem, RecordT) and i == len(segs) - 1:
                ft = t.elem.field(seg)
                if ft is None:
                    raise SpecError('resolves', line, f'records in {".".join(segs[:i])} have no field '
                                    f'"{seg}" (fields: {", ".join(t.elem.names())})')
                return Pluck(Ref(where), seg), ListT(ft)
            else:
                raise SpecError('types', line, f'{".".join(segs[:i])} is {t}; it has no field "{seg}"')
        return Ref(where), t

    def type_scope(self):
        return dict(self.roots)


def template(q, resolver, line):
    """A quoted string -> Str, or Fmt when it has {placeholders}."""
    body = unquote(q)
    parts, buf, i = [], [], 0
    while i < len(body):
        c = body[i]
        if body.startswith('{{', i) or body.startswith('}}', i):
            buf.append(c)
            i += 2
            continue
        if c == '{':
            j = body.find('}', i)
            if j < 0:
                raise SpecError('syntax', line, f'unclosed "{{" in {q}')
            name = body[i + 1:j].strip()
            if buf:
                parts.append(Str(''.join(buf)))
                buf = []
            if CONST.fullmatch(name):
                parts.append(resolver.constant(name, line)[0])
            elif PATH.fullmatch(name):
                parts.append(resolver.resolve(name, line)[0])
            else:
                raise SpecError('syntax', line, f'"{{{name}}}" is not a name')
            i = j + 1
            continue
        if c == '}':
            raise SpecError('syntax', line, f'stray "}}" in {q}; write "}}}}" for a literal brace')
        buf.append(c)
        i += 1
    if buf:
        parts.append(Str(''.join(buf)))
    if len(parts) == 1 and isinstance(parts[0], Str):
        return parts[0]
    return Fmt(tuple(parts)) if parts else Str('')


def value(raw, resolver, line, intent=None):
    """One value slot -> (IR2 expression, is_prose)."""
    from flow.spec.exact import parse_exact
    v = raw.strip()
    if len(v) >= 2 and v.startswith('`') and v.endswith('`'):
        return parse_exact(v[1:-1], resolver, line), False
    if QUOTED.fullmatch(v):
        return template(v, resolver, line), False
    if INTEGER.fullmatch(v):
        return Const(int(v)), False
    if v in ('true', 'false'):
        return Bool(v == 'true'), False
    if CONST.fullmatch(v):
        return resolver.constant(v, line)[0], False
    if PATH.fullmatch(v):
        return resolver.resolve(v, line)[0], False
    if not v:
        raise SpecError('syntax', line, 'a value is missing')
    if intent is not None:
        exact = intent(v)
        if exact is not None:
            return parse_exact(exact, resolver, line), True
    return Invalid(v, INTENT), True


def literal(raw, line):
    """A constant's value: integer, string or boolean -> (IR2 literal, type)."""
    v = raw.strip()
    if INTEGER.fullmatch(v):
        return Const(int(v)), INT
    if QUOTED.fullmatch(v):
        return Str(unquote(v)), STR
    if v in ('true', 'false'):
        return Bool(v == 'true'), BOOL
    raise SpecError('syntax', line, f'a constant is an integer, a string or true/false, not "{v}"')
