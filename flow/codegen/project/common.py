"""common.py - What every project generator shares: the system, names, types."""

import re
from dataclasses import dataclass, field
from typing import Dict, List

from flow.ir2 import graph as g
from flow.ir2.types import BOOL, INT, STR, ListT, RecordT


@dataclass
class System:
    """Everything a project is generated from."""
    name: str                                   # 'shop-orders'
    description: str
    flows: List[g.Flow]
    scenarios: Dict[str, dict] = field(default_factory=dict)     # flow name -> {scenario name: scenario}

    @property
    def services(self):
        out = {}
        for f in self.flows:
            for s, ops in f.services.items():
                out.setdefault(s, {}).update(ops)
        return out

    @property
    def entities(self):
        out = {}
        for f in self.flows:
            out.update(f.entities)
        return out


def snake(name):
    return re.sub(r'[^a-z0-9_]', '_', name.lower()).strip('_') or 'app'


def camel(name):
    return ''.join(p[:1].upper() + p[1:] for p in re.split(r'[^a-zA-Z0-9]', name) if p)


def trigger_line(flow):
    t = flow.trigger()
    if t.source == 'http':
        return f"{t.config['method']} {t.config['path']}"
    if t.source == 'event':
        return f"event {t.config['name']}"
    return f"schedule \"{t.config['cron']}\""


class TypeNamer:
    """Gives every record type a class name, and writes it once."""

    def __init__(self):
        self.defs = []                          # (name, RecordT), in dependency order
        self.names = {}

    def record(self, rec, name):
        key = rec
        if key in self.names:
            return self.names[key]
        base, n = name, 2
        while name in {d for d, _ in self.defs}:
            name, n = f'{base}{n}', n + 1
        for fname, ft in rec.fields:           # nested records first
            self.annotate(ft, name + camel(fname))
        self.names[key] = name
        self.defs.append((name, rec))
        return name

    def annotate(self, t, hint):
        """Register nested records; the language printers spell the type."""
        if isinstance(t, RecordT):
            return self.record(t, hint)
        if isinstance(t, ListT):
            return self.annotate(t.elem, hint + 'Item')
        return None


def py_type(t, namer):
    if t == INT:
        return 'int'
    if t == STR:
        return 'str'
    if t == BOOL:
        return 'bool'
    if isinstance(t, ListT):
        return f'List[{py_type(t.elem, namer)}]'
    if isinstance(t, RecordT):
        return namer.names[t]
    return 'Any'


def js_type(t, namer):
    if t == INT:
        return 'number'
    if t == STR:
        return 'string'
    if t == BOOL:
        return 'boolean'
    if isinstance(t, ListT):
        return f'{js_type(t.elem, namer)}[]'
    if isinstance(t, RecordT):
        return namer.names[t]
    return '*'


def collect_types(system):
    """Name every record the project exposes: entities, flow inputs, service contracts."""
    namer = TypeNamer()
    for e, rec in system.entities.items():
        namer.record(rec, camel(e))
    for f in system.flows:
        namer.record(f.trigger().payload, camel(f.name) + 'Input')
    for s, ops in system.services.items():
        for op, sig in ops.items():
            namer.record(sig.input, camel(s) + camel(op) + 'Args')
            namer.record(sig.output, camel(s) + camel(op) + 'Result')
    return namer


def outline(flow):
    """Indented step titles, for READMEs."""
    lines = []
    for s in flow.steps:
        if isinstance(s, (g.Trigger, g.Join)):
            continue
        label = s.title or s.kind
        lines.append(f'- {label} ({s.kind})')
    return lines
