"""document.py - Baby Spec text -> sections and a step tree, with line numbers.

Only layout is decided here: which section a line belongs to, which step
it is, what it is nested under. What a step *means* is steps.py's job.
Every node remembers its line, so every later error can point at it.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

VERSION = 'baby-spec/0.1'

_STEP = re.compile(r'^( *)(\d+|[a-z])\.\s+(.*)$')
_OTHERWISE = re.compile(r'^( *)Otherwise:\s*$')
_ITEM = re.compile(r'^( *)-\s+(.*)$')
_TITLE = re.compile(r'^\[([^\]]*)\]\s*(.*)$')


class SpecError(Exception):
    """One violated rule. `rule` is a name from LANGUAGE.md §9."""

    def __init__(self, rule, line, message):
        super().__init__(f'line {line}: [{rule}] {message}')
        self.rule, self.line, self.message = rule, line, message


@dataclass
class StepNode:
    line: int
    marker: str                          # '1', '2', ... or 'a', 'b', ...
    indent: int
    text: str                            # without the title
    title: str = ''
    children: List['StepNode'] = field(default_factory=list)
    otherwise: Optional[List['StepNode']] = None
    otherwise_line: int = 0
    otherwise_indent: int = -1
    items: List[Tuple[int, str]] = field(default_factory=list)     # compute '- var: value'


@dataclass
class FlowDoc:
    name: str
    line: int
    intent: str = ''
    trigger: Optional[Tuple[int, str]] = None
    mode: Optional[Tuple[int, str]] = None
    steps: List[StepNode] = field(default_factory=list)


@dataclass
class SpecDoc:
    system: str = ''
    system_line: int = 0
    description: str = ''
    constants: List[Tuple[int, str]] = field(default_factory=list)
    entities: List[Tuple[int, str]] = field(default_factory=list)
    services: List[Tuple[int, str]] = field(default_factory=list)
    flows: List[FlowDoc] = field(default_factory=list)


def _letter(i):
    return chr(ord('a') + i)


def parse_document(text):
    """Returns (SpecDoc, [SpecError]). Layout errors don't stop the parse
    where it can sensibly continue, so an analyst sees several at once."""
    doc, errors = SpecDoc(), []
    lines = text.split('\n')
    section, flow = None, None
    in_steps = False
    stack: List[StepNode] = []           # open steps, innermost last
    seen_sections = set()
    version_seen = False

    def err(rule, n, msg):
        errors.append(SpecError(rule, n, msg))

    def attach(node):
        """Place a step node under the right parent, by indentation."""
        while stack and stack[-1].indent >= node.indent:
            stack.pop()
        if not stack:
            if node.indent != 0 or not node.marker.isdigit():
                err('layout', node.line, 'a top-level step is numbered (1., 2., ...) and not indented')
            flow.steps.append(node)
        else:
            parent = stack[-1]
            if node.marker.isdigit():
                err('layout', node.line, 'a nested step is lettered (a., b., ...)')
            in_otherwise = parent.otherwise is not None and node.indent == parent.otherwise_indent
            target = parent.otherwise if in_otherwise else parent.children
            target.append(node)
        stack.append(node)

    for n, raw in enumerate(lines, 1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith('>'):
            continue
        if not version_seen:
            version_seen = True
            if stripped != f'Spec: {VERSION}':
                err('layout', n, f'the first line must be "Spec: {VERSION}"')
                if stripped.startswith('Spec:'):
                    continue
            else:
                continue

        if line.startswith('# '):
            m = re.fullmatch(r'# System:\s*([a-z][a-z0-9_-]*)', line)
            if not m:
                err('layout', n, 'expected "# System: <name>"')
            elif doc.system:
                err('layout', n, 'only one "# System" heading')
            else:
                doc.system, doc.system_line = m.group(1), n
            section, flow, in_steps, stack = 'system', None, False, []
            continue
        if line.startswith('## '):
            stack, in_steps = [], False
            m = re.fullmatch(r'## Flow:\s*([a-z][a-z0-9_]*)', line)
            if m:
                flow = FlowDoc(m.group(1), n)
                doc.flows.append(flow)
                section = 'flow'
                continue
            m = re.fullmatch(r'## (Constants|Entities|Services)', line)
            if not m:
                err('layout', n, 'expected "## Constants", "## Entities", "## Services" or "## Flow: <name>"')
                section = None
                continue
            name = m.group(1).lower()
            if name in seen_sections:
                err('layout', n, f'"## {m.group(1)}" appears more than once')
            if doc.flows:
                err('layout', n, f'"## {m.group(1)}" must come before the flows')
            seen_sections.add(name)
            section, flow = name, None
            continue
        if not doc.system:
            err('layout', n, 'expected "# System: <name>" before anything else')
            continue

        if section == 'system':
            doc.description += (' ' if doc.description else '') + stripped
            continue
        if section in ('constants', 'entities', 'services'):
            m = _ITEM.match(line)
            if not m or m.group(1):
                err('layout', n, f'each {section[:-1]} is a "- " line')
                continue
            getattr(doc, section).append((n, m.group(2).strip()))
            continue
        if section != 'flow' or flow is None:
            err('layout', n, 'text outside any section')
            continue

        if not in_steps:
            m = re.fullmatch(r'(Intent|Trigger|Mode):\s*(.*)', stripped)
            if stripped == 'Steps:':
                in_steps = True
            elif m and not line.startswith(' '):
                key, val = m.group(1), m.group(2).strip()
                if key == 'Intent':
                    flow.intent = val
                elif key == 'Trigger':
                    flow.trigger = (n, val)
                else:
                    flow.mode = (n, val)
            else:
                err('layout', n, 'a flow header has Intent:, Trigger:, Mode: and then Steps:')
            continue

        m = _STEP.match(line)
        if m:
            body = m.group(3).strip()
            t = _TITLE.match(body)
            title, body = (t.group(1).strip(), t.group(2).strip()) if t else ('', body)
            attach(StepNode(n, m.group(2), len(m.group(1)), body, title))
            continue
        m = _OTHERWISE.match(line)
        if m:
            indent = len(m.group(1))
            while stack and stack[-1].indent >= indent:
                stack.pop()
            if not stack:
                err('layout', n, '"Otherwise:" must belong to an If step')
                continue
            parent = stack[-1]
            if parent.otherwise is not None:
                err('layout', n, 'an If has at most one "Otherwise:"')
            parent.otherwise, parent.otherwise_line = [], n
            parent.otherwise_indent = indent
            continue
        m = _ITEM.match(line)
        if m and stack:
            indent = len(m.group(1))
            while stack and stack[-1].indent >= indent:
                stack.pop()
            if not stack:
                err('layout', n, 'a "- " line inside Steps must belong to a Compute step')
                continue
            stack[-1].items.append((n, m.group(2).strip()))
            continue
        if stack and line.startswith(' '):
            stack[-1].text += ' ' + stripped        # a step continued on the next line
            continue
        err('layout', n, 'expected a numbered or lettered step')

    if not version_seen:
        err('layout', 1, f'the first line must be "Spec: {VERSION}"')
    if version_seen and not doc.system:
        err('layout', 1, 'no "# System: <name>" heading')
    for f in doc.flows:
        _check_numbering(f.steps, errors, top=True)
    return doc, errors


def _check_numbering(steps, errors, top):
    for i, s in enumerate(steps):
        want = str(i + 1) if top else _letter(i)
        if s.marker != want:
            errors.append(SpecError('layout', s.line, f'expected step "{want}." here, found "{s.marker}."'))
        _check_numbering(s.children, errors, top=False)
        if s.otherwise is not None:
            _check_numbering(s.otherwise, errors, top=False)
