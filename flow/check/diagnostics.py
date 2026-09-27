"""diagnostics.py - What the checker reports.

Every finding names the step (and field) it belongs to, so an editor can put
it on the right box in the flowchart, and carries a stable `code` so tools
can react to a kind of problem without parsing the message.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Diagnostic:
    level: str                  # error warning
    code: str                   # e.g. unknown-ref, type-mismatch, unreachable
    message: str
    step: Optional[str] = None
    field: Optional[str] = None

    def __str__(self):
        where = self.step or 'flow'
        if self.field:
            where += f'.{self.field}'
        return f'{self.level}: {where}: {self.message} [{self.code}]'


class Report:
    def __init__(self):
        self.items = []

    def error(self, code, message, step=None, field=None):
        self.items.append(Diagnostic('error', code, message, step, field))

    def warn(self, code, message, step=None, field=None):
        self.items.append(Diagnostic('warning', code, message, step, field))

    @property
    def errors(self):
        return [d for d in self.items if d.level == 'error']

    @property
    def warnings(self):
        return [d for d in self.items if d.level == 'warning']

    @property
    def ok(self):
        return not self.errors

    def codes(self):
        return [d.code for d in self.items]

    def __str__(self):
        return '\n'.join(str(d) for d in self.items) or 'ok'
