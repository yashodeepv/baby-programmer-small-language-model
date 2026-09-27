"""spec - Baby Spec: the system specification language, and its compiler.

    LANGUAGE.md   the language reference (normative)
    examples/     example system specs
    document.py   text -> sections and a step tree, with line numbers
    text.py       splitting step text outside quotes, backquotes and brackets
    types.py      spec type syntax -> IR2 types
    values.py     value slots, names and templates -> IR2; the name resolver
    exact.py      exact expressions -> IR2
    compile.py    the whole document -> IR2 flows, or line-numbered errors

    python3.11 -m flow.spec check   flow/spec/examples/order_checkout.spec.md
    python3.11 -m flow.spec show    flow/spec/examples/order_checkout.spec.md
    python3.11 -m flow.spec compile flow/spec/examples/order_checkout.spec.md --out build/
"""

from flow.spec.compile import Compiled, Intent, compile_spec
from flow.spec.document import SpecError

__all__ = ['compile_spec', 'Compiled', 'Intent', 'SpecError']
