"""prompt.py - The text a user pastes into an LLM to get a Baby Spec back.

Assembled from LANGUAGE.md and an example spec at request time, so the
prompt can never drift from the grammar the compiler actually accepts.
"""

import os

SPEC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spec')

INSTRUCTIONS = '''You are the analyst for baby-programmer, a system that builds verified software
from a precise specification. Read the USE CASE at the end of this message and
write a complete system spec for it in Baby Spec 0.1, the language defined in
the LANGUAGE REFERENCE below.

Rules:
1. Reply with the spec only, inside a single ```markdown code block. The first
   line of the spec is exactly: Spec: baby-spec/0.1
2. Declare every entity and service operation the flows use, with exact types.
   If the use case doesn't name them, choose clear snake_case names yourself.
3. Write every value as a name, a literal, a template, or an exact expression
   in backquotes. Do NOT write prose values: this installation has no logic
   model yet, so prose can't be compiled.
4. Money is an int in minor units (cents). There are no decimals.
5. Give every step a short [title] in the business's own words. Business
   people review the flowchart drawn from these titles; they never read the spec.
6. Resolve every ambiguity yourself with a sensible default, and state the
   defaults you chose in the flow's Intent: line.
7. Stay inside Baby Spec 0.1. If the use case needs something it can't
   express (steps inside a loop, reading records back, timeouts), design the
   closest supported flow and explain the gap in a "> " comment line.
8. Check your spec against the validity rules in section 9 before replying:
   every name resolves, every call gives every parameter, every Save gives
   every field, and Respond with appears only in sync flows.
'''

# The example must follow rule 3, so its one prose value is swapped for the
# exact expression it stands for.
_EXAMPLE_FIX = ('   - enough_balance: days is at most balance, or kind is "sick"\n'
                '   > intent; expected: `days <= balance or kind == "sick"`\n',
                '   - enough_balance: `days <= balance or kind == "sick"`\n')


def example_spec():
    with open(os.path.join(SPEC_DIR, 'examples', 'leave_requests.spec.md')) as f:
        text = f.read()
    old, new = _EXAMPLE_FIX
    if old not in text:
        raise RuntimeError('the prompt example no longer matches leave_requests.spec.md')
    return text.replace(old, new)


def build_prompt():
    with open(os.path.join(SPEC_DIR, 'LANGUAGE.md')) as f:
        language = f.read()
    return (INSTRUCTIONS
            + '\n========== LANGUAGE REFERENCE ==========\n\n' + language
            + '\n========== EXAMPLE SPEC ==========\n\n```markdown\n' + example_spec() + '```\n'
            + '\n========== USE CASE ==========\n\n')
