"""text.py - Splitting step text without breaking quotes, backquotes or brackets.

A spec line like  Slack "#sales": "Large order {order_id}: {total}".
has a ':' inside a string, a ':' as the separator, and a '.' ending the
sentence. Every split here happens only at the top level: outside double
quotes, backquotes and parentheses.
"""

import re

QUOTED = re.compile(r'^"(?:[^"\\]|\\.)*"$')


def top_level(text):
    """Yields (index, char, at_top) for every character."""
    depth, quote, tick, esc = 0, False, False, False
    for i, c in enumerate(text):
        at_top = depth == 0 and not quote and not tick
        yield i, c, at_top and not esc
        if esc:
            esc = False
            continue
        if quote:
            if c == '\\':
                esc = True
            elif c == '"':
                quote = False
            continue
        if tick:
            if c == '`':
                tick = False
            continue
        if c == '"':
            quote = True
        elif c == '`':
            tick = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth = max(0, depth - 1)


def find_top(text, sub, start=0):
    for i, c, top in top_level(text):
        if i >= start and top and text.startswith(sub, i):
            return i
    return -1


def split_top(text, sep):
    parts, last = [], 0
    i = find_top(text, sep)
    while i >= 0:
        parts.append(text[last:i])
        last = i + len(sep)
        i = find_top(text, sep, last)
    parts.append(text[last:])
    return [p.strip() for p in parts]


def split_clauses(text, keywords):
    """Split at '. ' where the next words start one of `keywords` (top level only).

    'Call a.b with x. Retry 1 time. On failure stop ...' ->
    ['Call a.b with x', 'Retry 1 time', 'On failure stop ...']
    """
    cuts, i = [], find_top(text, '. ')
    while i >= 0:
        rest = text[i + 2:]
        if any(rest.startswith(k) for k in keywords):
            cuts.append(i)
        i = find_top(text, '. ', i + 1)
    parts, last = [], 0
    for c in cuts:
        parts.append(text[last:c])
        last = c + 2
    parts.append(text[last:])
    return [strip_period(p) for p in parts]


def strip_period(text):
    """Remove one sentence-ending '.', if it is at the top level."""
    text = text.strip()
    if text.endswith('.'):
        tops = [i for i, c, top in top_level(text) if top]
        if tops and tops[-1] == len(text) - 1:
            return text[:-1].rstrip()
    return text


def unquote(q):
    body, out, i = q[1:-1], [], 0
    while i < len(body):
        if body[i] == '\\' and i + 1 < len(body):
            out.append(body[i + 1])
            i += 2
        else:
            out.append(body[i])
            i += 1
    return ''.join(out)
