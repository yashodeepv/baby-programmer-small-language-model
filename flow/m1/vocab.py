"""vocab.py - Every label space the flow encoder predicts into.

All of them are closed and small: the model classifies, it never writes
free text. Values it needs verbatim (a path, a service name) are copied by
tagging the tokens that spell them.
"""

from flow.ir2 import graph as g

N_SLOTS = 16

EMPTY = 'empty'
KINDS = list(g.KINDS) + [EMPTY]

SETTINGS = {
    'source':   ['none'] + list(g.SOURCES),
    'method':   ['none', 'GET', 'POST', 'PUT', 'DELETE'],
    'retries':  ['none', '0', '1', '2', '3'],
    'op':       ['none'] + list(g.PERSIST_OPS),
    'channel':  ['none'] + list(g.CHANNELS),
    'status':   ['none'] + list(g.END_STATUS),
    'responds': ['none', 'no', 'yes'],
}

EDGE_LABELS = ['none', 'next', 'error', 'true', 'false', 'branch']

MODES = ['sync', 'async', 'abstain']

# Span roles, tagged B/I/O per (slot, token). A slot never has two spans of
# different roles over the same characters, so one tag per token suffices.
VALUE_ROLES = ['v_path', 'v_event', 'v_cron', 'v_service', 'v_op', 'v_entity',
               'v_message', 'v_var', 'v_key']
EXPR_ROLES = ['e_rule', 'e_cond', 'e_assign', 'e_arg', 'e_data', 'e_to', 'e_msg', 'e_resp']
ROLES = VALUE_ROLES + EXPR_ROLES
TAGS = ['O'] + [f'{bi}-{r}' for r in ROLES for bi in 'BI']
TAG_ID = {t: i for i, t in enumerate(TAGS)}

CLAUSE_TAGS = ['O', 'B', 'I']


def copy_role(key):
    """A corpus copy key ('path', 'name.total', ...) -> its span role, or None."""
    if key.startswith('name.'):
        return 'v_var'
    if key.startswith('key.'):
        return 'v_key'
    return {'path': 'v_path', 'name': 'v_event', 'cron': 'v_cron', 'service': 'v_service',
            'op': 'v_op', 'entity': 'v_entity', 'message': 'v_message'}.get(key)


def expr_role(field):
    """A corpus expression field ('assign.total', 'to', ...) -> its span role."""
    head = field.split('.')[0]
    return {'rule': 'e_rule', 'cond': 'e_cond', 'assign': 'e_assign', 'args': 'e_arg',
            'data': 'e_data', 'to': 'e_to', 'message': 'e_msg', 'response': 'e_resp'}[head]
