"""model.py - The flow encoder: a text encoder plus five kinds of head.

    kind      [slot]            n-way over the step kinds, plus EMPTY
    settings  [slot] x 7        n-way each: source, method, retries, op, channel, status, responds
    edges     [slot, slot]      n-way over {none, next, error, true, false, branch}  (biaffine)
    tags      [slot, token]     B/I/O over the span roles: what to copy, and each expression's text
    clause    [slot, token]     B/I/O: the step's whole clause
    mode      [request]         sync | async | abstain

Every output is a classification. The heads only see hidden states, so the
encoder underneath can be replaced without touching them.
"""

import torch
from torch import nn
from torch.nn import functional as F

from flow.m1.vocab import CLAUSE_TAGS, EDGE_LABELS, KINDS, MODES, SETTINGS, TAGS

PAIR = 256


class FlowEncoder(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.hidden_size
        self.drop = nn.Dropout(dropout)
        self.kind = nn.Linear(d, len(KINDS))
        self.settings = nn.ModuleDict({k: nn.Linear(d, len(v)) for k, v in SETTINGS.items()})
        self.mode = nn.Linear(d, len(MODES))
        # edges: biaffine over (source slot, destination slot)
        self.e_src = nn.Sequential(nn.Linear(d, PAIR), nn.GELU())
        self.e_dst = nn.Sequential(nn.Linear(d, PAIR), nn.GELU())
        self.e_bil = nn.Parameter(torch.zeros(len(EDGE_LABELS), PAIR, PAIR))
        self.e_bias = nn.Linear(2 * PAIR, len(EDGE_LABELS))
        # tags and clauses: additive (slot, token) attention
        self.t_tok = nn.Linear(d, PAIR)
        self.t_slot = nn.Linear(d, PAIR)
        self.t_out = nn.Linear(PAIR, len(TAGS))
        self.c_out = nn.Linear(PAIR, len(CLAUSE_TAGS))

    def forward(self, input_ids, attention_mask, slot_pos):
        h = self.drop(self.encoder(input_ids, attention_mask))           # [B, T, d]
        B = h.size(0)
        slots = h[torch.arange(B)[:, None], slot_pos]                    # [B, S, d]
        out = {
            'kind': self.kind(slots),
            'settings': {k: head(slots) for k, head in self.settings.items()},
            'mode': self.mode(h[:, 0]),
        }
        src, dst = self.e_src(slots), self.e_dst(slots)                  # [B, S, P]
        bil = torch.einsum('bip,cpq,bjq->bijc', src, self.e_bil, dst)
        S = slots.size(1)
        pair = torch.cat([src[:, :, None].expand(-1, -1, S, -1),
                          dst[:, None].expand(-1, S, -1, -1)], dim=-1)
        out['edges'] = bil + self.e_bias(pair)                           # [B, S, S, C]
        joint = torch.tanh(self.t_tok(h)[:, None] + self.t_slot(slots)[:, :, None])   # [B, S, T, P]
        out['tags'] = self.t_out(joint)
        out['clause'] = self.c_out(joint)
        return out


def loss(out, batch):
    parts = {
        'kind': F.cross_entropy(out['kind'].flatten(0, 1), batch['kind'].flatten()),
        'settings': sum(F.cross_entropy(out['settings'][k].flatten(0, 1), batch['settings'][k].flatten())
                        for k in SETTINGS) / len(SETTINGS),
        'edges': F.cross_entropy(out['edges'].flatten(0, 2), batch['edges'].flatten()),
        'tags': F.cross_entropy(out['tags'].flatten(0, 2), batch['tags'].flatten(), ignore_index=-100),
        'clause': F.cross_entropy(out['clause'].flatten(0, 2), batch['clause'].flatten(), ignore_index=-100),
        'mode': F.cross_entropy(out['mode'], batch['mode']),
    }
    return sum(parts.values()), {k: v.item() for k, v in parts.items()}
