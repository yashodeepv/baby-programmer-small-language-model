"""data.py - Corpus records -> training tensors for the flow encoder.

Each record's `m1` block already holds the labels in character offsets
(flow/gen/labels.py). Here they are aligned to the encoder's tokens: a span
covers every token whose characters overlap it, tagged B on the first and I
after. Records whose request doesn't fit the encoder are skipped and
counted, never truncated -- a truncated request would teach wrong labels.
"""

import json
import random
from dataclasses import dataclass
from typing import List

import torch

from flow.m1.vocab import (CLAUSE_TAGS, EDGE_LABELS, EMPTY, KINDS, MODES, N_SLOTS, SETTINGS,
                           TAG_ID, copy_role, expr_role)


@dataclass
class Example:
    record: dict
    input_ids: List[int]
    offsets: list
    n_text: int
    slot_pos: List[int]
    kind: List[int]                 # [S]
    settings: dict                  # name -> [S]
    edges: List[List[int]]          # [S][S]
    tags: List[List[int]]           # [S][T]
    clause: List[List[int]]         # [S][T]
    mode: int


def load_records(path, splits=None):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if splits is None or r['split'] in splits:
                out.append(r)
    return out


def _token_span(offsets, n_text, start, end):
    """Request-token positions overlapping characters [start, end)."""
    return [i for i in range(1, n_text + 1)
            if offsets[i][1] > start and offsets[i][0] < end]


def _tag(row, toks, role):
    for j, t in enumerate(toks):
        row[t] = TAG_ID[f'{"B" if j == 0 else "I"}-{role}']


def _setting(slot, name):
    v = slot['settings'].get(name)
    if v is None:
        return 0
    if isinstance(v, bool):
        v = 'yes' if v else 'no'
    return SETTINGS[name].index(str(v))


def make_example(record, enc):
    m1 = record['m1']
    tk = enc.tokenize(m1['text'])
    if tk is None or len(m1['slots']) > N_SLOTS:
        return None
    T = len(tk.input_ids)
    slots = m1['slots'] + [None] * (N_SLOTS - len(m1['slots']))
    kind = [KINDS.index(s['kind']) if s else KINDS.index(EMPTY) for s in slots]
    settings = {name: [_setting(s, name) if s else 0 for s in slots] for name in SETTINGS}
    edges = [[0] * N_SLOTS for _ in range(N_SLOTS)]
    for a, b, on in m1['edges']:
        edges[a][b] = EDGE_LABELS.index(on)
    tags = [[0] * T for _ in range(N_SLOTS)]
    clause = [[0] * T for _ in range(N_SLOTS)]
    for i, s in enumerate(slots):
        if s is None:
            continue
        for key, (a, b) in s['copy'].items():
            role = copy_role(key)
            if role:
                _tag(tags[i], _token_span(tk.offsets, tk.n_text, a, b), role)
        for field, (a, b) in s['exprs'].items():
            _tag(tags[i], _token_span(tk.offsets, tk.n_text, a, b), expr_role(field))
        if s['intent']:
            toks = _token_span(tk.offsets, tk.n_text, *s['intent'])
            for j, t in enumerate(toks):
                clause[i][t] = CLAUSE_TAGS.index('B' if j == 0 else 'I')
    return Example(record, tk.input_ids, tk.offsets, tk.n_text, tk.slot_pos, kind, settings,
                   edges, tags, clause, MODES.index(m1['mode']))


def make_examples(records, enc):
    out, skipped = [], 0
    for r in records:
        ex = make_example(r, enc)
        if ex is None:
            skipped += 1
        else:
            out.append(ex)
    return out, skipped


def collate(batch, pad_id):
    T = max(len(ex.input_ids) for ex in batch)
    B = len(batch)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    tags = torch.full((B, N_SLOTS, T), -100, dtype=torch.long)
    clause = torch.full((B, N_SLOTS, T), -100, dtype=torch.long)
    for b, ex in enumerate(batch):
        n = len(ex.input_ids)
        ids[b, :n] = torch.tensor(ex.input_ids)
        mask[b, :n] = 1
        text = slice(1, ex.n_text + 1)             # tags are scored on request tokens only
        tags[b, :, text] = torch.tensor(ex.tags)[:, text]
        clause[b, :, text] = torch.tensor(ex.clause)[:, text]
    return {
        'input_ids': ids,
        'attention_mask': mask,
        'slot_pos': torch.tensor([ex.slot_pos for ex in batch]),
        'kind': torch.tensor([ex.kind for ex in batch]),
        'settings': {k: torch.tensor([ex.settings[k] for ex in batch]) for k in SETTINGS},
        'edges': torch.tensor([ex.edges for ex in batch]),
        'tags': tags,
        'clause': clause,
        'mode': torch.tensor([ex.mode for ex in batch]),
    }


def batches(examples, size, shuffle, seed=0):
    idx = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), size):
        yield [examples[j] for j in idx[i:i + size]]
