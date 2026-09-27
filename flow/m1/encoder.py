"""encoder.py - The text encoder, behind the one interface the rest of M1 uses.

    enc.tokenize(text) -> Tokenized(input_ids, offsets, n_text, slot_pos)
    enc(input_ids, attention_mask) -> hidden states [batch, tokens, enc.hidden_size]

The sequence is  [CLS] request [SEP] [SLOT0] ... [SLOT15] [SEP] . The slot
tokens are learned queries: slot i's hidden state is read as "the i-th step
of the flow". `offsets` maps each request token back to its characters, so
copied values come from the original text, casing and all.

BertEncoder is a temporary stand-in for the POC. Our own encoder, trained
from scratch, replaces it by implementing the same two methods.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from flow.m1.vocab import N_SLOTS

SLOT_TOKENS = [f'[SLOT{i}]' for i in range(N_SLOTS)]


@dataclass
class Tokenized:
    input_ids: List[int]
    offsets: List[Tuple[int, int]]      # per token; (0, 0) for special tokens
    n_text: int                         # request tokens are positions 1 .. n_text
    slot_pos: List[int]


class BertEncoder(nn.Module):
    """A pretrained BERT-family model from Hugging Face (POC only)."""

    def __init__(self, name='bert-base-cased', max_len=512):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.name, self.max_len = name, max_len
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.add_special_tokens({'additional_special_tokens': SLOT_TOKENS})
        self.model = AutoModel.from_pretrained(name)
        self.model.resize_token_embeddings(len(self.tok))
        self.hidden_size = self.model.config.hidden_size
        self.slot_ids = self.tok.convert_tokens_to_ids(SLOT_TOKENS)

    def tokenize(self, text):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc['input_ids'], enc['offset_mapping']
        room = self.max_len - N_SLOTS - 3
        if len(ids) > room:
            return None                                  # too long; the caller counts it
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        input_ids = [cls] + ids + [sep] + self.slot_ids + [sep]
        offsets = [(0, 0)] + [tuple(o) for o in offs] + [(0, 0)] * (N_SLOTS + 2)
        first = len(ids) + 2
        return Tokenized(input_ids, offsets, len(ids), list(range(first, first + N_SLOTS)))

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def pad_id(self):
        return self.tok.pad_token_id

    def save(self, path):
        self.tok.save_pretrained(path)

    def config(self):
        return {'type': 'bert', 'name': self.name, 'max_len': self.max_len}


def build_encoder(cfg):
    if cfg['type'] == 'bert':
        return BertEncoder(cfg['name'], cfg.get('max_len', 512))
    raise ValueError(f'unknown encoder type {cfg["type"]!r}')
