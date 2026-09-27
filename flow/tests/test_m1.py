"""The flow encoder's data path, with a tiny fake encoder instead of BERT.

The real encoder is pretrained and large; these tests only check what M1
does around it: span alignment, labels, decoding, the gold replay, and that
the heads produce the right shapes. Skipped when torch isn't installed.
"""

import random
import re
import unittest

try:
    import torch
except ImportError:                      # the IR2 core doesn't need torch
    torch = None

from flow.gen.build import build_one
from flow.gen.catalog import all_catalogs
from flow.gen.sampler import sample_flows

CATS = all_catalogs()


class FakeEncoder(torch.nn.Module if torch else object):
    """Whitespace-and-punctuation tokens, a random embedding: same interface as BertEncoder."""
    hidden_size = 32

    def __init__(self):
        super().__init__()
        from flow.m1.vocab import N_SLOTS
        self.n_slots = N_SLOTS
        self.emb = torch.nn.Embedding(5000, self.hidden_size)

    def tokenize(self, text):
        from flow.m1.encoder import Tokenized
        toks = [(m.start(), m.end()) for m in re.finditer(r'\w+|[^\w\s]', text)]
        ids = [3 + (hash(text[a:b]) % 4000) for a, b in toks]
        input_ids = [1] + ids + [2] + [4990 + i % 10 for i in range(self.n_slots)] + [2]
        offsets = [(0, 0)] + toks + [(0, 0)] * (self.n_slots + 2)
        first = len(ids) + 2
        return Tokenized(input_ids, offsets, len(ids), list(range(first, first + self.n_slots)))

    def forward(self, input_ids, attention_mask):
        return self.emb(input_ids)

    def pad_id(self):
        return 0


def records(n, seed=0):
    rng, out = random.Random(seed), []
    for f, meta in sample_flows(n, CATS, seed=seed):
        rec, _ = build_one(rng, f, meta, CATS[meta['catalog']], 'logistics')
        if rec:
            out.append(rec)
    return out


@unittest.skipIf(torch is None, 'torch not installed')
class M1Data(unittest.TestCase):
    def setUp(self):
        self.enc = FakeEncoder()
        from flow.m1.data import make_examples
        self.exs, skipped = make_examples(records(60, seed=3), self.enc)
        self.assertEqual(skipped, 0)

    def test_gold_labels_decode_back_to_flows_that_replay(self):
        from flow.m1.decode import build_flow, gold
        from flow.m1.evaluate import compare, replays
        for ex in self.exs:
            with self.subTest(flow=ex.record['id']):
                gd = gold(ex)
                self.assertTrue(all(compare(gd, gd).values()))
                self.assertTrue(replays(build_flow(gd, ex.record), ex.record))

    def test_copied_values_come_back_exactly(self):
        from flow.m1.decode import gold
        for ex in self.exs:
            gd = gold(ex)
            for slot, gs in zip(ex.record['m1']['slots'], gd['slots']):
                text = ex.record['m1']['text']
                if 'service' in slot['copy']:
                    a, b = slot['copy']['service']
                    self.assertEqual(gs['spans']['v_service'], [text[a:b]])

    def test_heads_shapes_and_loss(self):
        from flow.m1.data import collate
        from flow.m1.model import FlowEncoder, loss
        from flow.m1.vocab import EDGE_LABELS, KINDS, N_SLOTS, TAGS
        model = FlowEncoder(self.enc)
        t = collate(self.exs[:4], 0)
        out = model(t['input_ids'], t['attention_mask'], t['slot_pos'])
        T = t['input_ids'].size(1)
        self.assertEqual(tuple(out['kind'].shape), (4, N_SLOTS, len(KINDS)))
        self.assertEqual(tuple(out['edges'].shape), (4, N_SLOTS, N_SLOTS, len(EDGE_LABELS)))
        self.assertEqual(tuple(out['tags'].shape), (4, N_SLOTS, T, len(TAGS)))
        total, parts = loss(out, t)
        total.backward()
        self.assertTrue(all(v > 0 for v in parts.values()))


if __name__ == '__main__':
    unittest.main()
