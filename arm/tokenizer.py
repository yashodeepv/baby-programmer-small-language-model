"""
tokenizer.py - Token-level vocabulary for the AArch64 corpus.

Character-level modelling wastes the regularity of assembly: `mov` is three
independent prediction steps, `ASSISTANT:` is ten. Assembly has a genuinely
closed vocabulary -- ~60 mnemonics, 64 registers, a handful of punctuation --
so tokens can be atomic without any learned merge table (no BPE needed).

Design decisions worth knowing:

  * Registers (`w0`, `x29`, `sp`) are single tokens. They are atomic operands,
    not spellings.

  * Conditional branches (`b.ge`) are single tokens, matched before bare words
    so the `.` never splits them.

  * Separators are absorbed rather than tokenized. Words carry an optional
    leading space and `\n    ` is one token, because bare spaces and indents
    were 26% of all tokens when emitted separately -- pure structural overhead
    the model would have to predict. Measured: 89 -> 68 tokens per Q&A pair.

  * NUMBERS ARE SPLIT INTO DIGITS. This is deliberate and is the most important
    choice here. The core skill being taught is "the integer in the question
    reappears after `#` in the answer" -- a copy. Digit tokens make that a copy
    over a 10-symbol alphabet that generalizes to values never seen in
    training. Whole-number tokens would instead force the model to memorize
    each constant independently and fail on any unseen one.

  * Tokenization is exactly reversible: decode(encode(t)) == t. Every token
    carries its literal text and decoding is concatenation. The character-level
    path in train_corpus.py silently dropped unknown characters (`if c in
    stoi`); here an unmapped token is a loud error unless <unk> is requested.

Usage:
    python3 arm/tokenizer.py --corpus data/arm_corpus.txt --stats
"""

import argparse
import json
import re
import sys

UNK = '<unk>'

# Longest-match-first. Order is load-bearing: multi-character tokens must be
# listed before the single-character classes that would otherwise consume them.
TOKEN_RE = re.compile(r'''
      (?P<role>USER:|ASSISTANT:)
    | (?P<newline_indent>\n\ {4})
    | (?P<indent>\ {4})
    | (?P<newline>\n)
    | (?P<condbranch>\ ?b\.(?:eq|ne|ge|gt|le|lt|mi|pl|hi|hs|lo|ls|vs|vc|al))
    | (?P<commasp>,\ )
    | (?P<word>\ ?[A-Za-z_][A-Za-z0-9_]*)
    | (?P<digit>[0-9])
    | (?P<punct>[^\sA-Za-z0-9_])
    | (?P<space>\ )
''', re.X)


def tokenize(text):
    """Split text into literal token strings. Raises on unmatched input."""
    out, pos = [], 0
    for m in TOKEN_RE.finditer(text):
        if m.start() != pos:
            bad = text[pos:m.start()]
            raise ValueError(f'untokenizable text at {pos}: {bad!r}')
        out.append(m.group(0))
        pos = m.end()
    if pos != len(text):
        raise ValueError(f'untokenizable trailing text: {text[pos:]!r}')
    return out


class ArmTokenizer:
    def __init__(self, vocab):
        self.itos = {i: t for i, t in enumerate(vocab)}
        self.stoi = {t: i for i, t in self.itos.items()}

    @property
    def vocab_size(self):
        return len(self.itos)

    @classmethod
    def build(cls, text):
        """Vocabulary is the sorted set of tokens present, with <unk> at 0."""
        toks = sorted(set(tokenize(text)))
        return cls([UNK] + toks)

    def encode(self, text, allow_unk=False):
        ids = []
        for t in tokenize(text):
            if t in self.stoi:
                ids.append(self.stoi[t])
            elif allow_unk:
                ids.append(0)
            else:
                raise KeyError(f'token {t!r} not in vocabulary '
                               f'(pass allow_unk=True to map it to {UNK})')
        return ids

    def decode(self, ids):
        return ''.join(self.itos[int(i)] for i in ids)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'vocab': [self.itos[i] for i in range(len(self.itos))]}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)['vocab'])


# --------------------------------------------------------------------------

def stats(corpus_path, block_size=256):
    text = open(corpus_path).read()
    tok  = ArmTokenizer.build(text)

    ids = tok.encode(text)
    assert tok.decode(ids) == text, 'round-trip failed'

    pairs = text.count('USER:')
    chars, ntok = len(text), len(ids)

    print(f"Corpus: {corpus_path}")
    print(f"  Q&A pairs        {pairs:>10,}")
    print()
    print(f"{'':18} {'char-level':>12} {'token-level':>12}")
    print(f"{'vocab size':18} {len(set(text)):>12,} {tok.vocab_size:>12,}")
    print(f"{'sequence length':18} {chars:>12,} {ntok:>12,}")
    print(f"{'per Q&A pair':18} {chars // pairs:>12,} {ntok // pairs:>12,}")
    print(f"{'compression':18} {'1.00x':>12} {f'{chars / ntok:.2f}x':>12}")
    print()
    print(f"At BLOCK_SIZE={block_size}, one window holds:")
    print(f"  char-level   {block_size / (chars / pairs):>5.1f} examples")
    print(f"  token-level  {block_size / (ntok / pairs):>5.1f} examples")

    # Vocabulary breakdown -- confirms the space is genuinely closed.
    kinds = {}
    for t in (tok.itos[i] for i in range(tok.vocab_size)):
        m = TOKEN_RE.fullmatch(t)
        kind = m.lastgroup if m else 'special'
        kinds.setdefault(kind, []).append(t)
    print("\nVocabulary by class:")
    for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
        show = ' '.join(repr(x) if x.isspace() else x for x in sorted(v)[:8])
        print(f"  {k:12} {len(v):>4}   {show}{' ...' if len(v) > 8 else ''}")

    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='data/arm_corpus.txt')
    ap.add_argument('--out',    default='data/arm_vocab.json')
    ap.add_argument('--stats',  action='store_true')
    ap.add_argument('--block-size', type=int, default=256)
    args = ap.parse_args()

    tok = stats(args.corpus, args.block_size) if args.stats \
        else ArmTokenizer.build(open(args.corpus).read())

    tok.save(args.out)
    print(f"\nSaved vocabulary ({tok.vocab_size} tokens) to {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
