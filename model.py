"""
model.py - BabyProgrammer SLM Architecture

Single source of truth for the model definition.
All training, inference, and distillation scripts import from here.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Fixed Architecture Spec (10.91M params) ---
# DO NOT change these between training stages or checkpoints will be incompatible.
N_EMBD     = 384
N_HEAD     = 6
N_LAYER    = 6
BLOCK_SIZE = 256
DROPOUT    = 0.2


def _pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():   # Apple Silicon
        return 'mps'
    return 'cpu'


device = _pick_device()


class CausalSelfAttention(nn.Module):
    """Fused multi-head causal self-attention.

    Replaces the previous per-head nn.Module list. That version ran 3 small
    matmuls per head (18 per layer) inside a Python loop; this does one fused
    QKV projection and hands the attention itself to PyTorch's kernel. Same
    math, far fewer dispatches -- which is what dominates on an Apple GPU.

    scale is head_size ** -0.5. An earlier version used n_embd ** -0.5, which
    divided the logits by 19.6 instead of 8 and flattened every softmax.
    """

    def __init__(self, n_head, block_size=BLOCK_SIZE, n_embd=N_EMBD):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd    = n_embd
        self.n_head    = n_head
        self.head_size = n_embd // n_head
        self.qkv       = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj      = nn.Linear(n_embd, n_embd)
        self.dropout   = nn.Dropout(DROPOUT)
        self.attn_p    = DROPOUT
        self.scale     = self.head_size ** -0.5

    def forward(self, x, past=None, use_cache=False):
        """past: (k, v) from earlier positions, shaped (B, n_head, T_past, hs).

        Causal masking means an earlier position's k and v never change once
        computed, so regenerating them every step is pure waste. With the cache
        only the newest token is projected and appended.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        shape   = lambda t: t.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q, k, v = shape(q), shape(k), shape(v)

        if past is not None:
            if T != 1:
                raise ValueError('cached attention expects one new token at a time')
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)

        # A single new query may attend to every cached key -- they are all in
        # its past already, so no triangular mask is needed (or valid) here.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(past is None and T > 1), scale=self.scale,
            dropout_p=self.attn_p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(y))
        return (out, (k, v)) if use_cache else out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size=BLOCK_SIZE):
        super().__init__()
        self.sa    = CausalSelfAttention(n_head, block_size, n_embd)
        self.ffwd  = FeedForward(n_embd)
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)

    def forward(self, x, past=None, use_cache=False):
        if use_cache:
            delta, present = self.sa(self.ln1(x), past, use_cache=True)
            x = x + delta
            x = x + self.ffwd(self.ln2(x))
            return x, present
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BabyProgrammer(nn.Module):
    """Dimensions are parameters so Phase 4 can sweep capacity. They default to
    the original spec, and every checkpoint records the ones it was built with,
    so an old file still reconstructs its own architecture."""

    def __init__(self, vocab_size, block_size=BLOCK_SIZE,
                 n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER):
        super().__init__()
        self.block_size = block_size
        self.dims = dict(n_embd=n_embd, n_head=n_head, n_layer=n_layer)
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # ModuleList rather than Sequential so per-block caches can be threaded
        # through. Parameter names are identical, so old checkpoints still load.
        self.blocks  = nn.ModuleList([Block(n_embd, n_head, block_size)
                                      for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, past=None, pos_offset=0, use_cache=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(pos_offset, pos_offset + T, device=idx.device))
        x = tok_emb + pos_emb

        presents = [] if use_cache else None
        for i, blk in enumerate(self.blocks):
            if use_cache:
                x, pres = blk(x, None if past is None else past[i], use_cache=True)
                presents.append(pres)
            else:
                x = blk(x)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            _, _, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return (logits, loss, presents) if use_cache else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 greedy=False, stop_token=None):
        """Sample a continuation.

        greedy=True takes the argmax instead of sampling -- used by the
        execution eval, where a stochastic answer would make the pass rate
        depend on luck rather than on what the model actually learned.
        stop_token halts once every row in the batch has emitted it.
        """
        self.eval()
        done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
        past, pos, chunk = None, 0, idx
        for _ in range(max_new_tokens):
            # Positions are learned per slot, so once the sequence reaches
            # block_size the cache is dropped and the last window is recomputed
            # -- correctness first, and it rarely fires since examples fit.
            if pos + chunk.shape[1] > self.block_size:
                past, pos, chunk = None, 0, idx[:, -self.block_size:]
            logits, _, past = self(chunk, past=past, pos_offset=pos, use_cache=True)
            pos      += chunk.shape[1]
            logits    = logits[:, -1, :]
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt   = torch.multinomial(probs, num_samples=1)
            idx   = torch.cat((idx, nxt), dim=1)
            chunk = nxt                       # only the new token from here on
            if stop_token is not None:
                done |= (nxt.squeeze(1) == stop_token)
                if bool(done.all()):
                    break
        return idx


# --- Checkpoint Utilities ---

def save_checkpoint(path, model, stoi, itos, vocab_size):
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi':       stoi,
        'itos':       itos,
        'vocab_size': vocab_size,
        'block_size': getattr(model, 'block_size', BLOCK_SIZE),
        'dims':       getattr(model, 'dims',
                              dict(n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER)),
    }, path)


def load_checkpoint(path):
    """Returns (model, stoi, itos, vocab_size).

    Checkpoints written before block_size was configurable predate the key and
    are all 256, so that is the fallback.
    """
    ckpt       = torch.load(path, map_location=device)
    vocab_size = ckpt['vocab_size']
    state      = ckpt['model_state_dict']

    # Pre-fusion checkpoints store per-head key/query/value and were trained
    # with the n_embd scaling; keep both so old models behave identically.
    dims  = ckpt.get('dims', dict(n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER))
    model = BabyProgrammer(vocab_size, ckpt.get('block_size', BLOCK_SIZE),
                           **dims).to(device)
    model.load_state_dict(state)
    return model, ckpt['stoi'], ckpt['itos'], vocab_size
