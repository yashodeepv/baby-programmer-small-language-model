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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key     = nn.Linear(N_EMBD, head_size, bias=False)
        self.query   = nn.Linear(N_EMBD, head_size, bias=False)
        self.value   = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return self.dropout(wei) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


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
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size  = n_embd // n_head
        self.sa    = MultiHeadAttention(n_head, head_size)
        self.ffwd  = FeedForward(n_embd)
        self.ln1   = nn.LayerNorm(n_embd)
        self.ln2   = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BabyProgrammer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks  = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x       = self.blocks(tok_emb + pos_emb)
        logits  = self.lm_head(self.ln_f(x))

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            idx    = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# --- Checkpoint Utilities ---

def save_checkpoint(path, model, stoi, itos, vocab_size):
    torch.save({
        'model_state_dict': model.state_dict(),
        'stoi':       stoi,
        'itos':       itos,
        'vocab_size': vocab_size,
    }, path)


def load_checkpoint(path):
    """Returns (model, stoi, itos, vocab_size)."""
    ckpt       = torch.load(path, map_location=device)
    vocab_size = ckpt['vocab_size']
    model      = BabyProgrammer(vocab_size).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt['stoi'], ckpt['itos'], vocab_size


def bridge_weights(checkpoint_path, new_vocab_size, new_stoi):
    """
    Transfer weights from a checkpoint trained on a different vocabulary.
    Handles vocab resizing safely — used when switching between training datasets.
    """
    ckpt      = torch.load(checkpoint_path, map_location=device)
    old_stoi  = ckpt['stoi']
    old_state = ckpt['model_state_dict']

    model     = BabyProgrammer(new_vocab_size).to(device)
    new_state = model.state_dict()

    vocab_layers = {'token_embedding_table.weight', 'lm_head.weight', 'lm_head.bias'}

    for name, param in old_state.items():
        if name in vocab_layers:
            new_w = new_state[name]
            for char, old_idx in old_stoi.items():
                if char in new_stoi:
                    new_w[new_stoi[char]] = param[old_idx]
            new_state[name] = new_w
        else:
            new_state[name] = param

    model.load_state_dict(new_state)
    print(f"Bridge: mapped {len(old_stoi)} -> {new_vocab_size} vocab tokens.")
    return model
