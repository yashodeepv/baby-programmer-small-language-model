"""
train_corpus.py - Direct training on the Java Q&A corpus (no Ollama needed)

This is the fastest way to teach BabyProgrammer Java syntax and Q&A format.
Run this BEFORE auto_train_v2.py for best results.

The corpus (java_corpus.txt) contains ~150 handcrafted USER/ASSISTANT pairs
covering all 5 curriculum stages: types, control flow, methods, OOP, advanced.

Strategy:
  - Train many epochs on the corpus so the model deeply memorizes the patterns.
  - Lower LR than scratch training to avoid overwriting existing weights.
  - The corpus is small enough to fit in one batch — each step sees all of it.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os

from model import (
    BabyProgrammer, load_checkpoint, save_checkpoint,
    bridge_weights, device, BLOCK_SIZE
)

# --- Config ---
CORPUS_PATH    = 'java_corpus.txt'
MODEL_PATH     = 'baby_programmer.pth'
MAX_ITERS      = 20000   # epochs over the corpus; small data = many passes needed
EVAL_INTERVAL  = 1000
LEARNING_RATE  = 5e-4
BATCH_SIZE     = 32      # random windows sampled from corpus each step


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i     : i + block_size    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    result = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = []
        for _ in range(50):
            x, y = get_batch(data, BLOCK_SIZE, BATCH_SIZE)
            _, loss = model(x, y)
            losses.append(loss.item())
        result[split] = sum(losses) / len(losses)
    model.train()
    return result


def run():
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: {CORPUS_PATH} not found.")
        return

    print(f"Loading corpus: {CORPUS_PATH}")
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Corpus size: {len(text):,} characters")
    print(f"Sample: {repr(text[:120])}")

    # Build vocab from corpus
    chars      = sorted(set(text))
    vocab_size = len(chars)
    stoi       = {ch: i for i, ch in enumerate(chars)}
    itos       = {i: ch for i, ch in enumerate(chars)}
    encode     = lambda s: [stoi[c] for c in s if c in stoi]
    decode     = lambda l: ''.join([itos[i] for i in l])

    print(f"Corpus vocab: {vocab_size} unique chars")

    data = torch.tensor(encode(text), dtype=torch.long)
    n    = int(0.95 * len(data))   # 95/5 split — corpus is small, keep most for training
    train_data, val_data = data[:n], data[n:]

    print(f"Train tokens: {len(train_data):,}   Val tokens: {len(val_data):,}")

    # --- Load or initialize model ---
    if os.path.exists(MODEL_PATH):
        print(f"\nWarm start: loading {MODEL_PATH}...")
        try:
            model = bridge_weights(MODEL_PATH, vocab_size, stoi)
            print("Bilingual bridge applied (vocab may have changed).")
        except Exception as e:
            print(f"Bridge failed ({e}), starting fresh for this corpus.")
            model = BabyProgrammer(vocab_size).to(device)
    else:
        print("No checkpoint found — training from scratch.")
        model = BabyProgrammer(vocab_size).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M on {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop ---
    print(f"Starting corpus training: {MAX_ITERS} steps, LR={LEARNING_RATE}")
    start = time.time()

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = (time.time() - start) / 60
            print(f"Step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.1f}m")
            save_checkpoint(MODEL_PATH, model, stoi, itos, vocab_size)

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final save
    save_checkpoint(MODEL_PATH, model, stoi, itos, vocab_size)
    total_mins = (time.time() - start) / 60
    print(f"\nCorpus training complete in {total_mins:.1f} minutes.")
    print(f"Checkpoint saved to {MODEL_PATH}")

    # --- Quick sanity generation ---
    print("\n--- Quick generation test ---")
    model.eval()
    test_prompts = [
        "USER: How do you declare an int named score?\nASSISTANT:",
        "USER: How do you write a for loop from 1 to 10?\nASSISTANT:",
        "USER: How does inheritance work in Java?\nASSISTANT:",
    ]
    for prompt in test_prompts:
        tokens = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            out    = model.generate(tokens, max_new_tokens=80, temperature=0.7)
            answer = decode(out[0].tolist())[len(prompt):]
        print(f"\nQ: {prompt.split(chr(10))[0].replace('USER: ', '')}")
        print(f"A: {answer.strip()[:150]}")


if __name__ == '__main__':
    run()
