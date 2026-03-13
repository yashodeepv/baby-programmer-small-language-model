"""
auto_train_v2.py - Curriculum-Based Distillation via Ollama (qwen3-coder)

Pipeline:
  1. Load the BabyProgrammer checkpoint (baby_programmer.pth).
  2. Read staged curriculum files from curriculum_configs/ (01_*.txt -> 05_*.txt).
  3. For each lesson prompt, ask qwen3-coder to generate a clean Q&A training example.
  4. Train on the ASSISTANT response only (masked loss — ignores the USER prompt).
  5. Save the checkpoint incrementally after every lesson.

Prerequisites:
  - Ollama running locally:  ollama serve
  - Qwen model pulled:       ollama pull qwen3-coder:30b   (or :8b for less VRAM)
  - baby_programmer.pth must exist (run slm_v3.py or slm_v4.py first)
  - curriculum_configs/ folder with *.txt lesson files
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import requests
import os
from pathlib import Path

from model import BabyProgrammer, load_checkpoint, save_checkpoint, device, BLOCK_SIZE

# --- Configuration ---
MODEL_PATH       = 'baby_programmer.pth'
CURRICULUM_DIR   = Path('curriculum_configs')
OLLAMA_URL       = 'http://localhost:11434/api/generate'
OLLAMA_MODEL     = 'qwen3-coder:30b'   # change to :8b if VRAM is tight
LEARNING_RATE    = 1e-4
EPOCHS_PER_FILE  = 1   # how many passes over each curriculum file


# --- Ollama Oracle ---

def ask_oracle(topic: str) -> str | None:
    """
    Ask qwen3-coder to generate one clean USER/ASSISTANT training pair
    for the given topic. Returns the raw text or None on failure.
    """
    system_prompt = (
        f"You are a Senior CS Professor creating a training example for a student learning Java.\n"
        f"Topic: {topic}\n\n"
        "Respond in exactly this format (no markdown, no extra text):\n"
        "USER: [one clear question about the topic]\n"
        "ASSISTANT: [the exact correct Java code or explanation, concise and accurate]"
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                'model':      OLLAMA_MODEL,
                'prompt':     system_prompt,
                'stream':     False,
                'keep_alive': -1,   # keep model loaded in VRAM between calls
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get('response', '').strip()
    except Exception as e:
        print(f"  [Oracle Error] {e}")
        return None


# --- Masked Training Step ---

def train_step(model: BabyProgrammer, optimizer, qa_text: str, stoi: dict) -> float:
    """
    Train on a Q&A pair, masking the USER prompt so the model only
    learns to predict the ASSISTANT answer tokens.
    """
    encode = lambda s: [stoi[c] for c in s if c in stoi]

    tokens = encode(qa_text)
    if len(tokens) < 10:
        return 0.0

    # Truncate to fit inside context window
    if len(tokens) > BLOCK_SIZE + 1:
        tokens = tokens[:BLOCK_SIZE + 1]

    # Find where the ASSISTANT answer starts
    assistant_tag = encode("ASSISTANT:")
    mask_start = len(tokens)  # default: no masking (shouldn't happen)
    for i in range(len(tokens) - len(assistant_tag)):
        if tokens[i: i + len(assistant_tag)] == assistant_tag:
            mask_start = i + len(assistant_tag)
            break

    if mask_start >= len(tokens) - 1:
        print("  [Warning] ASSISTANT: tag not found or answer is empty. Skipping.")
        return 0.0

    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([tokens[1:],  ], dtype=torch.long, device=device)

    model.train()
    logits, _ = model(x)   # (B, T, C) — no targets; we apply the mask manually

    B, T, C = logits.shape
    # Mask = 0 for the USER prompt, 1 for the ASSISTANT answer
    mask = torch.zeros(T, device=device)
    mask[mask_start:] = 1.0

    loss        = F.cross_entropy(logits.view(B * T, C), y.view(B * T), reduction='none')
    masked_loss = (loss * mask) .sum() / (mask.sum() + 1e-8)

    optimizer.zero_grad(set_to_none=True)
    masked_loss.backward()
    optimizer.step()

    return masked_loss.item()


# --- Main Pipeline ---

def run_curriculum_distillation():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Run slm_v3.py or slm_v4.py first.")
        return

    if not CURRICULUM_DIR.exists():
        print(f"Error: {CURRICULUM_DIR}/ folder not found.")
        print("Create it with lesson files named 01_*.txt, 02_*.txt, etc.")
        return

    # Load model and vocabulary from checkpoint
    print(f"Loading checkpoint from {MODEL_PATH}...")
    model, stoi, itos, vocab_size = load_checkpoint(MODEL_PATH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"Model loaded. Vocab size: {vocab_size}. Device: {device}")

    # Collect curriculum files in sorted order (01 -> 05)
    curriculum_files = sorted(CURRICULUM_DIR.glob('*.txt'))
    if not curriculum_files:
        print(f"No .txt files found in {CURRICULUM_DIR}/")
        return

    print(f"\nFound {len(curriculum_files)} curriculum stage(s):")
    for cf in curriculum_files:
        print(f"  {cf.name}")

    total_lessons = 0
    total_loss    = 0.0

    for stage_file in curriculum_files:
        prompts = [
            line.strip()
            for line in stage_file.read_text(encoding='utf-8').splitlines()
            if line.strip() and not line.startswith('#')
        ]
        print(f"\n{'='*60}")
        print(f"STAGE: {stage_file.name}  ({len(prompts)} lessons x {EPOCHS_PER_FILE} epoch(s))")
        print(f"{'='*60}")

        for epoch in range(EPOCHS_PER_FILE):
            for idx, prompt in enumerate(prompts):
                print(f"\n  Lesson {idx + 1}/{len(prompts)}: {prompt[:70]}...")

                qa_pair = ask_oracle(prompt)
                if not qa_pair:
                    print("  Skipped (Oracle returned nothing).")
                    continue

                # Show what Qwen generated (truncated)
                preview = qa_pair.replace('\n', ' ')[:120]
                print(f"  Oracle: {preview}...")

                loss = train_step(model, optimizer, qa_pair, stoi)
                print(f"  Loss: {loss:.4f}")

                total_lessons += 1
                total_loss    += loss

                # Save after every lesson so progress is never lost
                save_checkpoint(MODEL_PATH, model, stoi, itos, vocab_size)

    avg_loss = total_loss / max(total_lessons, 1)
    print(f"\nDistillation complete. {total_lessons} lessons processed. Avg loss: {avg_loss:.4f}")
    print(f"Updated checkpoint saved to {MODEL_PATH}")


if __name__ == '__main__':
    run_curriculum_distillation()
