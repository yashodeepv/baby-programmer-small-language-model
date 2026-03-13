"""
test_baby.py - BabyProgrammer Evaluation Suite

Tests the model across all 5 curriculum stages and produces a scored report.
Each question has a set of expected keywords/tokens. The model passes a question
if its generated answer contains at least MIN_KEYWORDS_TO_PASS of them.

Usage:
    python test_baby.py
    python test_baby.py --temp 0.7   # adjust generation temperature (default 0.8)
    python test_baby.py --tokens 200 # adjust max generated tokens (default 150)
"""

import torch
import argparse
import os
import sys

from model import load_checkpoint, device, BLOCK_SIZE

# --- Config ---
MODEL_PATH         = 'baby_programmer.pth'
MIN_KEYWORDS_PASS  = 2   # how many keywords must appear for a PASS verdict


# --- Test Suite ---
# Format: (question_prompt, [keywords that should appear in the answer])
# Keywords are checked case-insensitively. A single keyword may be a phrase.

TESTS = {
    "Stage 1: Variables & Types": [
        (
            "USER: How do you declare an int variable named score with value 100 in Java?\nASSISTANT:",
            ["int", "score", "100", ";"]
        ),
        (
            "USER: How do you declare a double variable named price set to 9.99?\nASSISTANT:",
            ["double", "price", "9.99", ";"]
        ),
        (
            "USER: How do you declare a String variable named name in Java?\nASSISTANT:",
            ["String", "name", "=", ";"]
        ),
        (
            "USER: What operator checks if an integer n is even in Java?\nASSISTANT:",
            ["%", "2", "==", "0"]
        ),
        (
            "USER: How do you declare a boolean named isActive set to true in Java?\nASSISTANT:",
            ["boolean", "isActive", "true", ";"]
        ),
    ],

    "Stage 2: Control Flow": [
        (
            "USER: Write an if-else block that prints Adult if age >= 18 else Minor.\nASSISTANT:",
            ["if", "age", "18", "else"]
        ),
        (
            "USER: Write a for loop that prints numbers 1 to 10 in Java.\nASSISTANT:",
            ["for", "int", "10", "System.out"]
        ),
        (
            "USER: Write a while loop that counts down from 10 to 1 in Java.\nASSISTANT:",
            ["while", "10", "--", "System.out"]
        ),
        (
            "USER: How do you use break inside a loop in Java?\nASSISTANT:",
            ["break", "for", "while", ";"]
        ),
        (
            "USER: Write a switch statement for an integer day with cases 1 and 2 in Java.\nASSISTANT:",
            ["switch", "case", "break", "day"]
        ),
    ],

    "Stage 3: Methods & Arrays": [
        (
            "USER: Write a Java method named square that takes an int and returns its square.\nASSISTANT:",
            ["int", "square", "return", "*"]
        ),
        (
            "USER: How do you declare an int array of size 5 in Java?\nASSISTANT:",
            ["int", "[", "]", "5"]
        ),
        (
            "USER: Write a Java method to find the maximum value in an int array.\nASSISTANT:",
            ["int", "max", "for", "return"]
        ),
        (
            "USER: How do you get the length of an array named data in Java?\nASSISTANT:",
            ["data", "length", "."]
        ),
        (
            "USER: Write a Java method named isPrime that returns true if a number is prime.\nASSISTANT:",
            ["boolean", "isPrime", "for", "return"]
        ),
    ],

    "Stage 4: OOP": [
        (
            "USER: Write a simple Java class Animal with a private String field name and a constructor.\nASSISTANT:",
            ["class", "Animal", "private", "String"]
        ),
        (
            "USER: How does a Dog class extend an Animal class in Java?\nASSISTANT:",
            ["class", "Dog", "extends", "Animal"]
        ),
        (
            "USER: What keyword is used to call a parent class constructor in Java?\nASSISTANT:",
            ["super", "(", ")"]
        ),
        (
            "USER: How do you define an interface named Drawable in Java?\nASSISTANT:",
            ["interface", "Drawable", "void", "draw"]
        ),
        (
            "USER: What is encapsulation in Java? Show a short example.\nASSISTANT:",
            ["private", "public", "get", "set"]
        ),
    ],

    "Stage 5: Advanced Topics": [
        (
            "USER: How do you add and iterate over elements in a Java ArrayList?\nASSISTANT:",
            ["ArrayList", "add", "for", "import"]
        ),
        (
            "USER: How do you use a HashMap to map names to grades in Java?\nASSISTANT:",
            ["HashMap", "put", "get", "String"]
        ),
        (
            "USER: Write a Java recursive method to compute the nth Fibonacci number.\nASSISTANT:",
            ["return", "fibonacci", "n", "-"]
        ),
        (
            "USER: How do you handle a NumberFormatException in Java?\nASSISTANT:",
            ["try", "catch", "NumberFormatException", "parseInt"]
        ),
        (
            "USER: What is the difference between == and .equals() for Strings in Java?\nASSISTANT:",
            ["equals", "==", "String", "reference"]
        ),
    ],
}


# --- Engine ---

def run_tests(max_tokens: int, temperature: float):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Train the model first.")
        sys.exit(1)

    print(f"Loading {MODEL_PATH} on {device}...")
    model, stoi, itos, vocab_size = load_checkpoint(MODEL_PATH)
    model.eval()

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])

    unknown_chars = set()
    total_q  = 0
    total_ok = 0
    stage_results = {}

    sep = "-" * 70

    for stage_name, questions in TESTS.items():
        print(f"\n{'='*70}")
        print(f"  {stage_name}")
        print(f"{'='*70}")

        stage_pass = 0

        for prompt, keywords in questions:
            total_q += 1

            # Track any chars not in vocab
            for ch in prompt:
                if ch not in stoi:
                    unknown_chars.add(ch)

            tokens = encode(prompt)
            if not tokens:
                print(f"\n  Q: {prompt[:60]}...")
                print(f"  [SKIP] Prompt encodes to empty (all chars OOV).")
                continue

            idx = torch.tensor([tokens], dtype=torch.long, device=device)

            with torch.no_grad():
                out     = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
                full    = decode(out[0].tolist())
                # Extract only the generated part after the prompt
                answer  = full[len(prompt):]

            # Score: count how many keywords appear in the answer
            answer_lower = answer.lower()
            hits = [kw for kw in keywords if kw.lower() in answer_lower]
            passed = len(hits) >= MIN_KEYWORDS_PASS

            if passed:
                stage_pass += 1
                total_ok   += 1
                verdict = "PASS"
            else:
                verdict = "FAIL"

            # Display
            q_short = prompt.replace("USER: ", "").replace("\nASSISTANT:", "").strip()
            print(f"\n  {sep}")
            print(f"  Q: {q_short}")
            print(f"  A: {answer.strip()[:200]}")
            print(f"  Keywords expected : {keywords}")
            print(f"  Keywords found    : {hits}")
            print(f"  Result            : [{verdict}]  ({len(hits)}/{len(keywords)} keywords)")

        stage_score = f"{stage_pass}/{len(questions)}"
        stage_results[stage_name] = (stage_pass, len(questions))
        print(f"\n  Stage Score: {stage_score}")

    # --- Final Report ---
    print(f"\n\n{'#'*70}")
    print(f"  FINAL REPORT")
    print(f"{'#'*70}")
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Device   : {device}")
    print(f"  Vocab    : {vocab_size} chars")
    print(f"  Temp     : {temperature}   Max tokens: {max_tokens}")
    if unknown_chars:
        print(f"  OOV chars: {sorted(unknown_chars)}  (these were skipped in prompts)")
    print()
    for stage, (ok, total) in stage_results.items():
        bar    = "#" * ok + "." * (total - ok)
        pct    = 100 * ok // total if total else 0
        print(f"  [{bar}]  {ok}/{total}  ({pct:3d}%)  {stage}")
    print()
    overall_pct = 100 * total_ok // total_q if total_q else 0
    print(f"  OVERALL: {total_ok}/{total_q} questions passed  ({overall_pct}%)")

    if overall_pct >= 80:
        print("  Grade: A  -- Baby is a solid programmer!")
    elif overall_pct >= 60:
        print("  Grade: B  -- Baby knows the basics, needs more distillation.")
    elif overall_pct >= 40:
        print("  Grade: C  -- Baby is learning. Run more Qwen distillation rounds.")
    elif overall_pct >= 20:
        print("  Grade: D  -- Baby needs more training data and Ollama sessions.")
    else:
        print("  Grade: F  -- Baby is just babbling. Continue pre-training first.")

    print(f"{'#'*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test BabyProgrammer SLM')
    parser.add_argument('--temp',   type=float, default=0.8,
                        help='Generation temperature (default 0.8). Lower = more focused.')
    parser.add_argument('--tokens', type=int,   default=150,
                        help='Max tokens to generate per answer (default 150).')
    args = parser.parse_args()

    run_tests(max_tokens=args.tokens, temperature=args.temp)
