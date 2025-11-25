# inspect_model.py
import os
import argparse
import torch

from model import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model/checkpoint set (e.g. dante_gpt2, shakes_char)",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="final.pt",
    help="Which checkpoint file to inspect (e.g. final.pt or latest.pt)",
)

args = parser.parse_args()

checkpoint_dir = os.path.join("checkpoints", args.model_name)
model_path = os.path.join(checkpoint_dir, args.checkpoint)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Checkpoint not found: {model_path}")

print(f"Inspecting model '{args.model_name}' from: {model_path}\n")

# Load model + tokenizer state (no need for optimizer here)
model, tokenizer_state, step, _ = GPT.load(model_path, map_location=device)

config = model.config

print("=== CONFIG ===")
print(config)
print()

print("=== TOKENIZER ===")
if tokenizer_state is None:
    print("No tokenizer state stored in this checkpoint.")
else:
    token_kind = tokenizer_state.get("token_kind", None)
    chars = tokenizer_state.get("chars", None)

    print(f"kind      : {token_kind}")
    if token_kind == "char":
        if chars is not None:
            print(f"vocab size: {len(chars)} (char-level)")
            # If you want, you can print a subset of chars:
            print("sample chars:", "".join(chars[:50]))
        else:
            print("WARNING: char tokenizer with no 'chars' list saved.")
    elif token_kind == "gpt2":
        print("vocab     : GPT-2 BPE (fixed ~50k tokens)")
    else:
        print("vocab     : unknown tokenizer kind")

print()

print("=== TRAINING STATE ===")
if step is None:
    print("No training step info stored.")
else:
    print(f"Last recorded training step: {step}")

print("\nInspection complete.")