# inspect_model.py
import argparse
import os
import json
import torch
from core.checkpoint import CheckpointManager
from core.model import GPTConfig
from core.banner import print_banner

print_banner("MyPT Inspector", "Model Checkpoint Analyzer")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Name of the model/checkpoint set (e.g. dante_gpt2, shakes_char)",
)
parser.add_argument(
    "--legacy_checkpoint",
    type=str,
    default=None,
    help="Optional: specific legacy checkpoint file (e.g. final.pt, latest.pt)",
)

args = parser.parse_args()

# Use checkpoint manager
ckpt_manager = CheckpointManager(args.model_name)

if not ckpt_manager.exists():
    raise FileNotFoundError(f"No checkpoint found for model '{args.model_name}'")

print(f"Inspecting model '{args.model_name}'\n")

# Detect format
if ckpt_manager.exists_new_format():
    print("=== FORMAT ===")
    print("New JSON-based format")
    print(f"Location: {ckpt_manager.checkpoint_dir}")
    print()
    
    # Load config from JSON
    config_path = os.path.join(ckpt_manager.checkpoint_dir, "config.json")
    config = GPTConfig.load_json(config_path)
    
    # Load tokenizer state
    tokenizer_path = os.path.join(ckpt_manager.checkpoint_dir, "tokenizer.json")
    tokenizer_state = None
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'r') as f:
            tokenizer_state = json.load(f)
    
    # Load training state
    training_state_path = os.path.join(ckpt_manager.checkpoint_dir, "training_state.json")
    step = None
    training_config = None
    if os.path.exists(training_state_path):
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
        step = training_state.get("step", None)
        training_config = training_state.get("training_config", None)

else:
    print("=== FORMAT ===")
    print("Legacy single-file format")
    
    # Determine which legacy file to use
    if args.legacy_checkpoint:
        legacy_file = args.legacy_checkpoint
    else:
        # Try to find a legacy file
        for filename in ["final.pt", "latest.pt"]:
            if ckpt_manager.exists_legacy_format(filename):
                legacy_file = filename
                break
    
    model_path = ckpt_manager.get_path(legacy_file)
    print(f"Location: {model_path}")
    print()
    
    # Load legacy checkpoint
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Legacy checkpoints contain dicts with config/tokenizer
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract info
    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)
    tokenizer_state = checkpoint.get("tokenizer", None)
    step = checkpoint.get("step", None)
    training_config = None  # Legacy format doesn't have this

# Now display the information
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

# Display training configuration if available
if training_config is not None:
    print("\nTraining hyperparameters:")
    print(f"  max_iters     : {training_config.get('max_iters', 'N/A')}")
    print(f"  eval_interval : {training_config.get('eval_interval', 'N/A')}")
    print(f"  eval_iters    : {training_config.get('eval_iters', 'N/A')}")
    print(f"  learning_rate : {training_config.get('learning_rate', 'N/A')}")
    print(f"  start_step    : {training_config.get('start_step', 'N/A')}")

print("\nInspection complete.")