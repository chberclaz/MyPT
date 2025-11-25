import torch
import os
import argparse
from model import GPT, GPTConfig
from tokenizer import Tokenizer  # if needed

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="default",
                    help="Name of the model/checkpoint set to load (e.g. dante, shakespeare)")
parser.add_argument("--checkpoint", type=str, default="latest.pt",
                    help="Which checkpoint file to load inside the model's folder (e.g. final.pt or latest.pt)")
parser.add_argument("--prompt", type=str, default="Die Nacht",
                    help="Prompt text to start generation from")
parser.add_argument("--max_new_tokens", type=int, default=100,
                    help="Number of tokens to generate")
args = parser.parse_args()

checkpoint_dir = os.path.join("checkpoints", args.model_name)
model_path = os.path.join(checkpoint_dir, args.checkpoint)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Checkpoint not found: {model_path}")

print(f"Loading model '{args.model_name}' from: {model_path}")


# Load model + tokenizer state
model, tokenizer_state, step, optim_state = GPT.load(model_path, map_location=device)

# Recreate tokenizer from saved state
config = model.config  # device/vocab_size are in config, but token_kind/ chars come from state

# rebuild tokenizer from saved state
if tokenizer_state is None:
    raise ValueError("No tokenizer state found in checkpoint.")
tokenizer = Tokenizer.from_state(config, tokenizer_state)
print(f"Tokenizer kind: {tokenizer.token_kind}")
print(f"Prompt: {args.prompt}")

# Prompt
ctx_ids = tokenizer.encode(args.prompt)
idx = torch.tensor([ctx_ids], dtype=torch.long, device=model.config.device)

print(f"Start generating...")

# Generate
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=args.max_new_tokens)

decoded = tokenizer.decode(out[0].tolist())
print(decoded)
print(f"!!! Finished !!!")
