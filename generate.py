import torch
import os
import argparse
from model import GPT, GPTConfig

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

def generate_answer(model, question: str, max_new_tokens=100):
    # Format Q&A prompt
    prompt = f"<Question> {question} </Question>\n<Answer> "       
    # Encode
    #ctx = model.encode(prompt)
    #idx = torch.tensor([ctx], dtype=torch.long, device=model.config.device)
    # Generate continuation
    out = model.generate(prompt, max_new_tokens=max_new_tokens)
    # Decode the output
    decoded = model.decode(out[0].tolist())
    # Strip everything before <Answer>
    answer = decoded.split("<Answer>")[-1]
    # Stop at end tag if model generated it
    answer = answer.split("</Answer>")[0]
    return answer.strip()

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Checkpoint not found: {model_path}")

print(f"Loading model '{args.model_name}' from: {model_path}")


# Load model (+ tokenizer is attached)
model, tokenizer_state, step, optim_state = GPT.load(model_path, map_location=device)
config = model.config
if model.tokenizer is None and tokenizer_state is None:
    raise ValueError("No tokenizer available. Check your checkpoint.")
print(f"Tokenizer kind: {getattr(model.tokenizer, 'token_kind', 'unknown')}")
print(f"Prompt: {args.prompt}")

# Prompt
#ctx_ids = model.encode(args.prompt)
#idx = torch.tensor([ctx_ids], dtype=torch.long, device=model.config.device)

print(f"Start generating...")

# Generate
with torch.no_grad():
    out = model.generate(args.prompt, max_new_tokens=args.max_new_tokens)

decoded = model.decode(out[0].tolist())
print(decoded)
print(f"!!! Finished !!!")



# Q&A prototype
print(f"Generating answer...")
answer = generate_answer(model, "How far is the moon?", 150)
print(answer)
print(f"!!! Finished !!!")

