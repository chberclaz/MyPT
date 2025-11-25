import torch
from model import GPT
from tokenizer import Tokenizer
from model import GPTConfig  # if needed

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model + tokenizer state
model, tokenizer_state, step, optim_state = GPT.load("checkpoints/gpt_step_140.pt", map_location=device)

# Recreate tokenizer from saved state
config = GPTConfig()  # device/vocab_size are in config, but token_kind/ chars come from state
tokenizer = Tokenizer.from_state(config, tokenizer_state)

# Prompt
prompt = "Die Nacht"
if tokenizer.token_kind == 'gpt2':
    ctx_ids = tokenizer.encode(prompt)
else:
    ctx_ids = tokenizer.encode(prompt)

idx = torch.tensor([ctx_ids], dtype=torch.long, device=model.config.device)
print(f"Prompt: {iter}")
print(f"Start generating...")

# Generate
out = model.generate(idx, max_new_tokens=50)
decoded = tokenizer.decode(out[0].tolist())
print(decoded)
print(f"!!! Finished !!!")
