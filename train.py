# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import datetime
import os
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPT, GPTConfig 
from loader import GPTDataLoader
from tokenizer import Tokenizer
#torch.manual_seed(1337)


# -----------------------------------------------------------------------------------
# argument parsing: allow different models / datasets
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="default",
                    help="Name of the model/checkpoint set (e.g. dante, shakespeare)")
parser.add_argument("--input_file", type=str, default="input_dante.txt",
                    help="Path to training text file")
parser.add_argument("--tokenization", type=str, default="gpt2",
                    choices=["gpt2", "char"],
                    help="Tokenizer type: gpt2 or char")
parser.add_argument("--init_from_model", type=str, default=None,
                    help="Optional: model_name to initialize weights from (e.g. dante_base)")
parser.add_argument("--max_iters", type=int, default=1000)
parser.add_argument("--eval_interval", type=int, default=50)
parser.add_argument("--eval_iters", type=int, default=200)
parser.add_argument("--learning_rate", type=float, default=3e-4)

# model hyperparams (per model)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--block_size", type=int, default=256)
parser.add_argument("--n_embd", type=int, default=384)
parser.add_argument("--n_head", type=int, default=6)
parser.add_argument("--n_layer", type=int, default=6)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--bias", action="store_true", help="Use bias in Linear/LayerNorm like GPT-2")

args = parser.parse_args()

# ----------------- CONFIG PER MODEL -----------------
# initial config from CLI (used only for first training run)
config = GPTConfig(
    batch_size=args.batch_size,
    block_size=args.block_size,
    vocab_size=50304,            # will adjust for char-level later
    n_embd=args.n_embd,
    n_head=args.n_head,
    n_layer=args.n_layer,
    dropout=args.dropout,
    bias=args.bias,
)
device = config.device
print("Initial CLI config:")
print(config)

data_loader=GPTDataLoader(config)
tokenizer=Tokenizer(config,args.tokenization) 

#for testing purposes, smaler sample
#myconfig.batch_size=20

# folder per model name
checkpoint_dir = os.path.join("checkpoints", args.model_name)
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")

print(f"Using model name: {args.model_name}")
print(f"Input file: {args.input_file}")
print(f"Tokenization: {args.tokenization}")
print(f"Checkpoint dir: {checkpoint_dir}")


max_iters = args.max_iters
eval_interval = args.eval_interval
eval_iters = args.eval_iters
learning_rate = args.learning_rate

# -----------------------------------------------------------------------------------------------------------------------
# data loding
def get_batch(config, split):
    #generate am small batch of data on inputs x and targets y
    batch_data= train_data if split=='train' else val_data
    ix = torch.randint(len(batch_data) -config.block_size, (config.batch_size,))
    x=torch.stack([batch_data[i:i+config.block_size] for i in ix])
    y=torch.stack([batch_data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(config, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# input: textfile
# plain text from a author (in our case sharespeare)
text=data_loader.read_textData(args.input_file)
# If using char-level tokenization, build vocab from full corpus
if args.tokenization == 'char':
    if args.init_from_model is None:
        # fresh char model: build vocab from this dataset
        tokenizer.build_char_vocab(text)
        config.vocab_size = len(tokenizer.chars)
    else:
        # init_from_model + char:
        # vocab will be taken from base_tokenizer_state in the init_from_path branch.
        # So we do NOT build a new vocab here.
        pass
else:
    config.vocab_size = 50304

# tokenization:
tokens=tokenizer.encode(text)
#load tokens into tensor
data= torch.tensor(tokens, dtype=torch.long)
# ---------------------------------------------------------------------------------------------------------------------
# splitt Innput Data in 2 set: trainig and validation Data
n = int(0.9*len(data))  #first 90% will be training Data
train_data= data[:n]
val_data = data[n:]

# ----------------- MODEL LOADING / RESUME / INIT-FROM -----------------
resume_from = None
init_from_path = None

if os.path.exists(checkpoint_path):
    # Case 1: resume this model
    resume_from = checkpoint_path
    print(f"Found checkpoint for this model at {checkpoint_path}, will resume training.")

elif args.init_from_model is not None:
    # Case 2: init from another model's checkpoint
    base_dir = os.path.join("checkpoints", args.init_from_model)
    base_ckpt = os.path.join(base_dir, "final.pt")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"--init_from_model given as '{args.init_from_model}', "
                                f"but {base_ckpt} does not exist.")
    init_from_path = base_ckpt
    print(f"Initializing weights from base model '{args.init_from_model}' at {base_ckpt}")
else:
    # Case 3: fresh
    print("No checkpoint for this model and no --init_from_model; starting from scratch.")


if resume_from is not None:
    # RESUME TRAINING
    model, tokenizer_state, start_step, optim_state = GPT.load(resume_from, map_location=device)
    config = model.config  # override CLI config
    print("Loaded config from this model checkpoint:")
    print(config)

    if tokenizer_state is not None:
        tokenizer = Tokenizer.from_state(config, tokenizer_state)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

elif init_from_path is not None:
    # INIT FROM ANOTHER MODEL (FINE-TUNE)
    model, base_tokenizer_state, _, _ = GPT.load(init_from_path, map_location=device)
    config = model.config  # use base model's config
    print("Loaded base model config:")
    print(config)


    if base_tokenizer_state is None:
        raise ValueError(
            f"Base model '{args.init_from_model}' has no tokenizer state saved. "
            "Cannot safely fine-tune from it."
        )

    base_kind = base_tokenizer_state.get("token_kind", None)
    if base_kind is None:
        raise ValueError(
            f"Base model '{args.init_from_model}' has unknown tokenizer kind."
        )

    # ---- VALIDATION: tokenization kind must match ----
    if base_kind != args.tokenization:
        raise ValueError(
            f"Tokenization mismatch when fine-tuning:\n"
            f"  base model '{args.init_from_model}' uses: {base_kind}\n"
            f"  this run requested: {args.tokenization}\n"
            f"For fine-tuning you must use the same tokenizer kind."
        )

    # For GPT-2: we're good, vocab is fixed
    if base_kind == "gpt2":
        # keep tokenizer as-is from this run, it already uses gpt2
        # (you don't need base_tokenizer_state except for sanity)
        pass

    # For char: this is trickier, see below
    if base_kind == "char":
        base_chars = base_tokenizer_state.get("chars", None)
        if base_chars is None:
            raise ValueError(
                f"Base char-level model '{args.init_from_model}' has no 'chars' list saved."
            )
        tokenizer.chars = base_chars
        config.vocab_size = len(base_chars)
        print(f"Using base model's char vocabulary of size {len(base_chars)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0  # new training run for this specialized model

else:
    # FRESH MODEL
    print("Using CLI config to initialize new model:")
    print(config)

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0



#m = model  # keep your alias if you still use it later
print("---------- model ready ----------")
print(config)
print("--------- start training ---------")
# Load our model with data and train it on itself
os.makedirs("checkpoints", exist_ok=True)


for iter in range(start_step, max_iters):
    #every once in a while output expected loss on train and val sets
    if iter % eval_interval == 0:
        ct = datetime.datetime.now()
        print(iter, " : ", ct)

        # OPTIONAL: quick eval + checkpoint
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        tokenizer_state = tokenizer.get_state()
        model.save(
            checkpoint_path,
            tokenizer_state=tokenizer_state,
            step=iter,
            optimizer_state=optimizer.state_dict(),
        )
    
    # sample a batch of data
    xb, yb = get_batch(config, 'train')
    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"final step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


final_path = os.path.join(checkpoint_dir, "final.pt")
print("----- saving final model ------")
tokenizer_state = tokenizer.get_state()
model.save(
    final_path,
    tokenizer_state=tokenizer_state,
    step=iter,
    optimizer_state=optimizer.state_dict(),
)
print(f"Training finished. Final model at: {final_path}")