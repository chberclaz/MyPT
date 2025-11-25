# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import datetime;
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig 
from loader import GPTDataLoader
from tokenizer import Tokenizer
import os

#torch.manual_seed(1337)

#hyperparameters
max_iters=1000
eval_interval= 20
learning_rate= 3e-4
device= 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
tokenization ='gpt2'
#tokenization ='char'

myconfig=GPTConfig()
mdl=GPTDataLoader(myconfig)
tokenizer=Tokenizer(myconfig,tokenization) 

#for testing purposes, smaler sample
#myconfig.batch_size=20

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
            X, Y = get_batch(myconfig, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# input: textfile
# plain text from a author (in our case sharespeare)
text=mdl.read_textData('input_dante.txt')
# If using char-level tokenization, build vocab from full corpus
if tokenization == 'char':
    tokenizer.build_char_vocab(text)
# tokenization:
tokens=tokenizer.encode(text)
#load tokens into tensor
data= torch.tensor(tokens, dtype=torch.long)
# ---------------------------------------------------------------------------------------------------------------------
# splitt Innput Data in 2 set: trainig and validation Data
n = int(0.9*len(data))  #first 90% will be training Data
train_data= data[:n]
val_data = data[n:]

# ----------------------------------------------------------------------------------------------------------------------
# start of model loading 
# model = BigramLanguageModel(vocab_size)
checkpoint_path = "checkpoints/gpt_step_100.pt"  # or whatever name you like

if os.path.exists(checkpoint_path):
    print(f"Found checkpoint at {checkpoint_path}, loading...")
    model, tokenizer_state, start_step, optim_state = GPT.load(checkpoint_path, map_location=myconfig.device)

    # Rebuild tokenizer from saved state
    tokenizer = Tokenizer.from_state(myconfig, tokenizer_state)

    # Build optimizer and load its state
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

else:
    print("No checkpoint found, starting fresh training...")
    model = GPT(myconfig).to(myconfig.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    start_step = 0

m = model  # keep your alias if you still use it later
print(f"Device: {device}")
print("---------- model ready ----------")
print(myconfig)
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
        model.save(f"checkpoints/gpt_step_{iter}.pt",
                   tokenizer_state=tokenizer_state,
                   step=iter)
    
    # sample a batch of data
    xb, yb = get_batch(myconfig, 'train')
    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"final step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

print("----- saving final model ------")
tokenizer_state = tokenizer.get_state()
model.save(
    "checkpoints/gpt_final.pt",
    tokenizer_state=tokenizer_state,
    step=iter,
    optimizer_state=optimizer.state_dict(),
)