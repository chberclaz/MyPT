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
#torch.manual_seed(1337)

#hyperparameters
max_iters=10000
eval_interval= 600
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
model = GPT(myconfig)
m=model.to(myconfig.device)

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("---------- model loaded ----------")
print(myconfig)
print("--------- start training ---------")
# Load our model with data and train it on itself
for iter in range(max_iters):
    #every once in a while output expected loss on train and val sets
    if iter % eval_interval== 0:
        ct = datetime.datetime.now()
        print(iter, " : ", ct)        
                
    # sample a batch of data
    xb, yb = get_batch(myconfig, 'train')
    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"step {iter}: train loss {losses['train']:.4f}")

print("----- start generation ------")
#generate predictiv output
#context= torch.zeros((1,1), dtype=torch.long, device=device) # set start at zero(zero represents a space or newline in our data set )
context= (torch.tensor(tokenizer.encode("Die Nacht"), dtype=torch.long, device=myconfig.device)[None, ...])
resulti=m.generate(context, max_new_tokens=5000)
# to decode results, it must be converted back to a list
decodes=tokenizer.decode(resulti[0].tolist())
# output decoded result
print(decodes)      