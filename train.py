# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# finding out how many different charactes and what kind will be used as input
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8')as f:
    text=f.read()
# length=len(text)
chars=sorted(list(set(text)))
vocab_size=len(chars)
# print('',join(chars))
# print(vocab_size)


# --------------------------------------------------------------------------------------------------------------------
# first we need strategy to tokenize input 
# (charactes to integers for us)
stoi = {ch:i for i, ch in enumerate(chars) }
itos = {i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encodeer: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string

print(encode("Blub gud"))
print(decode(encode("Blub gud")))
# let's encode the entire text dataset and store it into a torch.Tensor
# encode our whole text and wrap it in a torch tensor
data= torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000]) #the 1000 characters we looked at earlier will look to the GPT like this

# ---------------------------------------------------------------------------------------------------------------------
# splitt Innput Data in 2 set: trainig and validation Data
n = int(0.9*len(data))  #first 90% will be training Data
train_data= data[:n]
val_data = data[n:]

# ----------------------------------------------------------------------------------------------------------------------
# Data Loader: batches of chunks of data (start at 14:31)
# create data Batches:
batch_size = 4  # how many independent sequences will we process in parallel
block_size = 8  # what is the maximalum context length for predictions
train_data[:block_size+1]

# exampel of use of training Block sizes:
# print(train_data[:block_size+1])
# x= train_data[:block_size]
# y= train_data[1:block_size+1]
# for t in range(block_size):
#     context= x[:t+1]
#     target = y[t]
#     print(f"when input is {context} the target is: {target}")

def get_batch(splitt):
    #generate am small batch of data on inputs x and targets y
    batch_data= train_data if splitt=='train' else val_data
    ix = torch.randint(len(batch_data) -block_size, (batch_size,))
    x=torch.stack([batch_data[i:i+block_size] for i in ix])
    y=torch.stack([batch_data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# print('input:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)
# print('----------------')
# for b in range(batch_size): #batch dimension
#     for t in range(block_size):     #time dimension
#         context=xb[b, :t+1]
#         target= yb[b,t]
#         print(f"when input is {context.tolist()} the target is: {target}")

# feed data batch to simple Language Model "Bigram"
# see makemore video series of andrej for more informations

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C) --> batch by time by chanel (chanel = vocab_size)

        if targets is None:
            loss= None
        else:
            # reshape arraw from 3s to 2d to conform cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current  context
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (b,C)
            # applay softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next= torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running squence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

inputx= torch.zeros((1,1), dtype=torch.long) # set start at zero(zero represents a space or newline in our data set )
resultx=m.generate(inputx, max_new_tokens=100)

# to decode results, it must be converted back to a list
decodex=decode(resultx[0].tolist())
print(decodex)
