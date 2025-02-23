# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
import torch.nn as nn
from torch.nn import functional as F
#torch.manual_seed(1337)

#hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel
block_size = 192  # what is the maximalum context length for predictions
max_iters=5000
eval_interval= 500
lerning_rate= 3e-4
device= 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embd=384
n_head = 6
n_layer = 6
dropout = 0.2

# -----------------------------------------------------------------------------------------------------------------------
# data loding
def get_batch(split):
    #generate am small batch of data on inputs x and targets y
    batch_data= train_data if split=='train' else val_data
    ix = torch.randint(len(batch_data) -block_size, (batch_size,))
    x=torch.stack([batch_data[i:i+block_size] for i in ix])
    y=torch.stack([batch_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def context_preload(content):
    ix= torch.randint(len(content)-1, (len(content),))
    x=torch.stack([content[i:i+1] for i in ix])
    x = x.to(device)
    return x

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    # one Head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key= nn.Linear(n_embd, head_size, bias=False)
        self.query= nn.Linear(n_embd, head_size, bias=False)
        self.value= nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k= self.key(x)   #(B,T,16)
        q= self.query(x) #(B,T,16)
        v = self.value(x) # --> here is what im interested in, here is what i have and if you find me interesting, this is what i will communicate with you
 
        #compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1)* C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # elements can only look in the past --> decoder Block (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C) # --> Adding Values depending on how interesting the elements find each other (Q,K,V)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads], dim=-1) # concatenate over the chanel dimension
        out=self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
# Transformer BLock: communication followed by computation

    def __init__(self, n_embd, n_head):
        # n-embd: embedding dimension, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.fwd= FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.fwd(self.ln2(x))
        return x

# super simpel Bigram Model
# see makemore video series of andrej for more informations
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table= nn.Embedding(block_size, n_embd)
        self.blocks= nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f= nn.LayerNorm(n_embd) # final Layer norm
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) --> batch by time by chanel (chanel = vocab_size)
        pos_emb= self.position_embedding_table(torch.arange(T,device=device)) # (T,C) 
        x = tok_emb + pos_emb # (B,T,C) --> not only token identity but also position at which they accur
        x = self.blocks(x)
        x = self.ln_f(x)
        logits= self.lm_head(x) #(B,T,Vocab_size)

        if targets is None:
            loss= None
        else:
            # reshape array from 3d to 2d to conform cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) 
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current  context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (b,C)
            # applay softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next= torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running squence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# input: textfile
# plain text from a author (in our case sharespeare)

# finding out how many different charactes and what kind will be used as input
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8')as f:
    text=f.read()
chars=sorted(list(set(text)))
vocab_size=len(chars)
# print('',join(chars))
# print(vocab_size)


# --------------------------------------------------------------------------------------------------------------------
# tokenization:
# first we need strategy to tokenize input 
# (charactes to integers for us)
stoi = {ch:i for i, ch in enumerate(chars) }
itos = {i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encodeer: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take a list of integers, output a string
# print(encode("Blub gud"))
# print(decode(encode("Blub gud")))


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
# start of model loading 
model = BigramLanguageModel(vocab_size)
m=model.to(device)

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lerning_rate)
print("---------- model loaded ----------")
print("--------- start training ---------")
# Load our model with data and train it on itself
for iter in range(max_iters):

    #every once in a while output expected loss on train and val sets
    if iter % eval_interval== 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


#generate predictiv output
#context= torch.zeros((1,1), dtype=torch.long, device=device) # set start at zero(zero represents a space or newline in our data set )
print("----- start generation ------")

context= context_preload(torch.tensor(encode("The darkness shall beginn"), dtype=torch.long, device=device))

resulti=m.generate(context, max_new_tokens=500)

# to decode results, it must be converted back to a list
decodes=decode(resulti[0].tolist())

# output decoded result
print(decodes)      
