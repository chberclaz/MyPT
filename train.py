# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch


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
print(data.shape, data.dtype)
print(data[:1000]) #the 1000 characters we looked at earlier will look to the GPT like this

# ---------------------------------------------------------------------------------------------------------------------
# splitt Innput Data in 2 set: trainig and validation Data
n = int(0.9*len(data))  #first 90% will be training Data
train_data= data[:n]¨
val_data = data[n:]

# ----------------------------------------------------------------------------------------------------------------------
# Data Loader: batches of chunks of data
