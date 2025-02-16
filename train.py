# prerequisites
# finding out how many different charactes and what kind will be used as input

!wget https://raw.githubusercontent.com/karpathy/char-rrn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encode='utf-8')as f:
    text=f.read()
# length=len(text)
chars=sorted(list(set(text)))
vocab_size=len(chars)
# print('',join(chars))
# print(vocab_size)

# first we need strategy to tokenice input 
# (charactes to integers for us)

stoi = {ch:i for i, ch in enumerate(chars) }
itos = {i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] #encodeer: take a string, output a list of integers
decode = lambda l: ''.join([istos[i] for i in l]) #decoder: take a list of integers, output a string

print(encode("Blub gud"))
print(decode(encode("Blub gud")))
# let's encdoe the entire text dataset and store it into a torch.Tensor
# import torch