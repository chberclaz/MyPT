# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import torch
import torch.nn as nn
from torch.nn import functional as F

# averaging past context with for loops, the weakest form of aggreggation
def sa_v1(x):
    torch.manual_seed(1337)
    print(x.shape, x.dtype)

    # we want x[b,t] = mean_{i<=t} x[b,i]
    xbow = torch.zeros((B,T,C))
    for b in range(B):
        for t in range(T):
            xprev = x[b, :t+1] # (t,C)
            xbow[b,t] = torch.mean(xprev, 0)
            
    print(x[0])
    print(xbow[0])
    """ -------- output --------------
    tensor([[ 0.1808, -0.0700],
            [-0.3596, -0.9152],
            [ 0.6258,  0.0255],
            [ 0.9545,  0.0643],
            [ 0.3612,  1.1679],
            [-1.3499, -0.5102],
            [ 0.2360, -0.2398],
            [-0.9211,  1.5433]])
    tensor([[ 0.1808, -0.0700],
            [-0.0894, -0.4926],
            [ 0.1490, -0.3199],
            [ 0.3504, -0.2238],
            [ 0.3525,  0.0545],
            [ 0.0688, -0.0396],
            [ 0.0927, -0.0682],
            [-0.0341,  0.1332]]) 
        
        BSP: 0.3504= (0.1808 + -0.3596 + 0.6258 + 0.9545 ) / 4         
            """
    return xbow


# the trick in self attention: matrix multiply as weighted aggregation
def sa_v2(x):
    torch.manual_seed(42)
    a = torch.tril(torch.ones(3,3))
    b= torch.randint(0,10,(3,2)).float()
    c= a @ b #matrix multiplication

    print('a=')
    print(a)
    print('--')
    print('b=')
    print(b)
    print('--')
    print('c=')
    print(c)
    """ 
    a=
    tensor([[1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    --
    b=
    tensor([[2., 7.],
            [6., 4.],
            [6., 5.]])
    --
    c=
    tensor([[ 2.,  7.],
            [ 8., 11.],
            [14., 16.]]) 

            BSP: 8 = 1*2 +1*6 + 0*6 | 11 = 1*7 + 1*4 + 0*5        
    """
    a = a / torch.sum(a,1,keepdim=True)
    c= a @ b #matrix multiplication

    print('a=')
    print(a)
    print('--')
    print('b=')
    print(b)
    print('--')
    print('c=')
    print(c)
    """ 
    a=
    tensor([[1.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.0000],
            [0.3333, 0.3333, 0.3333]])
    --
    b=
    tensor([[2., 7.],
            [6., 4.],
            [6., 5.]])
    --
    c=
    tensor([[2.0000, 7.0000],
            [4.0000, 5.5000],
            [4.6667, 5.3333]])

            BSP: now we get average in each row:
            4= (2 + 6 )/2 | 5.333=(7+4+5)/3
    """
    torch.manual_seed(1337)
    wei = torch.tril(torch.ones(T,T)) # weights
    wei = wei / wei.sum(1, keepdim=True)
    xbow2 = wei @ x # (B,T,T) @ (B,T,C) ----> (B,T,C)
    return xbow2


#
# in a nutshell:
# you can do weighted aggregation of your past elements by using matrix multiplication of a lower triangular fashion
def sa_v3(x):
    torch.manual_seed(1337)
    tril = torch.tril(torch.ones(T,T)) # weights
    print("tril=",tril)
    wei = torch.zeros((T,T))    # how interesting one elements finds a other
    print("wei=",wei)
    wei = wei.masked_fill(tril==0, float('-inf'))   # elements can only look in the past
    print("wei after fill", wei)
    wei = F.softmax(wei, dim=-1)
    print("wei after softmax", wei)
    xbow3=wei @ x   # --> Adding Values depending on how interesting the elements find each other
    return xbow3

#
# version 4: self-attention!
#   key and querry --> what i am and what i'm searching for
def sa_v4(x):

    #lets see a single Head perform self-attention
    head_size = 16
    key= nn.Linear(C,head_size, bias=False)
    query= nn.Linear(C,head_size, bias=False)
    value= nn.Linear(C,head_size, bias=False)
    k= key(x)   #(B,T,16)
    q= query(x) #(B,T,16)
    v = value(x) # --> here is what im interested in, here is what i have and if you find me interesting, this is what i will communicate with you
    wei = q @ k.transpose(-2,-1) # (B,T,16) @ (B,16,T) --> (B,T,T)
    
    # "scaled" attention additionaly divides wei by 1/sqrt(head_size). This makes it so, that when input Q,K are unit variance, wei will be unit variance too and softmax will stay diffuse and not saturate too much.
    wei = wei * head_size**-0.5
    
    tril = torch.tril(torch.ones(T,T)) # weights    
    # if all the nodes need to be able to talk to each other, delete this line...
    # in encoder blocks delete this
    wei = wei.masked_fill(tril==0, float('-inf'))   # elements can only look in the past --> decoder Block
    wei = F.softmax(wei, dim=-1)
    print(wei)
    
 
    xbow4=wei @ v   # --> Adding Values depending on how interesting the elements find each other (Q,K,V)
    return xbow4

torch.manual_seed(1337)
B,T,C= 4,8,2 # batch,time,channels
x=torch.randn(B,T,C)

xbow=sa_v1(x)
xbow2=sa_v2(x)
xbow3=sa_v3(x)

print("xbow=",xbow[0])
print("xbow2=",xbow2[0])
print("xbow3=",xbow3[0])

torch.manual_seed(1337)
B,T,C= 4,8,32 # batch,time,channels
x=torch.randn(B,T,C)
xbow4=sa_v4(x)
#print("xbow4=",xbow4[0])

#print(torch.allclose(xbow,xbow2))