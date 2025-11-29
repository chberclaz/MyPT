import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from dataclasses import asdict
import os

# Local imports are placed here to avoid circulars when tooling loads files out of order
from .tokenizer import Tokenizer

@dataclass
class GPTConfig:
    batch_size: int = 32      # how many independent sequences will we process in parallel
    block_size: int = 256     # max context length
    vocab_size: int = 50304   # gpt2 vocabulary
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    bias: bool = False        # True: GPT-2 style, False: a bit better/faster
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __str__(self):
        return (
            f'batch_size:{self.batch_size}, block_size:{self.block_size}, '
            f'vocab_size:{self.vocab_size}, n_embd:{self.n_embd}, '
            f'n_head:{self.n_head}, n_layer:{self.n_layer}, '
            f'dropout:{self.dropout}, bias:{self.bias}, device:{self.device}'
        )

class Head(nn.Module):
    # one Head of self-attention
    def __init__(self, config):
        super().__init__()
        self.head_size= config.n_embd // config.n_head
        self.key= nn.Linear(config.n_embd, self.head_size, config.bias)
        self.query= nn.Linear(config.n_embd, self.head_size, config.bias)
        self.value= nn.Linear(config.n_embd, self.head_size, config.bias)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k= self.key(x)   #(B,T,16)
        q= self.query(x) #(B,T,16)
        v = self.value(x) # --> here is what im interested in, here is what i have and if you find me interesting, this is what i will communicate with you
 
        #compute attention scores ("affinities")
        #wei = q @ k.transpose(-2,-1)* C**-0.5 # (B,T,C) @ (B,C,T) --> (B,T,T)
        # use head_size for scaling, not full C
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # elements can only look in the past --> decoder Block (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        out = wei @ v # (B,T,T) @ (B,T,C) --> (B,T,C) # --> Adding Values depending on how interesting the elements find each other (Q,K,V)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out=torch.cat([h(x) for h in self.heads], dim=-1) # concatenate over the chanel dimension
        out=self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.ReLU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
# Transformer BLock: communication followed by computation

    def __init__(self, config):
        # n-embd: embedding dimension, n_head: number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.fwd= FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.fwd(self.ln2(x))
        return x


# super simpel Bigram Model
# see makemore video series of andrej for more informations
class GPT(nn.Module):

    def __init__(self, config, tokenizer: Tokenizer | None = None, token_kind: str | None = None):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.config = config

        # ---- Tokenizer ownership ----
        # If a tokenizer instance is provided, adopt it.
        # Else create one from token_kind (default to gpt2 if None).
        if tokenizer is not None:
            self.tokenizer: Tokenizer = tokenizer
        else:
            kind = token_kind if token_kind is not None else 'gpt2'
            self.tokenizer = Tokenizer(self.config, kind)

        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.position_embedding_table= nn.Embedding(self.config.block_size, self.config.n_embd)
        self.blocks= nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f= nn.LayerNorm(self.config.n_embd) # final Layer norm
        self.lm_head = nn.Linear(self.config.n_embd,self.config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) --> batch by time by chanel (chanel = vocab_size)
        #pos_emb= self.position_embedding_table(torch.arange(T,device=self.config.device)) # (T,C) 
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
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
    
    def save(self, path: str, tokenizer_state: dict | None = None, step: int | None = None, optimizer_state: dict | None = None):
        """
        Save model weights + config (+ optional tokenizer + step) to a single file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # If caller didn't pass tokenizer_state explicitly, take it from the owned tokenizer (if present)
        if tokenizer_state is None and hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                tokenizer_state = self.tokenizer.get_state()
            except Exception:
                tokenizer_state = None

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": asdict(self.config),
            "tokenizer": tokenizer_state,   # e.g. {"token_kind": "gpt2"} or {"token_kind": "char", "chars": [...]}
            "step": step,
            "optimizer_state_dict": optimizer_state,
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    @classmethod
    def load(cls, path: str, map_location: str | torch.device | None = None):
        """
        Load model + config (+ tokenizer + step) from checkpoint.
        Returns: (model, tokenizer_state, step, optimizer_state)
        """
        checkpoint = torch.load(path, map_location=map_location)
        config_dict = checkpoint["config"]
        config = GPTConfig(**config_dict)

        # Recreate model
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.device)
        model.eval()

        # Reattach tokenizer to model for first-class ownership
        tokenizer_state = checkpoint.get("tokenizer", None)
        if tokenizer_state is not None:
            try:
                model.tokenizer = Tokenizer.from_state(config, tokenizer_state)
            except Exception:
                # Fallback: keep existing tokenizer instance
                pass

        step = checkpoint.get("step", None)
        optim_state = checkpoint.get("optimizer_state_dict", None)
        return model, tokenizer_state, step, optim_state
    
    # ===== TRAINING METHODS =====
    def configure_optimizer(self, learning_rate=3e-4, optimizer_state=None):
        """
        Setup optimizer for this model.
        
        Args:
            learning_rate: Learning rate for AdamW optimizer
            optimizer_state: Optional state dict to restore optimizer state
        
        Returns:
            Configured optimizer
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        return optimizer
    
    @torch.no_grad()
    def estimate_loss(self, data_loader, eval_iters, splits=['train', 'val']):
        """
        Estimate loss on train/val sets.
        
        Args:
            data_loader: GPTDataLoader instance with get_batch method
            eval_iters: Number of iterations to average loss over
            splits: List of splits to evaluate (default: ['train', 'val'])
        
        Returns:
            Dict mapping split names to average losses
        """
        out = {}
        self.eval()
        for split in splits:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = data_loader.get_batch(split)
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
    
    def fit(self, data_loader, optimizer, max_iters, eval_interval=50, 
            eval_iters=200, checkpoint_dir=None, start_step=0):
        """
        Main training loop - the model trains itself!
        
        Args:
            data_loader: GPTDataLoader instance
            optimizer: Configured optimizer (from configure_optimizer)
            max_iters: Number of training iterations
            eval_interval: Evaluate every N steps
            eval_iters: Number of iterations for evaluation
            checkpoint_dir: Where to save checkpoints (None to skip saving)
            start_step: Starting iteration (for resuming)
        
        Returns:
            Dict with final train/val losses
        """
        import datetime
        
        self.train()
        checkpoint_path = os.path.join(checkpoint_dir, "latest.pt") if checkpoint_dir else None
        
        for iter in range(start_step, max_iters):
            # Evaluation and checkpointing
            if iter % eval_interval == 0:
                ct = datetime.datetime.now()
                print(f"{iter} : {ct}")
                
                losses = self.estimate_loss(data_loader, eval_iters)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                
                if checkpoint_path:
                    self.save(checkpoint_path, step=iter, 
                             optimizer_state=optimizer.state_dict())
            
            # Training step
            xb, yb = data_loader.get_batch('train')
            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        # Final evaluation
        final_losses = self.estimate_loss(data_loader, eval_iters)
        print(f"final step {iter}: train loss {final_losses['train']:.4f}, "
              f"val loss {final_losses['val']:.4f}")
        
        # Save final model
        if checkpoint_dir:
            final_path = os.path.join(checkpoint_dir, "final.pt")
            self.save(final_path, step=iter, optimizer_state=optimizer.state_dict())
            print(f"Training finished. Final model at: {final_path}")
        
        return final_losses
    
    # ===== GENERATION METHODS =====
    def generate(self, prompt, max_new_tokens):
        ctx_ids = self.encode(prompt)
        idx = torch.tensor([ctx_ids], dtype=torch.long, device=self.config.device)

        # idx is (B, T) array of indices in the current  context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            #get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (b,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            idx_next= torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running squence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            # decode the response
            response = self.decode(idx[0].tolist())
        return response

    # ---- Convenience helpers to avoid juggling tokenizer outside the model ----
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
    
