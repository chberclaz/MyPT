import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from dataclasses import asdict
import os
import json

# Local imports are placed here to avoid circulars when tooling loads files out of order
from .tokenizer import Tokenizer

def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """
    Map a human-friendly dtype string to a torch.dtype.
    Supports: fp32/float32, fp16/float16, bf16/bfloat16.
    """
    if dtype_str is None:
        raise ValueError("dtype_str must not be None")

    key = dtype_str.lower()
    if key in ("fp32", "float32", "f32"):
        return torch.float32
    if key in ("fp16", "float16", "f16", "half"):
        return torch.float16
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16

    raise ValueError(f"Unsupported dtype string: {dtype_str}")

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
    
    # SFT / loss-masking behavior
    use_loss_mask: bool = False   # if True, expect loss_mask from data loader during SFT
    
    # Checkpoint dtype (None = use current model dtype)
    save_dtype: str | None = None  # 'fp32', 'bf16', 'fp16', or None

    def __str__(self):
        return (
            f'batch_size:{self.batch_size}, block_size:{self.block_size}, '
            f'vocab_size:{self.vocab_size}, n_embd:{self.n_embd}, '
            f'n_head:{self.n_head}, n_layer:{self.n_layer}, '
            f'dropout:{self.dropout}, bias:{self.bias}, device:{self.device}, '
            f'use_loss_mask:{self.use_loss_mask}, save_dtype:{self.save_dtype}'
        )
    
    def to_dict(self):
        """Convert config to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary, ignoring unknown fields."""
        # Get valid field names from the dataclass
        import dataclasses
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        
        # Filter to only include valid fields
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        
        return cls(**filtered_dict)
    
    def save_json(self, path):
        """Save config to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved config to {path}")
    
    @classmethod
    def load_json(cls, path):
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

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

    def forward(self, idx, targets=None, loss_mask=None):
        """
        Forward pass with optional loss masking for SFT (supervised fine-tuning).
        
        Args:
            idx: Input token indices (B, T)
            targets: Target token indices (B, T), optional
            loss_mask: Binary mask for loss computation (B, T), optional
                       1 = compute loss, 0 = ignore position
                       Used for assistant-only training in chat/RAG scenarios
        
        Returns:
            logits: Predicted logits (B, T, vocab_size) or (B*T, vocab_size) if targets provided
            loss: Scalar loss (None if targets=None)
        """
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
            
            if loss_mask is not None:
                # Apply loss masking (for SFT with assistant-only loss)
                # loss_mask expected shape (B, T), values 0 or 1 (or floats)
                loss_mask = loss_mask.view(B * T).to(logits.device)
                
                # Compute per-token loss without reduction
                per_token_loss = F.cross_entropy(
                    logits, targets, reduction='none'
                )
                
                # Apply mask and normalize only by masked positions
                denom = loss_mask.sum()
                if denom.item() > 0:
                    loss = (per_token_loss * loss_mask).sum() / denom
                else:
                    # Fallback if mask is all zeros (shouldn't happen in practice)
                    loss = per_token_loss.mean()
            else:
                # Standard loss computation (all positions)
                loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def save(self, checkpoint_dir: str, save_dtype: str | None = None):
        """
        Save model weights only (to model.pt).
        Optionally cast floating-point tensors to a target dtype (e.g. 'float32', 'bfloat16').

        Config, tokenizer, and training state are saved separately via save_checkpoint_bundle().

        Args:
            checkpoint_dir: Directory to save checkpoint (e.g., "checkpoints/dante")
            save_dtype: Optional dtype string for checkpoint tensors
                        ('float32', 'fp32', 'bfloat16', 'bf16', 'float16', 'fp16').
                        If None, tensors are saved in their current dtype.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "model.pt")

        state_dict = self.state_dict()
        current_dtype = str(next(self.parameters()).dtype).replace("torch.", "")

        checkpoint_payload: dict

        if save_dtype is not None:
            target_dtype = _resolve_dtype(save_dtype)
            cast_state_dict = {}
            for k, v in state_dict.items():
                if torch.is_floating_point(v):
                    cast_state_dict[k] = v.to(target_dtype)
                else:
                    cast_state_dict[k] = v
            checkpoint_payload = {
                "state_dict": cast_state_dict,
                "checkpoint_dtype": save_dtype,
                "model_dtype": current_dtype,
            }
        else:
            # Backwards-compatible: still allow raw state_dict-only checkpoints
            checkpoint_payload = state_dict

        torch.save(checkpoint_payload, model_path)
        print(f"Saved model weights to {model_path} (model dtype={current_dtype}, "
              f"checkpoint dtype={save_dtype or current_dtype})")

    
    def save_checkpoint_bundle(self, checkpoint_dir: str, step: int | None = None,
                               optimizer_state: dict | None = None,
                               training_config: dict | None = None,
                               save_dtype: str | None = None):
        """
        Save complete checkpoint bundle: model weights + config + tokenizer + training state.

        File structure:
            checkpoint_dir/
                ├── model.pt              (model weights only; may be cast to save_dtype)
                ├── config.json           (architecture config)
                ├── tokenizer.json        (tokenizer state)
                └── training_state.json   (step, optimizer, training hyperparameters, dtype info)

        Args:
            checkpoint_dir: Directory to save all files
            step: Current training step (optional)
            optimizer_state: Optimizer state dict (optional, for resuming)
            training_config: Training hyperparameters dict (optional, e.g. max_iters, learning_rate, etc.)
            save_dtype: Optional target dtype for checkpoint weights
                        ('float32'/'fp32', 'bfloat16'/'bf16', 'float16'/'fp16').
                        If None, weights are saved in their current dtype.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        current_dtype = str(next(self.parameters()).dtype).replace("torch.", "")

        # 1. Save model weights (possibly cast to save_dtype)
        self.save(checkpoint_dir, save_dtype=save_dtype)

        # 2. Save config
        config_path = os.path.join(checkpoint_dir, "config.json")
        self.config.save_json(config_path)

        # 3. Save tokenizer state
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
            try:
                tokenizer_state = self.tokenizer.get_state()
                with open(tokenizer_path, 'w') as f:
                    json.dump(tokenizer_state, f, indent=2)
                print(f"Saved tokenizer to {tokenizer_path}")
            except Exception as e:
                print(f"Warning: Could not save tokenizer: {e}")

        # 4. Save training state (optional but recommended)
        if step is not None or optimizer_state is not None or training_config is not None:
            training_state_path = os.path.join(checkpoint_dir, "training_state.json")
            training_state: dict = {}

            # Save current step
            if step is not None:
                training_state["step"] = step

            # Save training hyperparameters (for resuming with same config)
            if training_config is not None:
                training_state["training_config"] = training_config

            # Record dtype info
            training_state["model_dtype_at_save"] = current_dtype
            if save_dtype is not None:
                training_state["checkpoint_dtype"] = save_dtype

            # Optimizer state is saved separately as .pt (it's large and contains tensors)
            if optimizer_state is not None:
                optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
                torch.save(optimizer_state, optimizer_path)
                training_state["optimizer_file"] = "optimizer.pt"
                print(f"Saved optimizer state to {optimizer_path}")

            with open(training_state_path, 'w') as f:
                json.dump(training_state, f, indent=2)
            print(f"Saved training state to {training_state_path}")


    @classmethod
    def load(cls, checkpoint_dir: str,
             map_location: str | torch.device | None = None,
             load_dtype: str | None = None):
        """
        Load model from checkpoint bundle (new JSON-based format).

        Expected structure:
            checkpoint_dir/
                ├── model.pt              (required)
                ├── config.json           (required)
                ├── tokenizer.json        (required)
                └── training_state.json   (optional)

        model.pt can be either:
            - a raw state_dict (old behavior)
            - a dict with keys: 'state_dict', 'checkpoint_dtype', 'model_dtype'

        Args:
            checkpoint_dir: Directory containing checkpoint files
            map_location: Device to load tensors to (for torch.load)
            load_dtype: Optional dtype string to cast model weights after loading
                        ('float32'/'fp32', 'bfloat16'/'bf16', 'float16'/'fp16').
                        If None, uses smart defaults based on checkpoint dtype and device:
                        - bf16 checkpoint on non-bf16 GPU → auto-convert to fp32
                        - fp16 checkpoint → keep as fp16
                        - fp32 checkpoint → keep as fp32

        Returns:
            Tuple of (model, tokenizer_state, step, optimizer_state)
        """
        # 1. Load config
        config_path = os.path.join(checkpoint_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = GPTConfig.load_json(config_path)

        # 2. Load tokenizer state
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
        tokenizer_state = None
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                tokenizer_state = json.load(f)

        # 3. Create model with config
        model = cls(config)

        # 4. Load model weights (handle both raw state_dict and dict-with-metadata)
        model_path = os.path.join(checkpoint_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        ckpt_obj = torch.load(model_path, map_location=map_location, weights_only=False)

        checkpoint_dtype = None
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            state_dict = ckpt_obj["state_dict"]
            checkpoint_dtype = ckpt_obj.get("checkpoint_dtype", None)
        else:
            # Backwards-compatible: old style state_dict-only
            state_dict = ckpt_obj
            # Detect dtype from first float tensor
            for v in state_dict.values():
                if torch.is_floating_point(v):
                    checkpoint_dtype = str(v.dtype).replace("torch.", "")
                    break

        # 5. Smart dtype handling
        effective_dtype = load_dtype
        
        if effective_dtype is None and checkpoint_dtype is not None:
            # Auto-detect: check if we need to convert bf16 on unsupported hardware
            device = config.device
            is_cuda = device.startswith('cuda') if isinstance(device, str) else (device.type == 'cuda')
            
            if checkpoint_dtype in ('bf16', 'bfloat16') and is_cuda:
                # Check if GPU supports bf16 (Ampere+ = compute capability 8.0+)
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    if capability[0] < 8:
                        print(f"⚠️  Checkpoint is bf16 but GPU (compute {capability[0]}.{capability[1]}) "
                              f"doesn't have native bf16 support.")
                        print(f"   Auto-converting to fp32 for compatibility.")
                        effective_dtype = 'fp32'
                    else:
                        print(f"✓ Loading bf16 checkpoint on bf16-capable GPU (compute {capability[0]}.{capability[1]})")
            
            elif checkpoint_dtype in ('fp16', 'float16'):
                print(f"✓ Loading fp16 checkpoint")
            
            elif checkpoint_dtype in ('fp32', 'float32'):
                print(f"✓ Loading fp32 checkpoint")
        
        if effective_dtype is not None:
            print(f"  Converting weights to {effective_dtype}")
            target_dtype = _resolve_dtype(effective_dtype)
            for k, v in state_dict.items():
                if torch.is_floating_point(v):
                    state_dict[k] = v.to(target_dtype)
        
        model.load_state_dict(state_dict)

        # Move model to device and dtype
        if effective_dtype is not None:
            target_dtype = _resolve_dtype(effective_dtype)
            model = model.to(device=config.device, dtype=target_dtype)
        else:
            model = model.to(config.device)

        model.eval()
        
        # Log final model dtype
        final_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        print(f"  Model loaded: {config.n_layer}L/{config.n_embd}E/{config.n_head}H, "
              f"dtype={final_dtype}, device={config.device}")

        # 5. Reattach tokenizer
        if tokenizer_state is not None:
            try:
                model.tokenizer = Tokenizer.from_state(config, tokenizer_state)
            except Exception as e:
                print(f"Warning: Could not restore tokenizer: {e}")

        # 6. Load training state (optional)
        step = None
        optimizer_state = None
        training_state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, 'r') as f:
                training_state = json.load(f)
            step = training_state.get("step", None)

            # Load optimizer state if it exists
            if "optimizer_file" in training_state:
                optimizer_path = os.path.join(checkpoint_dir, training_state["optimizer_file"])
                if os.path.exists(optimizer_path):
                    optimizer_state = torch.load(optimizer_path, map_location=map_location, weights_only=True)

        return model, tokenizer_state, step, optimizer_state

    
    @classmethod
    def load_legacy(cls, path: str, map_location: str | torch.device | None = None):
        """
        Load model from legacy checkpoint format (single .pt file with everything).
        For backwards compatibility with old checkpoints.
        
        Args:
            path: Path to legacy .pt checkpoint file
            map_location: Device to load model to
        
        Returns:
            Tuple of (model, tokenizer_state, step, optimizer_state)
        """
        # Legacy checkpoints contain dicts with config/tokenizer, need weights_only=False
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
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
        
        Automatically handles batches with or without loss masks.
        
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
                batch = data_loader.get_batch(split)
                
                # Handle both (X, Y) and (X, Y, loss_mask) formats
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    X, Y, loss_mask = batch
                    _, loss = self(X, Y, loss_mask=loss_mask)
                else:
                    X, Y = batch
                    _, loss = self(X, Y)
                
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
    
    def fit(self, data_loader, optimizer, max_iters, eval_interval=50, 
            eval_iters=200, checkpoint_dir=None, start_step=0, learning_rate=None,
            save_dtype='bf16', final_save_dtype='fp16', warmup_iters=0):
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
            learning_rate: Learning rate (optional, for recording in training state)
            save_dtype: Dtype for eval/training checkpoints (default: 'bf16').
                       Options: 'fp32', 'bf16', 'fp16'. Use 'bf16' for A100/modern GPUs.
            final_save_dtype: Dtype for final deployment checkpoint (default: 'fp16').
                             Use 'fp16' for older GPUs (2060, etc). Set to None to skip.
            warmup_iters: Learning rate warmup iterations. Can be:
                         - int: Absolute number of warmup steps (e.g., 1000)
                         - float (0-1): Fraction of max_iters (e.g., 0.05 for 5%)
                         Warmup uses linear scaling from 0 to target learning rate.
                         Recommended: 5-10% of max_iters for large models (750M+).
        
        Checkpoint Strategy:
            - Eval checkpoints: Saved to checkpoint_dir/ in bf16 (default)
                               Includes optimizer state for resuming training on A100.
            - Final checkpoints: Two versions are saved:
                1. checkpoint_dir/ in bf16 - for resume capability (A100)
                2. checkpoint_dir_fp16/ in fp16 - deployment version (2060, etc)
        
        Returns:
            Dict with final train/val losses
        """
        import datetime
        
        self.train()
        
        # Determine save dtype for eval checkpoints: explicit param > config > None (model dtype)
        effective_save_dtype = save_dtype if save_dtype is not None else self.config.save_dtype
        
        # Get target learning rate
        target_lr = learning_rate if learning_rate is not None else optimizer.param_groups[0]['lr']
        
        # Calculate warmup iterations
        if isinstance(warmup_iters, float) and 0 < warmup_iters < 1:
            # Fraction of max_iters
            warmup_steps = int(warmup_iters * max_iters)
        else:
            warmup_steps = int(warmup_iters)
        
        if warmup_steps > 0:
            print(f"Learning rate warmup: {warmup_steps} steps ({warmup_steps/max_iters*100:.1f}% of training)")
            print(f"  LR will ramp from 0 → {target_lr:.2e} over first {warmup_steps} iterations")
        
        # Prepare training configuration for saving
        training_config = {
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "learning_rate": target_lr,
            "start_step": start_step,
            "save_dtype": effective_save_dtype,
            "warmup_iters": warmup_steps,
        }
        
        def get_lr(step):
            """Calculate learning rate with linear warmup."""
            if step < warmup_steps:
                # Linear warmup: scale from 0 to target_lr
                return target_lr * (step + 1) / warmup_steps
            return target_lr
        
        for iter in range(start_step, max_iters):
            # Update learning rate (warmup or constant)
            current_lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            # Evaluation and checkpointing
            if iter % eval_interval == 0:
                ct = datetime.datetime.now()
                print(f"{iter} : {ct}")
                
                losses = self.estimate_loss(data_loader, eval_iters)
                lr_info = f" | lr {current_lr:.2e}" if warmup_steps > 0 else ""
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}{lr_info}")
                
                # Save eval checkpoint (fp32 by default, includes optimizer for resuming)
                if checkpoint_dir:
                    self.save_checkpoint_bundle(
                        checkpoint_dir, 
                        step=iter, 
                        optimizer_state=optimizer.state_dict(),
                        training_config=training_config,
                        save_dtype=effective_save_dtype
                    )
            
            # Training step
            batch = data_loader.get_batch('train')
            
            # Handle both (X, Y) and (X, Y, loss_mask) formats
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                xb, yb, loss_mask = batch
                _, loss = self(xb, yb, loss_mask=loss_mask)
            else:
                xb, yb = batch
                _, loss = self(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        # Final evaluation
        final_losses = self.estimate_loss(data_loader, eval_iters)
        print(f"final step {iter}: train loss {final_losses['train']:.4f}, "
              f"val loss {final_losses['val']:.4f}")
        
        # Save final model bundles
        if checkpoint_dir:
            # 1. Save full checkpoint (fp32) for potential training resume
            print("\n----- Saving final model (full, fp32) ------")
            self.save_checkpoint_bundle(
                checkpoint_dir, 
                step=iter, 
                optimizer_state=optimizer.state_dict(),
                training_config=training_config,
                save_dtype=effective_save_dtype
            )
            print(f"Full checkpoint saved to: {checkpoint_dir}")
            
            # 2. Save lightweight deployment checkpoint (fp16) - no optimizer state
            if final_save_dtype is not None:
                deploy_dir = f"{checkpoint_dir}_{final_save_dtype}"
                print(f"\n----- Saving deployment model ({final_save_dtype}) ------")
                self.save_checkpoint_bundle(
                    deploy_dir, 
                    step=iter, 
                    optimizer_state=None,  # No optimizer for deployment
                    training_config=training_config,
                    save_dtype=final_save_dtype
                )
                print(f"Deployment checkpoint saved to: {deploy_dir}")
            
            print(f"\nTraining finished!")
            print(f"  Resume training from: {checkpoint_dir}")
            if final_save_dtype:
                print(f"  Deploy inference from: {deploy_dir}")
        
        return final_losses
    
    # ===== GENERATION METHODS =====
    
    def compile_for_inference(self, mode: str = "default"):
        """
        Compile the model with torch.compile() for faster inference.
        
        Args:
            mode: Compilation mode:
                  - "default" (recommended, most compatible)
                  - "reduce-overhead" (faster but requires Triton)
                  - "max-autotune" (slower compile, potentially faster run)
        
        Returns:
            self (for chaining)
        
        Note: 
            - Requires PyTorch 2.0+
            - First generation will be slower due to JIT compilation
            - "reduce-overhead" and "max-autotune" require Triton (may not work on Windows)
        """
        if not hasattr(torch, 'compile'):
            print("torch.compile() not available (requires PyTorch 2.0+)")
            return self
        
        # Suppress dynamo errors to prevent crashes
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        
        print(f"Compiling model with mode='{mode}'...")
        try:
            # Try requested mode first
            self.forward = torch.compile(self.forward, mode=mode)
            print("Model compiled successfully!")
        except Exception as e:
            error_msg = str(e)
            if "triton" in error_msg.lower():
                print(f"Note: Triton not available, using eager mode fallback")
                print("  (Install triton for faster compilation: pip install triton)")
            else:
                print(f"Compilation warning: {error_msg[:100]}")
            # Model will fall back to eager mode automatically due to suppress_errors
        
        return self
    
    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.1, stop_tokens=None):
        """
        Generate text with advanced sampling controls.
        
        Args:
            prompt: Input text to continue from
            max_new_tokens: Maximum number of tokens to generate
            temperature: Controls randomness (0.0=deterministic, 1.0=neutral, >1.0=more random)
            top_k: Only sample from top K most likely tokens (0=disabled)
            top_p: Nucleus sampling - keep tokens with cumulative prob <= top_p (1.0=disabled)
            repetition_penalty: Penalize repeated tokens (1.0=disabled, >1.0=discourage repeats)
            stop_tokens: Optional list of token IDs to stop generation on
        
        Returns:
            Generated text string
        
        Sampling Strategy:
            1. Apply temperature scaling to logits
            2. Apply repetition penalty to already-seen tokens (VECTORIZED)
            3. Apply top-k filtering (keep only top K tokens)
            4. Apply top-p/nucleus filtering (only if needed after top-k)
            5. Sample from the filtered distribution
        
        Performance Tips:
            - Use top_k=20, top_p=1.0 for fastest generation
            - Call model.compile_for_inference() once before generating (PyTorch 2.0+)
            - Use fp16/bf16 model dtype for ~2x speedup
        """
        ctx_ids = self.encode(prompt)
        device = self.config.device
        
        # Pre-allocate output tensor with extra space to avoid repeated allocations
        max_len = len(ctx_ids) + max_new_tokens
        idx = torch.zeros(1, max_len, dtype=torch.long, device=device)
        idx[0, :len(ctx_ids)] = torch.tensor(ctx_ids, dtype=torch.long, device=device)
        cur_len = len(ctx_ids)
        
        # Track generated tokens as a GPU tensor for fast repetition penalty
        if repetition_penalty != 1.0:
            # Pre-allocate penalty tokens tensor
            penalty_tokens = torch.tensor(ctx_ids, dtype=torch.long, device=device)
        
        # Handle stop tokens
        if stop_tokens is None:
            stop_tokens = set()
        elif not isinstance(stop_tokens, set):
            stop_tokens = set(stop_tokens)

        for _ in range(max_new_tokens):
            # Crop to the last block_size tokens (sliding window)
            start_pos = max(0, cur_len - self.config.block_size)
            idx_cond = idx[:, start_pos:cur_len]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature (in-place for speed)
            if temperature != 1.0 and temperature > 0:
                logits.div_(temperature)
            
            # Apply repetition penalty (VECTORIZED)
            if repetition_penalty != 1.0:
                # Gather logits for penalty tokens
                penalty_logits = logits.index_select(1, penalty_tokens)
                # Apply penalty: divide positive, multiply negative
                penalty_logits = torch.where(
                    penalty_logits > 0,
                    penalty_logits / repetition_penalty,
                    penalty_logits * repetition_penalty
                )
                # Scatter back
                logits.scatter_(1, penalty_tokens.unsqueeze(0), penalty_logits)
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_actual = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k_actual)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            # Optimization: skip if top_k already reduced to very few tokens
            if top_p < 1.0 and (top_k == 0 or top_k > 10):
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Store in pre-allocated tensor (no cat!)
            idx[0, cur_len] = idx_next[0, 0]
            cur_len += 1
            
            # Update penalty tokens tensor
            if repetition_penalty != 1.0:
                penalty_tokens = torch.cat([penalty_tokens, idx_next[0]])
            
            # Check for stop tokens
            next_token = idx_next[0, 0].item()
            if next_token in stop_tokens:
                break
        
        # Decode only the generated portion
        return self.decode(idx[0, :cur_len].tolist())
    
    @torch.inference_mode()
    def generate_simple(self, prompt, max_new_tokens):
        """
        Simple generation without sampling controls (legacy behavior).
        Useful for testing or when you want pure model output.
        """
        ctx_ids = self.encode(prompt)
        idx = torch.tensor([ctx_ids], dtype=torch.long, device=self.config.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return self.decode(idx[0].tolist())

    # ---- Convenience helpers to avoid juggling tokenizer outside the model ----
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
    
