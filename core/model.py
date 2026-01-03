import torch
import torch._dynamo        
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
from dataclasses import dataclass
from dataclasses import asdict
from contextlib import nullcontext
import os
import json
# some tests
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
    use_checkpoint: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # SFT / loss-masking behavior
    use_loss_mask: bool = False   # if True, expect loss_mask from data loader during SFT
    
    # Checkpoint dtype (None = use current model dtype)
    save_dtype: str | None = None  # 'fp32', 'bf16', 'fp16', or None
    
    # Episode-indexed SFT data loading options
    # dataset_mode: 'token_stream' (default, random windows) or 'sft_episode' (episode-indexed)
    # Note: dataset_mode is auto-detected from dataset format, these are fallback/override options
    batch_sampling_mode: str = "epoch"    # 'epoch' (deterministic coverage) or 'random'
    pad_token_id: int | None = None       # Token ID for padding (None = use tokenizer EOS)
    episode_min_tokens: int = 2           # Minimum episode length (shorter episodes skipped)
    epoch_seed: int = 1337                # Seed for deterministic epoch shuffling (reproducibility)
    epoch_shuffle: bool = True            # Shuffle episode order each epoch
    epoch_drop_last: bool = True          # Drop incomplete final batch in epoch mode

    def __str__(self):
        return (
            f'batch_size:{self.batch_size}, block_size:{self.block_size}, '
            f'vocab_size:{self.vocab_size}, n_embd:{self.n_embd}, '
            f'n_head:{self.n_head}, n_layer:{self.n_layer}, '
            f'dropout:{self.dropout}, bias:{self.bias}, device:{self.device}, '
            f'use_loss_mask:{self.use_loss_mask}, save_dtype:{self.save_dtype}, '
            f'batch_sampling_mode:{self.batch_sampling_mode}'
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

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout_p = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x,
        past_kv=None,
        use_cache: bool = False,
        kv_cache=None,
        cache_pos: int = 0,
    ):
        """
        Supports two cache modes:

        (A) Old mode (cat-based):
            past_kv = (pk, pv) where pk/pv are (B, nh, T_past, hs)
            -> returns present = (k_cat, v_cat) if use_cache

        (B) Fast mode (preallocated):
            kv_cache = (k_cache, v_cache) where k_cache/v_cache are (B, nh, block_size, hs)
            cache_pos = int start position to write current tokens
            -> writes into cache in-place, returns present=("kv_cache", new_cache_pos)
        """
        B, T, C = x.shape

        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)

        # (B,T,C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # ----------------------------
        # FAST PATH: preallocated KV cache (no cat)
        # ----------------------------
        if kv_cache is not None:
            if not use_cache:
                raise ValueError("kv_cache provided but use_cache=False. Set use_cache=True.")

            k_cache, v_cache = kv_cache  # (B, nh, block_size, hs)
            end_pos = cache_pos + T
            if end_pos > k_cache.size(2):
                raise ValueError(
                    f"KV cache overflow: need end_pos={end_pos}, but cache has size={k_cache.size(2)}. "
                    f"Increase block_size or trim prompt."
                )

            # Write new keys/values into cache
            k_cache[:, :, cache_pos:end_pos, :] = k
            v_cache[:, :, cache_pos:end_pos, :] = v

            # Use cache up to end_pos
            k_used = k_cache[:, :, :end_pos, :]
            v_used = v_cache[:, :, :end_pos, :]

            # IMPORTANT:
            # With KV-cache decode, q is only the "new" token(s) and k_used contains only <= current time.
            # Using is_causal=True when q_len != k_len can mis-mask and break quality.
            # So: causal for prefill (cache_pos==0 and T>1), non-causal for incremental decode.
            use_causal = (cache_pos == 0 and T > 1)

            y = F.scaled_dot_product_attention(
                q, k_used, v_used,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=use_causal,
            )  # (B, nh, T, hs)


            y = y.transpose(1, 2).contiguous().view(B, T, C)
            y = self.dropout(self.proj(y))

            present = (kv_cache, end_pos)  # (cache_tensors, new_cache_pos)
            return y, present

        # ----------------------------
        # OLD PATH: cat-based cache (kept for compatibility)
        # ----------------------------
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        # Same issue: during decode with cache, q_len is often 1 while k_len is past+1.
        # Causal masking can misbehave in that rectangular case, so disable it when past exists.
        use_causal = (past_kv is None)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=use_causal,
        )


        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.proj(y))

        present = (k, v) if use_cache else None
        return y, present



    
class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4*config.n_embd),
            nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = CausalSelfAttention(config)
        self.fwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.use_checkpoint = config.use_checkpoint  # toggle

    def forward(
        self,
        x,
        past_kv=None,
        use_cache: bool = False,
        kv_cache=None,
        cache_pos: int = 0,
    ):
        # Training checkpoint path (no cache)
        if self.training and self.use_checkpoint and (past_kv is None) and (not use_cache) and (kv_cache is None):
            def sa_fn(inp):
                y, _ = self.sa(self.ln1(inp), past_kv=None, use_cache=False)
                return y

            def ff_fn(inp):
                return self.fwd(self.ln2(inp))

            x = x + cp.checkpoint(sa_fn, x, use_reentrant=False)
            x = x + cp.checkpoint(ff_fn, x, use_reentrant=False)
            return x, None

        # Normal path (supports both cache modes)
        attn_out, present = self.sa(
            self.ln1(x),
            past_kv=past_kv,
            use_cache=use_cache,
            kv_cache=kv_cache,
            cache_pos=cache_pos,
        )
        x = x + attn_out
        x = x + self.fwd(self.ln2(x))
        return x, present





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
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f= nn.LayerNorm(self.config.n_embd) # final Layer norm
        self.lm_head = nn.Linear(self.config.n_embd,self.config.vocab_size)

    def forward(
        self,
        idx,
        targets=None,
        loss_mask=None,
        past_kv=None,
        use_cache: bool = False,
        kv_cache=None,
        cache_pos: int = 0,
    ):
        """
        Cache modes:

        (A) past_kv (old cat-based):
            past_kv: list length n_layer, each element (k,v) or None
            returns present_kv: list length n_layer of (k,v) if use_cache

        (B) kv_cache (fast preallocated):
            kv_cache: list length n_layer, each element (k_cache, v_cache)
                     where caches are (B, nh, block_size, hs)
            cache_pos: int position to write current tokens
            returns present_kv: (kv_cache, new_cache_pos)
        """
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)

        # Position handling:
        # - fast cache mode uses cache_pos
        # - cat cache mode uses past_len inferred from past_kv
        # - no cache uses 0..T-1
        if kv_cache is not None and use_cache:
            pos_start = cache_pos
        else:
            pos_start = 0
            if past_kv is not None and len(past_kv) > 0 and past_kv[0] is not None:
                pos_start = past_kv[0][0].size(2)

        pos = torch.arange(pos_start, pos_start + T, device=device)
        pos_emb = self.position_embedding_table(pos)  # (T,C)

        x = tok_emb + pos_emb

        # Run blocks
        if use_cache:
            if kv_cache is not None:
                # FAST MODE: preallocated caches; same cache_pos for all layers
                for i, block in enumerate(self.blocks):
                    x, _ = block(
                        x,
                        past_kv=None,
                        use_cache=True,
                        kv_cache=kv_cache[i],
                        cache_pos=cache_pos,
                    )
                present_kv = (kv_cache, cache_pos + T)

            else:
                # OLD MODE: cat-based caches
                present_kv = []
                for i, block in enumerate(self.blocks):
                    layer_past = None if past_kv is None else past_kv[i]
                    x, pkv = block(x, past_kv=layer_past, use_cache=True, kv_cache=None, cache_pos=0)
                    present_kv.append(pkv)
        else:
            # No cache
            for block in self.blocks:
                x, _ = block(x, past_kv=None, use_cache=False, kv_cache=None, cache_pos=0)
            present_kv = None

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,V)

        loss = None
        if targets is not None:
            Bt, Tt, Ct = logits.shape
            logits2 = logits.view(Bt * Tt, Ct)
            targets2 = targets.view(Bt * Tt)

            if loss_mask is not None:
                loss_mask2 = loss_mask.view(Bt * Tt).to(logits2.device)
                per_token_loss = F.cross_entropy(logits2, targets2, reduction='none')
                denom = loss_mask2.sum().clamp_min(1.0)
                loss = (per_token_loss * loss_mask2).sum() / denom
            else:
                loss = F.cross_entropy(logits2, targets2)

        return logits, loss, present_kv



    def _cpu_state_dict(self, dtype: torch.dtype | None = None) -> dict:
        """Return a CPU state_dict snapshot. Optionally cast floating tensors."""
        sd = self.state_dict()
        out = {}
        for k, v in sd.items():
            if torch.is_floating_point(v):
                vv = v.detach()
                if dtype is not None:
                    vv = vv.to(dtype)
                out[k] = vv.cpu()
            else:
                out[k] = v.detach().cpu()
        return out

    def save_checkpoint_bundle(self, checkpoint_dir: str, step: int | None = None,
                            optimizer_state: dict | None = None,
                            training_config: dict | None = None,
                            save_dtype: str | None = None):
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 1) CPU snapshot of weights (and optional cast) — no VRAM spike
        target_dtype = _resolve_dtype(save_dtype) if save_dtype is not None else None
        payload = {
            "state_dict": self._cpu_state_dict(dtype=target_dtype),
            "checkpoint_dtype": save_dtype,
            "model_dtype": str(next(self.parameters()).dtype).replace("torch.", ""),
        }
        torch.save(payload, os.path.join(checkpoint_dir, "model.pt"))

        # 2) config
        self.config.save_json(os.path.join(checkpoint_dir, "config.json"))

        # 3) tokenizer
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
            with open(tok_path, "w") as f:
                json.dump(self.tokenizer.get_state(), f, indent=2)

        # 4) training_state.json
        training_state_path = os.path.join(checkpoint_dir, "training_state.json")
        training_state = {}
        if step is not None:
            training_state["step"] = step
        if training_config is not None:
            training_state["training_config"] = training_config
        training_state["model_dtype_at_save"] = payload["model_dtype"]
        if save_dtype is not None:
            training_state["checkpoint_dtype"] = save_dtype

        # 5) optimizer — convert to CPU *before* saving
        if optimizer_state is not None:
            def _to_cpu(obj):
                if torch.is_tensor(obj):
                    return obj.detach().cpu()
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return type(obj)(_to_cpu(v) for v in obj)
                return obj

            opt_cpu = _to_cpu(optimizer_state)
            opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
            torch.save(opt_cpu, opt_path)
            training_state["optimizer_file"] = "optimizer.pt"

        with open(training_state_path, "w") as f:
            json.dump(training_state, f, indent=2)

        print(f"Saved checkpoint bundle to {checkpoint_dir}")

    
    def save(self, checkpoint_dir: str, save_dtype: str | None = None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, "model.pt")

        state_dict = self.state_dict()
        current_dtype = str(next(self.parameters()).dtype).replace("torch.", "")

        if save_dtype is not None:
            target_dtype = _resolve_dtype(save_dtype)
            cast_state_dict = {}
            for k, v in state_dict.items():
                if torch.is_floating_point(v):
                    # ✅ detach + move to CPU + cast on CPU (no VRAM duplication)
                    cast_state_dict[k] = v.detach().to(device="cpu", dtype=target_dtype)
                else:
                    cast_state_dict[k] = v.detach().to(device="cpu")
            checkpoint_payload = {
                "state_dict": cast_state_dict,
                "checkpoint_dtype": save_dtype,
                "model_dtype": current_dtype,
            }
        else:
            # also move to CPU to avoid GPU spikes during serialization
            checkpoint_payload = {k: v.detach().to("cpu") for k, v in state_dict.items()}

        torch.save(checkpoint_payload, model_path)
        print(f"Saved model weights to {model_path} (model dtype={current_dtype}, "
            f"checkpoint dtype={save_dtype or current_dtype})")


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
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                if capability[0] < 8:
                    print(f"⚠️  Checkpoint is bf16 but GPU (compute {capability[0]}.{capability[1]}) "
                        f"doesn't have native bf16 support.")
                    print("   Auto-converting to fp16 for faster CUDA inference.")
                    effective_dtype = 'fp16'
                else:
                    print(f"✓ Loading bf16 checkpoint on bf16-capable GPU (compute {capability[0]}.{capability[1]})")
                    effective_dtype = 'bf16'

            
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
    def configure_optimizer(self, learning_rate=3e-4, weight_decay=0.1, optimizer_state=None):
        """
        Setup optimizer for this model with proper weight decay.
        
        Uses the "no decay" convention: don't apply weight decay to biases, 
        LayerNorm weights, or embedding weights.
        
        Args:
            learning_rate: Learning rate for AdamW optimizer
            weight_decay: Weight decay (L2 regularization) for non-embedding weights
            optimizer_state: Optional state dict to restore optimizer state
        
        Returns:
            Configured optimizer
        """
        # Separate parameters into decay and no-decay groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Don't decay: biases, LayerNorm, embeddings
            if param.dim() == 1 or 'embedding' in name.lower() or 'ln' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)
        print(f"Optimizer: {num_decay:,} params with weight_decay={weight_decay}, "
              f"{num_no_decay:,} params without decay")
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
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
                    _, loss, _ = self(X, Y, loss_mask=loss_mask)
                else:
                    X, Y = batch
                    _, loss, _ = self(X, Y)
                
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
    
    def fit(self, data_loader, optimizer, max_iters, eval_interval=20, 
            eval_iters=50, checkpoint_dir=None, start_step=0, learning_rate=None,
            save_dtype='bf16', final_save_dtype='fp16', warmup_iters=0, grad_clip=1.0,
            use_amp=True, amp_dtype='bf16'):
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
            grad_clip: Maximum gradient norm for clipping (default: 1.0).
                      Prevents gradient explosions. Set to 0 to disable.
            use_amp: Enable automatic mixed precision training (default: True).
                    Significantly reduces memory usage and speeds up training on modern GPUs.
            amp_dtype: Dtype for AMP ('bf16' for A100/H100, 'fp16' for older GPUs).
        
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
        
        # Setup automatic mixed precision (AMP)
        device_type = 'cuda' if 'cuda' in str(self.config.device) else 'cpu'
        if use_amp and device_type == 'cuda':
            amp_dtype_torch = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
            # Check if bf16 is supported
            if amp_dtype == 'bf16' and not torch.cuda.is_bf16_supported():
                print(f"[WARN] bf16 not supported on this GPU, falling back to fp16")
                amp_dtype_torch = torch.float16
                amp_dtype = 'fp16'
            
            ctx = torch.amp.autocast(device_type=device_type, dtype=amp_dtype_torch)
            # GradScaler only needed for fp16, not bf16
            scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == 'fp16'))
            print(f"Mixed precision training: {amp_dtype.upper()} enabled")
        else:
            ctx = nullcontext()
            scaler = None
            if use_amp:
                print("Mixed precision disabled (CPU training)")
        
        # Prepare training configuration for saving
        training_config = {
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "learning_rate": target_lr,
            "start_step": start_step,
            "save_dtype": effective_save_dtype,
            "warmup_iters": warmup_steps,
            "grad_clip": grad_clip,
        }
        
        def get_lr(step):
            """Calculate learning rate with linear warmup."""
            if step < warmup_steps:
                # Linear warmup: scale from 0 to target_lr
                return target_lr * (step + 1) / warmup_steps
            return target_lr

        def _mem(tag):
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                resv  = torch.cuda.memory_reserved() / 1024**3
                peak  = torch.cuda.max_memory_allocated() / 1024**3
                print(f"[GPU {tag}] alloc={alloc:.2f}GB reserved={resv:.2f}GB peak={peak:.2f}GB")

        def gpu_mem(tag: str):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                alloc = torch.cuda.memory_allocated() / 1024**3
                resv  = torch.cuda.memory_reserved() / 1024**3
                free, total = torch.cuda.mem_get_info()
                free_gb = free / 1024**3
                total_gb = total / 1024**3
                print(f"[GPU {tag}] alloc={alloc:.2f}GB reserved={resv:.2f}GB free={free_gb:.2f}GB total={total_gb:.2f}GB")

        
        for iter in range(start_step, max_iters):
            # Update learning rate (warmup or constant)
            current_lr = get_lr(iter)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            # Evaluation and checkpointing
            if iter % eval_interval == 0:
                ct = datetime.datetime.now()
                print(f"{iter} : {ct}")
                _mem("before_estimate")
                gpu_mem("before_estimate")
                losses = self.estimate_loss(data_loader, eval_iters, splits=['val'])
                _mem("after_estimate_before_save")
                gpu_mem("after_estimate_before_save")
                lr_info = f" | lr {current_lr:.2e}" if warmup_steps > 0 else ""

                print(f"step {iter}: Validation loss {losses['val']:.4f}{lr_info}")
                
                # Save eval checkpoint (fp32 by default, includes optimizer for resuming)

                if checkpoint_dir:
                    self.save_checkpoint_bundle(
                        checkpoint_dir, 
                        step=iter, 
                        optimizer_state=optimizer.state_dict(),
                        training_config=training_config,
                        save_dtype=effective_save_dtype
                    )
                _mem("after_save")
                gpu_mem("after_save")

                import gc
                gc.collect()
                torch.cuda.empty_cache()
                gpu_mem("after_gc_and_empty_cache")

            # Training step with optional AMP
            optimizer.zero_grad(set_to_none=True)
            batch = data_loader.get_batch('train')
            
            #gpu_mem("after_zero_grad")
            # Handle both (X, Y) and (X, Y, loss_mask) formats
            with ctx:
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    xb, yb, loss_mask = batch
                    _, loss, _ = self(xb, yb, loss_mask=loss_mask)
                else:
                    xb, yb = batch
                    _, loss, _ = self(xb, yb)
                    # debug once
                    if iter == start_step:
                        print("Inside autocast dtype:", self.lm_head.weight.dtype)
            #gpu_mem("after_forward")
            
            if scaler is not None:
                # FP16 path with GradScaler
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # BF16 or FP32 path (no scaler needed)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
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
    def to_inference(self, device: str | None = None):
        """
        Make inference dtype sane.
        - CUDA consumer GPUs (e.g. RTX 2060): fp16
        - CPU: fp32
        """
        if device is None:
            device = self.config.device

        self.eval()
        self.config.device = device

        if isinstance(device, str) and "cuda" in device:
            self.to(device=device, dtype=torch.float16)
        else:
            self.to(device=device, dtype=torch.float32)
        return self

    
    def compile_for_inference(self, mode: str = "default"):
        """
        Compile the model with torch.compile() for faster inference.
        Robust against accidental local 'torch' shadowing.
        """
        import torch as _torch

        if not hasattr(_torch, "compile"):
            print("torch.compile() not available (requires PyTorch 2.0+)")
            return self

        # Suppress dynamo errors to prevent crashes
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True

        print(f"Compiling model with mode='{mode}'...")
        try:
            self.forward = torch.compile(self.forward, mode=mode)  # inductor default
            print("Model compiled successfully!")
        except Exception:
            print("Note: Triton not available, using eager mode fallback")
            self.forward = torch.compile(self.forward, backend="aot_eager")

        return self

    def _ban_repeat_ngrams(self, logits: torch.Tensor, out_ids: list[int], n: int):
        if n <= 0:
            return
        if len(out_ids) < n:
            return

        seen = {}
        for i in range(len(out_ids) - n + 1):
            prefix = tuple(out_ids[i:i+n-1])
            nxt = out_ids[i+n-1]
            if prefix not in seen:
                seen[prefix] = set()
            seen[prefix].add(nxt)

        current_prefix = tuple(out_ids[-(n-1):])
        banned = seen.get(current_prefix)
        if not banned:
            return

        banned_idx = torch.tensor(list(banned), device=logits.device, dtype=torch.long)
        logits[0, banned_idx] = float("-inf")



    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
                repetition_penalty=1.1, stop_tokens=None, recent_penalty_window: int = 256):
        """
        KV-cache generation (fast).
        Fixes repetition penalty (no growing cat) using a bounded recent window.
        """
        device = self.config.device
        device_type = "cuda" if isinstance(device, str) and "cuda" in device else "cpu"

        # Autocast for inference: fp16 on consumer CUDA; otherwise no autocast
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16) if device_type == "cuda" else nullcontext()

        ctx_ids = self.encode(prompt)
        if len(ctx_ids) == 0:
            ctx_ids = [0]

        # Stop tokens handling
        if stop_tokens is None:
            stop_tokens = set()
        elif not isinstance(stop_tokens, set):
            stop_tokens = set(stop_tokens)

        # We'll do KV-cache prefill on at most block_size tokens
        if len(ctx_ids) > self.config.block_size:
            ctx_ids = ctx_ids[-self.config.block_size:]

        idx_prompt = torch.tensor([ctx_ids], dtype=torch.long, device=device)  # (1, L)

        # --------- Preallocate KV caches (FAST) ----------
        # caches: list length n_layer, each entry (k_cache, v_cache)
        # shapes: (B=1, nh, block_size, hs)
        nh = self.config.n_head
        hs = self.config.n_embd // self.config.n_head
        cache_dtype = next(self.parameters()).dtype  # match model dtype (fp16 on 2060)

        kv_cache = []
        for _ in range(self.config.n_layer):
            k_cache = torch.empty((1, nh, self.config.block_size, hs), device=device, dtype=cache_dtype)
            v_cache = torch.empty((1, nh, self.config.block_size, hs), device=device, dtype=cache_dtype)
            kv_cache.append((k_cache, v_cache))
        cache_pos = 0


        # Ring buffer for repetition penalty (bounded)
        use_rep = (repetition_penalty is not None) and (repetition_penalty != 1.0)
        recent_n = max(0, int(recent_penalty_window))
        if use_rep and recent_n > 0:
            recent = torch.empty(recent_n, dtype=torch.long, device=device)
            recent_len = 0
            recent_pos = 0

            # seed with prompt tokens (last recent_n)
            seed = ctx_ids[-recent_n:] if len(ctx_ids) > recent_n else ctx_ids
            for t in seed:
                recent[recent_pos] = int(t)
                recent_pos = (recent_pos + 1) % recent_n
                recent_len = min(recent_len + 1, recent_n)

        # Prefill (build cache)
        with ctx:
            logits, _, present = self(
                idx_prompt,
                past_kv=None,
                use_cache=True,
                kv_cache=kv_cache,
                cache_pos=cache_pos,
            )
        kv_cache, cache_pos = present


        # Start from last token's logits
        logits = logits[:, -1, :]  # (1, vocab)

        # --- DEBUG: verify cache path matches non-cache next-token ---
        with torch.no_grad():
            logits_nc, _, _ = self(idx_prompt, use_cache=False, kv_cache=None, cache_pos=0)
            a_nc = int(torch.argmax(logits_nc[:, -1, :], dim=-1).item())
            a_c  = int(torch.argmax(logits[:, :], dim=-1).item())
            if a_nc != a_c:
                print(f"[WARN] cache vs non-cache argmax mismatch: no_cache={a_nc} cache={a_c}")


        out_ids = list(ctx_ids)

        for _ in range(max_new_tokens):
            base_logits = logits  # fp16/fp32 model dtype
            work_logits = logits.float()  # fp32 for filtering/sampling stability

            # temperature
            if temperature is not None and temperature > 0 and temperature != 1.0:
                work_logits = work_logits / float(temperature)

            # repetition penalty (bounded + unique)
            if use_rep and recent_n > 0 and recent_len > 0:
                recent_tokens = recent[:recent_len] if recent_len < recent_n else recent
                uniq = torch.unique(recent_tokens)
                # operate on fp32 work_logits
                penalty_logits = work_logits.index_select(1, uniq)
                penalty_logits = torch.where(
                    penalty_logits > 0,
                    penalty_logits / float(repetition_penalty),
                    penalty_logits * float(repetition_penalty),
                )
                work_logits.scatter_(1, uniq.unsqueeze(0), penalty_logits)

            # top-k
            if top_k and top_k > 0:
                top_k_actual = min(int(top_k), work_logits.size(-1))
                v, _ = torch.topk(work_logits, top_k_actual)
                work_logits = work_logits.masked_fill(work_logits < v[:, [-1]], float("-inf"))

            # top-p (robust mask construction)
            if top_p is not None and float(top_p) < 1.0:
                sorted_logits, sorted_indices = torch.sort(work_logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs_sorted, dim=-1)

                sorted_indices_to_remove = cumulative_probs > float(top_p)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                remove_mask = torch.zeros_like(work_logits, dtype=torch.bool)
                remove_mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
                work_logits = work_logits.masked_fill(remove_mask, float("-inf"))
            
            no_repeat_ngram=3
            if no_repeat_ngram and no_repeat_ngram > 0:
                # this expects logits shape (1, V)
                self._ban_repeat_ngrams(work_logits, out_ids, int(no_repeat_ngram))


            # --- SAFETY: if everything got masked, fall back ---
            if not torch.isfinite(work_logits).any():
                # fall back to base logits (also fp32), no filtering
                work_logits = base_logits.float()

            # If temperature <= 0, do GREEDY decoding (deterministic)
            if temperature is None or float(temperature) <= 0.0:
                idx_next = torch.argmax(work_logits, dim=-1, keepdim=True)
            else:
                # convert to probs safely
                probs = F.softmax(work_logits, dim=-1)

                # if probs are bad, force greedy
                if (not torch.isfinite(probs).all()) or (probs.sum(dim=-1) <= 0).any():
                    idx_next = torch.argmax(work_logits, dim=-1, keepdim=True)
                else:
                    idx_next = torch.multinomial(probs, num_samples=1)




            next_token = int(idx_next.item())
            # if the last 32 tokens are identical pattern, break
            if len(out_ids) > 64 and out_ids[-32:] == out_ids[-64:-32]:
                break


            out_ids.append(next_token)

            # update ring buffer
            if use_rep and recent_n > 0:
                recent[recent_pos] = next_token
                recent_pos = (recent_pos + 1) % recent_n
                recent_len = min(recent_len + 1, recent_n)

            # stop
            if next_token in stop_tokens:
                break

            # KV step: feed ONLY the new token
            with ctx:
                logits_step, _, present = self(
                    idx_next,
                    past_kv=None,
                    use_cache=True,
                    kv_cache=kv_cache,
                    cache_pos=cache_pos,
                )
            kv_cache, cache_pos = present
            logits = logits_step[:, -1, :]


        return self.decode(out_ids)

    
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
    
