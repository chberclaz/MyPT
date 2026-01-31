"""
Episode-Indexed SFT Data Loader

Handles episode-aware data loading for SFT (Supervised Fine-Tuning) training.
Never crosses episode boundaries, supports fixed-shape outputs with padding,
and provides deterministic epoch-based coverage.

Dataset format:
    dataset_dir/
        train/tokens.bin      # uint32 token IDs (concatenated episodes)
        train/mask.bin        # uint8 loss mask (optional, aligned to tokens)
        train/episodes.idx    # uint64 pairs: (start, length) per episode
        val/tokens.bin
        val/mask.bin
        val/episodes.idx

Multi-shard format (for very large datasets):
    dataset_dir/
        train/shard_00000/tokens.bin
        train/shard_00000/mask.bin
        train/shard_00000/episodes.idx
        train/shard_00001/...
        val/...

Usage:
    from core.episode_data_loader import GPTEpisodeDataLoader
    
    loader = GPTEpisodeDataLoader(
        config, tokenizer, dataset_dir="data/chat_sft",
        use_loss_mask=True,
        batch_sampling_mode="epoch"
    )
    
    x, y, mask = loader.get_batch("train")  # Fixed shapes: (batch_size, block_size)
"""

import os
import sys
import json
import glob
import numpy as np
import torch
from typing import Optional, Tuple, List

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass  # Python < 3.7

# Audit logging for compliance
try:
    from .compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


class GPTEpisodeDataLoader:
    """
    Episode-indexed SFT data loader.
    
    Key features:
    - Never crosses episode boundaries during sampling
    - Returns fixed-shape tensors padded/truncated to block_size
    - Supports epoch-based deterministic coverage (each episode seen once per epoch)
    - Supports optional loss masking aligned to y
    - Supports multi-shard datasets for very large corpora
    
    Args:
        config: GPTConfig with batch_size, block_size, device
        tokenizer: Tokenizer instance (for pad_token_id fallback)
        dataset_dir: Path to episode-indexed dataset
        use_loss_mask: Whether to load and return loss masks
        batch_sampling_mode: 'epoch' for deterministic coverage, 'random' for uniform sampling
        epoch_shuffle: Whether to shuffle episode order each epoch (only for epoch mode)
        epoch_drop_last: Whether to drop incomplete final batch (only for epoch mode)
        epoch_seed: Random seed for epoch shuffling (for reproducibility)
        pad_token_id: Token ID for padding (defaults to tokenizer's EOS)
        episode_min_tokens: Minimum episode length (shorter episodes are skipped)
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        dataset_dir: str,
        use_loss_mask: bool = None,
        batch_sampling_mode: str = None,
        epoch_shuffle: bool = None,
        epoch_drop_last: bool = None,
        epoch_seed: int = None,
        pad_token_id: Optional[int] = None,
        episode_min_tokens: int = None
    ):
        """
        Initialize the episode data loader.
        
        All parameters default to values from config if not explicitly provided.
        This ensures the config file is the single source of truth for reproducibility.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_dir = dataset_dir
        
        # Use config values as defaults (config is single source of truth)
        self.use_loss_mask = use_loss_mask if use_loss_mask is not None else getattr(config, 'use_loss_mask', False)
        self.batch_sampling_mode = batch_sampling_mode if batch_sampling_mode is not None else getattr(config, 'batch_sampling_mode', 'epoch')
        self.epoch_shuffle = epoch_shuffle if epoch_shuffle is not None else getattr(config, 'epoch_shuffle', True)
        self.epoch_drop_last = epoch_drop_last if epoch_drop_last is not None else getattr(config, 'epoch_drop_last', True)
        self.epoch_seed = epoch_seed if epoch_seed is not None else getattr(config, 'epoch_seed', 1337)
        self.episode_min_tokens = episode_min_tokens if episode_min_tokens is not None else getattr(config, 'episode_min_tokens', 2)
        
        # Determine pad token ID
        if pad_token_id is not None:
            self.pad_token_id = pad_token_id
        elif hasattr(tokenizer, 'special_tokens') and 'myPT_eot' in tokenizer.special_tokens:
            # Use EOT token as pad token
            self.pad_token_id = tokenizer.special_tokens['myPT_eot']
        elif hasattr(tokenizer, 'enc') and tokenizer.enc is not None:
            # Fall back to GPT-2 EOS token
            self.pad_token_id = tokenizer.enc.eot_token  # 50256 for GPT-2
        else:
            raise ValueError(
                "pad_token_id not specified and could not be inferred from tokenizer. "
                "Please set pad_token_id explicitly."
            )
        
        # Data storage (per split)
        self._data = {
            'train': {'shards': [], 'loaded': False},
            'val': {'shards': [], 'loaded': False}
        }
        
        # Epoch state (per split)
        self._epoch_state = {
            'train': {'epoch': 0, 'pos': 0, 'order': None, 'rng': None},
            'val': {'epoch': 0, 'pos': 0, 'order': None, 'rng': None}
        }
        
        # Load dataset
        self._load_dataset(dataset_dir)
        
        # Initialize epoch state for both splits
        for split in ['train', 'val']:
            self._init_epoch_state(split)
    
    def _load_dataset(self, dataset_dir: str):
        """Load dataset metadata and memory-map data files."""
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Load metadata if present
        metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📦 Loaded episode-indexed dataset:")
            print(f"   Schema: {metadata.get('schema', 'unknown')}")
            print(f"   Episodes: {metadata.get('num_train_episodes', '?')} train, "
                  f"{metadata.get('num_val_episodes', '?')} val")
            print(f"   Tokens: {metadata.get('num_train_tokens', 0):,} train, "
                  f"{metadata.get('num_val_tokens', 0):,} val")
        
        # Load train and val splits
        for split in ['train', 'val']:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.exists(split_dir):
                if split == 'train':
                    raise FileNotFoundError(f"Training data not found: {split_dir}")
                continue
            
            # Check for multi-shard format
            shard_dirs = sorted(glob.glob(os.path.join(split_dir, "shard_*")))
            
            if shard_dirs:
                # Multi-shard format
                for shard_dir in shard_dirs:
                    shard = self._load_shard(shard_dir)
                    if shard is not None:
                        self._data[split]['shards'].append(shard)
            else:
                # Single-file format
                shard = self._load_shard(split_dir)
                if shard is not None:
                    self._data[split]['shards'].append(shard)
            
            self._data[split]['loaded'] = len(self._data[split]['shards']) > 0
            
            # Print stats
            total_episodes = sum(len(s['valid_ids']) for s in self._data[split]['shards'])
            total_tokens = sum(len(s['tokens']) for s in self._data[split]['shards'])
            print(f"   [{split}] {len(self._data[split]['shards'])} shard(s), "
                  f"{total_episodes:,} episodes, {total_tokens:,} tokens")
        
        if not self._data['train']['loaded']:
            raise FileNotFoundError("No training data loaded")
        
        # Check mask availability
        if self.use_loss_mask:
            has_masks = all(
                s['mask'] is not None 
                for s in self._data['train']['shards']
            )
            if not has_masks:
                print("   ⚠️ Warning: use_loss_mask=True but mask files not found/incomplete")
                print("   Disabling loss masking")
                self.use_loss_mask = False
            else:
                print(f"   Using loss masking (assistant-only training)")
        
        # Audit: Dataset loaded
        if AUDIT_AVAILABLE:
            train_episodes = sum(len(s['valid_ids']) for s in self._data['train']['shards'])
            val_episodes = sum(len(s['valid_ids']) for s in self._data['val']['shards']) if self._data['val']['loaded'] else 0
            train_tokens = sum(len(s['tokens']) for s in self._data['train']['shards'])
            val_tokens = sum(len(s['tokens']) for s in self._data['val']['shards']) if self._data['val']['loaded'] else 0
            
            audit.training(
                "dataset_load",
                dataset_dir=dataset_dir,
                dataset_format="episode_indexed",
                num_train_episodes=train_episodes,
                num_val_episodes=val_episodes,
                num_train_tokens=train_tokens,
                num_val_tokens=val_tokens,
                use_loss_mask=self.use_loss_mask,
                batch_sampling_mode=self.batch_sampling_mode,
                epoch_seed=self.epoch_seed,
                epoch_shuffle=self.epoch_shuffle,
                details=f"Episode-indexed dataset loaded: {train_episodes} train episodes, {train_tokens:,} tokens, seed={self.epoch_seed}"
            )
    
    def _load_shard(self, shard_dir: str) -> Optional[dict]:
        """Load a single shard (tokens, mask, episodes index)."""
        tokens_path = os.path.join(shard_dir, "tokens.bin")
        episodes_path = os.path.join(shard_dir, "episodes.idx")
        mask_path = os.path.join(shard_dir, "mask.bin")
        
        if not os.path.exists(tokens_path):
            return None
        if not os.path.exists(episodes_path):
            return None
        
        # Memory-map tokens
        tokens = np.memmap(tokens_path, dtype=np.uint32, mode='r')
        
        # Load episodes index into RAM (small: N_episodes × 16 bytes)
        episodes_raw = np.memmap(episodes_path, dtype=np.uint64, mode='r')
        episodes = episodes_raw.reshape(-1, 2)  # (N, 2) where each row is (start, length)
        
        # Filter episodes by minimum length
        valid_ids = []
        for i in range(len(episodes)):
            start, length = episodes[i]
            if length >= self.episode_min_tokens:
                valid_ids.append(i)
        
        # Memory-map mask if available
        mask = None
        if os.path.exists(mask_path):
            mask = np.memmap(mask_path, dtype=np.uint8, mode='r')
        
        return {
            'tokens': tokens,
            'mask': mask,
            'episodes': episodes,
            'valid_ids': np.array(valid_ids, dtype=np.int64),
            'path': shard_dir
        }
    
    def _init_epoch_state(self, split: str):
        """Initialize or reset epoch state for a split."""
        shards = self._data[split]['shards']
        if not shards:
            return
        
        # Build global episode list: (shard_idx, episode_idx_in_shard)
        global_episodes = []
        for shard_idx, shard in enumerate(shards):
            for ep_idx in shard['valid_ids']:
                global_episodes.append((shard_idx, ep_idx))
        
        self._epoch_state[split]['global_episodes'] = global_episodes
        self._epoch_state[split]['epoch'] = 0
        self._epoch_state[split]['pos'] = 0
        
        # Create RNG for this split
        self._epoch_state[split]['rng'] = np.random.RandomState(self.epoch_seed)
        
        # Shuffle order if enabled
        self._shuffle_epoch(split)
    
    def _shuffle_epoch(self, split: str):
        """Shuffle episode order for a new epoch."""
        state = self._epoch_state[split]
        n_episodes = len(state['global_episodes'])
        
        if self.batch_sampling_mode == 'epoch' and self.epoch_shuffle:
            # Deterministic shuffle based on epoch seed + epoch number
            effective_seed = self.epoch_seed + state['epoch']
            rng = np.random.RandomState(effective_seed)
            state['order'] = rng.permutation(n_episodes)
            
            # Audit: Log epoch start with episode order for traceability
            if AUDIT_AVAILABLE:
                # Log first 10 episode IDs for traceability (full order can be reconstructed from seed)
                first_episodes = state['order'][:min(10, len(state['order']))].tolist()
                audit.training(
                    "epoch_start",
                    split=split,
                    epoch=state['epoch'],
                    seed=effective_seed,
                    base_seed=self.epoch_seed,
                    num_episodes=n_episodes,
                    first_episode_ids=str(first_episodes),
                    details=f"Epoch {state['epoch']} started for {split}: seed={effective_seed}, "
                            f"{n_episodes} episodes, first IDs: {first_episodes}"
                )
        else:
            # Sequential order
            state['order'] = np.arange(n_episodes)
            
            if AUDIT_AVAILABLE:
                audit.training(
                    "epoch_start",
                    split=split,
                    epoch=state['epoch'],
                    seed=None,
                    num_episodes=n_episodes,
                    shuffle=False,
                    details=f"Epoch {state['epoch']} started for {split}: sequential order, {n_episodes} episodes"
                )
    
    def _get_episode_sample(self, split: str, global_idx: int) -> Tuple[np.ndarray, ...]:
        """Get padded/truncated sample from a single episode.
        
        Includes defensive assertions for loss-mask alignment.
        """
        state = self._epoch_state[split]
        shard_idx, ep_idx = state['global_episodes'][global_idx]
        shard = self._data[split]['shards'][shard_idx]
        
        # Get episode bounds
        start, length = shard['episodes'][ep_idx]
        start, length = int(start), int(length)
        
        L = self.config.block_size + 1  # Need L tokens for x and y
        
        # Slice tokens
        actual_len = min(length, L)
        seq = np.array(shard['tokens'][start:start + actual_len], dtype=np.int64)
        
        # Track original length before padding
        original_len = len(seq)
        
        # Pad if needed
        if len(seq) < L:
            pad_len = L - len(seq)
            seq = np.pad(seq, (0, pad_len), mode='constant', constant_values=self.pad_token_id)
        
        # Form x and y
        x = seq[:-1]  # (block_size,)
        y = seq[1:]   # (block_size,)
        
        if self.use_loss_mask and shard['mask'] is not None:
            # Slice mask
            m_raw = np.array(shard['mask'][start:start + actual_len], dtype=np.float32)
            
            # DEFENSIVE ASSERTION 1: mask length must match tokens length
            assert len(m_raw) == original_len, (
                f"Mask/token length mismatch in episode {ep_idx} of shard {shard_idx}: "
                f"tokens={original_len}, mask={len(m_raw)}"
            )
            
            # Pad with zeros (don't train on padding)
            if len(m_raw) < L:
                m = np.pad(m_raw, (0, L - len(m_raw)), mode='constant', constant_values=0)
            else:
                m = m_raw
            
            mask_y = m[1:]  # Aligned with y
            
            # DEFENSIVE ASSERTION 2: padding positions must have mask=0
            if original_len < L:
                pad_start = original_len - 1  # -1 because mask_y is shifted
                if pad_start < len(mask_y):
                    pad_mask_values = mask_y[pad_start:]
                    assert np.all(pad_mask_values == 0), (
                        f"Non-zero mask in padding region of episode {ep_idx}: "
                        f"pad_mask_values={pad_mask_values[:5]}..."
                    )
            
            # Debug mode: print token/mask info if enabled
            if getattr(self, '_debug_mask_alignment', False):
                self._print_mask_debug(ep_idx, shard_idx, seq, m, original_len)
            
            return x, y, mask_y
        
        return x, y
    
    def _print_mask_debug(self, ep_idx: int, shard_idx: int, seq: np.ndarray, mask: np.ndarray, original_len: int):
        """Print debug info for mask alignment (first/last 10 tokens)."""
        print(f"\n[DEBUG MASK] Episode {ep_idx} (shard {shard_idx}), original_len={original_len}")
        print("  First 10 tokens with mask:")
        for i in range(min(10, len(seq))):
            tok_id = seq[i]
            m_val = mask[i] if i < len(mask) else 0
            tok_str = f"ID={tok_id}"
            if hasattr(self.tokenizer, 'decode'):
                try:
                    tok_str = repr(self.tokenizer.decode([int(tok_id)]))
                except:
                    pass
            print(f"    [{i}] {tok_str:30s} mask={m_val}")
        
        if original_len > 20:
            print("  ...")
        
        print("  Last 10 tokens with mask:")
        start_idx = max(0, original_len - 10)
        for i in range(start_idx, original_len):
            tok_id = seq[i]
            m_val = mask[i] if i < len(mask) else 0
            tok_str = f"ID={tok_id}"
            if hasattr(self.tokenizer, 'decode'):
                try:
                    tok_str = repr(self.tokenizer.decode([int(tok_id)]))
                except:
                    pass
            marker = " <- LAST ORIGINAL" if i == original_len - 1 else ""
            print(f"    [{i}] {tok_str:30s} mask={m_val}{marker}")
    
    def enable_mask_debug(self, enabled: bool = True):
        """Enable/disable mask alignment debug output."""
        self._debug_mask_alignment = enabled
    
    def get_batch(self, split: str = 'train') -> Tuple[torch.Tensor, ...]:
        """
        Get a batch of episode samples.
        
        Returns:
            - (x, y) if use_loss_mask=False
            - (x, y, mask) if use_loss_mask=True
            
            All tensors have shape (batch_size, block_size)
        """
        if not self._data[split]['loaded']:
            if split == 'val':
                # Fall back to train if val not available
                split = 'train'
            else:
                raise ValueError(f"No data loaded for split: {split}")
        
        state = self._epoch_state[split]
        batch_size = self.config.batch_size
        
        # Select episode indices for this batch
        if self.batch_sampling_mode == 'epoch':
            # Epoch mode: deterministic coverage
            indices = []
            n_episodes = len(state['order'])
            
            for _ in range(batch_size):
                if state['pos'] >= n_episodes:
                    # Epoch complete - log it
                    completed_epoch = state['epoch']
                    completed_seed = self.epoch_seed + completed_epoch
                    state['epoch'] += 1
                    state['pos'] = 0
                    self._shuffle_epoch(split)  # This logs epoch_start for the NEW epoch
                    n_episodes = len(state['order'])
                    
                    # Audit: Epoch completed
                    if AUDIT_AVAILABLE:
                        audit.training(
                            "epoch_complete",
                            split=split,
                            epoch=completed_epoch,
                            seed_used=completed_seed,
                            episodes_seen=n_episodes,
                            details=f"Epoch {completed_epoch} complete for {split} split (seed={completed_seed})"
                        )
                
                idx = state['order'][state['pos']]
                indices.append(idx)
                state['pos'] += 1
        else:
            # Random mode: uniform sampling with replacement
            n_episodes = len(state['global_episodes'])
            indices = state['rng'].randint(0, n_episodes, size=batch_size)
        
        # Build batch
        batch_x = []
        batch_y = []
        batch_mask = [] if self.use_loss_mask else None
        
        for idx in indices:
            sample = self._get_episode_sample(split, idx)
            batch_x.append(sample[0])
            batch_y.append(sample[1])
            if self.use_loss_mask:
                batch_mask.append(sample[2])
        
        # Stack and convert to tensors
        x = torch.from_numpy(np.stack(batch_x)).to(self.config.device)
        y = torch.from_numpy(np.stack(batch_y)).to(self.config.device)
        
        if self.use_loss_mask:
            mask = torch.from_numpy(np.stack(batch_mask)).to(self.config.device)
            return x, y, mask
        
        return x, y
    
    def get_epoch_info(self, split: str = 'train') -> dict:
        """Get current epoch state information."""
        state = self._epoch_state[split]
        n_episodes = len(state.get('global_episodes', []))
        batches_per_epoch = n_episodes // self.config.batch_size
        
        return {
            'epoch': state.get('epoch', 0),
            'position': state.get('pos', 0),
            'episodes_total': n_episodes,
            'batches_per_epoch': batches_per_epoch,
            'episodes_seen_this_epoch': state.get('pos', 0),
        }
    
    def reset_epoch(self, split: str = 'train'):
        """Reset epoch state to start fresh."""
        self._init_epoch_state(split)


def is_episode_indexed_dataset(dataset_dir: str) -> bool:
    """
    Check if a dataset directory contains episode-indexed format.
    
    Returns True if episodes.idx exists in train/ (or train/shard_*/),
    False otherwise (assume shard-based format).
    """
    if not os.path.exists(dataset_dir):
        return False
    
    train_dir = os.path.join(dataset_dir, "train")
    if not os.path.exists(train_dir):
        return False
    
    # Check for single-file format
    if os.path.exists(os.path.join(train_dir, "episodes.idx")):
        return True
    
    # Check for multi-shard format
    shard_dirs = glob.glob(os.path.join(train_dir, "shard_*"))
    if shard_dirs:
        # Check first shard
        first_shard = sorted(shard_dirs)[0]
        if os.path.exists(os.path.join(first_shard, "episodes.idx")):
            return True
    
    return False

