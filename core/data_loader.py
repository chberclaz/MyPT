import torch
import numpy as np
import os
import glob
import json

class GPTDataLoader:
    """
    Handles data loading, tokenization, and batching for GPT training.
    
    Supports two modes:
    1. In-memory: Loads entire dataset into memory (good for small datasets < 100M tokens)
    2. Sharded: Memory-maps binary shards from disk (good for large datasets >= 100M tokens)
    
    Usage:
        # In-memory mode (legacy, for small datasets)
        data_loader = GPTDataLoader(config, tokenizer)
        text = data_loader.read_text("input.txt")
        data_loader.prepare_data(text)
        
        # Sharded mode (for large datasets)
        data_loader = GPTDataLoader(config, tokenizer, dataset_dir="data/my_dataset")
    """
    def __init__(self, config, tokenizer, dataset_dir=None):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_dir = dataset_dir
        
        # In-memory data (legacy mode)
        self.train_data = None
        self.val_data = None
        
        # Shard-based data (new mode)
        self.train_shards = []
        self.val_shards = []
        self.shard_cache = {}  # Cache for loaded shards
        self.use_shards = False
        
        # If dataset_dir provided, use shard mode
        if dataset_dir is not None:
            self._load_sharded_dataset(dataset_dir)
    
    def _load_sharded_dataset(self, dataset_dir):
        """Load metadata for sharded dataset."""
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
        # Load metadata
        metadata_path = os.path.join(dataset_dir, "dataset_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📦 Loaded sharded dataset:")
            print(f"   Total tokens: {metadata['total_tokens']:,}")
            print(f"   Train shards: {metadata['train_shards']}")
            print(f"   Val shards: {metadata['val_shards']}")
        
        # Find shard files
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        
        if os.path.exists(train_dir):
            self.train_shards = sorted(glob.glob(os.path.join(train_dir, "*.bin")))
        if os.path.exists(val_dir):
            self.val_shards = sorted(glob.glob(os.path.join(val_dir, "*.bin")))
        
        if not self.train_shards:
            raise FileNotFoundError(f"No training shards found in {train_dir}")
        
        self.use_shards = True
        print(f"   Using shard-based loading (low memory mode)")
    
    def _load_shard(self, shard_path):
        """Load a shard from disk (with caching)."""
        if shard_path in self.shard_cache:
            return self.shard_cache[shard_path]
        
        # Memory-map the shard (doesn't load into RAM immediately)
        data = np.memmap(shard_path, dtype=np.uint32, mode='r')
        
        # Cache the memmap
        self.shard_cache[shard_path] = data
        
        # Limit cache size to avoid memory issues
        if len(self.shard_cache) > 10:  # Keep at most 10 shards in cache
            # Remove oldest entry
            oldest = next(iter(self.shard_cache))
            del self.shard_cache[oldest]
        
        return data
    
    @staticmethod
    def read_text(file_path):
        """Read text file (for in-memory mode)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def prepare_data(self, text, train_ratio=0.9):
        """
        Tokenize text and split into train/val sets (in-memory mode).
        
        ⚠️ Warning: For large datasets (>100M tokens), use sharded mode instead:
            data_loader = GPTDataLoader(config, tokenizer, dataset_dir="data/my_dataset")
        
        Args:
            text: Input text to tokenize
            train_ratio: Fraction of data to use for training (default 0.9)
        """
        tokens = self.tokenizer.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        
        n = int(train_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.use_shards = False
        
        print(f"Data prepared (in-memory mode): {len(self.train_data)} train tokens, "
              f"{len(self.val_data)} val tokens")
    
    def get_batch(self, split='train'):
        """
        Generate a batch of data.
        
        Automatically uses shard-based or in-memory mode depending on initialization.
        
        Args:
            split: 'train' or 'val'
        
        Returns:
            Tuple of (x, y) tensors on device
        """
        if self.use_shards:
            return self._get_batch_sharded(split)
        else:
            return self._get_batch_memory(split)
    
    def _get_batch_memory(self, split='train'):
        """Get batch from in-memory data (legacy mode)."""
        batch_data = self.train_data if split == 'train' else self.val_data
        
        if batch_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        ix = torch.randint(len(batch_data) - self.config.block_size, 
                          (self.config.batch_size,))
        x = torch.stack([batch_data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([batch_data[i + 1:i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y
    
    def _get_batch_sharded(self, split='train'):
        """Get batch from sharded data (new mode for large datasets)."""
        shards = self.train_shards if split == 'train' else self.val_shards
        
        if not shards:
            raise ValueError(f"No shards available for split '{split}'")
        
        # Randomly select a shard
        shard_idx = np.random.randint(0, len(shards))
        shard_path = shards[shard_idx]
        
        # Load shard (memory-mapped, doesn't load all into RAM)
        shard_data = self._load_shard(shard_path)
        
        # Sample random positions from this shard
        max_start = len(shard_data) - self.config.block_size - 1
        if max_start < self.config.batch_size:
            # Shard too small, fall back to sequential sampling
            indices = np.arange(min(self.config.batch_size, max_start))
        else:
            indices = np.random.randint(0, max_start, size=self.config.batch_size)
        
        # Extract sequences
        x = np.stack([shard_data[i:i+self.config.block_size] for i in indices])
        y = np.stack([shard_data[i+1:i+self.config.block_size+1] for i in indices])
        
        # Convert to torch tensors
        x = torch.from_numpy(x.astype(np.int64)).to(self.config.device)
        y = torch.from_numpy(y.astype(np.int64)).to(self.config.device)
        
        return x, y

