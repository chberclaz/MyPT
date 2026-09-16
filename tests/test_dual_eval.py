#!/usr/bin/env python
"""
Tests for Phase 1 (single eval) and Phase 2 (dual eval) training modes.

Verifies:
1. Phase 1: Standard training with single dataset (train + val)
2. Phase 2: Domain adaptation with dual eval sets (domain + general)

Run with:
    python -m pytest tests/test_dual_eval.py -v
    
Or standalone:
    python tests/test_dual_eval.py
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import numpy as np
import torch

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import GPTConfig, GPTDataLoader, Tokenizer
from core.model import GPT


def create_mock_shards(output_dir: str, num_train_shards: int = 2, num_val_shards: int = 1,
                       tokens_per_shard: int = 10000, vocab_size: int = 50304):
    """Create mock binary shards for testing."""
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create train shards
    for i in range(num_train_shards):
        data = np.random.randint(0, vocab_size, size=tokens_per_shard, dtype=np.uint32)
        shard_path = os.path.join(train_dir, f"shard_{i:05d}.bin")
        data.tofile(shard_path)
    
    # Create val shards
    for i in range(num_val_shards):
        data = np.random.randint(0, vocab_size, size=tokens_per_shard, dtype=np.uint32)
        shard_path = os.path.join(val_dir, f"shard_{i:05d}.bin")
        data.tofile(shard_path)
    
    # Create metadata
    metadata = {
        "total_tokens": (num_train_shards + num_val_shards) * tokens_per_shard,
        "train_shards": num_train_shards,
        "val_shards": num_val_shards,
        "tokens_per_shard": tokens_per_shard,
        "tokenization": "gpt2",
        "vocab_size": vocab_size,
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create tokenizer state
    tokenizer_state = {
        "token_kind": "gpt2",
        "base_vocab_size": 50257,
    }
    with open(os.path.join(output_dir, "tokenizer_state.json"), 'w') as f:
        json.dump(tokenizer_state, f, indent=2)
    
    return output_dir


def create_tiny_config():
    """Create a tiny model config for fast testing."""
    return GPTConfig(
        batch_size=2,
        block_size=64,
        vocab_size=50304,
        n_embd=64,
        n_head=2,
        n_layer=2,
        dropout=0.0,
        bias=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


class TestPhase1SingleEval:
    """Test Phase 1 training: single dataset with train + val."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "phase1_dataset")
        create_mock_shards(self.dataset_dir)
        
    def teardown_method(self):
        """Cleanup."""
        import time
        time.sleep(0.1)  # Give Windows time to release file handles
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors on Windows
    
    def test_dataloader_loads_train_and_val(self):
        """Verify data loader loads both train and val shards."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        
        loader = GPTDataLoader(config, tokenizer, dataset_dir=self.dataset_dir)
        
        assert len(loader.train_shards) > 0, "Should have train shards"
        assert len(loader.val_shards) > 0, "Should have val shards"
        assert loader.use_shards is True
        assert loader.eval_only is False
    
    def test_get_batch_train(self):
        """Verify can get training batches."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        loader = GPTDataLoader(config, tokenizer, dataset_dir=self.dataset_dir)
        
        x, y = loader.get_batch('train')
        
        assert x.shape == (config.batch_size, config.block_size)
        assert y.shape == (config.batch_size, config.block_size)
    
    def test_get_batch_val(self):
        """Verify can get validation batches."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        loader = GPTDataLoader(config, tokenizer, dataset_dir=self.dataset_dir)
        
        x, y = loader.get_batch('val')
        
        assert x.shape == (config.batch_size, config.block_size)
        assert y.shape == (config.batch_size, config.block_size)
    
    def test_model_estimate_loss_single_eval(self):
        """Verify model can estimate loss on single dataset."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        loader = GPTDataLoader(config, tokenizer, dataset_dir=self.dataset_dir)
        
        model = GPT(config, tokenizer)
        model = model.to(config.device)
        
        losses = model.estimate_loss(loader, eval_iters=5, splits=['val'])
        
        assert 'val' in losses
        assert isinstance(losses['val'].item(), float)
        assert losses['val'] > 0
    
    def test_fit_without_eval_data_loaders(self):
        """Verify fit() works without additional eval_data_loaders (Phase 1 mode)."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        loader = GPTDataLoader(config, tokenizer, dataset_dir=self.dataset_dir)
        
        model = GPT(config, tokenizer)
        model = model.to(config.device)
        optimizer = model.configure_optimizer(learning_rate=1e-3)
        
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        
        # Run minimal training (2 iters, eval at 0)
        final_losses = model.fit(
            data_loader=loader,
            optimizer=optimizer,
            max_iters=2,
            eval_interval=1,
            eval_iters=2,
            checkpoint_dir=checkpoint_dir,
            start_step=0,
            use_amp=False,
            eval_data_loaders=None  # No additional eval sets (Phase 1)
        )
        
        assert 'val' in final_losses
        assert os.path.exists(os.path.join(checkpoint_dir, "model.pt"))


class TestPhase2DualEval:
    """Test Phase 2 training: domain dataset + multiple eval sets."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create domain dataset (Phase 2 train + val)
        self.domain_dataset_dir = os.path.join(self.temp_dir, "domain_dataset")
        create_mock_shards(self.domain_dataset_dir, num_train_shards=2, num_val_shards=1)
        
        # Create general eval dataset (Phase 1 val only for eval)
        self.general_eval_dir = os.path.join(self.temp_dir, "general_eval")
        create_mock_shards(self.general_eval_dir, num_train_shards=1, num_val_shards=1)
        
    def teardown_method(self):
        """Cleanup."""
        import time
        time.sleep(0.1)  # Give Windows time to release file handles
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors on Windows
    
    def test_eval_only_loader(self):
        """Verify eval_only loader only loads val shards."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        
        loader = GPTDataLoader(
            config, tokenizer, 
            dataset_dir=self.general_eval_dir, 
            eval_only=True
        )
        
        assert len(loader.val_shards) > 0, "Should have val shards"
        assert len(loader.train_shards) == 0, "Should NOT have train shards in eval_only mode"
        assert loader.eval_only is True
    
    def test_eval_only_loader_get_batch_val(self):
        """Verify eval_only loader can get val batches."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        
        loader = GPTDataLoader(
            config, tokenizer, 
            dataset_dir=self.general_eval_dir, 
            eval_only=True
        )
        
        x, y = loader.get_batch('val')
        
        assert x.shape == (config.batch_size, config.block_size)
        assert y.shape == (config.batch_size, config.block_size)
    
    def test_model_estimate_loss_multiple_loaders(self):
        """Verify model can estimate loss on multiple eval sets."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        
        # Domain loader (train + val)
        domain_loader = GPTDataLoader(config, tokenizer, dataset_dir=self.domain_dataset_dir)
        
        # General eval loader (eval_only)
        general_loader = GPTDataLoader(
            config, tokenizer, 
            dataset_dir=self.general_eval_dir, 
            eval_only=True
        )
        
        model = GPT(config, tokenizer)
        model = model.to(config.device)
        
        # Estimate on domain
        domain_loss = model.estimate_loss(domain_loader, eval_iters=5, splits=['val'])
        assert 'val' in domain_loss
        
        # Estimate on general
        general_loss = model.estimate_loss(general_loader, eval_iters=5, splits=['val'])
        assert 'val' in general_loss
    
    def test_fit_with_dual_eval(self):
        """Verify fit() works with dual eval sets (Phase 2 mode)."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        
        # Domain loader (train + val)
        domain_loader = GPTDataLoader(config, tokenizer, dataset_dir=self.domain_dataset_dir)
        
        # General eval loader (eval_only)
        general_loader = GPTDataLoader(
            config, tokenizer, 
            dataset_dir=self.general_eval_dir, 
            eval_only=True
        )
        
        model = GPT(config, tokenizer)
        model = model.to(config.device)
        optimizer = model.configure_optimizer(learning_rate=1e-3)
        
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        log_file = os.path.join(self.temp_dir, "train_log.jsonl")
        
        # Run minimal training with dual eval
        final_losses = model.fit(
            data_loader=domain_loader,
            optimizer=optimizer,
            max_iters=2,
            eval_interval=1,
            eval_iters=2,
            checkpoint_dir=checkpoint_dir,
            start_step=0,
            use_amp=False,
            eval_data_loaders={"general": general_loader},  # Additional eval set
            log_file=log_file,
            eval_seed=42
        )
        
        assert 'val' in final_losses
        assert os.path.exists(os.path.join(checkpoint_dir, "model.pt"))
        assert os.path.exists(log_file), "JSONL log file should be created"
    
    def test_jsonl_log_format(self):
        """Verify JSONL log contains correct format with dual eval."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        
        domain_loader = GPTDataLoader(config, tokenizer, dataset_dir=self.domain_dataset_dir)
        general_loader = GPTDataLoader(
            config, tokenizer, 
            dataset_dir=self.general_eval_dir, 
            eval_only=True
        )
        
        model = GPT(config, tokenizer)
        model = model.to(config.device)
        optimizer = model.configure_optimizer(learning_rate=1e-3)
        
        log_file = os.path.join(self.temp_dir, "train_log.jsonl")
        
        model.fit(
            data_loader=domain_loader,
            optimizer=optimizer,
            max_iters=3,
            eval_interval=1,
            eval_iters=2,
            checkpoint_dir=None,  # No checkpoints for this test
            start_step=0,
            use_amp=False,
            eval_data_loaders={"general": general_loader},
            log_file=log_file
        )
        
        # Verify log file content
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) >= 1, "Should have at least one log entry"
        
        # Parse first entry
        entry = json.loads(lines[0])
        assert 'iter' in entry, "Log entry should have 'iter'"
        assert 'val_loss' in entry, "Log entry should have 'val_loss'"
        assert 'eval_general' in entry, "Log entry should have 'eval_general'"
    
    def test_eval_does_not_affect_training_state(self):
        """Verify eval does not affect model training state."""
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        
        domain_loader = GPTDataLoader(config, tokenizer, dataset_dir=self.domain_dataset_dir)
        general_loader = GPTDataLoader(
            config, tokenizer, 
            dataset_dir=self.general_eval_dir, 
            eval_only=True
        )
        
        model = GPT(config, tokenizer)
        model = model.to(config.device)
        
        # Set to training mode
        model.train()
        assert model.training is True
        
        # Run eval
        _ = model.estimate_loss(domain_loader, eval_iters=5, splits=['val'])
        _ = model.estimate_loss(general_loader, eval_iters=5, splits=['val'])
        
        # Should be back in training mode
        assert model.training is True, "Model should be in training mode after estimate_loss"


class TestEvalSeedReproducibility:
    """Test that eval_seed produces reproducible results."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = os.path.join(self.temp_dir, "dataset")
        create_mock_shards(self.dataset_dir)
        
    def teardown_method(self):
        """Cleanup."""
        import time
        time.sleep(0.1)  # Give Windows time to release file handles
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors on Windows
    
    def test_eval_seed_reproducibility(self):
        """Verify same eval_seed produces reproducible results.
        
        Note: Full reproducibility requires controlling both torch and numpy RNG.
        The data loader uses np.random for shard/batch selection.
        """
        config = create_tiny_config()
        tokenizer = Tokenizer(config, 'gpt2')
        loader = GPTDataLoader(config, tokenizer, dataset_dir=self.dataset_dir)
        
        model = GPT(config, tokenizer)
        model = model.to(config.device)
        
        # Run eval with both torch and numpy seeds
        torch.manual_seed(42)
        np.random.seed(42)
        losses1 = model.estimate_loss(loader, eval_iters=10, splits=['val'])
        
        # Run eval with same seeds
        torch.manual_seed(42)
        np.random.seed(42)
        losses2 = model.estimate_loss(loader, eval_iters=10, splits=['val'])
        
        # Should be identical when both RNGs are seeded
        assert abs(losses1['val'] - losses2['val']) < 1e-6, \
            f"Losses should be identical with same seed: {losses1['val']} vs {losses2['val']}"


def run_tests():
    """Run all tests manually (without pytest)."""
    print("=" * 70)
    print("Running Dual Eval Tests")
    print("=" * 70)
    
    # Phase 1 Tests
    print("\n--- Phase 1 (Single Eval) Tests ---")
    phase1 = TestPhase1SingleEval()
    
    phase1.setup_method()
    try:
        phase1.test_dataloader_loads_train_and_val()
        print("✓ test_dataloader_loads_train_and_val")
    finally:
        phase1.teardown_method()
    
    phase1.setup_method()
    try:
        phase1.test_get_batch_train()
        print("✓ test_get_batch_train")
    finally:
        phase1.teardown_method()
    
    phase1.setup_method()
    try:
        phase1.test_get_batch_val()
        print("✓ test_get_batch_val")
    finally:
        phase1.teardown_method()
    
    phase1.setup_method()
    try:
        phase1.test_model_estimate_loss_single_eval()
        print("✓ test_model_estimate_loss_single_eval")
    finally:
        phase1.teardown_method()
    
    phase1.setup_method()
    try:
        phase1.test_fit_without_eval_data_loaders()
        print("✓ test_fit_without_eval_data_loaders")
    finally:
        phase1.teardown_method()
    
    # Phase 2 Tests
    print("\n--- Phase 2 (Dual Eval) Tests ---")
    phase2 = TestPhase2DualEval()
    
    phase2.setup_method()
    try:
        phase2.test_eval_only_loader()
        print("✓ test_eval_only_loader")
    finally:
        phase2.teardown_method()
    
    phase2.setup_method()
    try:
        phase2.test_eval_only_loader_get_batch_val()
        print("✓ test_eval_only_loader_get_batch_val")
    finally:
        phase2.teardown_method()
    
    phase2.setup_method()
    try:
        phase2.test_model_estimate_loss_multiple_loaders()
        print("✓ test_model_estimate_loss_multiple_loaders")
    finally:
        phase2.teardown_method()
    
    phase2.setup_method()
    try:
        phase2.test_fit_with_dual_eval()
        print("✓ test_fit_with_dual_eval")
    finally:
        phase2.teardown_method()
    
    phase2.setup_method()
    try:
        phase2.test_jsonl_log_format()
        print("✓ test_jsonl_log_format")
    finally:
        phase2.teardown_method()
    
    phase2.setup_method()
    try:
        phase2.test_eval_does_not_affect_training_state()
        print("✓ test_eval_does_not_affect_training_state")
    finally:
        phase2.teardown_method()
    
    # Reproducibility Tests
    print("\n--- Reproducibility Tests ---")
    repro = TestEvalSeedReproducibility()
    
    repro.setup_method()
    try:
        repro.test_eval_seed_reproducibility()
        print("✓ test_eval_seed_reproducibility")
    finally:
        repro.teardown_method()
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

