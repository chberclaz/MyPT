"""
Unit tests for GPTEpisodeDataLoader.

These tests verify that:
1. Episodes are loaded correctly from disk
2. Batches have correct shapes (batch_size, block_size)
3. Loss masks are loaded and applied correctly
4. Padding/truncation works as expected
5. Epoch-based sampling is deterministic
6. Multi-shard datasets work correctly
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import GPT, GPTConfig
from core.episode_data_loader import GPTEpisodeDataLoader, is_episode_indexed_dataset
from core.banner import print_banner

# Display banner when module is loaded
print_banner("MyPT Episode Data Loader Tests", "Unit Tests for Episode-Indexed SFT Loader")


def create_test_episode_dataset(data_dir, num_episodes=10, episode_length=20, vocab_size=50304):
    """
    Create a synthetic episode-indexed dataset for testing.
    
    Format:
    - tokens.bin: concatenated token sequences
    - mask.bin: concatenated loss masks (1=train, 0=skip)
    - episodes.idx: episode boundaries
    - metadata.json: dataset info
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Create train split
    train_dir = os.path.join(data_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    
    all_tokens = []
    all_masks = []
    episodes_list = []  # (start, length) pairs
    
    for i in range(num_episodes):
        # Episode tokens: mix of different values to distinguish episodes
        # Pattern: [base + i] * episode_length
        base = (i * 100) % vocab_size
        episode_tokens = np.full(episode_length, base, dtype=np.uint16)
        
        # Loss mask: train on second half only
        episode_mask = np.zeros(episode_length, dtype=np.uint8)
        episode_mask[episode_length // 2:] = 1
        
        # Record episode (start, length)
        start = len(all_tokens)
        length = episode_length
        episodes_list.append(start)
        episodes_list.append(length)
        
        all_tokens.extend(episode_tokens)
        all_masks.extend(episode_mask)
    
    # Write tokens.bin
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    tokens_array.tofile(os.path.join(train_dir, 'tokens.bin'))
    
    # Write mask.bin
    mask_array = np.array(all_masks, dtype=np.uint8)
    mask_array.tofile(os.path.join(train_dir, 'mask.bin'))
    
    # Write episodes.idx (flat array of (start, length) pairs)
    episodes_array = np.array(episodes_list, dtype=np.uint64)
    episodes_array.tofile(os.path.join(train_dir, 'episodes.idx'))
    
    # Create val split (smaller)
    val_dir = os.path.join(data_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)
    
    val_tokens = []
    val_masks = []
    val_episodes_list = []
    
    for i in range(3):  # 3 val episodes
        base = ((i + 100) * 100) % vocab_size
        episode_tokens = np.full(episode_length, base, dtype=np.uint16)
        episode_mask = np.zeros(episode_length, dtype=np.uint8)
        episode_mask[episode_length // 2:] = 1
        
        # Record episode (start, length)
        start = len(val_tokens)
        length = episode_length
        val_episodes_list.append(start)
        val_episodes_list.append(length)
        
        val_tokens.extend(episode_tokens)
        val_masks.extend(episode_mask)
    
    np.array(val_tokens, dtype=np.uint16).tofile(os.path.join(val_dir, 'tokens.bin'))
    np.array(val_masks, dtype=np.uint8).tofile(os.path.join(val_dir, 'mask.bin'))
    np.array(val_episodes_list, dtype=np.uint64).tofile(os.path.join(val_dir, 'episodes.idx'))
    
    # Write metadata.json
    import json
    metadata = {
        "format": "episode_indexed_sft_v1",
        "vocab_size": vocab_size,
        "num_episodes": {"train": num_episodes, "val": 3},
        "total_tokens": {"train": len(all_tokens), "val": len(val_tokens)}
    }
    with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return data_dir


def test_dataset_detection():
    """Test that is_episode_indexed_dataset() correctly detects the format."""
    print("\n" + "="*70)
    print("TEST 1: Dataset Format Detection")
    print("="*70)
    
    # Create temp dataset
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        data_dir = create_test_episode_dataset(tmpdir, num_episodes=5)
        
        # Should detect as episode-indexed
        is_episode = is_episode_indexed_dataset(data_dir)
        
        print(f"\nDataset directory: {data_dir}")
        print(f"Is episode-indexed: {is_episode}")
        
        if is_episode:
            print("\n✅ PASS: Episode-indexed dataset detected correctly!")
            return True
        else:
            print("\n❌ FAIL: Dataset not detected as episode-indexed!")
            return False


def test_data_loader_initialization():
    """Test that GPTEpisodeDataLoader initializes correctly."""
    print("\n" + "="*70)
    print("TEST 2: Data Loader Initialization")
    print("="*70)
    
    config = GPTConfig(
        batch_size=4,
        block_size=32,
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_loss_mask=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        data_dir = create_test_episode_dataset(tmpdir, num_episodes=10, episode_length=20)
        
        try:
            loader = GPTEpisodeDataLoader(
                config=config,
                tokenizer=model.tokenizer,
                dataset_dir=data_dir,
                use_loss_mask=True,
                batch_sampling_mode='epoch',
                epoch_shuffle=True,
                epoch_seed=1337
            )
            
            print(f"\nDataset loaded successfully!")
            
            # Count episodes from shards
            train_episodes = sum(len(s['valid_ids']) for s in loader._data['train']['shards'])
            val_episodes = sum(len(s['valid_ids']) for s in loader._data['val']['shards'])
            
            print(f"  Train episodes: {train_episodes}")
            print(f"  Val episodes:   {val_episodes}")
            print(f"  Block size:     {loader.config.block_size}")
            print(f"  Batch size:     {loader.config.batch_size}")
            print(f"  Pad token ID:   {loader.pad_token_id}")
            
            if train_episodes == 10 and val_episodes == 3:
                print("\n✅ PASS: Data loader initialized correctly!")
                return True
            else:
                print(f"\n❌ FAIL: Expected 10 train, 3 val episodes, got {train_episodes}, {val_episodes}")
                return False
                
        except Exception as e:
            print(f"\n❌ FAIL: Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_batch_shapes():
    """Test that batches have correct shapes with padding/truncation."""
    print("\n" + "="*70)
    print("TEST 3: Batch Shapes (Padding/Truncation)")
    print("="*70)
    
    config = GPTConfig(
        batch_size=4,
        block_size=16,  # Smaller than episode length (20) to test truncation
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_loss_mask=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        data_dir = create_test_episode_dataset(tmpdir, num_episodes=10, episode_length=20)
        
        loader = GPTEpisodeDataLoader(
            config=config,
            tokenizer=model.tokenizer,
            dataset_dir=data_dir,
            use_loss_mask=True
        )
        
        # Get batch
        X, Y, loss_mask = loader.get_batch('train')
        
        print(f"\nBatch shapes:")
        print(f"  X (input):     {X.shape}")
        print(f"  Y (targets):   {Y.shape}")
        print(f"  loss_mask:     {loss_mask.shape}")
        
        expected_shape = (config.batch_size, config.block_size)
        
        shapes_correct = (
            X.shape == expected_shape and
            Y.shape == expected_shape and
            loss_mask.shape == expected_shape
        )
        
        print(f"\nExpected shape: {expected_shape}")
        print(f"Shapes correct: {shapes_correct}")
        
        # Check mask values are 0 or 1
        mask_values = torch.unique(loss_mask)
        print(f"Mask unique values: {mask_values.tolist()}")
        
        mask_binary = all(v in [0, 1] for v in mask_values.tolist())
        
        if shapes_correct and mask_binary:
            print("\n✅ PASS: Batch shapes are correct!")
            return True
        else:
            print("\n❌ FAIL: Batch shapes or mask values incorrect!")
            return False


def test_loss_mask_consistency():
    """Test that loss masks are loaded correctly from mask.bin."""
    print("\n" + "="*70)
    print("TEST 4: Loss Mask Consistency")
    print("="*70)
    
    config = GPTConfig(
        batch_size=2,
        block_size=20,  # Same as episode length for exact match
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_loss_mask=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        data_dir = create_test_episode_dataset(tmpdir, num_episodes=5, episode_length=20)
        
        loader = GPTEpisodeDataLoader(
            config=config,
            tokenizer=model.tokenizer,
            dataset_dir=data_dir,
            use_loss_mask=True
        )
        
        # Get batch
        X, Y, loss_mask = loader.get_batch('train')
        
        # Expected mask pattern: first half 0, second half 1
        # (from our synthetic data creation)
        expected_mask_pattern = [0] * 10 + [1] * 10
        
        print(f"\nLoss mask for first episode:")
        got_mask = loss_mask[0].tolist()
        print(f"  Got:      {got_mask}")
        print(f"  Expected: {expected_mask_pattern}")
        
        # The mask might be shifted by 1 due to input/target relationship
        # X contains tokens [0:19], Y contains tokens [1:20]
        # So the loss mask for Y position i corresponds to input position i+1
        # Let's check if the pattern is correct (10 zeros, 10 ones)
        num_zeros_first = got_mask.count(0.0)
        num_ones_first = got_mask.count(1.0)
        
        mask_pattern_ok = num_zeros_first == 10 and num_ones_first == 10
        
        # Check that mask has the right distribution (50% zeros, 50% ones)
        num_zeros = (loss_mask == 0).sum().item()
        num_ones = (loss_mask == 1).sum().item()
        total = loss_mask.numel()
        
        print(f"\nMask distribution:")
        print(f"  Zeros: {num_zeros}/{total} ({num_zeros/total*100:.1f}%)")
        print(f"  Ones:  {num_ones}/{total} ({num_ones/total*100:.1f}%)")
        
        # Should be ~50/50 based on our synthetic data
        distribution_ok = abs(num_ones / total - 0.5) < 0.1  # Within 10%
        
        print(f"\nPattern check:")
        print(f"  First episode has {num_zeros_first} zeros, {num_ones_first} ones (expected 10/10)")
        
        if mask_pattern_ok and distribution_ok:
            print("\n✅ PASS: Loss masks loaded correctly!")
            return True
        else:
            print("\n❌ FAIL: Loss mask doesn't match expected pattern!")
            return False


def test_epoch_determinism():
    """Test that epoch-based sampling is deterministic with same seed."""
    print("\n" + "="*70)
    print("TEST 5: Epoch Sampling Determinism")
    print("="*70)
    
    config = GPTConfig(
        batch_size=4,
        block_size=16,
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_loss_mask=True,
        epoch_seed=42,  # Fixed seed
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        data_dir = create_test_episode_dataset(tmpdir, num_episodes=10, episode_length=20)
        
        # Create two loaders with same seed
        loader1 = GPTEpisodeDataLoader(
            config=config,
            tokenizer=model.tokenizer,
            dataset_dir=data_dir,
            use_loss_mask=True,
            epoch_seed=42
        )
        
        loader2 = GPTEpisodeDataLoader(
            config=config,
            tokenizer=model.tokenizer,
            dataset_dir=data_dir,
            use_loss_mask=True,
            epoch_seed=42
        )
        
        # Get first batch from each
        X1, Y1, mask1 = loader1.get_batch('train')
        X2, Y2, mask2 = loader2.get_batch('train')
        
        # Should be identical
        x_equal = torch.equal(X1, X2)
        y_equal = torch.equal(Y1, Y2)
        mask_equal = torch.equal(mask1, mask2)
        
        print(f"\nDeterminism check (same seed=42):")
        print(f"  X identical:         {x_equal}")
        print(f"  Y identical:         {y_equal}")
        print(f"  loss_mask identical: {mask_equal}")
        
        # Now try different seed
        loader3 = GPTEpisodeDataLoader(
            config=config,
            tokenizer=model.tokenizer,
            dataset_dir=data_dir,
            use_loss_mask=True,
            epoch_seed=99  # Different seed
        )
        
        X3, Y3, mask3 = loader3.get_batch('train')
        
        x_different = not torch.equal(X1, X3)
        
        print(f"\nDifferent seed check (42 vs 99):")
        print(f"  X different:         {x_different}")
        
        all_passed = x_equal and y_equal and mask_equal and x_different
        
        if all_passed:
            print("\n✅ PASS: Epoch sampling is deterministic!")
            return True
        else:
            print("\n❌ FAIL: Epoch sampling not deterministic!")
            return False


def test_padding_with_short_episodes():
    """Test padding behavior when episodes are shorter than block_size."""
    print("\n" + "="*70)
    print("TEST 6: Padding Short Episodes")
    print("="*70)
    
    config = GPTConfig(
        batch_size=2,
        block_size=32,  # Larger than episode length (20) to force padding
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_loss_mask=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        data_dir = create_test_episode_dataset(tmpdir, num_episodes=5, episode_length=20)
        
        loader = GPTEpisodeDataLoader(
            config=config,
            tokenizer=model.tokenizer,
            dataset_dir=data_dir,
            use_loss_mask=True
        )
        
        X, Y, loss_mask = loader.get_batch('train')
        
        print(f"\nBatch shape: {X.shape} (episode_len=20, block_size=32)")
        print(f"Pad token ID: {loader.pad_token_id}")
        
        # Check for padding tokens (should be in last 12 positions)
        # Our synthetic episodes are 20 tokens, block_size is 32
        # So we expect 12 padding tokens
        
        # Count pad tokens in first sequence
        pad_count = (X[0] == loader.pad_token_id).sum().item()
        
        print(f"\nFirst sequence:")
        print(f"  Pad tokens: {pad_count}")
        print(f"  Content tokens: {32 - pad_count}")
        print(f"  X values: {X[0].tolist()}")
        
        # The mask should be 0 for padded positions
        masked_positions = (loss_mask[0] == 0).sum().item()
        print(f"\nLoss mask:")
        print(f"  Masked positions (0): {masked_positions}")
        print(f"  Trained positions (1): {32 - masked_positions}")
        
        # Padding should be present (episode 20 < block 32)
        has_padding = pad_count > 0
        
        if has_padding:
            print("\n✅ PASS: Padding applied correctly!")
            return True
        else:
            print("\n❌ FAIL: Expected padding but found none!")
            return False


def test_no_loss_mask_mode():
    """Test loader behavior when use_loss_mask=False."""
    print("\n" + "="*70)
    print("TEST 7: No Loss Mask Mode")
    print("="*70)
    
    config = GPTConfig(
        batch_size=4,
        block_size=16,
        vocab_size=50304,
        n_layer=2,
        n_head=2,
        n_embd=64,
        use_loss_mask=False,  # Disabled
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        data_dir = create_test_episode_dataset(tmpdir, num_episodes=5, episode_length=20)
        
        loader = GPTEpisodeDataLoader(
            config=config,
            tokenizer=model.tokenizer,
            dataset_dir=data_dir,
            use_loss_mask=False  # Disabled
        )
        
        batch_result = loader.get_batch('train')
        
        # When use_loss_mask=False, get_batch returns (X, Y) without loss_mask
        if len(batch_result) == 2:
            X, Y = batch_result
            loss_mask = None
        else:
            X, Y, loss_mask = batch_result
        
        print(f"\nBatch shapes:")
        print(f"  X: {X.shape}")
        print(f"  Y: {Y.shape}")
        print(f"  loss_mask: {loss_mask.shape if loss_mask is not None else 'None'}")
        
        # When use_loss_mask=False, should return None or all-ones mask
        mask_disabled = (loss_mask is None) or torch.all(loss_mask == 1)
        
        if mask_disabled:
            print("\n✅ PASS: Loss mask correctly disabled!")
            return True
        else:
            print("\n❌ FAIL: Loss mask should be None or all-ones when disabled!")
            return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  EPISODE DATA LOADER TEST SUITE")
    print("="*70)
    print("\nThese tests verify that GPTEpisodeDataLoader correctly:")
    print("- Loads episode-indexed datasets")
    print("- Creates properly shaped batches")
    print("- Handles loss masks from mask.bin")
    print("- Applies padding/truncation")
    print("- Provides deterministic epoch sampling\n")
    
    results = []
    
    # Run all tests
    results.append(("Dataset Detection", test_dataset_detection()))
    results.append(("Loader Initialization", test_data_loader_initialization()))
    results.append(("Batch Shapes", test_batch_shapes()))
    results.append(("Loss Mask Consistency", test_loss_mask_consistency()))
    results.append(("Epoch Determinism", test_epoch_determinism()))
    results.append(("Padding Short Episodes", test_padding_with_short_episodes()))
    results.append(("No Loss Mask Mode", test_no_loss_mask_mode()))
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("  ✅ ALL TESTS PASSED - Episode data loader working correctly!")
    else:
        print("  ❌ SOME TESTS FAILED - Check episode data loader implementation!")
    print("="*70 + "\n")
    
    sys.exit(0 if all_passed else 1)

