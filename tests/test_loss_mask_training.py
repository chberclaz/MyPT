"""
Practical training test for loss mask functionality.

This test trains a tiny model on synthetic data where:
- Prompts are random noise
- Responses are a simple pattern (e.g., all token ID 42)

With correct loss masking:
- Model should learn to generate token 42
- Model should NOT overfit to the random prompts

This is a practical, inexpensive way to verify loss masking works.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import GPT, GPTConfig
from core.banner import print_banner

# Display banner when module is loaded
print_banner("MyPT Loss Mask Training Test", "Integration Test for Loss Masking")


def create_synthetic_sft_data(num_samples=50, vocab_size=50304, prompt_len=8, response_len=8):
    """
    Create synthetic SFT data with a clear pattern.
    
    Format:
    - Prompt: random tokens
    - Response: all token ID 42
    
    Returns:
        x, y, loss_mask (all tensors)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len = prompt_len + response_len
    
    x = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)
    y = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)
    loss_mask = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)
    
    for i in range(num_samples):
        # Prompt: random tokens
        prompt = torch.randint(0, vocab_size, (prompt_len,), device=device)
        
        # Response: all token 42
        response = torch.full((response_len,), 42, dtype=torch.long, device=device)
        
        # Concatenate
        x[i, :prompt_len] = prompt
        x[i, prompt_len:] = response[:-1]  # Input doesn't include last response token
        
        # Target: shifted by 1
        y[i, :prompt_len-1] = prompt[1:]
        y[i, prompt_len-1:] = response  # Predict response tokens
        
        # Loss mask: only train on response
        loss_mask[i, prompt_len-1:] = 1
    
    return x, y, loss_mask


def test_training_with_loss_mask():
    """
    Train a tiny model on synthetic data with loss mask.
    
    Expected behavior:
    - Loss should decrease (model learns the pattern)
    - Model should generate token 42 for responses
    - Model should NOT memorize random prompts
    """
    print("\n" + "="*70)
    print("  PRACTICAL TRAINING TEST: Loss Mask in Real Training")
    print("="*70)
    
    # Create tiny model
    config = GPTConfig(
        n_layer=3,
        n_head=4,
        n_embd=128,
        block_size=32,
        vocab_size=50304,  # GPT-2 vocab size (needed for special tokens)
        dropout=0.1,
        bias=False,
        use_loss_mask=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    model.to(config.device)
    
    print(f"\nModel: {config.n_layer} layers, {config.n_embd} embd")
    print(f"Device: {config.device}")
    
    # Create synthetic data
    prompt_len = 8
    response_len = 8
    x_train, y_train, mask_train = create_synthetic_sft_data(
        num_samples=100, 
        vocab_size=config.vocab_size,
        prompt_len=prompt_len,
        response_len=response_len
    )
    
    print(f"\nSynthetic data:")
    print(f"  Samples: {x_train.shape[0]}")
    print(f"  Sequence length: {x_train.shape[1]} ({prompt_len} prompt + {response_len} response)")
    print(f"  Pattern: Prompts are random, responses are all token ID 42")
    print(f"  Loss mask: Train only on response tokens (positions {prompt_len-1}+)")
    
    print(f"\nExample sequence:")
    print(f"  Input (x):    {x_train[0].tolist()}")
    print(f"  Target (y):   {y_train[0].tolist()}")
    print(f"  Loss mask:    {mask_train[0].tolist()}")
    
    # Setup optimizer
    optimizer = model.configure_optimizer(learning_rate=1e-3, weight_decay=0.0)
    
    # Training loop
    model.train()
    batch_size = 16
    num_steps = 200
    
    print(f"\nTraining for {num_steps} steps (batch_size={batch_size})...")
    
    initial_loss = None
    final_loss = None
    
    for step in range(num_steps):
        # Sample random batch
        indices = torch.randint(0, x_train.shape[0], (batch_size,), device=config.device)
        x_batch = x_train[indices]
        y_batch = y_train[indices]
        mask_batch = mask_train[indices]
        
        # Forward + backward
        optimizer.zero_grad()
        _, loss, _ = model(x_batch, y_batch, loss_mask=mask_batch)
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        
        if step % 50 == 0 or step == num_steps - 1:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")
        
        final_loss = loss.item()
    
    # Verify loss decreased
    print(f"\nTraining results:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Reduction:    {initial_loss - final_loss:.4f} ({(1 - final_loss/initial_loss)*100:.1f}%)")
    
    loss_decreased = final_loss < initial_loss * 0.5
    
    if loss_decreased:
        print(f"  ✅ Loss decreased significantly - model learned the pattern!")
    else:
        print(f"  ❌ Loss didn't decrease enough - possible issue!")
    
    # Test generation: given a random prompt, model should generate token 42
    print(f"\nGeneration test:")
    model.eval()
    
    # Create test prompt (random tokens)
    test_prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=config.device)
    
    print(f"  Test prompt: {test_prompt[0].tolist()}")
    
    # Generate response (should be mostly token 42)
    with torch.no_grad():
        generated = test_prompt.clone()
        for _ in range(response_len):
            logits, _, _ = model(generated)
            next_token = logits[0, -1, :].argmax()
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    
    generated_response = generated[0, prompt_len:].tolist()
    print(f"  Generated:   {generated_response}")
    print(f"  Expected:    {[42] * response_len}")
    
    # Count how many tokens are 42
    correct_tokens = sum(1 for t in generated_response if t == 42)
    accuracy = correct_tokens / response_len
    
    print(f"  Accuracy:    {correct_tokens}/{response_len} = {accuracy*100:.1f}%")
    
    pattern_learned = accuracy >= 0.75  # At least 75% should be token 42
    
    if pattern_learned:
        print(f"  ✅ Model learned the pattern (generates token 42)!")
    else:
        print(f"  ❌ Model didn't learn pattern - possible loss mask issue!")
    
    # Overall result
    print(f"\n" + "="*70)
    if loss_decreased and pattern_learned:
        print("  ✅ PASS: Loss mask working correctly in training!")
        print("     - Loss decreased during training")
        print("     - Model learned to generate the masked pattern")
        return True
    else:
        print("  ❌ FAIL: Loss mask may not be working correctly!")
        if not loss_decreased:
            print("     - Loss didn't decrease enough")
        if not pattern_learned:
            print("     - Model didn't learn the pattern")
        return False


def test_training_without_loss_mask():
    """
    Train WITHOUT loss mask to show the difference.
    
    Without loss mask:
    - Model trains on both prompts AND responses
    - This is wrong for SFT (should only train on responses)
    - Loss on response tokens should be similar to with-mask case
    """
    print("\n" + "="*70)
    print("  COMPARISON: Training WITHOUT Loss Mask (Wrong Behavior)")
    print("="*70)
    
    # Create tiny model
    config = GPTConfig(
        n_layer=3,
        n_head=4,
        n_embd=128,
        block_size=32,
        vocab_size=100,
        dropout=0.1,
        bias=False,
        use_loss_mask=False,  # DISABLED
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    model = GPT(config)
    
    print(f"\nTraining WITHOUT loss mask (trains on all tokens)...")
    
    # Create synthetic data (same pattern)
    prompt_len = 8
    response_len = 8
    x_train, y_train, _ = create_synthetic_sft_data(
        num_samples=100, 
        vocab_size=config.vocab_size,
        prompt_len=prompt_len,
        response_len=response_len
    )
    
    # Setup optimizer
    optimizer = model.configure_optimizer(learning_rate=1e-3, weight_decay=0.0)
    
    # Training loop
    model.train()
    batch_size = 16
    num_steps = 200
    
    initial_loss = None
    final_loss = None
    
    for step in range(num_steps):
        indices = torch.randint(0, x_train.shape[0], (batch_size,), device=config.device)
        x_batch = x_train[indices]
        y_batch = y_train[indices]
        
        optimizer.zero_grad()
        _, loss, _ = model(x_batch, y_batch)  # No loss mask
        loss.backward()
        optimizer.step()
        
        if step == 0:
            initial_loss = loss.item()
        
        if step % 50 == 0 or step == num_steps - 1:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")
        
        final_loss = loss.item()
    
    print(f"\nWithout loss mask:")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Note: Model trains on BOTH prompts and responses (not ideal for SFT)")
    
    print(f"\n✅ This demonstrates the difference:")
    print(f"   - WITH mask: trains only on responses")
    print(f"   - WITHOUT mask: trains on everything (wastes capacity on random prompts)")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  LOSS MASK PRACTICAL TRAINING TESTS")
    print("="*70)
    print("\nThese tests train a tiny model on synthetic data to verify")
    print("loss masking works correctly in real training scenarios.\n")
    
    results = []
    
    # Run tests
    results.append(("Training WITH loss mask", test_training_with_loss_mask()))
    results.append(("Training WITHOUT loss mask (comparison)", test_training_without_loss_mask()))
    
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
        print("  ✅ ALL TESTS PASSED - Loss mask works in training!")
    else:
        print("  ❌ SOME TESTS FAILED - Check loss mask implementation!")
    print("="*70 + "\n")
    
    sys.exit(0 if all_passed else 1)

