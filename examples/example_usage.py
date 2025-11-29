"""
Example usage of MyPT package - demonstrates the clean public API.

This script shows how to use MyPT programmatically without relying on CLI scripts.
"""

# Example 1: Using convenience functions
print("=" * 60)
print("Example 1: Create and use a model with convenience functions")
print("=" * 60)

from core import create_model

# Create a small model for testing
model = create_model(
    n_layer=2,           # Small for demo
    n_head=2,
    n_embd=128,
    block_size=64,
    tokenization="gpt2"
)

print(f"✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Device: {model.config.device}")

# Generate some text (will be random since untrained)
prompt = "Hello world"
output = model.generate(prompt, max_new_tokens=20)
print(f"\nGenerated (untrained model):\n{output[:100]}...")

# =============================================================================
# Example 2: Using core classes directly
print("\n" + "=" * 60)
print("Example 2: Using core classes directly")
print("=" * 60)

from core import GPT, GPTConfig, Tokenizer, GPTDataLoader

# Create custom config
config = GPTConfig(
    batch_size=16,
    block_size=128,
    n_embd=256,
    n_head=4,
    n_layer=4,
    dropout=0.1,
)

print(f"Config: {config}")

# Create tokenizer
tokenizer = Tokenizer(config, kind="gpt2")

# Create model
model = GPT(config, tokenizer=tokenizer)
print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# =============================================================================
# Example 3: Training a model (programmatically)
print("\n" + "=" * 60)
print("Example 3: Training a small model (demo)")
print("=" * 60)

from core import CheckpointManager

# Prepare some dummy data
sample_text = "Hello world! " * 100  # Tiny dataset for demo

# Build char-level tokenizer for demo
tokenizer_char = Tokenizer(config, kind="char")
tokenizer_char.build_char_vocab(sample_text)
config.vocab_size = len(tokenizer_char.chars)

# Create model with char tokenizer
model_train = GPT(config, tokenizer=tokenizer_char)

# Prepare data loader
data_loader = GPTDataLoader(config, tokenizer_char)
data_loader.prepare_data(sample_text)

# Setup optimizer
import torch
optimizer = torch.optim.AdamW(model_train.parameters(), lr=3e-4)

# Train for a few steps (demo)
print("Training for 10 steps...")
try:
    model_train.fit(
        data_loader=data_loader,
        optimizer=optimizer,
        max_iters=10,
        eval_interval=5,
        eval_iters=5,
        checkpoint_dir=None,  # Don't save checkpoints in demo
        start_step=0
    )
    print("✅ Training completed!")
except Exception as e:
    print(f"Note: Training may fail with tiny dataset: {e}")

# =============================================================================
# Example 4: Load a trained model
print("\n" + "=" * 60)
print("Example 4: Load a trained model")
print("=" * 60)

from core import load_model, get_model_info

# Uncomment if you have a trained model:
# model = load_model("dante")
# output = model.generate("Nel mezzo del cammin", max_new_tokens=100)
# print(output)

# Get model info without loading full model
# info = get_model_info("dante")
# print(f"Model info: {info['config']}")

print("(Skipped - no trained model available)")
print("To use: Uncomment the code above after training a model")

# =============================================================================
# Example 5: Checkpoint management
print("\n" + "=" * 60)
print("Example 5: Checkpoint management")
print("=" * 60)

ckpt_manager = CheckpointManager("demo_model")
print(f"Checkpoint directory: {ckpt_manager.checkpoint_dir}")
print(f"New format exists: {ckpt_manager.exists_new_format()}")
print(f"Legacy format exists: {ckpt_manager.exists_legacy_format()}")

# =============================================================================
print("\n" + "=" * 60)
print("✅ All examples completed successfully!")
print("=" * 60)
print("\nNext steps:")
print("1. Train a real model: python train.py --model_name my_model --input_file input.txt")
print("2. Generate text: python generate.py --model_name my_model --prompt 'Hello'")
print("3. Use the API in your own code!")

