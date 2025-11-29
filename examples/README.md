# MyPT Examples

Example code demonstrating MyPT usage and concepts.

## Available Examples

### `example_usage.py`
Comprehensive examples of using MyPT programmatically.

**What it covers:**
1. **Convenience functions** - Using `create_model()`, `load_model()`
2. **Core classes** - Using `GPT`, `GPTConfig`, `Tokenizer` directly
3. **Training** - Training a model programmatically
4. **Loading models** - Loading trained models
5. **Checkpoint management** - Using `CheckpointManager`

**Run it:**
```bash
python examples/example_usage.py
```

**Learn:**
- How to create models without CLI
- How to train programmatically
- How to use the public API
- Best practices for model management

---

### `helper_selfaggregation.py`
Educational code exploring self-attention mechanisms.

**What it covers:**
- Different implementations of self-attention
- Performance comparisons
- Matrix multiplication approaches
- Understanding attention mechanisms

**Purpose:**
- Educational reference
- Understanding transformer internals
- Performance optimization techniques

**Note:** This is educational code from the original tutorial, not part of the production API.

---

## Running Examples

All examples can be run directly:

```bash
# API usage examples
python examples/example_usage.py

# Self-attention educational code
python examples/helper_selfaggregation.py
```

---

## Creating Your Own Examples

Feel free to add your own examples here! Good examples to create:

1. **Fine-tuning example** - Show how to fine-tune a model
2. **Custom tokenizer** - Implement a custom tokenization scheme
3. **Generation strategies** - Different ways to generate text
4. **Model comparison** - Compare different model sizes
5. **Evaluation** - Evaluate model performance

---

## See Also

- **Main scripts**: `train.py`, `generate.py` (in root)
- **Utility scripts**: `scripts/` folder
- **Documentation**: `docs/` folder
- **API Reference**: See `core/__init__.py` for public API

