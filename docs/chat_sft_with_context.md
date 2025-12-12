# Chat SFT with Context (RAG Training)

This guide covers how to create and train on Supervised Fine-Tuning (SFT) datasets with context-aware conversations for RAG applications.

## Overview

The chat SFT pipeline:
1. Takes JSONL conversations with optional `context` (from RAG retrieval)
2. Serializes to special token format using `core/special_tokens.py`
3. Creates token shards with parallel **loss mask** files
4. Trains only on assistant responses (mask=1) while using context as input

## JSONL Input Format

Each line is a JSON object with:

```json
{
  "system": "You are a helpful assistant that answers based on the provided context.",
  "context": "[1] (document.md) The quick brown fox jumps over the lazy dog...\n[2] (facts.txt) Foxes are omnivorous mammals...",
  "messages": [
    {"role": "user", "content": "What animals are mentioned?"},
    {"role": "assistant", "content": "Based on the context, the animals mentioned are: a fox and a dog."}
  ]
}
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `system` | No | System prompt setting assistant behavior |
| `context` | No | Retrieved documents (RAG context) |
| `messages` | **Yes** | Array of user/assistant turns |

### Messages Format

```json
{"role": "user", "content": "User's question or input"}
{"role": "assistant", "content": "Assistant's response (TRAINED ON)"}
```

## Serialization Format

The pipeline serializes to this exact format (using `core/special_tokens.py`):

```
<myPT_system>You are a helpful assistant.</myPT_system>
<myPT_user_context>
[1] (document.md) The quick brown fox...
[2] (facts.txt) Foxes are omnivorous...
</myPT_user_context>
<myPT_user>What animals are mentioned?</myPT_user>
<myPT_assistant>Based on the context, the animals mentioned are: a fox and a dog.</myPT_assistant>
<myPT_eot>
```

### Special Tokens Used

From `core/special_tokens.py`:

| Token | Purpose |
|-------|---------|
| `<myPT_system>` | System prompt open |
| `</myPT_system>` | System prompt close |
| `<myPT_user_context>` | RAG context open |
| `</myPT_user_context>` | RAG context close |
| `<myPT_user>` | User message open |
| `</myPT_user>` | User message close |
| `<myPT_assistant>` | Assistant response open |
| `</myPT_assistant>` | Assistant response close |
| `<myPT_eot>` | End of turn marker |

## Loss Masking

The loss mask determines which tokens are trained on:

| Content | Mask Value | Trained? |
|---------|------------|----------|
| System prompt | 0 | No |
| Context (RAG docs) | 0 | No |
| User messages | 0 | No |
| **Assistant responses** | **1** | **Yes** |
| EOT marker | 0 | No |

This means:
- The model learns to **generate** assistant responses
- The model uses system/context/user as **input context**
- The model doesn't try to "learn" the user's questions

## Usage

### 1. Prepare Dataset

```bash
python scripts/prepare_chat_sft.py \
    --input data/conversations.jsonl \
    --output_dir data/chat_sft \
    --tokenization gpt2 \
    --val_split 0.1
```

### 2. Train with Loss Masking

```bash
python train.py \
    --model_name my_rag_model \
    --dataset_dir data/chat_sft \
    --config_file configs/sft1/micro.json \
    --max_iters 5000
```

Note: Use a config with `"use_loss_mask": true`.

### 3. Use for RAG Generation

```bash
python scripts/rag_chat.py \
    --model_name my_rag_model \
    --index_dir workspace/index/latest
```

## Creating Training Data

### From Existing Documents + Questions

```python
import json

# You have documents and Q&A pairs
data = [
    {
        "context": "Paris is the capital of France. It has a population of 2.1 million.",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France."}
        ]
    },
    # ... more examples
]

with open("data/qa_sft.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

### From RAG Retrieval Logs

If you have a running RAG system, log the queries, retrieved contexts, and (human-verified) answers:

```python
def log_for_sft(query: str, retrieved_docs: list, answer: str):
    """Log a RAG interaction for future SFT training."""
    context = "\n".join(
        f"[{i+1}] ({d['source']['filename']}) {d['text']}"
        for i, d in enumerate(retrieved_docs)
    )
    
    item = {
        "context": context,
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]
    }
    
    with open("data/rag_logs.jsonl", "a") as f:
        f.write(json.dumps(item) + "\n")
```

## Output Structure

After running `prepare_chat_sft.py`:

```
output_dir/
├── train/
│   ├── shard_00000.bin         # Token IDs (uint16)
│   ├── shard_00000_mask.bin    # Loss mask (uint8: 0 or 1)
│   └── ...
├── val/
│   ├── shard_00000.bin
│   ├── shard_00000_mask.bin
│   └── ...
├── tokenizer_state.json        # Tokenizer config
└── dataset_metadata.json       # Dataset statistics
```

## Best Practices

### Data Quality

1. **Verify assistant responses** - Only train on correct answers
2. **Match format** - Responses should match your desired output style
3. **Include diverse examples** - Cover different question types
4. **Context relevance** - Include both relevant and slightly-relevant context

### Training Tips

1. **Start small** - Use micro config first to validate pipeline
2. **Monitor loss** - Should decrease on assistant tokens
3. **Check generation** - Test with `rag_chat.py` periodically
4. **Fine-tune from pretrained** - Start from a base model, not random init:
   ```bash
   python train.py --init_from_model pretrained_base --config_file configs/sft1/small.json ...
   ```

### Context Length

- Keep total sequence (system + context + user + assistant) under `block_size`
- Typical: 3-5 context chunks work well
- Truncate context if needed, not assistant response

## Troubleshooting

### "Mask ratio too low"

If mask ratio is <10%, you have very short assistant responses or very long contexts. This is fine but training will be slower.

### "Model outputs garbage"

- Check tokenizer state was saved/loaded correctly
- Verify the base model was trained with same tokenization
- Ensure special tokens are registered

### "Context not used"

- Verify the model learned to read `<myPT_user_context>` blocks
- May need more training data with context-dependent answers
- Check that context-to-answer examples are clear

## Example: Full Pipeline

```bash
# 1. Build RAG index from your docs
python scripts/build_rag_index.py \
    --docs_dir workspace/docs \
    --out_dir workspace/index/v1

# 2. Create SFT dataset (you provide the JSONL)
python scripts/prepare_chat_sft.py \
    --input data/my_qa_pairs.jsonl \
    --output_dir data/my_sft

# 3. Train (fine-tune from base model)
python train.py \
    --model_name my_rag_assistant \
    --init_from_model my_base_model \
    --dataset_dir data/my_sft \
    --config_file configs/sft1/small.json \
    --max_iters 10000

# 4. Chat with RAG
python scripts/rag_chat.py \
    --model_name my_rag_assistant \
    --index_dir workspace/index/v1
```

## See Also

- [SFT Loss Masking](SFT_LOSS_MASKING.md) - Core loss masking implementation
- [Special Tokens](SPECIAL_TOKENS.md) - Token definitions
- [Tokenization Comparison](TOKENIZATION_COMPARISON.md) - GPT-2 vs character tokenization

