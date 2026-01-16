# Toolcall SFT Training

This guide covers how to create and train on SFT datasets for agentic tool-calling behavior.

## Overview

Toolcall SFT teaches your model to:

1. Recognize when to call workspace tools
2. Format toolcall blocks with correct JSON
3. Interpret tool results
4. Generate final answers based on tool outputs

---

## Complete Pipeline

The agentic RAG SFT pipeline has three steps:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENTIC RAG SFT PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Generate Synthetic Data
────────────────────────────────
    python scripts/generate_agent_sft.py \
        --docs_dir workspace/docs \
        --output data/agent_sft.jsonl \
        --num_examples 500
                    │
                    ▼
            ┌───────────────────┐
            │  agent_sft.jsonl  │  ← Human-readable intermediate format
            │  (JSONL format)   │    Uses: "role", "content", "messages"
            └───────────────────┘
                    │
Step 2: Convert to Training Format
──────────────────────────────────
    python scripts/prepare_tool_sft.py \
        --input data/agent_sft.jsonl \
        --output_dir data/agent_sft
                    │
                    ▼
            ┌───────────────────────────────────────────┐
            │  data/agent_sft/                          │
            │    train/shard_00000.bin      (tokens)    │
            │    train/shard_00000_mask.bin (loss mask) │  ← Tokenized format
            │    val/shard_00000.bin                    │    Uses: <myPT_*> tags
            │    val/shard_00000_mask.bin               │
            │    tokenizer_state.json                   │
            └───────────────────────────────────────────┘
                    │
Step 3: Train Model
───────────────────
    python train.py \
        --model_name my_agent \
        --init_from_model base_model \
        --dataset_dir data/agent_sft \
        --config_file configs/sft2/toolchat.json
```

---

## Special Tokens (from `core/special_tokens.py`)

The training data uses these special tokens:

| Token Name              | String               | Used For                 |
| ----------------------- | -------------------- | ------------------------ |
| `myPT_system_open`      | `<myPT_system>`      | System message start     |
| `myPT_system_close`     | `</myPT_system>`     | System message end       |
| `myPT_user_open`        | `<myPT_user>`        | User message start       |
| `myPT_user_close`       | `</myPT_user>`       | User message end         |
| `myPT_assistant_open`   | `<myPT_assistant>`   | Assistant response start |
| `myPT_assistant_close`  | `</myPT_assistant>`  | Assistant response end   |
| `myPT_toolcall_open`    | `<myPT_toolcall>`    | Tool call start          |
| `myPT_toolcall_close`   | `</myPT_toolcall>`   | Tool call end            |
| `myPT_toolresult_open`  | `<myPT_toolresult>`  | Tool result start        |
| `myPT_toolresult_close` | `</myPT_toolresult>` | Tool result end          |
| `myPT_eot`              | `<myPT_eot>`         | End of turn/conversation |

---

## Data Format Conversion

### Input: JSONL (Human-Readable)

The JSONL format is an **intermediate representation** for humans to read and edit:

```json
{
  "system": "You are MyPT assistant.",
  "messages": [
    { "role": "user", "content": "Search for Python docs" },
    {
      "role": "assistant_toolcall",
      "name": "workspace.search",
      "arguments": { "query": "Python" }
    },
    {
      "role": "toolresult",
      "name": "workspace.search",
      "content": { "documents": [{ "text": "Python is great" }] }
    },
    { "role": "assistant", "content": "Found Python documentation." }
  ]
}
```

### Output: Training Text (What LLM Sees)

`prepare_tool_sft.py` converts the JSONL to actual training text using special tokens:

```
<myPT_system>You are MyPT assistant.</myPT_system>
<myPT_user>Search for Python docs</myPT_user>
<myPT_assistant><myPT_toolcall>{"name":"workspace.search","query":"Python"}</myPT_toolcall></myPT_assistant>
<myPT_toolresult>{"documents":[{"text":"Python is great"}]}</myPT_toolresult>
<myPT_assistant>Found Python documentation.</myPT_assistant>
<myPT_eot>
```

### Conversion Rules

| JSONL Role                                                          | Serialized As                                                                                |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `"system": "text"`                                                  | `<myPT_system>text</myPT_system>`                                                            |
| `{"role": "user", "content": "text"}`                               | `<myPT_user>text</myPT_user>`                                                                |
| `{"role": "assistant", "content": "text"}`                          | `<myPT_assistant>text</myPT_assistant>`                                                      |
| `{"role": "assistant_toolcall", "name": "...", "arguments": {...}}` | `<myPT_assistant><myPT_toolcall>{"name":"...","arg":"val"}</myPT_toolcall></myPT_assistant>` |
| `{"role": "toolresult", "content": {...}}`                          | `<myPT_toolresult>{...json...}</myPT_toolresult>`                                            |

---

## JSONL Input Format

Each line is a JSON object representing a conversation with tool usage:

```json
{
  "system": "You are MyPT workspace assistant with access to search and document tools.",
  "messages": [
    {
      "role": "user",
      "content": "Find docs about machine learning and summarize them"
    },
    {
      "role": "assistant_toolcall",
      "name": "workspace.search",
      "arguments": { "query": "machine learning", "top_k": 3 }
    },
    {
      "role": "toolresult",
      "name": "workspace.search",
      "content": {
        "documents": [{ "doc_id": "abc", "text": "ML is..." }],
        "total": 3
      }
    },
    {
      "role": "assistant_toolcall",
      "name": "workspace.summarize",
      "arguments": { "doc_id": "abc" }
    },
    {
      "role": "toolresult",
      "name": "workspace.summarize",
      "content": { "summary": "Machine learning is..." }
    },
    {
      "role": "assistant",
      "content": "Based on the documents, machine learning is a subset of AI that..."
    }
  ]
}
```

### Message Roles

| Role                 | Description                   | Mask  |
| -------------------- | ----------------------------- | ----- |
| `user`               | User's question/request       | 0     |
| `assistant`          | Final natural language answer | **1** |
| `assistant_toolcall` | Assistant calling a tool      | **1** |
| `toolresult`         | Tool execution result         | 0     |

### Role Details

**`assistant_toolcall`**:

- `name`: Tool name (e.g., `workspace.search`)
- `arguments`: Dict of tool arguments

**`toolresult`**:

- `name`: Tool name (should match the toolcall)
- `content`: Tool result (dict or JSON string)

## Serialization Format

The pipeline serializes to this format (using `core/special_tokens.py`):

```
<myPT_system>You are MyPT workspace assistant...</myPT_system>
<myPT_user>Find docs about machine learning</myPT_user>
<myPT_assistant><myPT_toolcall>{"name": "workspace.search", "query": "machine learning", "top_k": 3}</myPT_toolcall></myPT_assistant>
<myPT_toolresult>{"documents": [...], "total": 3}</myPT_toolresult>
<myPT_assistant><myPT_toolcall>{"name": "workspace.summarize", "doc_id": "abc"}</myPT_toolcall></myPT_assistant>
<myPT_toolresult>{"summary": "Machine learning is..."}</myPT_toolresult>
<myPT_assistant>Based on the documents, machine learning is a subset of AI that...</myPT_assistant>
<myPT_eot>
```

## Loss Masking

The loss mask ensures the model only learns to generate:

- Tool calls (including the JSON formatting)
- Final answers

| Content                 | Mask  | Trained? |
| ----------------------- | ----- | -------- |
| System prompt           | 0     | No       |
| User messages           | 0     | No       |
| **Assistant messages**  | **1** | **Yes**  |
| **Assistant toolcalls** | **1** | **Yes**  |
| Tool results            | 0     | No       |

### Visual Loss Mask Example

```
                                                        LOSS MASK
                                                        ─────────
<myPT_system>You are MyPT assistant.</myPT_system>      0000000... (don't train)
<myPT_user>Search for Python docs</myPT_user>           0000000... (don't train)
<myPT_assistant><myPT_toolcall>{"name":"workspace.search"...}</myPT_toolcall></myPT_assistant>
                                                        1111111... (TRAIN!)
<myPT_toolresult>{"documents":[...]}</myPT_toolresult>  0000000... (don't train)
<myPT_assistant>Found Python documentation.</myPT_assistant>
                                                        1111111... (TRAIN!)
<myPT_eot>                                              0000000... (don't train)
```

**Why mask tool results?**

- Tool results come from executing code, not from the model
- We don't want the model to "learn" tool outputs
- The model should learn to _use_ tool outputs, not _generate_ them

## Available Tools

The default workspace tools are:

### `workspace.search`

Search documents by semantic similarity.

```json
{ "name": "workspace.search", "query": "search terms", "top_k": 5 }
```

Returns:

```json
{
  "documents": [
    { "chunk_id": "...", "doc_id": "...", "text": "...", "score": 0.85 }
  ],
  "total": 5
}
```

### `workspace.list_docs`

List all documents in workspace.

```json
{ "name": "workspace.list_docs" }
```

Returns:

```json
{
  "documents": [{ "doc_id": "...", "title": "...", "path": "..." }],
  "total": 10
}
```

### `workspace.get_doc`

Get full document text.

```json
{ "name": "workspace.get_doc", "doc_id": "abc123" }
```

Returns:

```json
{
  "doc_id": "abc123",
  "title": "Document Title",
  "text": "Full document text...",
  "length": 5000
}
```

### `workspace.summarize`

Summarize a document or text.

```json
{ "name": "workspace.summarize", "doc_id": "abc123" }
```

Returns:

```json
{
  "summary": "This document discusses...",
  "source": "doc_id",
  "original_length": 5000
}
```

## Usage

### Step 0. Prepare Workspace Documents

Add documents to your workspace:

```bash
mkdir -p workspace/docs
# Add .txt and .md files to workspace/docs/
```

### Step 1. Generate Synthetic Training Data (Optional)

If you don't have hand-crafted training data, generate synthetic examples:

```bash
python scripts/generate_agent_sft.py \
    --docs_dir workspace/docs \
    --output data/agent_sft.jsonl \
    --num_examples 500 \
    --include_errors
```

This creates JSONL with realistic tool-use patterns:

- `workspace.search` → find relevant chunks → answer
- `workspace.list_docs` → enumerate docs → describe
- `workspace.get_doc` → read full doc → quote/summarize
- `workspace.summarize` → condense content → report
- Multi-step: search → get_doc → answer

### Step 2. Convert to Tokenized Shards

```bash
python scripts/prepare_tool_sft.py \
    --input data/agent_sft.jsonl \
    --output_dir data/agent_sft \
    --tokenization gpt2 \
    --val_split 0.1
```

This converts JSONL to:

- `train/shard_*.bin` - tokenized text with `<myPT_*>` tags
- `train/shard_*_mask.bin` - loss mask (1=train, 0=context)
- `tokenizer_state.json` - tokenizer configuration

### Step 3. Train with Loss Masking

```bash
python train.py \
    --model_name my_agent \
    --init_from_model base_model \
    --dataset_dir data/agent_sft \
    --config_file configs/sft2/toolchat.json \
    --max_iters 5000
```

### Step 4. Test Agent

```bash
python scripts/workspace_chat.py \
    --model_name my_agent \
    --workspace_dir workspace/ \
    --index_dir workspace/index/latest
```

## Creating Training Data

You have three options for creating training data:

### Option A: Synthetic Generation (Recommended for Starting)

Use `generate_agent_sft.py` to create synthetic examples from your documents:

```bash
python scripts/generate_agent_sft.py \
    --docs_dir workspace/docs \
    --output data/agent_sft.jsonl \
    --num_examples 500 \
    --seed 42
```

Options:

- `--num_examples`: Number of synthetic conversations (default: 500)
- `--include_errors`: Add examples with "no results found"
- `--max_doc_chars`: Truncate long documents (default: 2000)

### Option B: Manual Creation

Create JSONL files manually with diverse tool-use patterns:

```python
import json

examples = [
    {
        "system": "You are a helpful workspace assistant.",
        "messages": [
            {"role": "user", "content": "What documents do we have?"},
            {"role": "assistant_toolcall", "name": "workspace.list_docs", "arguments": {}},
            {"role": "toolresult", "name": "workspace.list_docs", "content": {
                "documents": [{"doc_id": "a", "title": "Python Guide"}],
                "total": 1
            }},
            {"role": "assistant", "content": "You have 1 document: Python Guide."}
        ]
    },
    # More examples...
]

with open("data/tool_sft.jsonl", "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")
```

### Option C: From Agent Logs

If you have a working agent, log interactions for training:

```python
def log_for_training(history, result):
    """Log agent interaction for future training."""
    item = {
        "messages": history + [{"role": "assistant", "content": result["content"]}]
    }

    # Include tool calls
    for tc in result.get("tool_calls", []):
        # These are already in history from the agent
        pass

    with open("data/agent_logs.jsonl", "a") as f:
        f.write(json.dumps(item) + "\n")
```

## Training Tips

### Data Quality

1. **Diverse patterns** - Include various tool combinations:

   - Search only
   - Search → summarize
   - List → get_doc → answer
   - Multi-step reasoning

2. **Error cases** - Include examples where tools return errors:

   ```json
   {
     "role": "toolresult",
     "name": "workspace.get_doc",
     "content": { "error": "Document not found" }
   }
   ```

3. **No-tool cases** - Include conversations without tools:
   ```json
   {"role": "user", "content": "Hello!"},
   {"role": "assistant", "content": "Hello! How can I help you today?"}
   ```

### Training Path

Recommended training sequence:

1. **Phase 1/2**: Pretrain base model
2. **Phase 3A**: Chat SFT (without tools)
3. **Phase 3B**: Toolcall SFT (this guide)

```bash
# Fine-tune from chat model
python train.py \
    --model_name my_agent \
    --init_from_model my_chat_model \
    --dataset_dir data/tool_sft \
    --config_file configs/sft2/toolchat.json
```

### Config Settings

Use a config with `use_loss_mask: true`:

```json
{
  "name": "Toolchat-Small",
  "use_loss_mask": true,
  "dropout": 0.05,
  "batch_size": 8,
  ...
}
```

## Troubleshooting

### "Model doesn't call tools"

- Check your training data has enough toolcall examples
- Ensure mask ratio is reasonable (20-50% for tool-heavy data)
- Train longer or with more examples

### "Malformed JSON in toolcalls"

- Increase training iterations
- Add more examples with varied JSON structures
- Check tokenizer handles special characters correctly

### "Model calls wrong tool"

- Add more diverse examples showing when to use each tool
- Include negative examples (when NOT to call tools)
- Consider adding tool descriptions to system prompt

## Example Dataset

A minimal example dataset:

```jsonl
{"messages": [{"role": "user", "content": "List documents"}, {"role": "assistant_toolcall", "name": "workspace.list_docs", "arguments": {}}, {"role": "toolresult", "name": "workspace.list_docs", "content": {"documents": [{"doc_id": "1", "title": "Doc A"}], "total": 1}}, {"role": "assistant", "content": "You have 1 document: Doc A."}]}
{"messages": [{"role": "user", "content": "Search for Python"}, {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": "Python", "top_k": 3}}, {"role": "toolresult", "name": "workspace.search", "content": {"documents": [{"text": "Python is..."}], "total": 1}}, {"role": "assistant", "content": "I found information about Python..."}]}
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hello! How can I help you with your documents today?"}]}
```

## See Also

- [Workspace API](workspace_api.md) - Tool implementations
- [SFT Loss Masking](SFT_LOSS_MASKING.md) - Loss masking details
- [Chat SFT with Context](chat_sft_with_context.md) - Phase 3A RAG training
