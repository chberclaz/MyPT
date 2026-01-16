# Special Tokens Guide

This guide documents the special token system in MyPT, which allows you to define and use custom tokens for structured text like conversations, tool calls, and other formatted content.

---

## Table of Contents

1. [Overview](#overview)
2. [Available Special Tokens](#available-special-tokens)
3. [How It Works](#how-it-works)
4. [Usage](#usage)
5. [Adding Custom Special Tokens](#adding-custom-special-tokens)
6. [Implementation Details](#implementation-details)
7. [Best Practices](#best-practices)

---

## Overview

Special tokens are reserved tokens that have specific meanings in your model's vocabulary. They're used to:

- **Structure conversations** - Mark user/assistant turns
- **Define boundaries** - Separate different parts of input
- **Enable tool use** - Mark tool calls and results
- **Control generation** - Signal end of turn, context boundaries

### Key Features

- ✅ **Reserved IDs** - Special tokens get IDs after the base vocabulary (50257+ for GPT-2)
- ✅ **Automatic encoding** - `<user>` in text → special token ID
- ✅ **Automatic decoding** - Special token ID → `<user>` in output
- ✅ **Named access** - Get token IDs by name (`user_open`, `assistant_close`)
- ✅ **Serializable** - State saved/restored with model checkpoints

---

## Available Special Tokens

Special tokens are defined in `core/special_tokens.py`:

| Name | String | Purpose |
|------|--------|---------|
| `system_open` | `<system>` | Start of system message |
| `system_close` | `</system>` | End of system message |
| `user_open` | `<user>` | Start of user message |
| `user_close` | `</user>` | End of user message |
| `user_context_open` | `<user_context>` | Start of user context |
| `user_context_close` | `</user_context>` | End of user context |
| `assistant_open` | `<assistant>` | Start of assistant message |
| `assistant_close` | `</assistant>` | End of assistant message |
| `assistant_context_open` | `<assistant_context>` | Start of assistant context |
| `assistant_context_close` | `</assistant_context>` | End of assistant context |
| `tool_call_open` | `<tool_call>` | Start of tool call |
| `tool_call_close` | `</tool_call>` | End of tool call |
| `tool_result_open` | `<tool_result>` | Start of tool result |
| `tool_result_close` | `</tool_result>` | End of tool result |
| `eot` | `<eot>` | End of turn / message separator |

---

## How It Works

### Token ID Assignment

For GPT-2 tokenization:

```
Base vocabulary:     IDs 0 - 50256    (50257 tokens)
Special tokens:      IDs 50257 - 50269 (13 tokens)
Reserved for future: IDs 50270 - 50303 (34 slots)
Model vocab_size:    50304 (padded for GPU efficiency)
```

When the tokenizer is initialized, special tokens are registered in order:

```python
system_open:           50257
system_close:          50258
user_open:             50259
user_close:            50260
user_context_open:     50261
user_context_close:    50262
assistant_open:        50263
assistant_close:       50264
assistant_context_open: 50265
assistant_context_close: 50266
tool_call_open:        50267
tool_call_close:       50268
tool_result_open:      50269
tool_result_close:     50270
eot:                   50271
```

### Encoding Process

When you encode text containing special tokens:

```python
text = "<user>Hello!</user>"

# 1. Find special token "<user>" at position 0
# 2. Encode nothing before it
# 3. Add special token ID 50259
# 4. Find "</user>" 
# 5. Encode "Hello!" with tiktoken → [15496, 0]
# 6. Add special token ID 50260

result = [50259, 15496, 0, 50260]
```

### Decoding Process

When you decode token IDs:

```python
ids = [50259, 15496, 0, 50260]

# 1. See 50259 → special token "<user>"
# 2. See 15496, 0 → regular tokens, decode with tiktoken → "Hello!"
# 3. See 50260 → special token "</user>"

result = "<user>Hello!</user>"
```

---

## Usage

### Basic Encoding/Decoding

```python
from core import GPTConfig
from core.tokenizer import Tokenizer

# Create tokenizer
config = GPTConfig(vocab_size=50304)
tokenizer = Tokenizer(config, 'gpt2')
# Output: Registered 13 special tokens (IDs 50257-50269)

# Encode text with special tokens
text = "<user>What is 2+2?</user><assistant>4</assistant>"
tokens = tokenizer.encode(text)
print(tokens)
# [50259, 2061, 318, 362, 10, 17, 30, 50260, 50263, 19, 50264]

# Decode back
decoded = tokenizer.decode(tokens)
print(decoded)
# "<user>What is 2+2?</user><assistant>4</assistant>"
```

### Accessing Special Token IDs

```python
# Get ID by name
user_open_id = tokenizer.get_special_token_id("user_open")
print(user_open_id)  # 50259

# Get string by name
user_open_str = tokenizer.get_special_token_string("user_open")
print(user_open_str)  # "<user>"

# Check if ID is a special token
print(tokenizer.is_special_token(50259))  # True
print(tokenizer.is_special_token(1234))   # False
```

### Building Prompts Programmatically

```python
# Build a chat prompt
user_open = tokenizer.get_special_token_id("user_open")
user_close = tokenizer.get_special_token_id("user_close")
assistant_open = tokenizer.get_special_token_id("assistant_open")

# Method 1: Encode text with special tokens
prompt = "<user>Hello!</user><assistant>"
tokens = tokenizer.encode(prompt)

# Method 2: Build token list manually
question = "Hello!"
tokens = (
    [user_open] + 
    tokenizer.encode_ordinary(question) + 
    [user_close, assistant_open]
)
```

### Encoding Without Special Token Handling

If you want to encode text literally (treating `<user>` as regular text):

```python
# This will NOT recognize special tokens
tokens = tokenizer.encode_ordinary("<user>Hello</user>")
# Encodes "<user>Hello</user>" as regular BPE tokens
```

---

## Adding Custom Special Tokens

### Step 1: Edit `core/special_tokens.py`

```python
SPECIAL_TOKEN_STRINGS = {
    # Existing tokens...
    "system_open": "<system>",
    "system_close": "</system>",
    # ...
    
    # Add your custom tokens
    "code_open": "<code>",
    "code_close": "</code>",
    "think_open": "<think>",
    "think_close": "</think>",
}
```

### Step 2: Ensure Enough Vocab Slots

Make sure your `config.vocab_size` has enough room:

```python
# GPT-2 base vocab: 50257
# Your special tokens: N
# Need vocab_size >= 50257 + N

# Default 50304 gives 47 slots for special tokens
config = GPTConfig(vocab_size=50304)
```

### Step 3: Retrain or Fine-tune

Special tokens need to be learned by the model:

1. **New model:** Include special tokens in training data
2. **Fine-tuning:** The new token embeddings start random, need training

---

## Implementation Details

### Tokenizer Initialization

```python
class Tokenizer:
    def __init__(self, config, kind):
        # ...
        
        # Get base vocab size
        if self.token_kind == 'gpt2':
            self.base_vocab_size = self.enc.n_vocab  # 50257
        
        # Mappings
        self.special_token_encoder = {}  # str -> id
        self.special_token_decoder = {}  # id -> str
        self.special_tokens = {}         # name -> id
        
        # Register special tokens
        if self.token_kind == 'gpt2':
            self._register_special_tokens()
```

### Registration Logic

```python
def _register_special_tokens(self):
    next_id = self.base_vocab_size  # Start at 50257
    
    for name, tok_str in SPECIAL_TOKEN_STRINGS.items():
        if next_id >= self.model_vocab_size:
            raise ValueError("Not enough vocab slots!")
        
        self.special_token_encoder[tok_str] = next_id
        self.special_token_decoder[next_id] = tok_str
        self.special_tokens[name] = next_id
        next_id += 1
```

### Encode with Special Tokens

```python
def __encode_gpt2_with_special(self, text):
    result = []
    remaining = text
    
    while remaining:
        # Find earliest special token
        earliest_pos = len(remaining)
        earliest_token = None
        
        for tok_str in self.special_token_encoder:
            pos = remaining.find(tok_str)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
                earliest_token = tok_str
        
        if earliest_token is None:
            # No more special tokens
            result.extend(self.enc.encode_ordinary(remaining))
            break
        else:
            # Encode text before special token
            if earliest_pos > 0:
                result.extend(self.enc.encode_ordinary(remaining[:earliest_pos]))
            
            # Add special token ID
            result.append(self.special_token_encoder[earliest_token])
            
            # Continue after special token
            remaining = remaining[earliest_pos + len(earliest_token):]
    
    return result
```

### Decode with Special Tokens

```python
def __decode_gpt2_with_special(self, ids):
    result = []
    current_chunk = []
    
    for token_id in ids:
        if token_id in self.special_token_decoder:
            # Decode accumulated regular tokens
            if current_chunk:
                result.append(self.enc.decode(current_chunk))
                current_chunk = []
            # Add special token string
            result.append(self.special_token_decoder[token_id])
        else:
            current_chunk.append(token_id)
    
    # Decode remaining regular tokens
    if current_chunk:
        result.append(self.enc.decode(current_chunk))
    
    return ''.join(result)
```

---

## Best Practices

### 1. Use Consistent Formatting

Always use complete open/close pairs:

```python
# ✅ Good
"<user>Hello</user><assistant>Hi!</assistant>"

# ❌ Bad - missing close tag
"<user>Hello<assistant>Hi!"
```

### 2. Don't Nest Same-Type Tags

```python
# ✅ Good
"<user>Question 1</user><user>Question 2</user>"

# ❌ Bad - nested user tags
"<user>Outer <user>Inner</user></user>"
```

### 3. Include Special Tokens in Training Data

For the model to learn special tokens, include them in training:

```python
training_text = """
<user>What is Python?</user>
<assistant>Python is a programming language.</assistant>
<eot>
<user>How do I install it?</user>
<assistant>Use your package manager or download from python.org.</assistant>
<eot>
"""
```

### 4. Use End-of-Turn Markers

Use `<eot>` to separate conversation turns:

```python
conversation = """
<user>Hello!</user>
<assistant>Hi there!</assistant>
<eot>
<user>How are you?</user>
<assistant>I'm doing well, thanks!</assistant>
<eot>
"""
```

### 5. Handle Unknown Characters Gracefully

Special tokens not in your vocabulary will be encoded as regular text:

```python
# If <custom> is not registered
tokenizer.encode("<custom>text</custom>")
# Encodes "<custom>" as regular BPE tokens, not as special token
```

---

## Character-Level Tokenization

Special tokens are currently only implemented for GPT-2 tokenization. For character-level:

- Special token registration is skipped
- You can still include special token strings in text
- They'll be encoded character-by-character

If you need special tokens with character-level, you could:

1. Add them to the character vocabulary
2. Implement similar logic in `__encode_char` / `__decode_char`

---

## Troubleshooting

### "Not enough reserved vocab slots!"

Your `config.vocab_size` is too small for all special tokens.

```python
# Check how many slots you need
from core.special_tokens import SPECIAL_TOKEN_STRINGS
print(f"Need {len(SPECIAL_TOKEN_STRINGS)} slots")  # e.g., 13

# GPT-2 base: 50257
# Your config needs: vocab_size >= 50257 + 13 = 50270
# Default 50304 is fine
```

### Special tokens not being recognized

Make sure you're using `encode()` not `encode_ordinary()`:

```python
# ✅ Recognizes special tokens
tokenizer.encode("<user>Hello</user>")

# ❌ Treats as regular text
tokenizer.encode_ordinary("<user>Hello</user>")
```

### Model generates broken special tokens

If the model outputs `<use` or `er>` separately, it hasn't learned the special tokens well:

1. Include more examples in training data
2. Train for more iterations
3. Consider using a special token as generation start (e.g., always start with `<assistant>`)

---

## See Also

- [Tokenization Comparison](TOKENIZATION_COMPARISON.md) - GPT-2 vs character-level
- [SFT Loss Masking](SFT_LOSS_MASKING.md) - Using special tokens for chat training
- [core/special_tokens.py](../core/special_tokens.py) - Token definitions
- [core/tokenizer.py](../core/tokenizer.py) - Implementation

