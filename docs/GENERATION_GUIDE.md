# Text Generation Guide

MyPT provides advanced sampling controls for high-quality text generation. This guide explains the parameters and recommended presets for different use cases.

## Quick Start

```python
from core import load_model

model = load_model("my_model")

# Default generation (recommended for most cases)
output = model.generate("Hello, how are you?", max_new_tokens=200)
```

## Parameters

| Parameter            | Default | Range          | Description                    |
| -------------------- | ------- | -------------- | ------------------------------ |
| `temperature`        | `0.8`   | 0.0 - 2.0+     | Controls randomness/creativity |
| `top_k`              | `50`    | 0 - vocab_size | Only consider top K tokens     |
| `top_p`              | `0.95`  | 0.0 - 1.0      | Nucleus sampling threshold     |
| `repetition_penalty` | `1.1`   | 1.0 - 2.0      | Penalize repeated tokens       |
| `stop_tokens`        | `None`  | list[int]      | Token IDs to stop on           |

---

## Parameter Deep Dive

### Temperature

Controls the "sharpness" of the probability distribution.

| Value  | Effect                                  | Use Case                 |
| ------ | --------------------------------------- | ------------------------ |
| `0.0`  | Deterministic (always pick most likely) | Testing, reproducibility |
| `0.3`  | Very focused                            | Factual answers, code    |
| `0.7`  | Balanced                                | Chat, general use        |
| `1.0`  | Neutral (model's natural distribution)  | Creative writing         |
| `1.5+` | Very random                             | Brainstorming, variety   |

```python
# Low temperature = focused, predictable
model.generate("The capital of France is", temperature=0.3)
# → "The capital of France is Paris."

# High temperature = creative, varied
model.generate("The capital of France is", temperature=1.2)
# → "The capital of France is a city of lights and romance..."
```

### Top-K Filtering

Only consider the K most likely tokens at each step.

| Value | Effect                         |
| ----- | ------------------------------ |
| `0`   | Disabled (all tokens possible) |
| `10`  | Very restrictive               |
| `50`  | Balanced (default)             |
| `100` | More variety                   |

```python
# Restrictive top-k = safer, more coherent
model.generate("Once upon a time", top_k=20)

# Larger top-k = more creative possibilities
model.generate("Once upon a time", top_k=100)
```

### Top-P (Nucleus Sampling)

Keep the smallest set of tokens whose cumulative probability exceeds `top_p`.

**Why it's better than top-k alone:** Adapts to model confidence!

| Value  | Effect                                  |
| ------ | --------------------------------------- |
| `1.0`  | Disabled                                |
| `0.95` | Keep 95% probability mass (recommended) |
| `0.9`  | More focused                            |
| `0.5`  | Very focused                            |

```python
# High confidence prediction: "Paris is the capital of ___"
# → top_p=0.95 might only keep "France" (it has 98% prob)

# Low confidence prediction: "I like to eat ___"
# → top_p=0.95 keeps many food tokens (probability spread out)
```

### Repetition Penalty

Reduces the probability of tokens that have already appeared.

| Value | Effect                            |
| ----- | --------------------------------- |
| `1.0` | Disabled                          |
| `1.1` | Mild penalty (recommended)        |
| `1.3` | Strong penalty                    |
| `2.0` | Very strong (may break coherence) |

```python
# Without repetition penalty
"The cat sat on the mat. The cat sat on the mat. The cat..."

# With repetition_penalty=1.2
"The cat sat on the mat. It stretched lazily and yawned..."
```

### Stop Tokens

Stop generation when specific tokens are produced.

```python
# Stop at newline (useful for single-line answers)
newline_token = model.tokenizer.encode("\n")[0]
output = model.generate("Q: What is 2+2?\nA:", stop_tokens=[newline_token])

# Stop at end-of-text token
eos_token = model.tokenizer.encode("<|endoftext|>")[0]
output = model.generate("Hello", stop_tokens=[eos_token])
```

---

## Recommended Presets

### Chat / RAG (Default)

Balanced settings for conversational AI and retrieval-augmented generation.

```python
output = model.generate(
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1
)
```

### Creative Writing

More variety and unexpected turns for stories, poetry, etc.

```python
output = model.generate(
    "Once upon a time in a land far away,",
    max_new_tokens=500,
    temperature=1.0,
    top_k=100,
    top_p=0.95,
    repetition_penalty=1.2
)
```

### Code Generation

Focused and precise for programming tasks.

```python
output = model.generate(
    "def fibonacci(n):\n    ",
    max_new_tokens=200,
    temperature=0.2,
    top_k=40,
    top_p=0.95,
    repetition_penalty=1.0  # Repetition often valid in code
)
```

### Factual / Q&A

Tight filtering for accurate, factual responses.

```python
output = model.generate(
    "The chemical formula for water is",
    max_new_tokens=50,
    temperature=0.3,
    top_k=10,
    top_p=0.8,
    repetition_penalty=1.0
)
```

### Deterministic

Always produces the same output (for testing/debugging).

```python
output = model.generate(
    "2 + 2 =",
    max_new_tokens=10,
    temperature=0.0,  # or very low like 0.001
    top_k=1,          # Only pick the top token
    top_p=1.0,
    repetition_penalty=1.0
)
```

### Brainstorming

Maximum variety for generating diverse ideas.

```python
output = model.generate(
    "Ideas for a new mobile app:",
    max_new_tokens=300,
    temperature=1.3,
    top_k=0,          # No top-k filtering
    top_p=0.98,
    repetition_penalty=1.3
)
```

---

## Preset Summary Table

| Preset            | temp | top_k | top_p | rep_penalty | Best For           |
| ----------------- | ---- | ----- | ----- | ----------- | ------------------ |
| **Chat/RAG**      | 0.7  | 50    | 0.9   | 1.1         | Conversations, Q&A |
| **Creative**      | 1.0  | 100   | 0.95  | 1.2         | Stories, poetry    |
| **Code**          | 0.2  | 40    | 0.95  | 1.0         | Programming        |
| **Factual**       | 0.3  | 10    | 0.8   | 1.0         | Facts, definitions |
| **Deterministic** | 0.0  | 1     | 1.0   | 1.0         | Testing, debugging |
| **Brainstorm**    | 1.3  | 0     | 0.98  | 1.3         | Idea generation    |

---

## Legacy Generation

For testing or when you want pure model output without any filtering:

```python
output = model.generate_simple("Hello", max_new_tokens=100)
```

This uses basic sampling without temperature scaling, top-k/p filtering, or repetition penalty.

---

## Troubleshooting

### Output is repetitive

- Increase `repetition_penalty` (try 1.2 - 1.5)
- Decrease `temperature` slightly

### Output is nonsensical / too random

- Decrease `temperature` (try 0.5 - 0.7)
- Decrease `top_k` (try 20 - 40)
- Decrease `top_p` (try 0.8 - 0.9)

### Output is too boring / predictable

- Increase `temperature` (try 0.9 - 1.1)
- Increase `top_k` or set to 0
- Increase `top_p` (try 0.95 - 0.98)

### Output cuts off mid-sentence

- Increase `max_new_tokens`
- Remove or adjust `stop_tokens`

### Same output every time (unintentionally)

- Increase `temperature` above 0
- Increase `top_k` above 1
