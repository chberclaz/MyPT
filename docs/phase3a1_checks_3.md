## TASK: Fix Phase3a curriculum robustness (echo/gibberish) + add eval gates

Context:
- Current failure: "gibberish echo" degraded overall format+echo and caused safe-token collapse (e.g. Zurich).
- Hypothesis: GPT-2 BPE makes random gibberish hard/unlearnable (too many rare sub-tokens). Model minimizes loss via frequent short tokens.
- Goal: Keep staged curriculum (3a-1α, β, γ, 3a-2) but make γ learnable + prevent echo mode-collapse.

### 0) Add new evaluation gates (must run after each phase)
Create scripts/sft_eval_suite.py that loads a model and runs deterministic generation (temperature=0, top_k=0, top_p=1, repetition_penalty=1.0) on a fixed prompt set.

Add 4 eval buckets with pass/fail thresholds:
A) format_strict:
  - must output within <myPT_assistant> ... </myPT_assistant> and stop at </myPT_assistant> or <myPT_eot>
  - must not produce malformed tags
B) echo_basic:
  - "Say Hello" -> "Hello"
  - "Repeat: Hello world" -> "Hello world"
C) anti_echo:
  - "What is 'Blurpix'?" -> must NOT output "Blurpix" (expect "Unknown." or "No meaning.")
  - "Is 'Zanthor quexling' a real word?" -> must NOT echo full string
D) regression_basic:
  - simple math: 5+7 -> 12
  - simple fact: capital of Germany -> Berlin
  (These can be weak early, but must not collapse into constant token like "Zurich" for all inputs.)

Output a JSON summary with per-bucket pass rates and show failing examples.

### 1) Fix gibberish generation: make it BPE-friendly
Modify scripts/generate_echo_dataset.py (or create generate_echo_dataset_bpe_safe.py):

Add option:
  --gibberish_mode {exclude, include, only}
  --bpe_safe true/false (default true for gibberish_mode include/only)
  --max_target_tokens N (default 4)

Implementation idea (fast):
  - For each candidate echo target string, run tokenizer.encode(target)
  - If len(encoded) > max_target_tokens: reject and resample
  - Also reject targets that encode to any special token ids (>=50257)
  - Prefer targets made from a syllable list (blur/pix/zan/thor/quex/ling/vex/tron/neo/proto etc.)
  - Ensure targets include a mix of:
      single-word, two-word, with digits, with hyphen (but still BPE-safe via token-length filter)

This guarantees γ examples are learnable and prevents loss spikes that cause "safe token" collapse.

### 2) Add anti-echo negatives into β and γ
In echo dataset generation, add a configurable fraction (e.g. 20%) of "anti-echo" templates:
  - "What is '{X}'?" -> "Unknown."
  - "Is '{X}' a real word?" -> "No."
  - "Translate '{X}' to German." -> "Unknown."
Where X is the same echo target pool (including gibberish).
IMPORTANT: these are assistant-span targets, still short.

### 2b) Add "contrast pairs" to force abstract instruction following (CRITICAL)
Implement paired examples where the SAME payload string X appears in multiple instruction frames:

For each sampled payload X (including normal + BPE-safe gibberish), generate 4 examples:

1) echo_quote:
   User: Say "{X}"
   Assistant: {X}

2) echo_prefix:
   User: Repeat: {X}
   Assistant: {X}

3) answer_if_question:
   If X is a known factual question template, ask it as a real question:
   User: What is the capital city of Germany?
   Assistant: Berlin
   (Use a small controlled set: capitals, 1-step arithmetic, simple translations)

4) anti_echo_meta:
   User: Is "{X}" the answer to the question "What is the capital of Germany?"
   Assistant: No.
   OR (for gibberish X):
   User: Does "{X}" have a meaning in English?
   Assistant: No.

Key rule:
- The exact string X must appear in BOTH echo and non-echo contexts.
- This prevents the model from learning "quoted text => copy" or "colon => copy".
- Mix ratio: ensure ≥20% of echo dataset are these contrast groups.


### 3) Enforce mix ratios during dataset mixing
Update scripts/mix_sft_jsonl.py to support explicit weights:
  --inputs fileA fileB ...
  --weights 0.85 0.15 ...
so we can guarantee replay ratios (not just concatenation + shuffle).

Recommended ratios:
- 3a-1β: 70% echo_beta + 20% format_lock_alpha + 10% anti_echo (if separate file, else baked in)
- 3a-1γ: 35% gibberish_echo_bpe_safe + 50% echo_beta + 15% format_lock_alpha
- 3a-2: keep replay from gamma/beta at 15-25% (avoid forgetting format/echo)

### 4) Tie-weights correctness audit
Add a small script scripts/check_tie_weights.py:
- Load a checkpoint
- Print config.tie_weights
- Verify that model.lm_head.weight and model.token_embedding_table.weight are the SAME tensor (data_ptr equality)
- If tie_weights true but pointers differ -> fail loudly

Also ensure train.py:
- If enabling tie_weights mid-init, do it before training starts, once.
- Ensure config saved with tie_weights=true.

### 5) Add "train vs eval parity" sanity check (dropout / mode bugs)
Add scripts/debug_train_eval_parity.py:
- Load model
- Take a fixed batch from val loader
- Compute loss in model.eval() and model.train() with dropout disabled/enabled
- Print both. They should differ slightly (dropout), but not explode.
- Also print a forward-pass parity in eval with use_cache=False to rule out inference-only bugs.

### Deliverables
- New/updated scripts:
  - scripts/sft_eval_suite.py
  - scripts/generate_echo_dataset.py (or *_bpe_safe.py) with BPE-safe gibberish via token-length filtering
  - scripts/mix_sft_jsonl.py updated with --weights
  - scripts/check_tie_weights.py
  - scripts/debug_train_eval_parity.py
- Update docs/Phase3a pipeline markdown to:
  - set recommended ratios
  - replace "gibberish" with "BPE-safe gibberish" definition
  - add eval gates as mandatory step after each phase
