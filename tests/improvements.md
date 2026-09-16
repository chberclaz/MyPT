# MyPT — Bug Fixes & Improvements

All issues confirmed against actual source files. **Tasks 1–2 must be done before Phase 3
dataset generation. Tasks 3–8 are ordered by severity (crash → correctness →
performance → cleanup). Tasks 9–10 are additional findings not in the original review.
Task 11 is a pipeline correctness fix required before any dataset is packed.**

---

## Task 1 — `train.py`: Remove blocking `input()` calls

**Severity: RUNPOD BLOCKER**

### Problem

`train.py` has two `input("Continue anyway? (y/n): ")` calls inside the dataset coverage
check. They block on stdin and stall unattended RunPod runs silently — the process hangs,
no error, no timeout, GPU idles and burns money.

Locations (both in `main()`, inside the `if total_tokens:` and
`if is_episode_dataset and total_episodes:` branches):

```python
# Line ~480 (token-stream path)
if coverage['coverage_ratio'] < 1.0:
    print("⚠️  Your model will not see the entire dataset!")
    response = input("Continue anyway? (y/n): ")  # <-- BLOCKS
    if response != 'y' and response != 'yes':
        print("Training cancelled.")
        return

# Line ~460 (episode-indexed path)
if coverage['coverage_ratio'] < 1.0:
    print("⚠️  Your model will not see all episodes!")
    response = input("Continue anyway? (y/n): ")  # <-- BLOCKS
    if response != 'y' and response != 'yes':
        print("Training cancelled.")
        return
```

### Approach

Add a CLI flag `--auto_confirm` (boolean, default `False`). When set, skip the `input()`
call and print a warning instead. Local interactive runs retain the prompt. RunPod launch
scripts pass `--auto_confirm`. Do **not** silently remove the coverage warning — it is
useful signal. Only remove the blocking gate.

### Solution

**1. Add the argument to `parse_args()`:**

```python
parser.add_argument("--auto_confirm", action="store_true",
                    help="Skip interactive coverage confirmation prompts "
                         "(required for unattended RunPod runs).")
```

**2. Add `"auto_confirm"` to `training_keys` in `main()`** so it gets extracted from
the config dict before `GPTConfig` is constructed, and resolve it:

```python
effective_auto_confirm = args.auto_confirm or bool(config_training.get("auto_confirm", False))
```

**3. Replace both `input()` blocks with:**

```python
if coverage['coverage_ratio'] < 1.0:
    print("⚠️  Your model will not see all episodes!")
    if not effective_auto_confirm:
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response not in ('y', 'yes'):
            print("Training cancelled. Adjust --max_iters and try again.")
            return
    else:
        print("  [auto_confirm] Continuing without interactive prompt.")
```

Apply the same pattern to both coverage blocks (token-stream and episode-indexed).

### Testing

```bash
# 1. Confirm interactive prompt still appears locally (no flag)
python train.py --config_file configs/sft/phase3_chat_sft_110k.json \
    --dataset_dir data/sft_phase3_chat

# 2. Confirm RunPod path skips prompt
python train.py --config_file configs/sft/phase3_chat_sft_110k.json \
    --dataset_dir data/sft_phase3_chat --auto_confirm

# 3. Confirm config-file path works ("auto_confirm": true in JSON)
python train.py --config_file configs/sft/phase3_chat_sft_110k.json \
    --dataset_dir data/sft_phase3_chat
# Should skip prompt without --auto_confirm on CLI
```

Expected: No hanging. Coverage warning still printed. Training starts immediately when
`--auto_confirm` is set.

---

## Task 2 — `prepare_chat_sft.py`: Fix `strict_json_schema` assistant-token loop

**Severity: PHASE 3 CORRECTNESS BLOCKER**

### Problem

Episodes where the assistant response begins immediately with `{` (strict JSON output,
no preamble) trigger a token-boundary misalignment in `char_mask_to_token_mask()` in
`scripts/sft/prepare_chat_sft.py`.

**Root cause:** The masking logic scans token IDs looking for `ASSISTANT_OPEN_ID`
(50261) to flip `in_assistant_response = True`. The GPT-2 BPE tokenizer encodes
`<myPT_assistant>{` as a single fused token in some contexts because the tokenizer
sees the `{` as part of the same text chunk passed to `encode()`.

When this fusion happens:

- `ASSISTANT_OPEN_ID` never appears as a standalone token
- `in_assistant_response` is never set to `True`
- The entire assistant response gets `mask=0` — not trained on
- At inference time the model loops or emits nothing after `<myPT_assistant>`

**Affected episodes:** Every episode in `generate_phase3_json_sft.py` (all start with
`{`). Also affects future toolcall and RAG episodes that open with a special character.

**Verification:** Run `validate_sft_episode_masks.py` on a Phase 3 JSON dataset.
Episodes with `mask_ratio ≈ 0.0` are broken.

### Solution

Fix `serialize_conversation()` in `scripts/sft/prepare_chat_sft.py` to insert a space
sentinel between the opening tag and content. This guarantees `ASSISTANT_OPEN_ID`
always tokenizes as a standalone token.

Add module-level constant:

```python
# Sentinel inserted between <myPT_assistant> and response content to prevent
# BPE fusion of the opening tag with the first content character.
# Guarantees ASSISTANT_OPEN_ID always appears as a standalone token.
ASSISTANT_OPEN_SENTINEL = " "
```

**Before:**

```python
ac = ASSISTANT_OPEN
content_and_close = content + ASSISTANT_CLOSE
text_parts.append(ac)
mask_parts.append("0" * len(ac))
text_parts.append(content_and_close)
mask_parts.append("1" * len(content_and_close))
```

**After:**

```python
ac = ASSISTANT_OPEN
content_and_close = content + ASSISTANT_CLOSE
text_parts.append(ac)
mask_parts.append("0" * len(ac))
text_parts.append(ASSISTANT_OPEN_SENTINEL)
mask_parts.append("0" * len(ASSISTANT_OPEN_SENTINEL))  # structural, not trained
text_parts.append(content_and_close)
mask_parts.append("1" * len(content_and_close))
```

The sentinel is `mask=0`. The model trains to emit the content, not the space.

### Testing

```bash
# Step 1 — reproduce before fix
python scripts/sft/generate_phase3_json_sft.py \
    --output data/test_json_fix/raw.jsonl --num_examples 200 --seed 42
python scripts/sft/prepare_chat_sft.py \
    --input data/test_json_fix/raw.jsonl --output_dir data/test_json_fix/before
python scripts/sft/validate_sft_episode_masks.py --dataset_dir data/test_json_fix/before
# Expected: some episodes with mask_ratio ~0.0

# Step 2 — verify after fix
python scripts/sft/prepare_chat_sft.py \
    --input data/test_json_fix/raw.jsonl --output_dir data/test_json_fix/after
python scripts/sft/validate_sft_episode_masks.py --dataset_dir data/test_json_fix/after
# Expected: all episodes mask_ratio > 0

# Step 3 — token boundary sanity check
from core.tokenizer import Tokenizer
from core.model import GPTConfig
from core.special_tokens import get_special_token_ids
config = GPTConfig(vocab_size=50304)
tok = Tokenizer(config, 'gpt2')
ids = tok.encode('<myPT_assistant> {"label":"A","score":1}</myPT_assistant>')
IDS = get_special_token_ids()
assert IDS["myPT_assistant_open"] in ids, "FAIL: assistant open fused!"
print("PASS: standalone at index", ids.index(IDS["myPT_assistant_open"]))

# Step 4 — regression check on Phase 2 gold dataset
python scripts/sft/validate_sft_episode_masks.py \
    --dataset_dir data/sft_phase2_unified_rebuild
# Mask ratios must be unchanged vs pre-fix baseline
```

---

## Task 3 — `train.py`: `args.vocab_size` AttributeError on CLI-only path

**Severity: CRASH**

### Problem

In `main()`, the `else` branch (no `--config_file`) constructs `GPTConfig` with
`vocab_size=args.vocab_size`. That argument is never added to the argparse parser. Any
invocation without `--config_file` raises `AttributeError: Namespace object has no
attribute 'vocab_size'` immediately.

```python
# train.py ~line 207
config = GPTConfig(
    ...
    vocab_size=args.vocab_size,  # <-- AttributeError: not in parser
    ...
)
```

### Fix

Add to `parse_args()`:

```python
parser.add_argument("--vocab_size", type=int, default=50304,
                    help="Vocabulary size (default: 50304 = GPT-2 base + special tokens). "
                         "Ignored when --config_file is used.")
```

### Testing

```bash
python train.py --input_file data/some_text.txt --max_iters 10
# Must not crash
```

---

## Task 4 — `model.py load()`: broken scoping causes NameError

**Severity: CRASH on specific load paths**

### Problem

In `GPT.load()`, `is_cuda` is defined inside `if effective_dtype is None` but used
outside it. When `load_dtype` is explicitly passed by the caller, `effective_dtype` is
already set, the `if` block is skipped, and `is_cuda` is never defined — causing
`NameError` on the next line.

Additionally, the `elif checkpoint_dtype in ('fp16'...)` and
`elif checkpoint_dtype in ('fp32'...)` branches are dangling — logically they belong
at the outer level but their indentation puts them inside the
`if torch.cuda.is_available()` block, making the fp16/fp32 log messages unreachable.

```python
# Current broken code (~line 973)
if effective_dtype is None and checkpoint_dtype is not None:
    device = config.device
    is_cuda = device.startswith('cuda') if isinstance(device, str) else (device.type == 'cuda')

if checkpoint_dtype in ('bf16', 'bfloat16') and is_cuda:  # NameError if effective_dtype was set
    if torch.cuda.is_available():
        ...
        elif checkpoint_dtype in ('fp16', 'float16'):  # unreachable
            print(f"✓ Loading fp16 checkpoint")
        elif checkpoint_dtype in ('fp32', 'float32'):  # unreachable
            print(f"✓ Loading fp32 checkpoint")
```

### Fix

```python
# 5. Smart dtype handling
effective_dtype = load_dtype

# is_cuda must be unconditional — used regardless of effective_dtype
_device = config.device
is_cuda = _device.startswith('cuda') if isinstance(_device, str) else (_device.type == 'cuda')

if effective_dtype is None and checkpoint_dtype is not None:
    if checkpoint_dtype in ('bf16', 'bfloat16') and is_cuda:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] < 8:
                print(f"⚠️  Checkpoint is bf16 but GPU (compute {capability[0]}.{capability[1]}) "
                      f"doesn't have native bf16 support.")
                print("   Auto-converting to fp16 for faster CUDA inference.")
                effective_dtype = 'fp16'
            else:
                print(f"✓ Loading bf16 checkpoint on bf16-capable GPU "
                      f"(compute {capability[0]}.{capability[1]})")
                effective_dtype = 'bf16'
    elif checkpoint_dtype in ('fp16', 'float16'):
        print(f"✓ Loading fp16 checkpoint")
    elif checkpoint_dtype in ('fp32', 'float32'):
        print(f"✓ Loading fp32 checkpoint")
```

### Testing

```python
# All three must complete without NameError
model, _, _, _ = GPT.load("checkpoints/my_model", load_dtype=None)    # auto-detect
model, _, _, _ = GPT.load("checkpoints/my_model", load_dtype='fp32')  # explicit — was crashing
model, _, _, _ = GPT.load("checkpoints/my_model", map_location='cpu') # CPU path
```

---

## Task 5 — `model.py compile_for_inference()`: compiles `self.forward` instead of `self`

**Severity: CORRECTNESS**

### Problem

`torch.compile` is designed to wrap an `nn.Module`, not a bound method. Compiling
`self.forward` bypasses `__call__` hooks (training/eval mode check,
`register_forward_hook`, AMP autocast integration). It also stores the compiled object
as an instance attribute that shadows the method, causing unexpected behaviour on
subsequent `.to()` or `.eval()` calls.

Furthermore, `nn.Module.__call__` routes through `type(self).forward(self, ...)` — not
the instance attribute — so `model(x)` silently calls the **uncompiled** forward even
after `compile_for_inference()` runs. The compilation is a silent no-op for normal
inference.

```python
# Current broken code
self.forward = torch.compile(self.forward, mode=mode)     # wrong
self.forward = torch.compile(self.forward, backend="aot_eager")  # wrong
```

### Fix

Compile `self`, return the compiled module, and let the caller use the return value:

```python
def compile_for_inference(self, mode: str = "default"):
    import torch as _torch
    import torch._dynamo
    if not hasattr(_torch, "compile"):
        print("torch.compile() not available (requires PyTorch 2.0+)")
        return self
    torch._dynamo.config.suppress_errors = True
    print(f"Compiling model with mode='{mode}'...")
    try:
        compiled = torch.compile(self, mode=mode)
        print("Model compiled successfully!")
        return compiled
    except Exception:
        print("Note: Triton not available, using eager mode fallback")
        return torch.compile(self, backend="aot_eager")
```

Update **all call sites** to capture the return value:

```python
model = model.compile_for_inference()  # not just model.compile_for_inference()
```

**Known call site that must be fixed:** `webapp/routers/chat.py` line 127:

```python
# BROKEN — discards return value, stores uncompiled model
model.compile_for_inference(mode="default")
_loaded_models[model_name] = model

# FIXED
model = model.compile_for_inference(mode="default")
_loaded_models[model_name] = model
```

Search `generate.py` and all `scripts/` for `compile_for_inference` and verify return
value is captured everywhere.

### Testing

```python
model = GPT.load("checkpoints/my_model")[0]
model = model.compile_for_inference()
output = model.generate("Test prompt", max_new_tokens=20)
# Must not raise, must produce coherent output
```

---

## Task 6 — `data_loader.py _get_batch_sharded()`: double shard load per batch

**Severity: PERFORMANCE**

### Problem

`_get_batch_sharded()` selects and loads a shard twice per call. The first load is
immediately thrown away when the `while True` loop re-selects:

```python
# Dead load — result immediately overwritten
shard_idx = np.random.randint(0, len(shards))
shard_path = shards[shard_idx]
shard_data = self._load_shard(shard_path)  # <-- wasted disk hit

while True:
    shard_idx = np.random.randint(0, len(shards))
    shard_data = self._load_shard(shards[shard_idx])  # actual load
    ...
```

`_load_shard` uses `np.fromfile` (not mmap), so every call hits disk. On RunPod with
network storage this is measurable latency on every single batch.

### Fix

Remove the three dead lines before the `while True`:

```python
def _get_batch_sharded(self, split='train'):
    shards = self.train_shards if split == 'train' else self.val_shards
    if not shards:
        raise ValueError(f"No shards available for split '{split}'")
    while True:
        shard_idx = np.random.randint(0, len(shards))
        shard_data = self._load_shard(shards[shard_idx])
        max_start = len(shard_data) - self.config.block_size - 1
        if max_start > 0:
            break
    # ... rest unchanged
```

### Testing

Instrument `_load_shard` with a call counter and verify exactly one call per
`_get_batch_sharded` invocation. Or measure tokens/sec before and after.

---

## Task 7 — `model.py fit()`: `gpu_mem()` debug call left in eval path

**Severity: NOISE**

### Problem

`gpu_mem("after_gc_and_empty_cache")` fires unconditionally after every eval checkpoint
save. Its siblings `_mem()` and `gpu_mem("after_save")` are correctly commented out,
but this one was missed:

```python
# _mem("after_save")             # correctly commented out
# gpu_mem("after_save")          # correctly commented out
import gc
gc.collect()
torch.cuda.empty_cache()
gpu_mem("after_gc_and_empty_cache")  # <-- NOT commented out
```

This clutters stdout at every eval interval and calls `torch.cuda.synchronize()`
inside `gpu_mem`, adding an unnecessary sync point in the training loop.

### Fix

```python
import gc
gc.collect()
torch.cuda.empty_cache()
# gpu_mem("after_gc_and_empty_cache")
```

If GPU memory monitoring is wanted long-term, it belongs in the JSONL log entry behind
a `--debug_gpu_mem` flag.

### Testing

Run a short training loop. Confirm `[GPU after_gc_and_empty_cache]` lines no longer
appear in stdout.

---

## Task 8 — `model.py generate()`: `no_repeat_ngram` is hardcoded dead code

**Severity: CLEANUP**

### Problem

Inside the sampling loop, `no_repeat_ngram` is assigned `0` locally and the guard
`if no_repeat_ngram and no_repeat_ngram > 0` is always `False`. `_ban_repeat_ngrams`
is implemented correctly but permanently unreachable:

```python
no_repeat_ngram=0   # always 0
if no_repeat_ngram and no_repeat_ngram > 0:  # always False
    self._ban_repeat_ngrams(work_logits, out_ids, int(no_repeat_ngram))
```

### Fix

Wire it up as a proper `generate()` parameter and remove the local assignment:

```python
def generate(self, prompt, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
             repetition_penalty=1.1, stop_tokens=None, recent_penalty_window=256,
             use_default_stop_tokens=True, no_repeat_ngram: int = 0):  # <-- add
    ...
    # Delete the local: no_repeat_ngram=0
    if no_repeat_ngram and no_repeat_ngram > 0:
        self._ban_repeat_ngrams(work_logits, out_ids, int(no_repeat_ngram))
```

### Testing

```python
output = model.generate("Test", max_new_tokens=50, no_repeat_ngram=3)
# Manually verify no 3-gram repeats in output
```

---

## Task 9 — `prepare_tool_sft.py`: same BPE fusion bug as Task 2

**Severity: PHASE 5 & 6 CORRECTNESS BLOCKER**

### Problem

`prepare_tool_sft.py` has the same BPE fusion bug as Task 2, but it's not fixable by
the same sentinel because `assistant_toolcall` turns are built as a raw f-string,
bypassing any `serialize_conversation()` equivalent:

```python
# Lines 219 and 235 — direct f-string, no sentinel
a = f"{ASSISTANT_OPEN}{inner}{ASSISTANT_CLOSE}\n"
```

If `inner` starts with `{` (which every toolcall does — it's a JSON arguments block),
`<myPT_assistant>{` fuses into a single BPE token. `ASSISTANT_OPEN_ID` never appears
standalone. `in_assistant_response` stays `False`. The entire toolcall turn is
`mask=0`. The model never trains on when or how to emit tool calls.

**Affected phases:** Phase 5 (all episodes), Phase 6 (all agentic episodes).

### Fix

Apply the same `ASSISTANT_OPEN_SENTINEL` pattern as Task 2. Add the constant at
module level in `prepare_tool_sft.py`:

```python
ASSISTANT_OPEN_SENTINEL = " "
```

Find every location that builds an assistant or assistant_toolcall block and insert
the sentinel between the opening tag and content:

```python
# Before (assistant turn)
a = f"{ASSISTANT_OPEN}{inner}{ASSISTANT_CLOSE}\n"

# After
a = f"{ASSISTANT_OPEN}{ASSISTANT_OPEN_SENTINEL}{inner}{ASSISTANT_CLOSE}\n"
```

Apply the same fix to `assistant_toolcall` turns:

```python
# Before
a = f"{ASSISTANT_OPEN}{toolcall_str}{ASSISTANT_CLOSE}\n"

# After
a = f"{ASSISTANT_OPEN}{ASSISTANT_OPEN_SENTINEL}{toolcall_str}{ASSISTANT_CLOSE}\n"
```

The sentinel is structural (`mask=0`). The model trains to emit the content, not
the space.

### Testing

```bash
python scripts/sft/generate_sft_tool_episodes.py \
    --output data/test_tool_fix/raw.jsonl --num_examples 200 --seed 42
python scripts/sft/prepare_tool_sft.py \
    --input data/test_tool_fix/raw.jsonl --output_dir data/test_tool_fix/before
python scripts/sft/validate_sft_episode_masks.py --dataset_dir data/test_tool_fix/before
# Expected before fix: toolcall episodes with mask_ratio ~0.0

python scripts/sft/prepare_tool_sft.py \
    --input data/test_tool_fix/raw.jsonl --output_dir data/test_tool_fix/after
python scripts/sft/validate_sft_episode_masks.py --dataset_dir data/test_tool_fix/after
# Expected after fix: all episodes mask_ratio > 0
```

---

## Task 10 — `webapp/routers/chat.py`: `compile_for_inference()` return value not captured

**Severity: CORRECTNESS (silent failure)**

### Problem

`webapp/routers/chat.py` line 127 calls `compile_for_inference()` but discards the
return value. After the Task 5 fix, `compile_for_inference()` returns the compiled
module — but the caller stores the original uncompiled model:

```python
# BROKEN — compile result discarded, uncompiled model stored
model.compile_for_inference(mode="default")
_loaded_models[model_name] = model
```

This is a **silent** failure — no error is raised, but the webapp inference never uses
the compiled model, defeating the purpose of compilation entirely.

### Fix

```python
# FIXED — capture return value
model = model.compile_for_inference(mode="default")
_loaded_models[model_name] = model
```

### Testing

Add a log line verifying `type(model)` after compilation is a torch-compiled wrapper,
not the original `GPT` class. Or time a generation before and after to confirm speedup.

---

## Task 11 — Pipeline: wrong packer for Phase 6, missing `--enable_rag_tags` for Phase 3

**Severity: PHASE 3 & 6 SILENT CORRECTNESS BUG**

### Problem

Two packer selection mistakes that produce silently broken datasets — no error at pack
time, but the model never trains on the intended signal:

**Problem A — Phase 3 missing `--enable_rag_tags`:**
`generate_rag_chat_sft.py` produces episodes with `context` fields on user messages and
`cite`/`think` fields on assistant messages. These are the fields that drive the
`context_citation` eval bucket. When `prepare_chat_sft.py` is called **without**
`--enable_rag_tags`, these fields are silently ignored — the context and citation tags
are never written into the packed binary. The `context_citation` bucket cannot pass
because the signal was never in the training data.

**Problem B — Phase 6 using wrong packer:**
`generate_agent_sft.py` produces `assistant_toolcall` and `toolresult` roles.
`prepare_chat_sft.py` does not know these roles — it will either skip them or serialize
them incorrectly. `prepare_tool_sft.py` is the correct packer: it handles
`assistant_toolcall`, `toolresult`, and also `cite`/`think`/`context` — making it a
full superset. Using `prepare_chat_sft.py` for Phase 6 silently drops all toolcall
turns from training.

### Understanding the tag split

```
chat tags (prepare_chat_sft.py):
  system, user, assistant
  user.context  → <myPT_user_context>      (RAG: document fed to model)
  assistant.think → <myPT_think>           (RAG: reasoning before answer)
  assistant.cite  → <myPT_cite>            (RAG: source attribution)

toolcall tags (prepare_tool_sft.py — superset of chat tags, adds):
  assistant_toolcall → <myPT_toolcall>     (model decides to call a tool)
  toolresult         → <myPT_toolresult>   (tool returns data to model)
```

RAG uses toolcall tags — `workspace.search`, `workspace.get`, `workspace.list`,
`workspace.sum` are how the model retrieves. They are not separate from RAG; they are
how RAG works. The packer split is:

- **Any episode with `assistant_toolcall` or `toolresult` roles → `prepare_tool_sft.py`**
- **All other episodes → `prepare_chat_sft.py` (with `--enable_rag_tags` if RAG fields present)**

### Fix

**Phase 3 — add `--enable_rag_tags`:**

```bash
# WRONG
python scripts/sft/prepare_chat_sft.py \
    --input data/raw/phase3_mixed.jsonl \
    --output_dir data/sft_phase3_chat

# CORRECT
python scripts/sft/prepare_chat_sft.py \
    --input data/raw/phase3_mixed.jsonl \
    --output_dir data/sft_phase3_chat \
    --enable_rag_tags
```

**Phase 6 — use `prepare_tool_sft.py`:**

```bash
# WRONG
python scripts/sft/prepare_chat_sft.py \
    --input data/raw/phase6_mixed.jsonl \
    --output_dir data/sft_phase6_agentic_rag

# CORRECT
python scripts/sft/prepare_tool_sft.py \
    --input data/raw/phase6_mixed.jsonl \
    --output_dir data/sft_phase6_agentic_rag
```

**Phase 4** — `prepare_chat_sft.py` correct, no `--enable_rag_tags` needed
(multiturn has no RAG fields).

**Phase 5** — `prepare_tool_sft.py` correct as documented.

### Testing

```bash
# Phase 3: verify cite tags appear in packed binary after fix
python scripts/sft/validate_sft_dataset.py --dataset_dir data/sft_phase3_chat
# Should show episodes containing <myPT_cite> tokens

# Phase 6: verify toolcall tags appear after fix
python scripts/sft/validate_sft_dataset.py --dataset_dir data/sft_phase6_agentic_rag
# Should show episodes containing <myPT_toolcall> tokens
```

---

## Completion Checklist

### Phase 3 blockers — do first:

- [x] Task 1: `--auto_confirm` added to parser + both `input()` blocks replaced
- [x] Task 1: `"auto_confirm"` added to `training_keys`; `effective_auto_confirm` used
- [x] Task 2: `ASSISTANT_OPEN_SENTINEL` constant defined in `prepare_chat_sft.py`
- [x] Task 2: Sentinel inserted in `serialize_conversation()` with `mask=0`
- [x] Task 2: JSON episodes show non-zero mask ratio; Phase 2 episodes unchanged
- [x] Task 11: `--enable_rag_tags` added to Phase 3 pack command
- [x] Task 11: Phase 6 pack command uses `prepare_tool_sft.py`

### Crash fixes:

- [x] Task 3: `--vocab_size` argument added to `parse_args()` in `train.py`
- [x] Task 4: `is_cuda` moved unconditional; `elif` chain fixed in `load()`

### Correctness:

- [x] Task 5: `compile_for_inference()` compiles `self`; return value captured at all call sites
- [x] Task 9: `ASSISTANT_OPEN_SENTINEL` added to `prepare_tool_sft.py` for all assistant/toolcall blocks
- [x] Task 10: `webapp/routers/chat.py` captures `compile_for_inference()` return value

### Performance:

- [x] Task 6: Three dead lines removed from `_get_batch_sharded()`

### Cleanup:

- [x] Task 7: `gpu_mem("after_gc_and_empty_cache")` commented out
- [x] Task 8: `no_repeat_ngram` wired as `generate()` parameter
