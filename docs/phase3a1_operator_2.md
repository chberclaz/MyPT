# CURSOR SPEC — Operator Dataset: Preserve Train/Val Separation (No Split)

## Goal
Ensure **operator_train.jsonl** and **operator_val.jsonl** stay **strictly separated** through preparation so validation truly tests:
- **template generalization** (val templates differ from train)
- **payload generalization** (payloads are unique)

We must NOT do `--val_split` on a combined file for this dataset.

---

## Problem Statement
Current pipeline likely does:
- load one JSONL
- randomly split with `val_split`
This breaks the operator benchmark because train/val may share instruction templates.

We need:
- explicit `--val_file` support in `prepare_chat_sft.py`
- deterministic, reproducible prepare
- sanity checks (no overlap, template difference preserved)
- metadata that records both sources and hash/fingerprint

---

## Required Changes

### 1) Update `scripts/prepare_chat_sft.py` CLI
Add optional argument:

- `--val_file PATH` (optional)
  - If provided: **do not** use `--val_split`
  - Prepare train from `--input`, val from `--val_file`

Rules:
- If `--val_file` is provided AND `--val_split > 0`: **error**
- If `--val_file` is provided: set val split to 0 internally
- If `--val_file` is not provided: keep existing `--val_split` behavior unchanged

Example error message:
> "Invalid args: --val_file and --val_split are mutually exclusive. Use --val_file to preserve fixed validation templates."

---

### 2) Implement Separate Loading Paths
Refactor the script so it can tokenize **two independent conversation lists**:

- `train_convs = load_jsonl(input_path)`
- `val_convs = load_jsonl(val_path)` (if `--val_file` provided)
Else:
- split `all_convs` by `val_split` (existing behavior)

Ensure:
- no shuffling across splits when `--val_file` used
- optional shuffle within each split is OK, but default should remain current behavior

---

### 3) Add Safety Checks (Hard Fails)
When `--val_file` is used, enforce these checks before writing shards:

#### 3.1 Payload overlap check (hard fail)
This dataset relies on unique payloads. Implement conservative overlap detection:

- Extract payload from user message using heuristics:
  - COPY templates: last token-span after colon / after "Return:" / after "Say this back:"
  - WRAP templates: same
  - EXTRACT templates: inside quotes "..."
- Normalize payload: strip, collapse whitespace, keep exact case
- Build `set(train_payloads)` and `set(val_payloads)`
- If intersection non-empty: raise error and print a few overlaps

Fail message:
> "Train/Val payload overlap detected: N overlaps. Operator benchmark invalid."

(If you don’t want heuristic fragility: also store `payload` explicitly in generator JSONL going forward, but implement heuristic now.)

#### 3.2 Template overlap smoke check (warn or fail)
We want template generalization; payload uniqueness alone is not enough.

Implement lightweight template signature:
- Replace the payload span with `{PAYLOAD}`
- Replace any quoted payload with `"{PAYLOAD}"`
- Collapse whitespace
- Use resulting string as a “template signature”

Compute:
- `train_templates` set
- `val_templates` set
- Intersection size
If intersection > 0:
- Print warning with count and 5 examples
- Optionally hard fail when `--strict_operator` flag enabled (add flag if you want)

Default: WARN (to avoid breaking non-operator datasets), but print loudly when `--val_file` used.

---

### 4) Write Train/Val Shards Independently
Keep current episode-indexed output format.

Write:
- `output_dir/train/shard_00000/...`
- `output_dir/val/shard_00000/...`

Ensure `episodes.idx` is correct per split.

Do NOT merge splits into a single token stream.

---

### 5) Metadata & Provenance
Extend `dataset_metadata.json` to include:

- `source_train_file`: absolute/relative path
- `source_val_file`: path (if used)
- `prepare_mode`: `"explicit_val_file"` or `"val_split"`
- `val_split`: value used (0 if explicit val)
- `num_train_conversations`, `num_val_conversations`
- `train_payload_overlap_count`, `val_template_overlap_count`
- hashes:
  - sha256 of `operator_train.jsonl`
  - sha256 of `operator_val.jsonl` (if provided)

Add `tokenizer_state.json` as usual.

---

## Implementation Notes

### A) Maintain Backward Compatibility
Do not break existing datasets that rely on `--val_split`.
Only activate the new behavior when `--val_file` is provided.

### B) Determinism
When `--val_file` is provided:
- do not randomly split
- do not shuffle across splits
- optional: shuffle inside each split with fixed `--seed` (if script already has seed support)

### C) Console Output (must be explicit)
When `--val_file` used, print:

- "VAL SOURCE: using explicit val file (no split)"
- "Payload overlap: 0"
- "Template overlap: X (warn)"
- "Prepared train tokens/episodes..."
- "Prepared val tokens/episodes..."

---

## Expected Usage (Operator Dataset)

### Prepare (STRICT separation)
```bash
python scripts/prepare_chat_sft.py \
  --input data/sft_operator/operator_train.jsonl \
  --val_file data/sft_operator/operator_val.jsonl \
  --output_dir data/sft_operator/prepared \
  --tokenization gpt2
