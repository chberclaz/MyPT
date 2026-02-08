# Phase 1.5: Induction Strengthening Pre-Training

## Rationale

After extensive SFT experimentation (Phase 3a), we identified that the `domain_v5` base model has **weak induction heads** — attention patterns that enable multi-token copying. The model was pre-trained on 1.8B tokens (vs. 15B optimal for 750M parameters per Chinchilla scaling), with 90% encyclopedic prose, 5% dialogue, and 0% code.

**Evidence:**
- Direct prompting tests on `domain_v5` showed it could complete simple patterns ("tiger elephant → tiger elephant") but failed abstract copying ("Copy: red mountain peak → ???")
- SFT fine-tuning for operator tasks (COPY, WRAP, EXTRACT) consistently achieved ~80% on single-word payloads but 0-17% on multi-word payloads — regardless of data augmentation, segment attention masking, or hyperparameter tuning
- This indicates the base model lacks the circuit-level capability that SFT is trying to leverage

**Root cause:** Pre-training data composition lacked repetition-rich content (code, dialogue, structured Q&A) that naturally develops induction heads during training.

## Approach

Continue pre-training from the `domain_v5` checkpoint (not from scratch) with **3-5B tokens of repetition-rich data**:

| Data Type | Share | Why It Helps |
|-----------|-------|-------------|
| Code (Python, JS, Java) | ~50% | Variable reuse, function calls, imports, docstrings — massive natural repetition |
| Dialogue (subtitles, Reddit) | ~25% | Names repeated, quote-reply patterns, conversational echoing |
| Structured (SO Q&A, READMEs) | ~12% | Code snippets quoted in answers, references repeated |
| Wiki replay | ~10% | Prevent forgetting Phase 1 general knowledge |
| Domain replay | ~3% | Preserve Phase 2 IT security domain knowledge |

## Data Sources

Defined in `data/sources/phase1_5_induction.json`:

| Source | HF Dataset ID | Est. Tokens | Type |
|--------|--------------|-------------|------|
| CodeParrot Clean | `codeparrot/codeparrot-clean` | ~1B | Python code |
| StarCoderData (Python) | `bigcode/starcoderdata` [python] | ~400M | Python code |
| StarCoderData (JavaScript) | `bigcode/starcoderdata` [javascript] | ~300M | JS code |
| StarCoderData (Java) | `bigcode/starcoderdata` [java] | ~300M | Java code |
| OpenSubtitles EN | OPUS mono download | ~500M | Dialogue |
| Reddit Comments | `webis/tldr-17` | ~500M | Discussion |
| StackOverflow Q&A | `kensho/spuq` | ~400M | Structured Q&A |
| GitHub READMEs | `bigcode/starcoderdata` [markdown] | ~300M | Documentation |

## Pipeline

### Architecture

```
Source Config (JSON)
    ↓
fetch_and_prepare_phase1_5.py
    ├── Download (HuggingFace streaming / OPUS)
    ├── Clean (clean_code_corpus.py / clean_dialogue_corpus.py)
    └── Tokenize (prepare_weighted_dataset.py)
    ↓
mix_tokenized_datasets.py (add wiki + domain replay)
    ↓
train.py --init_from_model domain_v5
```

### Step 1: Fetch and Prepare

```bash
python scripts/fetch_and_prepare_phase1_5.py \
    --sources_file data/sources/phase1_5_induction.json \
    --out_dir data/phase1_5_induction_raw \
    --total_tokens 3500000000
```

This downloads, cleans, and tokenizes all sources. Use `--skip_download` to resume if downloads were interrupted. Use `--sources codeparrot opensub_en` to process specific sources.

**Disk space required:** ~40GB for raw downloads, ~16GB for tokenized shards.

### Step 2: Mix with Replay

```bash
python scripts/mix_tokenized_datasets.py \
    --domain_dir data/phase1_5_induction_raw \
    --replay_dir data/multilingual_1.5B_wiki90 \
    --replay2_dir data/domain_161M_corpus_tokenized \
    --output_dir data/phase1_5_mixed \
    --replay_ratio 0.10 \
    --replay2_ratio 0.03
```

The `--replay2_dir` and `--replay2_ratio` flags are new additions to `mix_tokenized_datasets.py` to support dual replay sources (wiki knowledge preservation + domain knowledge preservation).

### Step 3: Train

```bash
python train.py \
    --model_name domain_v6_induction \
    --dataset_dir data/phase1_5_mixed \
    --config_file configs/base/750M_phase1_5_induction.json \
    --init_from_model domain_v5
```

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `init_from_model` | `domain_v5` | Continue from best domain-adapted checkpoint |
| `learning_rate` | 3e-5 | Same as Phase 2 continuation — stable for continued pre-training |
| `max_iters` | 500,000 | ~4B tokens / (12 batch × 1024 block) ≈ 325k iters/epoch × 1.5 epochs |
| `warmup_iters` | 5,000 | Long warmup for stability with new data distribution |
| `eval_interval` | 2,000 | ~250 evals over the run, sufficient for trend detection |
| `weight_decay` | 0.1 | Standard for pre-training |
| `dropout` | 0.1 | Match existing model config |
| Gold checkpoint | Enabled | Automatically saves best validation loss checkpoint |

**Estimated training time:** ~500k iterations at ~0.8-1.0 sec/iter (standard data loader, no loss masking or segment attention) = ~5-6 days on A100. Checkpointing allows pause/resume.

## Data Cleaning

### Code Cleaning (`tools/clean_code_corpus.py`)

- **Size filter:** Remove files <100 bytes (junk) and >100KB (generated/vendored)
- **Quality filter:** Min 5 lines, max 10% very long lines (minified detection), no auto-generated files
- **Dedup:** Exact SHA-256 hash deduplication
- **License strip:** Remove common copyright headers (reduces boilerplate)
- **Output:** Sharded text files with `==== DOC START ====` / `==== DOC END ====` markers

### Dialogue Cleaning (`tools/clean_dialogue_corpus.py`)

- **OpenSubtitles:** Parse line-based subtitles into conversation windows (20 lines, 50% overlap), remove timestamps/formatting
- **Reddit:** Preserve quote structure (`>` prefix — shows explicit repetition), remove URLs, filter by word count
- **StackOverflow:** Strip HTML tags, filter by length
- **Output:** Same sharded text format as code cleaner

## Validation Strategy

After Phase 1.5 training, verify induction head development:

```bash
# Test 1: Pattern completion (should already work on domain_v5)
python generate.py --model_name domain_v6_induction \
    --prompt "The password is: tiger elephant. Please repeat the password: tiger" \
    --temperature 0

# Test 2: Sequence continuation
python generate.py --model_name domain_v6_induction \
    --prompt "A B C D. A B C" \
    --temperature 0

# Test 3: Multi-word copying (the failure case that motivated Phase 1.5)
python generate.py --model_name domain_v6_induction \
    --prompt "Copy exactly: red mountain peak. Output: red" \
    --temperature 0
```

**Success criteria:**
- Test 1 & 2: Continue to work (no regression)
- Test 3: Model outputs "mountain peak" (or close) — this is the key capability that was missing

If tests pass, proceed to SFT Phase 3a using the full infrastructure (segment attention, gold checkpoint, cross-operator data).

## Files Created

| File | Purpose |
|------|---------|
| `data/sources/phase1_5_induction.json` | Source definitions and weights |
| `scripts/fetch_and_prepare_phase1_5.py` | Main orchestrator script |
| `tools/clean_code_corpus.py` | Code cleaning and filtering |
| `tools/clean_dialogue_corpus.py` | Dialogue extraction and cleaning |
| `configs/base/750M_phase1_5_induction.json` | Training configuration |
| `docs/training/PHASE1_5_INDUCTION_TRAINING.md` | This documentation |

**Modified:**
| File | Change |
|------|--------|
| `scripts/mix_tokenized_datasets.py` | Added `--replay2_dir` and `--replay2_ratio` for dual replay |

## Key Considerations

- **HuggingFace access:** Some datasets may require a HuggingFace account. Install with `pip install datasets`.
- **Disk space:** Total ~60GB (raw + tokenized + mixed). Clean up raw downloads after tokenization if space is tight.
- **Training cost:** ~12 days on A100. The Gold checkpoint feature ensures you always have the best model saved.
- **Replay ratios:** 10% wiki + 3% domain = 13% total replay. This is conservative — increase if validation loss on general prompts degrades.
