# Phase 1b: Context Extension (1024 → 4096)

> **Config:** `configs/phase1b_context_extension.json`  
> **Architecture:** LLaMA-2 style (RoPE + SwiGLU + RMSNorm), ~699M parameters, Position Interpolation (PI)  
> **Dataset:** ~800M tokens: 6 HuggingFace QA sources (40%) + pre-training general text (60%), episode-indexed with packing  
> **Training:** Continued pre-training, LR 3e-5, 12,000 optimizer steps, rope_scale=4.0  
> **Input checkpoint:** `checkpoints/GOLD_unified_v1` (pre-trained at 1024 context)  
> **Date:** February 2026

---

## Background: Why This Training Run Exists

### The Problem

The pre-trained model (`GOLD_unified_v1`) has a 1024-token context window. This is insufficient for production RAG:

- **System prompt:** ~100-200 tokens (instructions, persona, constraints)
- **Retrieved passages:** 2-4 passages from RAG, each 200-500 tokens = 400-2000 tokens
- **User query:** ~50-200 tokens
- **Tool calls + results:** ~200-500 tokens per tool round-trip
- **Model response:** ~100-500 tokens

Total: **850-3400 tokens** for a typical production episode. At 1024, only the simplest single-passage queries fit. Multi-passage retrieval, multi-turn conversations, and tool chains are impossible.

### The Solution: Position Interpolation

Extending from 1024 to 4096 using Position Interpolation (Chen et al., 2023). PI is the simplest proven method for RoPE-based context extension:

1. Divide all position indices by the scaling factor (4.0) before computing RoPE frequencies
2. Fine-tune on diverse text at the new context length (we use 12,000 steps on ~800M tokens)
3. The model interpolates between positions it already learned, rather than extrapolating to unseen positions

Why PI over alternatives:

- **vs. NTK-aware scaling:** NTK modifies frequency bases non-uniformly. More complex, marginal gains for 4x extension. PI is proven at 4x with minimal risk.
- **vs. YaRN:** Combines NTK + attention temperature. Overkill for 4x, adds hyperparameters. Best suited for >8x extension.
- **vs. Direct fine-tuning (no scaling):** Would require learning entirely new positional patterns for positions 1024-4095. Needs 10x more data and steps. PI compresses these into the already-learned [0, 1024) range.

### Dual Purpose: Context Extension + Retrieval Strengthening

This phase serves two purposes simultaneously:

1. **Context extension:** Train the model to attend across 4096 positions with PI-compressed RoPE
2. **Retrieval strengthening:** 40% of the dataset is structured reading comprehension (passage + question + answer), which reinforces the retrieval circuits formed during pre-training

This works because PI fine-tuning requires the model to process real text at the new context length. The 60% general text ensures broad positional encoding adaptation across all domains; the 40% QA portion adds targeted retrieval signal on top, so every QA training step simultaneously adapts the positional encoding AND practices the passage → answer extraction pattern.

---

## Architecture Changes

### Position Interpolation Implementation

Single change in `core/model.py`, function `precompute_rope_frequencies`:

```python
# Before (pre-training):
t = torch.arange(max_seq_len, device=device).float()

# After (context extension):
t = torch.arange(max_seq_len, device=device).float() / scale
```

With `scale=4.0`, position 4095 maps to effective position 1023.75 -- safely within the pre-trained range [0, 1024).

### Config Additions

| Field | Value | Effect |
|:--|:--|:--|
| `rope_scale` | 4.0 | Position Interpolation factor (new_context / old_context) |
| `block_size` | 4096 | New context window |

All other architecture parameters (n_embd, n_head, n_layer, etc.) remain identical to pre-training. No new parameters are introduced -- PI only changes how existing RoPE buffers are computed.

### Checkpoint Loading

`core/checkpoint.py` `initialize_for_training()` handles the context extension automatically when `init_from_model` is used:

1. Loads the pre-trained model (block_size=1024, no rope_scale)
2. Detects that the new config has different block_size/rope_scale
3. Updates `model.config.block_size`, `model.config.rope_scale`
4. Recomputes `rope_cos` and `rope_sin` buffers at the new size with PI scaling
5. Logs the context extension change

This works because RoPE buffers are `persistent=False` (not saved in state_dict), so they are always recomputed from config values.

---

## Dataset: Two-Part Design (~800M Tokens)

The dataset has two complementary parts:

- **QA episodes (40%, ~320M tokens):** 6 HuggingFace QA datasets, providing structured retrieval training signal
- **General text (60%, ~480M tokens):** Sampled from pre-training shards, providing diverse text for positional encoding adaptation

PI's primary job is adapting the RoPE frequencies to work at 4096 positions. That requires diverse text at 4096 -- not just QA. The general text is the vehicle for positional encoding adaptation; the QA portion is the targeted retrieval bonus that strengthens reading comprehension circuits formed during pre-training.

### Design Principles

1. **Plain text only:** No `<myPT_*>` tags. The model has not seen SFT data yet -- it only knows plain text from pre-training. Tags are introduced in Phase 1 (Format Lock).
2. **Reading comprehension format (QA portion):** Every QA episode is `Passage(s) + Question + Answer`. This is the fundamental skill underlying RAG: locate relevant information in provided context and use it to answer.
3. **Position-varied gold content:** For multi-passage sources (HotpotQA, MuSiQue, MS MARCO), gold/selected passages are placed in 4 uniform position buckets (beginning, early-middle, late-middle, end). This prevents the "Lost in the Middle" bias (Liu et al., 2023) where models attend strongly to beginning/end but weakly to middle positions.
4. **General text (60%):** Sampled from existing pre-training shards (`data/unified_6B`), split at EOS token boundaries to preserve document integrity. Short documents are concatenated (with EOS separators) into ~4096-token mega-episodes so each episode fills a full block, giving the model full-length attention spans for RoPE adaptation. Covers all pre-training domains: code, Wikipedia, StackExchange, domain text -- ensuring the model's positional encoding adapts broadly, not just on QA patterns.
5. **Episode-indexed format with packing and segment isolation:** All episodes (QA + general) are packed into fixed 4096-token blocks using greedy bin-packing. Each block contains complete episodes only -- no mid-episode cuts. Padding at block end is masked out (mask=0). All real tokens are loss targets (mask=1). Episodes are diversity-interleaved before packing so each block mixes short, medium, and long episodes from both QA and general sources. Each episode gets a unique segment ID (`segment_ids.bin`), enabling **segment-isolated attention**: tokens can only attend to other tokens within the same episode, preventing cross-episode attention bleeding. This ensures a SQuAD answer about geography never attends to a HotpotQA passage about history that happened to land in the same block.

6. **Absolute positions across the full block (`segment_position_reset: false`):** While segment isolation prevents cross-episode attention, positions are NOT reset at episode boundaries. The first episode gets positions 0-199, the second gets 200-1499, the third gets 1500-3800, etc. This is critical: the entire point of Phase 1b is to train the model on positions 0-4095 with PI-compressed RoPE. If positions were reset per episode (as they are during SFT), the model would only ever see positions 0 to ~2000 (the length of the longest single episode), completely defeating context extension. Later SFT phases use `segment_position_reset: true` (the default), since during SFT each conversation should start at position 0 to match inference.

### QA Source Detail (40% of dataset)

| # | Source | Episodes | Est. Tokens | Role |
|:--|:--|:--|:--|:--|
| 1 | **HotpotQA (distractor)** | ~90K | ~108M | Long multi-passage, position-varied gold paragraphs |
| 2 | **MS MARCO v2.1** | ~100K | ~100M | 10-passage search results, position-varied, real Bing queries |
| 3 | **TriviaQA (evidence)** | ~130K | ~78M | Medium-long trivia grounded in Wikipedia |
| 4 | **SQuAD v2** | ~130K | ~32M | Short extractive EN QA (incl. unanswerable) |
| 5 | **MuSiQue** | ~20K | ~20M | Hard multi-hop EN (2-4 hops) |
| 6 | **GermanQuAD** | ~6K | ~1.5M | Short extractive DE QA (reduced) |

### General Text (60% of dataset)

~480M tokens sampled from `data/unified_6B` pre-training shards. These are the same 10-source mix the model was pre-trained on (FineWeb-Edu, Wikipedia, Python code, StackExchange, etc.), re-chunked at document boundaries (EOS token 50256). Short documents are concatenated (with EOS separators preserved) into ~4096-token "mega-episodes" that fill entire blocks, ensuring full-length attention spans for RoPE adaptation. Long documents (>= 4096 tokens) become their own full-length episode.

### Source Analysis: Why Each Dataset Was Chosen

#### SQuAD v2 (~32M tokens) — Foundation Retrieval Signal

**What it is:** 130K+ questions on Wikipedia paragraphs. Each question is paired with a single paragraph. ~80% answerable (answer is a span in the paragraph), ~20% unanswerable (paragraph does not contain the answer).

**Why it's here:**
- Provides the highest volume of clean, short passage → question → answer episodes
- The unanswerable subset (~26K questions) teaches a critical RAG skill: recognizing when provided context does NOT contain the answer. Without this signal, the model learns to always extract something, even when nothing is relevant (a form of hallucination)
- Short episodes (~150-300 tokens) pack efficiently into 4096 blocks, giving the model many training examples per step
- English, high quality, widely benchmarked

**Episode format:**
```
Passage: In meteorology, precipitation is any product of the condensation
of atmospheric water vapor that falls under gravity. The main forms of
precipitation include drizzle, rain, sleet, snow, graupel and hail...

Question: What causes precipitation to fall?
Answer: gravity
```

**Unanswerable format:**
```
Passage: The Norman dynasty had a major political, cultural and military
impact on medieval Europe and even the Near East...

Question: What was the name of the Norman king who conquered England?
Answer: The passage does not contain enough information to answer this question.
```

#### HotpotQA Distractor (~108M tokens) — Multi-Passage Position Training

**What it is:** 90K+ multi-hop questions requiring reasoning over exactly 2 Wikipedia paragraphs, presented alongside 8 distractor paragraphs (10 total). Each example has explicit `supporting_facts` annotations identifying which paragraphs and sentences are needed.

**Why it's here:**
- Provides the primary signal for **long-context attention training**. With 10 paragraphs per episode (~800-2000 tokens), these episodes actually exercise the extended context window
- The distractor setting is critical: the model must learn to attend to the RIGHT paragraphs among irrelevant ones -- exactly the skill needed for RAG where retrieved passages vary in relevance
- **Position control:** We shuffle distractor paragraphs randomly and place gold paragraphs in 4 uniform position buckets. This forces the model to find relevant content regardless of where it appears in the 4096-token window. Without this, the model would develop a "primacy/recency" bias (attend to beginning and end, ignore middle)
- Multi-hop: some questions require combining facts from 2 different paragraphs, training cross-passage attention

**Episode format (gold at end bucket):**
```
[Distractor Title 1]
Distractor paragraph text...

[Distractor Title 2]
Another distractor paragraph...

... (more distractors) ...

[Gold Title 1]
Paragraph containing first supporting fact...

[Gold Title 2]
Paragraph containing second supporting fact...

Question: Which country hosted the 2014 FIFA World Cup?
Answer: Brazil
```

**Position bucket distribution:**
- Bucket 0 (beginning, 0-25%): ~25% of episodes — gold paragraphs first
- Bucket 1 (early-middle, 25-50%): ~25% — gold in first third of distractors
- Bucket 2 (late-middle, 50-75%): ~25% — gold in second third
- Bucket 3 (end, 75-100%): ~25% — gold paragraphs last

#### MuSiQue (~20M tokens) — Hard Multi-Hop Reasoning

**What it is:** ~25K questions requiring 2-4 genuine reasoning hops across multiple documents. Created by composing single-hop questions to ensure each hop is necessary. Includes both answerable and unanswerable variants.

**Why it's here:**
- **Hardest retrieval signal** in the dataset. While HotpotQA mostly requires 2-hop reasoning, MuSiQue includes 3-hop and 4-hop chains where missing any step produces a wrong answer
- Tests whether the model can maintain attention chains across multiple documents: Document A mentions entity X → Document B links X to Y → Document C answers about Y
- Same position-control strategy as HotpotQA (4 uniform buckets) to prevent positional bias
- Lower share (10%) because: (a) smaller source dataset, (b) too much hard multi-hop could overwhelm the simpler extraction patterns from SQuAD v2 and bias the model toward over-reasoning

**Why not more:**
At 750M parameters, there's a ceiling on multi-hop reasoning depth. Over-weighting hard multi-hop (>15%) risks the model learning to guess when it can't genuinely chain, producing more hallucinations. 10% provides the signal without overwhelming.

#### GermanQuAD (~1.5M tokens) — Bilingual Maintenance

**What it is:** ~13.7K German extractive QA questions over German Wikipedia paragraphs, following the SQuAD format but in German.

**Why it's here:**
- The pre-trained model was trained on 15% bilingual Wikipedia (EN+DE) and 7% Swiss German law text. German capability is a first-class feature
- Without German data in this phase, the extended context would only be exercised on English text, and the model might lose some German positional patterns during PI adaptation
- Uses German-language labels (`Frage:` / `Antwort:` instead of `Question:` / `Answer:`) to preserve the model's German language patterns

**Why reduced to ~6K:**
GermanQuAD has ~13.7K training examples. We use ~6K to maintain bilingual presence without dominating the short-episode bucket. German positional patterns are also reinforced by the 60% general text portion (which includes German Wikipedia and Swiss law from the pre-training mix).

#### MS MARCO v2.1 (~100M tokens) — Multi-Passage Search Results

**What it is:** Microsoft's MS MARCO v2.1 dataset. Real Bing search queries paired with 10 web-retrieved passages per query. Each passage has a binary relevance label (`is_selected`). ~1M training examples; we cap at ~100K for balance.

**Why it's here:**
- **Closest match to production RAG:** Each example mirrors the actual retrieval pipeline -- a user query with multiple retrieved passages of varying relevance. This is exactly the skill the model needs for `toolresult`-based RAG in production
- **10 passages per example** produce naturally long episodes (~1500-3000 tokens), filling 4096-token blocks far better than single-passage datasets
- **Position control:** Same 4-bucket strategy as HotpotQA -- selected (relevant) passages are placed at beginning, early-middle, late-middle, or end among unselected passages. This trains position-invariant attention across the long context
- **Negative retrieval signal:** ~20% of kept examples have "No Answer Present" answers, teaching the model to recognize when none of the retrieved passages contain the answer
- **Clean Parquet format** on HuggingFace -- no schema issues, reliable loading

**Episode format (selected passage at late-middle bucket):**
```
[Source 1]
Passage text from unselected search result...

[Source 2]
Another unselected passage...

... (more unselected) ...

[Source 7]
Passage text from the SELECTED search result containing the answer...

[Source 8]
More unselected passage...

... (remaining unselected) ...

Query: what is the capital of france
Answer: Paris
```

**Why it replaces Natural Questions:**
NQ requires a 45+ GB download, has a complex nested schema that fails with modern HuggingFace `datasets` versions, and only produces ~400-900 token episodes. MS MARCO is smaller to download, has a clean flat schema, produces 2-3x longer episodes, and directly matches the production RAG pattern.

#### TriviaQA Evidence (~78M tokens) — Medium-Long Grounded Trivia

**What it is:** Trivia questions paired with Wikipedia evidence passages that contain the answer. ~130K training examples. Only grounded examples (answer appears in passage) are used.

**Why it's here:**
- Medium-length passages (~600 tokens average) fill the gap between short SQuAD episodes and long HotpotQA episodes
- The grounding filter ensures 100% of examples are genuinely extractive (no hallucination signal)
- Already proven effective in pre-training (used in the retrieval portion of the pre-training mix)

### Cross-Source Signal Analysis

The six QA sources provide overlapping and complementary training signals:

| Signal | SQuAD v2 | HotpotQA | MuSiQue | MS MARCO | TriviaQA | GermanQuAD |
|:--|:--|:--|:--|:--|:--|:--|
| Short-range extraction | **Strong** | Moderate | Moderate | **Strong** | **Strong** | **Strong** |
| Long-range attention | Weak | **Strong** | **Strong** | **Strong** | Moderate | Weak |
| Position invariance | N/A | **Strong** (4 buckets) | **Strong** (4 buckets) | **Strong** (4 buckets) | N/A | N/A |
| Multi-hop reasoning | None | Moderate (2 hops) | **Strong** (2-4 hops) | None | None | None |
| Multi-passage retrieval | None | **Strong** (10 passages) | Moderate | **Strong** (10 passages) | None | None |
| Negative retrieval | **Strong** (20% unans.) | Weak | Moderate | **Strong** (20% no-ans.) | None | None |
| Bilingual | None | None | None | None | None | **Strong** |

**Combined coverage:** Every critical retrieval skill is covered by at least 2 sources. MS MARCO adds the strongest multi-passage retrieval signal (10 real search results per query with position control), complementing HotpotQA's multi-hop focus. TriviaQA fills the medium-length gap.

---

## Training Configuration

### Hyperparameters

| Parameter | Value | Rationale |
|:--|:--|:--|
| `learning_rate` | 3e-5 | 5x lower than pre-training (1.5e-4). PI fine-tuning is a small adaptation, not training from scratch. Too high would destroy pre-trained knowledge. |
| `warmup_iters` | 400 | ~3% of training. Short warmup sufficient for fine-tuning. |
| `max_iters` | 12,000 | Optimizer steps. ~800M tokens total. Enough for ~1 full epoch of the combined QA+general dataset. |
| `batch_size` | 4 | Micro-batch of 4 sequences at 4096 tokens = 16,384 tokens. Fits in VRAM with bf16. |
| `grad_accum_steps` | 4 | Effective batch: 4 * 4 = 16 sequences = 65,536 tokens/step. |
| `dropout` | 0.05 | Lower than pre-training (0.1). Fine-tuning benefits from less regularization -- the model has already learned features. |
| `weight_decay` | 0.05 | Lower than pre-training (0.1). Same reasoning as dropout. |
| `grad_clip` | 1.0 | Standard gradient clipping. |
| `use_amp` | true (bf16) | Mixed precision for speed and memory. |
| `eval_interval` | 500 | Every 500 steps = 24 evaluations over 12,000 steps. |
| `use_loss_mask` | true | Mask=1 on all real tokens, mask=0 on block padding. Every content token trains the model. |
| `batch_sampling_mode` | "epoch" | Deterministic epoch-based coverage. Each packed block seen exactly once per epoch. |
| `pad_token_id` | 50256 | GPT-2 EOS used as padding in packed blocks. |
| `rope_scale` | 4.0 | Position Interpolation factor: 4096 / 1024 = 4.0. |
| `segment_position_reset` | false | Keep absolute positions 0..4095 across packed blocks (do NOT reset per episode). Critical for actually training on extended positions. |

### VRAM Estimation

```
Model parameters:    ~699M * 2 bytes (bf16) = ~1.4 GB
Optimizer state:     ~699M * 8 bytes (Adam fp32 m+v + params) = ~5.6 GB
Activations (4096):  ~4x the 1024 baseline ≈ 3-5 GB (with grad checkpointing if needed)
Batch (B=4, T=4096): 4 * 4096 * 2 bytes * layers = moderate

Estimated total: ~12-16 GB
```

This fits comfortably on 24GB GPUs (RTX 3090/4090) and well within 48GB RunPod instances.

### LR Schedule

Linear warmup (0 to 3e-5 over 400 steps) followed by cosine decay to 3e-6 (10% of peak). Continuous, no restarts.

### Effective Batch Size Calculation

```
micro_batch     = 4 sequences * 4096 tokens = 16,384 tokens
effective_batch = 16,384 * 4 (grad_accum)   = 65,536 tokens/step
total_tokens    = 65,536 * 12,000 steps      = ~786M tokens
```

---

## Data Pipeline

### Step 1: Build Raw Text Dataset

```bash
python scripts/data_prep/build_context_extension_dataset.py
```

Downloads all 6 HuggingFace QA datasets (SQuAD v2, GermanQuAD, HotpotQA, MuSiQue, MS MARCO v2.1, TriviaQA), adapts them to plain text format, applies position-control shuffling for HotpotQA, MuSiQue, and MS MARCO, combines and shuffles, and writes a single text file with `<|endoftext|>` delimiters:

```
data/context_extension_raw/context_ext.txt
```

### Step 2: Tokenize, Combine with General Text, and Pack

```bash
python scripts/data_prep/prepare_context_extension.py \
    --general_shards_dir data/unified_6B \
    --general_target_tokens 480000000
```

Tokenizes QA episodes, samples ~480M tokens of general text from pre-training shards (split at EOS document boundaries, short documents concatenated into ~4096-token mega-episodes), combines both, diversity-interleaves (short/medium/long round-robin), then greedy-packs into 4096-token blocks with padding masked out. Prints length distribution histograms for both QA and general episodes to verify long-context coverage:

```
data/context_extension/
  train/
    tokens.bin          # uint32 token IDs (packed blocks concatenated)
    mask.bin            # uint8 loss mask (1=real, 0=padding)
    segment_ids.bin     # uint8 segment IDs for segment-isolated attention
    episodes.idx        # uint64 (start, length) per packed block
  val/
    tokens.bin
    mask.bin
    segment_ids.bin
    episodes.idx
  dataset_metadata.json
  tokenizer_state.json
```

**Why episode-indexed (not token-stream)?**

Token-stream mode samples random windows from a continuous byte stream. A random 4096-token window can start mid-episode, so the model would see a question without its passage (teaching hallucination) or a passage without its question (wasted signal). Episode-indexed packing guarantees every block contains complete episodes only.

**Why a custom tokenizer script instead of `prepare_chat_sft.py`?**

Two reasons: (1) `tiktoken.encode_ordinary()` treats `<|endoftext|>` as literal characters, not as token 50256 -- the script handles this by explicit splitting and ID insertion. (2) `prepare_chat_sft.py` expects `<myPT_*>` tagged JSONL conversations, which don't exist at this stage.

**Diversity interleaving:** Before packing, episodes are classified by length (short < 500 tokens, medium 500-1200, long > 1200) and round-robin interleaved. This ensures each 4096-token block contains a mix of episode types -- short SQuAD extractions alongside longer HotpotQA multi-passage reasoning -- rather than homogeneous blocks of all-short or all-long.

### Step 3: Upload to RunPod

```
data/context_extension/            (tokenized episode-indexed dataset)
configs/phase1b_context_extension.json
```

### Step 4: Train

```bash
python train.py \
    --model_name phase1b_context_ext \
    --config_file configs/phase1b_context_extension.json \
    --dataset_dir data/context_extension \
    --init_from_model GOLD_unified_v1
```

The `init_from_model` path triggers checkpoint loading, which detects the block_size change (1024 → 4096) and automatically recomputes RoPE buffers with `rope_scale=4.0`.

---

## Evaluation Strategy

### Primary Metrics

1. **Training loss:** Should decrease steadily. Expect initial spike (new context length) followed by rapid convergence (PI typically converges in ~500-1000 steps). Loss should reach or beat the pre-training final loss by step ~3000-4000.

2. **Validation loss:** Should track training loss closely. Watch for train/val divergence after step ~6000 (halfway).

### What to Watch For

| Signal | Healthy | Concerning |
|:--|:--|:--|
| Initial loss | ~4-6 (spike from new context) | >8 (RoPE scaling broken) |
| Loss at step 1000 | Decreasing rapidly, ~3-4 | Still >5 (PI not converging) |
| Loss at step 4000 | Near pre-training level (~2.5-3) | Plateaued >3.5 |
| Loss at step 12000 | Stable or slowly improving | Rising (overfitting) |
| Train/val gap | <0.15 | >0.3 (overfitting) |

### Post-Training Validation

After Phase 1b completes, before proceeding to SFT:

1. **Perplexity on pre-training eval sets:** Run eval on `data/general`, `data/code_eval_tokenized`, `data/retrieval_eval_tokenized`. Context extension should NOT significantly degrade general perplexity (tolerance: <5% increase). If code or retrieval eval degrades >10%, the PI adaptation may have been too aggressive.

2. **Manual sampling at long context:** Generate text with 2000+ token prompts. The model should produce coherent continuations, not degenerate (repetition loops, garbage tokens).

---

## Expected Outcomes

### Quantitative

- Context window: 1024 → 4096 (4x)
- Training tokens: ~800M (40% QA + 60% general, ~11% of pre-training volume)
- Expected training time: ~6-10 hours on 1x A100/H100, ~18-30 hours on 1x RTX 3090
- General perplexity: <5% degradation from pre-training baseline
- Retrieval perplexity: expected improvement (dataset is retrieval-focused)

### Qualitative

After Phase 1b, the model should:

1. Process 4096-token inputs without degeneration
2. Attend to information across the full context window (not just beginning/end)
3. Have strengthened passage → answer extraction patterns from the QA dataset
4. Maintain bilingual (EN+DE) capability
5. Still have NO knowledge of `<myPT_*>` tags (introduced in Phase 1: Format Lock)

### What This Phase Does NOT Do

- Does not teach the model about special tokens or conversation structure
- Does not teach tool calling or agentic behavior
- Does not replace SFT -- it prepares the foundation for SFT at 4096 context
- Does not guarantee perfect "Lost in the Middle" robustness -- only provides the training signal. Full robustness depends on continued exposure during SFT phases 3-6

---

## Downstream Impact

After Phase 1b, ALL subsequent SFT phases run at 4096 context:

| Phase | Config Change | Effect |
|:--|:--|:--|
| Phase 1 (Format Lock) | `rope_scale: 4.0` (block_size stays 512) | Short episodes, but RoPE scaling matches |
| Phase 2 (Operators) | `block_size: 4096`, `rope_scale: 4.0`, `batch_size: 16→8`, `grad_accum: +2` | Operators at full context |
| Phase 3 (Chat SFT) | `block_size: 4096`, `rope_scale: 4.0`, `batch_size: 10→4`, `grad_accum: +3` | RAG episodes can include full passages |
| Phase 4 (Multi-turn) | `block_size: 4096`, `rope_scale: 4.0`, `batch_size: 8→4`, `grad_accum: +2` | Multi-turn conversations fit completely |
| Phase 5 (Toolcall) | `block_size: 4096`, `rope_scale: 4.0`, `batch_size: 6→3`, `grad_accum: +2` | Tool chains with context fit |
| Phase 6 (Agentic RAG) | `block_size: 4096`, `rope_scale: 4.0`, `batch_size: 6→3`, `grad_accum: +2` | Full agentic episodes at 4096 |

Batch sizes are reduced to maintain VRAM budget (4x context = 4x activation memory per sequence). `grad_accum_steps` increased to maintain comparable effective batch sizes.

---

## File References

| File | Description |
|:--|:--|
| `configs/phase1b_context_extension.json` | Training configuration |
| `scripts/data_prep/build_context_extension_dataset.py` | HuggingFace QA dataset builder (6 sources → plain text) |
| `scripts/data_prep/prepare_context_extension.py` | Tokenizer + packer (QA text + general shards → episode-indexed with segment IDs) |
| `core/model.py` | PI implementation (`rope_scale` in `GPTConfig`, `precompute_rope_frequencies`) |
| `core/checkpoint.py` | Context extension handling in `initialize_for_training` |
| `data/context_extension_raw/context_ext.txt` | Raw QA text with EOS delimiters (generated) |
| `data/context_extension/` | Episode-indexed dataset: tokens.bin, mask.bin, segment_ids.bin, episodes.idx (generated) |
| `logs/train/phase1b_context_extension.jsonl` | Training log output |
| `docs/sft/SFT_PIPELINE_GUIDE.md` | Updated pipeline guide with Phase 1b section |
