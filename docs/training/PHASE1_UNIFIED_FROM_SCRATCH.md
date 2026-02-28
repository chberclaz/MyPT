# Phase 1: Unified From-Scratch Training (v1)

> **Config:** `configs/base/750M_unified_v1.json`  
> **Architecture:** LLaMA-2 style (RoPE + SwiGLU + RMSNorm), ~699M parameters  
> **Dataset:** ~6B tokens, 10 sources, RAG-optimized mix  
> **Training:** Two-stage curriculum, gradient accumulation 8x, 75K optimizer steps  
> **Date:** February 2026

---

## Background: Why This Training Run Exists

### The Problem

The previous 750M model (GPT-2 architecture, prose-heavy dataset) failed at RAG generalization.
When given a retrieval-augmented prompt -- context passage + question -- the model would either:

- **Memorize** specific training examples instead of learning the abstract "look up answer in context" operation
- **Ignore** the provided context and hallucinate from parametric memory
- **Fail to copy** relevant spans from the context into the answer

Root cause analysis identified two issues:

1. **Architectural limitation:** GPT-2's learned absolute positional embeddings force the model to learn position-specific attention patterns. An induction head that learns "copy from 50 tokens back" at position 100 must re-learn it separately at position 500. This makes retrieval heads fragile and position-dependent.

2. **Data composition:** The pre-training dataset was ~70% general prose (Wikipedia, web text) with minimal code or structured Q&A. Code is the strongest driver of induction head formation (variable binding, function calls force exact-copy attention patterns). Without enough code, the model never developed strong copying circuits.

### The Solution

Two changes, applied together:

1. **Architecture modernization:** Replace GPT-2 components with LLaMA-2 equivalents (RoPE, SwiGLU, RMSNorm). RoPE encodes relative distances directly in the Q\*K dot product, so a retrieval pattern learned once works at any position.

2. **RAG-optimized dataset with curriculum training:** A 6B-token dataset with 23% code (induction signal) and 20% structured retrieval Q&A (retrieval signal), trained with a two-stage curriculum that front-loads high-signal data during the critical circuit formation window.

---

## Architecture

| Component         | Old (GPT-2)           | New (LLaMA-2 style)       | Why                                                                                    |
| ----------------- | --------------------- | ------------------------- | -------------------------------------------------------------------------------------- |
| Position encoding | Learned absolute      | Rotary (RoPE)             | Position-invariant attention patterns; single retrieval circuit works at all positions |
| MLP activation    | GELU (2 matrices)     | SwiGLU (3 gated matrices) | Selective information routing; more expressive per parameter                           |
| Normalization     | LayerNorm             | RMSNorm                   | Faster (no mean subtraction), no bias params                                           |
| Batch scaling     | Fixed (batch_size=12) | Gradient accumulation 8x  | Effective batch 96, smoother gradients during circuit formation                        |

**Parameter count:** ~699M (vs ~695M old). Essentially parameter-neutral.

### Model Dimensions

| Parameter     | Value | Rationale                                                                |
| ------------- | ----- | ------------------------------------------------------------------------ |
| `n_embd`      | 1280  | Standard for 750M class                                                  |
| `n_head`      | 20    | 64 dims/head, optimal for RoPE                                           |
| `n_layer`     | 32    | Deep enough for circuit depth                                            |
| `block_size`  | 1024  | Full context for RAG passages                                            |
| `vocab_size`  | 50304 | GPT-2 tokenizer + special tokens, padded to 64 for tensor core alignment |
| `dropout`     | 0.1   | Mild regularization, standard                                            |
| `bias`        | false | Modern practice, slightly faster                                         |
| `tie_weights` | true  | Shared embedding/lm_head, fewer params                                   |

---

## Dataset: 10 Sources, 5 Categories

### Category Rollup

| Category                 | Sources                        | %   | Function                                                                           |
| ------------------------ | ------------------------------ | --- | ---------------------------------------------------------------------------------- |
| **Induction (code)**     | Python, JS/Java                | 23% | Forces exact-copy attention patterns via variable binding, function calls, imports |
| **Retrieval (Q&A)**      | StackExchange (17%), TriviaQA/SQuAD v2 (3%) | 20% | Teaches passage-grounded answer extraction + unanswerable recognition           |
| **General language**     | FineWeb-Edu, Wikipedia, Reddit | 45% | Broad knowledge, writing quality, conversational fluency                           |
| **Domain**               | IT-sec + Swiss law             | 7%  | Target domain baked in natively from pre-training                                  |
| **Structured/technical** | peS2o, GitHub README           | 5%  | Document structure, citation patterns, code-NL bridging                            |

### Source Detail

| #   | Source                                   | Tokens | %   | Category   | Function / Reason                                                                                                                                                                                |
| --- | ---------------------------------------- | ------ | --- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **FineWeb-Edu**                          | 1.62B  | 27% | General    | Educational web content filtered by quality classifier. Largest source: provides diverse world knowledge, writing styles, and basic language capability.                                         |
| 2   | **Bilingual Wikipedia** (EN+DE)          | 900M   | 15% | General    | Encyclopedic, entity-dense, factual. Bilingual for DE capability. Already tokenized from prior runs.                                                                                             |
| 3   | **Python code** (CodeParrot + StarCoder) | 900M   | 15% | Induction  | **Primary induction driver.** Variable reuse, function calls, import chains force the model to learn exact-copy attention patterns. These become induction heads.                                |
| 4   | **StackExchange Q&A**                    | 1.01B  | 17% | Retrieval  | **Conversational retrieval.** Full Q&A threads where answers reference code, commands, and errors from the question. Increased from 13% to 17% to absorb TriviaQA/SQuAD v2 shortfall.            |
| 5   | **JS + Java code** (StarCoder)           | 480M   | 8%  | Induction  | Secondary induction from different languages. Java's verbosity provides longer-range copy patterns.                                                                                              |
| 6   | **TriviaQA + SQuAD v2**                  | 192M   | 3%  | Retrieval  | **Extractive retrieval.** TriviaQA (~150M): trivia questions grounded in Wikipedia evidence passages. SQuAD v2 (~42M): precise span extraction + ~20% unanswerable questions. Reduced from 7% to 3% (only 192M tokens passed grounding filter). |
| 7   | **IT-sec + Swiss law**                   | 420M   | 7%  | Domain     | Target deployment domain. IT security protocols, Swiss Federal Law (DE/EN). Upsampled ~2.6x from 161M base corpus.                                                                               |
| 8   | **Reddit threaded**                      | 180M   | 3%  | General    | Conversational language patterns. Reduced from 5% after audit (only ~30-40% has useful quote-reply structure).                                                                                   |
| 9   | **peS2o scientific**                     | 180M   | 3%  | Structured | Scientific papers with structured sections, citations, cross-references. Teaches document structure understanding.                                                                               |
| 10  | **GitHub README**                        | 120M   | 2%  | Structured | Bridges code and natural language. Reduced after audit (non-English content, changelogs, frontmatter).                                                                                           |

### Source Audit Decisions

- **OpenSubtitles REMOVED:** Quality audit found literal line duplication from windowing overlap, extremely low information density, broken/translated English. Near-zero value for induction or retrieval.
- **Reddit reduced 5% to 3%:** Only ~30-40% of content has the quote-reply patterns that support retrieval. Remainder is flat debate.
- **GitHub README reduced 4% to 2%:** Contains non-English content (Chinese READMEs), changelogs, Jekyll frontmatter.
- **StackExchange increased 10% to 17%:** Strongest conversational retrieval source after audit. Absorbed 228M shortfall from TriviaQA/SQuAD v2 (only 192M passed grounding filter vs 420M target).
- **TriviaQA + SQuAD v2 at 3% (192M):** v3.1 adjustment. TriviaQA's `answer.lower() in passage.lower()` grounding filter reduces usable tokens significantly, but this ensures 100% of examples are genuinely grounded retrieval. SQuAD v2 supplements with ~20% unanswerable questions -- critical negative signal for RAG. NQ was dropped: 42GB download for ~250M useful tokens.
- **FineWeb-Edu reduced 30% to 27%:** Funds the retrieval source additions. Still largest single source.

### Cross-Source Signal Analysis

The category labels above (Induction, Retrieval, General, etc.) are a simplification. In practice, induction and retrieval signals are distributed across most sources, not isolated to the ones we labeled. Understanding the true signal coverage is important for evaluating whether the dataset is one-sided or balanced.

**Induction = "I've seen pattern A before; copy/complete what followed it."** This is not just code.

| Source        | Category Label | %   | Induction mechanism                                                        | Signal strength      |
| ------------- | -------------- | --- | -------------------------------------------------------------------------- | -------------------- |
| Python code   | Induction      | 15% | Exact-copy: `x = 5; print(x)` -- variable binding, function calls, imports | **Strong** (primary) |
| JS/Java code  | Induction      | 8%  | Same as Python; Java verbosity gives longer-range copy patterns            | **Strong**           |
| StackExchange | Retrieval      | 17% | Answers echo code snippets, error messages, commands from the question     | **Strong** (applied) |
| TriviaQA + SQuAD v2 | Retrieval | 3%  | Answer is a span copied from passage -- extractive induction. SQuAD v2 adds unanswerable = negative induction signal | **Strong** (applied) |
| Wikipedia     | General        | 15% | Entity coreference: "Einstein was born in Ulm. **Einstein** later..."      | Moderate             |
| Swiss law     | Domain         | 7%  | Legal cross-reference: "pursuant to **Article 5**, paragraph 2..."         | Moderate             |
| peS2o         | Structured     | 3%  | Citation patterns: "As shown in **Theorem 3.1**..."                        | Moderate             |
| Reddit        | General        | 3%  | `>` quote-reply patterns -- explicit context echoing (~30-35% of data)     | Moderate (sparse)    |
| GitHub README | Structured     | 2%  | Code-prose bridging: explaining `function_name` referenced in examples     | Weak-moderate        |
| FineWeb-Edu   | General        | 27% | Occasional back-references in educational text                             | Weak (diffuse)       |

**True induction coverage (Phase 2):** ~43% of the dataset carries meaningful induction signal, not just the 23% labeled as "Induction (code)." The retrieval sources (20%) ARE induction training -- they teach the same copying operation applied to natural language passages rather than code tokens. This provides two complementary forms: code induction (short, exact-match, token-level) and retrieval induction (longer spans, natural language, semantic-level). Both are needed for RAG.

**Retrieval = "Given context + query, locate relevant information and extract/use it."** This is not just Q&A.

| Source        | Category Label | %   | Retrieval mechanism                                                                | Signal strength     |
| ------------- | -------------- | --- | ---------------------------------------------------------------------------------- | ------------------- |
| TriviaQA + SQuAD v2 | Retrieval | 3%  | Explicit extractive: passage → question → answer-from-passage (100% retrieval). SQuAD v2's unanswerable questions teach "answer NOT here" (negative retrieval) | **Strong** (purest) |
| StackExchange | Retrieval      | 17% | Conversational: answer references Q's code, errors, context (~75% retrieval)       | **Strong**          |
| Python code   | Induction      | 15% | `import X; ... X.method()` = retrieve what was imported; definition → usage (~28%) | Moderate            |
| JS/Java code  | Induction      | 8%  | Same as Python: import/definition retrieval (~28%)                                 | Moderate            |
| Swiss law     | Domain         | 7%  | "As defined in Article 3..." = structured legal retrieval (~38%)                   | Moderate-strong     |
| peS2o         | Structured     | 3%  | "From equation (3)...", "see Section 2.1" = citation cross-ref (~28%)              | Moderate            |
| Reddit        | General        | 3%  | `>` quote then respond = explicit context retrieval (~33%)                         | Moderate (sparse)   |
| Wikipedia     | General        | 15% | Entity fact recall across paragraphs, pronoun resolution (~18%)                    | Weak-moderate       |
| GitHub README | Structured     | 2%  | API docs reference earlier-defined code patterns (~23%)                            | Weak-moderate       |
| FineWeb-Edu   | General        | 27% | "As we discussed...", "recall from section..." (~8%)                               | Weak (diffuse)      |

**True retrieval coverage (Phase 2):** ~33% of the dataset carries meaningful retrieval signal, not just the 20% labeled as "Retrieval (Q&A)." Code provides a different but complementary form of retrieval (lookup-by-name rather than lookup-by-meaning), and domain/scientific sources provide structured cross-referencing that mirrors how RAG contexts are organized.

**Phase 1 signal coverage (circuit formation window):**

| Signal type | Labeled %       | Estimated true % | Sources contributing                                         |
| ----------- | --------------- | ---------------- | ------------------------------------------------------------ |
| Induction   | 40% (code only) | ~70%             | Code 40% + Retrieval Q&A 30% + soft signals from general ~5% |
| Retrieval   | 30% (Q&A only)  | ~44%             | Q&A 30% + code retrieval ~11% + domain/scientific ~3%        |

**Why this matters:** The dataset is NOT one-sided. Code provides the strongest and cleanest induction signal (exact-copy), but 30% of Phase 1 is retrieval Q&A which trains the same circuits on natural language. This prevents "locking in" to code-only copy patterns. Conversely, retrieval training isn't limited to Q&A -- code teaches lookup-by-name, and domain/scientific text teaches structured cross-referencing. The model forms diverse, complementary circuits across both signal types simultaneously.

---

## Two-Stage Curriculum Training

Based on Anthropic (2022) research showing that induction heads form during a sharp phase transition in the first 5-15% of training, we use a two-stage curriculum.

### Stage 1: Circuit Formation (steps 0 - 9,125 | ~0.9B tokens)

| Category         | Phase 1 % | Phase 2 % | Change                                             |
| ---------------- | --------- | --------- | -------------------------------------------------- |
| Code (induction) | **40%**   | 23%       | +17pp -- maximize induction head formation         |
| Retrieval (Q&A)  | **30%**   | 20%       | +10pp -- maximize retrieval head formation         |
| General + domain | **30%**   | 57%       | -27pp -- reduced but sufficient for basic language |

**Rationale:** During the critical circuit formation window, bias attention heads toward copying and retrieval patterns. These circuits, once formed, persist through Phase 2.

### Stage 2: Balanced (steps 9,125 - 75,000 | ~6.5B tokens)

Normal v3.1 mix (the percentages in the table above). Restores full general language exposure for broad capability while the retrieval circuits formed in Phase 1 continue to strengthen.

**Implementation:** Single training run, single continuous LR schedule. Only the data loader switches at step 9,125. Managed by the `curriculum` section in the config JSON.

### Data Overlap

Phase 1 sources are a subset of Phase 2 sources (the mixer copies shards). This means ~882M tokens (12.8% of total processed) are seen twice -- once in each phase. This is intentional: mild repetition of high-value code/retrieval data reinforces the circuits formed in Phase 1, and Phase 2 retains full source diversity.

---

## Training Configuration

### Hyperparameters

| Parameter          | Value       | Rationale                                                  |
| ------------------ | ----------- | ---------------------------------------------------------- |
| `learning_rate`    | 1.5e-4      | Standard for 750M scale (GPT-3 paper range)                |
| `warmup_iters`     | 3,750       | 5% of training, linear warmup to peak LR                   |
| `max_iters`        | 75,000      | Optimizer steps. Total tokens: 75K _ 96 _ 1024 = ~7.37B    |
| `grad_accum_steps` | 8           | Effective batch: 12 \* 8 = 96 sequences = ~98K tokens/step |
| `grad_clip`        | 1.0         | Standard gradient clipping                                 |
| `weight_decay`     | 0.1         | AdamW regularization                                       |
| `use_amp`          | true (bf16) | Mixed precision for speed and memory                       |
| `eval_interval`    | 350         | Every ~34M tokens. ~214 evals over full run                |
| `eval_iters`       | 200         | Batches per eval split                                     |
| `eval_seed`        | 1337        | Reproducible eval across runs                              |

### LR Schedule

Linear warmup (0 to 1.5e-4 over 3,750 steps) followed by cosine decay to 1.5e-5 (10% of peak). Continuous across both curriculum phases -- no LR reset at phase boundary.

### Effective Batch Size Calculation

```
micro_batch     = 12 sequences * 1024 tokens = 12,288 tokens
effective_batch = 12,288 * 8 (grad_accum)    = 98,304 tokens/step
```

This matches the optimal range for 750M-scale models (64K-128K tokens/step), providing smooth gradient estimates critical for stable circuit formation.

---

## Evaluation Sets (4 categories, held out before training)

| Eval Set    | Source                                        | Purpose                                              |
| ----------- | --------------------------------------------- | ---------------------------------------------------- |
| `general`   | Bilingual Wikipedia                           | Detect catastrophic forgetting of general capability |
| `domain`    | IT-sec + Swiss law                            | Monitor domain-specific knowledge retention          |
| `code`      | CodeParrot + StarCoder (held-out shards)      | Monitor induction circuit health                     |
| `retrieval` | StackExchange + TriviaQA + SQuAD v2 (held-out shards) | Monitor retrieval circuit health               |

All eval shards are held back **before** tokenization and mixing. They are never seen during training.

### GOLD Checkpoint System

The best checkpoint is saved with three guards to prevent false "best" saves:

1. **Overfit guard:** val/train ratio must not exceed 5x
2. **Trend guard:** val loss must not have risen 2+ consecutive evals before this dip (prevents saving noise/flapping)
3. **Eval regression guard:** no eval set may degrade >20% from its initial baseline (prevents catastrophic forgetting)

---

## Commands

### Dataset Generation (run locally)

```bash
# Full pipeline: download, holdback eval shards, tokenize, mix both phases
py scripts/unified_build/build_unified_dataset.py
```

This produces:

- `data/unified_6B/` -- Phase 2 balanced dataset (~6B tokens)
- `data/unified_phase1_circuit/` -- Phase 1 circuit formation (~0.9B tokens)
- `data/code_eval_tokenized/` -- Code eval set (held out)
- `data/retrieval_eval_tokenized/` -- Retrieval eval set (held out)

### Upload to RunPod

```
data/unified_6B/
data/unified_phase1_circuit/
data/code_eval_tokenized/
data/retrieval_eval_tokenized/
data/multilingual_1.5B_wiki90/        (general eval, if not already there)
data/domain_161M_corpus_tokenized/    (domain eval, if not already there)
```

### Training (run on RunPod)

```bash
python train.py \
    --config_file configs/base/750M_unified_v1.json \
    --model_name unified_v1_llama \
    --dataset_dir data/unified_phase1_circuit
```

`--dataset_dir` points to Phase 1 (the starting dataset). The config's `curriculum` section auto-switches to `data/unified_6B` at step 9,125.

---

## Expected Outcomes

### Quantitative (conservative estimates from published ablations)

- Expected significant improvement in retrieval consistency and position invariance.
- General perplexity: **3-7% improvement** (SwiGLU ~2%, gradient accumulation ~2-3%, RMSNorm ~0.5%)
- Training stability: significantly smoother loss curves (8x larger effective batch)

### Qualitative

- Model should learn position-invariant retrieval: "find the answer in the context" works regardless of where in the 1024-token window the answer appears
- RAG grounding: given relevant context, extract and use information rather than hallucinating
- Curriculum should accelerate induction/retrieval head formation in the first 15% of training
- Not expected to match 7B+ models on general reasoning -- this is a RAG execution engine, not a general chatbot

### What Success Looks Like

1. `eval_retrieval` loss decreasing steadily, not plateauing early
2. `eval_code` loss tracking close to `val_loss` (code circuits healthy)
3. `eval_general` and `eval_domain` stable (no catastrophic forgetting)
4. GOLD checkpoint selected with all three guards passing
5. Qualitative: model extracts answers from provided passages rather than ignoring context

---

## File References

| File                                             | Description                                                 |
| ------------------------------------------------ | ----------------------------------------------------------- |
| `configs/base/750M_unified_v1.json`              | Training configuration                                      |
| `data/sources/unified_from_scratch.json`         | Phase 2 dataset source definitions and percentages          |
| `data/sources/unified_phase1_circuit.json`       | Phase 1 circuit formation source definitions                |
| `scripts/unified_build/build_unified_dataset.py` | Dataset generation pipeline                                 |
| `scripts/unified_build/download_nq_triviaqa.py`  | TriviaQA + SQuAD v2 downloader with grounded-retrieval formatting   |
| `scripts/unified_build/mix_multi_source.py`      | Multi-source shard mixer                                    |
| `core/dataset_lineage.py`                        | Pipeline-core recursive dataset lineage + provenance helper |
| `docs/training/DATASET_LINEAGE_STANDARD.md`      | Global lineage schema and audit interpretation guide        |
| `scripts/model/smoke_test_arch.py`               | Architecture verification test suite                        |
| `core/model.py`                                  | Model implementation (RoPE, SwiGLU, RMSNorm, grad accum)    |
| `logs/train/unified_v1_eval.jsonl`               | Training log output (JSONL with timestamps, losses, phases) |
