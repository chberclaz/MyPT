# Changelog

All notable changes to MyPT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-15 — Complete SFT Pipeline

Design and implementation of the complete 6-phase Supervised Fine-Tuning pipeline
that transforms the 750M base model into a bilingual (EN/DE) agentic RAG assistant
with tool-calling, reasoning, and citation capabilities.

### Added

#### SFT Pipeline Architecture
- **6-phase sequential curriculum** taking the base model from raw token prediction to agentic RAG:
  1. **Format Lock** — learn `<myPT_assistant>...<myPT_eot>` skeleton, stop generation
  2. **Operators** — generalize abstract operators (COPY/WRAP/EXTRACT) to unseen payloads
  3. **Chat SFT** — natural bilingual conversation, `<myPT_think>` reasoning, `<myPT_cite>` attribution
  4. **Multi-turn** — multi-turn conversations with clean turn boundaries
  5. **Simple Toolcall** — single-step tool use with `<myPT_toolcall>` / `<myPT_toolresult>`
  6. **Agentic RAG** — multi-step tool chains (search → get_doc → answer), error recovery
- **Standardized data flow** for all phases: Generator → Mix → Tokenize → Train
- **Unified JSONL schema** with role types (`user`, `assistant`, `assistant_toolcall`, `toolresult`) and optional `think`, `cite`, `context` fields
- **19 special tokens** (IDs 50257-50275) with documented loss masking rules: model-generated tags trained, system-injected tags masked
- **Phase-by-phase tag introduction** so the model learns incrementally (system/user/assistant/eot in P1, think/cite/user_context in P3, toolcall/toolresult in P5)
- **Complete pipeline guide:** `docs/sft/SFT_PIPELINE_GUIDE.md` (17 sections, ~940 lines) covering architecture, data flow, every phase with exact commands, success gates, HuggingFace dataset integration, anti-forgetting strategies, and troubleshooting

#### Data Generation Scripts
- `scripts/sft/generate_rag_chat_sft.py` — Phase 3 RAG chat episodes with 6 episode patterns (context_answer, context_think, multiturn, extraction, insufficient_context, no_context), 30 EN + 30 DE context question templates, follow-up questions, extraction directives, insufficient-context refusals
- `scripts/sft/generate_agent_sft.py` — Phase 5-6 agentic tool-calling episodes with 8 patterns (search, list_docs, get_doc, summarize, multi_step, no_results, no_tool, contrastive), think/cite blocks, 4 EN + 4 DE contrastive wrong-tool-then-correction scenarios
- `scripts/sft/generate_pretrain_replay.py` — samples raw text from pre-training shards for anti-forgetting replay buffers

#### Tokenization & Packing
- `scripts/sft/prepare_chat_sft.py` — Phase 1-4 tokenizer with episode packing, loss masking, weighted masks
- `scripts/sft/prepare_tool_sft.py` — Phase 5-6 tokenizer handling all 19 tags including toolcall/toolresult roles, packing support, segment_ids for attention isolation
- `scripts/sft/pack_utils.py` — shared packing utilities: `greedy_bin_pack()`, `group_by_field()`, `compute_packing_stats()`
- **Episode packing** fills each 1024-token block with multiple episodes (25-50x efficiency gain for Phase 1-2 where episodes are ~30 tokens)

#### Training Configs
- 6 canonical configs in `configs/sft/`: `phase1_format_lock.json` through `phase6_agentic_rag.json`
- Monotonically decreasing learning rates: 7e-5 → 3e-5 → 3e-5 → 2.5e-5 → 2e-5 → 1.5e-5
- Phase-appropriate dropout: 0.10 (synthetic) → 0.08 (operators) → 0.12 (chat) → 0.10 → 0.08 → 0.08
- Cumulative `eval_sets` per phase for cross-phase OOD forgetting detection
- NEFTune noise (`neftune_alpha`) per phase

#### Three-Tier System Prompt Strategy
- Phase 1-2: `"You are MyPT."` (~4 tokens) — maximizes loss mask % on ultra-short episodes
- Phase 3-4: `CHAT_SYSTEM_PROMPT` (~15-20 tokens, 4 short variants) — episodes long enough to absorb
- Phase 5-6: `AGENTIC_STANDARD_PROMPT` (~80 tokens) — tool episodes are long, need tool list
- Defined in `core/system_prompts.py`

#### Anti-Forgetting & Evaluation
- **Pre-training replay:** `generate_pretrain_replay.py` for 1-5% raw text mixing per phase
- **Cross-phase replay:** mandatory schedule (Phase 2: 5% P1, Phase 3: 3% P1 + 3% P2, etc.)
- **Regression gate:** `scripts/eval/run_regression_gate.py` — automated pass/fail after each phase with cumulative bucket thresholds
- **Pre-training skills eval:** `data/eval_pretrain_skills/skills_eval.jsonl` — 32 held-out prompts (math, German, facts, code)
- **OOD generalization eval:** `data/eval_ood/` — 38 prompts across 4 files using novel phrasings absent from training templates, auto-detected by regression gate for phases ≥ 3

#### Weighted Loss Masking (WeFT-style)
- Control tokens (`<myPT_eot>`, `<myPT_toolcall>`, `<myPT_think>`, `<myPT_cite>`) get 1.5-2.0x loss weight
- `--weighted_mask` flag writes `mask_weighted.bin` (float32) alongside backward-compatible `mask.bin` (uint8)
- `core/episode_data_loader.py` auto-detects and prioritizes float32 weights

#### NEFTune Embedding Noise
- `core/model.py`: `neftune_alpha` in `GPTConfig`, uniform noise injection in `forward()` during training only
- Configurable per phase via config JSON

#### Future-Proof Special Token ID Resolution
- `core/special_tokens.py`: `BASE_VOCAB_SIZE` + `get_special_token_ids()` — single source of truth
- Migrated 15+ scripts from hardcoded numeric IDs to dynamic lookups

#### Data Quality & Diversity
- **Template diversity:** All template lists in both generators roughly doubled (think templates 2-3→5-7, answer templates 3→6-12, question templates 4-8→8-30)
- **Question style diversity:** Added casual, imperative, task-oriented, analytical, and extraction styles (previously all neutral/polite)
- **Contrastive examples:** 8% of Phase 5-6 episodes are wrong-tool-then-correction scenarios
- **Dynamic think omission:** `--think_omit_ratio` (15%) randomly omits think blocks so model learns thinking is contextual
- **German balance:** Default `--de_ratio` raised to 0.40 in both generators, full EN/DE parity on all templates
- **HuggingFace dataset recommendations** documented per phase (OASST2, alpaca-gpt4_de, Dolci-Instruct, OpenSchnabeltier, ultra-chat_de, German function calling, German RAG SFT)

### Changed

- Merged `configs/sft1/` + `configs/sft2/` → single `configs/sft/` directory (legacy configs archived to `_archive/`)
- `core/episode_data_loader.py`: prioritizes `mask_weighted.bin` (float32) over `mask.bin` (uint8)
- All 15+ scripts with hardcoded token IDs updated to use `get_special_token_ids()`

### Notes

- Complete pipeline documented in `docs/sft/SFT_PIPELINE_GUIDE.md` (17 sections)
- No breaking changes to existing training or inference workflows
- Backward-compatible: `mask.bin` (uint8) still works; `mask_weighted.bin` (float32) is optional
- All SFT Quality Audit items at 100%: C1-C8, A1, A4, A5, A6, B4, B5, B7

## [0.3.0] - 2026-02-08

### Added

#### Architecture Modernization (LLaMA-style)

- **Rotary Position Embeddings (RoPE):** Replaces learned absolute positional embeddings with rotation-based relative encoding. Enables position-invariant attention patterns critical for RAG retrieval heads. Supports future context-length extension without retraining.
- **SwiGLU Activation:** Gated MLP with SiLU activation replaces standard GELU feed-forward. Three weight matrices (gate, up, down) with auto-computed intermediate dimension (rounded to 64 for tensor core alignment). Approximately parameter-neutral vs standard 4x MLP.
- **RMSNorm:** Root Mean Square normalization replaces LayerNorm in all transformer blocks and final layer norm. Faster (skips mean subtraction) with no bias parameters.
- **Gradient Accumulation:** Configurable micro-step accumulation (`grad_accum_steps`) enables larger effective batch sizes without increasing GPU memory. Loss is normalized across micro-steps; optimizer steps once per accumulation cycle.

#### Training Infrastructure

- **Two-stage curriculum training:** Circuit formation phase (configurable code/retrieval-heavy mix) followed by balanced phase, managed via single config file with iteration-based data loader switching
- **Per-category eval holdout:** Separate evaluation sets for induction, retrieval, general, domain, and structured categories with independent monitoring
- **GOLD checkpoint system:** 3-guard validation (overfit ratio, consecutive val loss rises, eval set regression) for best-checkpoint selection

#### Config & Compatibility

- New `GPTConfig` fields: `pos_encoding` ("learned"/"rope"), `mlp_type` ("gelu"/"swiglu"), `norm_type` ("layernorm"/"rmsnorm"), `rope_theta` (base frequency), all with backward-compatible defaults
- All existing configs and checkpoints continue to work without modification (defaults match GPT-2 architecture)
- New 750M unified training config (`configs/base/750M_unified_v1.json`) with all modernizations enabled

### Changed

- `CausalSelfAttention.forward()` accepts optional `rope_cos`/`rope_sin` tensors for rotary embeddings (all cache paths: training, preallocated KV, cat-based KV)
- `Block` selects norm and MLP type from config flags
- `GPT.__init__()` conditionally creates RoPE buffers (non-learnable) or learned position embedding table
- `GPT.forward()` computes position-indexed RoPE cos/sin and passes through all blocks; supports per-sample positions for packed SFT
- `GPT.fit()` wraps training step in micro-batch accumulation loop with proper scaler handling
- `train.py` extracts `grad_accum_steps` from config and passes to `model.fit()`
- Training config iteration counts recalculated for 8x gradient accumulation (same total tokens, fewer optimizer steps)

### Notes

- Model parameter count with all modernizations: ~699M (vs ~695M baseline, +0.5%)
- SFT loss masking and segment attention masking fully compatible with all changes
- Smoke test suite (`scripts/model/smoke_test_arch.py`) validates all 10 verification points
- No breaking changes to existing training or inference workflows

## [0.2.1] - 2026-01-16

### Changed

- Phase 2 domain adaptation pipeline validated on large-scale corpora
- Dual-evaluation mechanism (domain + general) stabilized for catastrophic-forgetting detection
- Training constraints and helper utilities added to preserve base model capability during continued pretraining
- Documentation updated to reflect completed Phase 2 workflow and best practices

### Added

- Official Phase 2 Domain Adaptation Guide
- Helper tooling for controlled replay ratios and learning-rate safety

### Notes

- Phase 2 (Domain Adaptation) is considered complete and production-stable for internal and pilot deployments.
- No breaking changes.

## [0.2.0] - 2026-01-14

### Added

#### Core Training

- GPT model architecture with configurable layers, heads, and embedding dimensions
- Support for both character-level and GPT-2 BPE tokenization
- Sharded dataset format for large-scale training (100M+ tokens)
- Mixed precision training (FP16/BF16) with automatic mixed precision
- Gradient checkpointing for memory efficiency
- Learning rate scheduling with warmup and cosine decay

#### Domain Adaptation

- Continued pretraining with `--init_from_model` for domain adaptation
- Weighted multi-source dataset preparation (`prepare_weighted_dataset.py`)
- Phase 2 domain corpus builder for IT security, protocols, and programming docs
- Swiss Federal Law corpus scraper (Fedlex) with DE/EN support

#### Data Pipeline

- Multi-source data acquisition (Git cloning, direct downloads)
- Text transformers for various formats:
  - Markdown/reStructuredText → plaintext
  - Man pages → plaintext
  - RFC XML → plaintext
  - HTML → plaintext
  - JavaDoc → plaintext
  - Texinfo → plaintext
  - Go/TypeScript doc comments → plaintext
- Aggressive deduplication (exact hash + near-duplicate detection)
- Deterministic builds with configurable random seed

#### Generation & Inference

- Text generation with temperature and top-k/top-p sampling
- Interactive generation mode
- Batch generation support

#### Web Application

- Interactive chat interface
- Workspace document integration
- RAG (Retrieval-Augmented Generation) support
- User management system

#### Tools & Utilities

- Model parameter calculator (`calculate_params.py`)
- Dataset inspection and statistics
- Checkpoint conversion (legacy → JSON format)
- Configuration presets for various model sizes (150M, 350M, 750M, 1.5B)

### Infrastructure

- Comprehensive logging with audit trail
- Compliance tracking for data provenance
- Cross-platform support (Windows/Linux)
- Modular configuration system (JSON-based)

---

## [Unreleased]

### Planned

- Unified 6B token dataset generation (from-scratch training run)
- RAG evaluation benchmarks
- RAG Tool-Calling agent training
