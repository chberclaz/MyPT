# Changelog

All notable changes to MyPT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- Fine-tuning pipeline (SFT, RLHF)
- RAG
- RAG Tool-Calling
