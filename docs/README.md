# MyPT Documentation

Welcome to the MyPT documentation! This folder contains comprehensive guides and documentation for the MyPT project.

---

## 📚 Documentation Index

### Getting Started

- **[INSTALL.md](INSTALL.md)** - Comprehensive installation guide
  - Multiple installation methods
  - Platform-specific instructions (Windows, Linux, macOS)
  - CUDA setup and troubleshooting
  - Virtual environment setup

### Core Features

- **[CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md)** - JSON-based checkpoint system

  - New checkpoint file structure
  - Benefits and rationale
  - Backwards compatibility
  - Migration instructions

- **[JSON_CHECKPOINT_MIGRATION.md](JSON_CHECKPOINT_MIGRATION.md)** - Migration guide

  - How to migrate from legacy format
  - Step-by-step instructions
  - Automatic conversion tools
  - FAQ

- **[CONFIG_PRESETS.md](CONFIG_PRESETS.md)** - Configuration presets system

  - Predefined model sizes (150M, 200M, 250M)
  - Easy model creation with JSON configs
  - Parameter count estimation
  - Custom config creation

- **[TRAINING_CONFIG.md](TRAINING_CONFIG.md)** - Training configuration storage

  - Complete configuration tracking
  - What gets saved where
  - Reproducibility guide
  - Training lifecycle examples

- **[CONFIGURATION_STORAGE.md](CONFIGURATION_STORAGE.md)** - Configuration storage details

  - Fixed vs mutable configurations
  - File structure breakdown
  - Complete checkpoint contents
  - Usage examples

- **[PARAMETER_CALCULATION.md](PARAMETER_CALCULATION.md)** - Parameter calculation guide

  - How to calculate model parameters
  - Formula breakdown
  - Memory estimation
  - Using the calculator tool

- **[LARGE_DATASET_TRAINING.md](LARGE_DATASET_TRAINING.md)** - Large dataset training guide

  - Sharded dataset system for 100M+ tokens
  - Minimal RAM usage (memory-mapped shards)
  - Step-by-step examples
  - Performance comparison

- **[DATA_SOURCES_CONFIG.md](DATA_SOURCES_CONFIG.md)** - Data sources JSON configuration

  - JSON format for defining data sources
  - Configuring download URLs and weights
  - Creating custom source files
  - Usage with fetch_and_prepare_multilingual.py

- **[SHARDED_DATASET_IMPLEMENTATION.md](SHARDED_DATASET_IMPLEMENTATION.md)** - Sharded dataset implementation

  - Technical implementation details
  - Memory mapping explained
  - API reference
  - Use cases and examples

- **[SHARDED_TOKENIZER_FIX.md](SHARDED_TOKENIZER_FIX.md)** - Sharded dataset tokenizer fix

  - Fix for character-level tokenization with sharded datasets
  - Root cause analysis (gibberish output issue)
  - Implementation details
  - Verification steps

- **[DATASET_COVERAGE_ANALYSIS.md](DATASET_COVERAGE_ANALYSIS.md)** - Dataset coverage analysis

  - Automatic coverage calculation
  - Optimal training iterations
  - Coverage recommendations (2-5x)
  - Interactive warnings for low coverage

- **[VOCAB_SIZE_EXPLAINED.md](VOCAB_SIZE_EXPLAINED.md)** - Vocabulary size and parameters

  - GPT-2 BPE vs character-level tokenization
  - Impact on parameter count (~20-40M difference)
  - Why configs show 50304
  - How to calculate correctly for char-level

- **[TOKENIZATION_COMPARISON.md](TOKENIZATION_COMPARISON.md)** - Complete tokenization comparison

  - Detailed GPT-2 BPE vs char-level comparison
  - Parameter breakdown by tokenization
  - Sequence length differences
  - Trade-offs and when to use each

- **[SPECIAL_TOKENS.md](SPECIAL_TOKENS.md)** - Special tokens for structured text

  - Available special tokens (user, assistant, tool calls, etc.)
  - Token ID assignment and encoding/decoding
  - Usage examples and helper methods
  - Adding custom special tokens

- **[SFT_LOSS_MASKING.md](SFT_LOSS_MASKING.md)** - Supervised fine-tuning with loss masking

  - What is loss masking and why use it
  - Assistant-only training for chat models
  - Implementation details and usage
  - Creating masked datasets
  - Best practices and examples

- **[CONFIG_INHERITANCE_FIX.md](CONFIG_INHERITANCE_FIX.md)** - Config inheritance during fine-tuning
  - How configs are handled when using --init_from_model
  - Which parameters are inherited vs updated
  - Enabling SFT during fine-tuning
  - Dropout and batch size updates

### RAG (Retrieval-Augmented Generation)

- **[chat_sft_with_context.md](chat_sft_with_context.md)** - Chat SFT with RAG context

  - JSONL input format for conversations
  - Loss masking for assistant-only training
  - Creating datasets from RAG logs
  - Full training pipeline example

- **[DATA_PERSISTENCE.md](DATA_PERSISTENCE.md)** - Data persistence and reusability
  - Which data can be reused across model runs
  - Tokenization compatibility constraints
  - RAG index independence from models
  - Recommended workflow for experimentation

### Development & Architecture

- **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Initial refactoring details

  - Modular architecture overview
  - Code structure improvements
  - Design decisions

- **[PACKAGING_SUMMARY.md](PACKAGING_SUMMARY.md)** - Packaging system

  - pyproject.toml configuration
  - Public API design
  - Distribution details

- **[CLI_REFACTORING.md](CLI_REFACTORING.md)** - CLI enhancements
  - How CLI scripts use convenience functions
  - Enhanced output and features
  - Usage examples

### Technical Details

- **[PYTORCH_SECURITY_FIX.md](PYTORCH_SECURITY_FIX.md)** - Security improvements

  - Fix for torch.load warnings
  - weights_only parameter usage
  - Security best practices

- **[VERIFICATION.md](VERIFICATION.md)** - Refactoring verification
  - Line count comparisons
  - Testing checklist
  - Quality assurance

### Project Summary

- **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - Complete project overview
  - All features and improvements
  - Usage guide
  - Future enhancements

---

## 📖 Quick Reference

### For New Users

1. Start with **[INSTALL.md](INSTALL.md)** for installation
2. Read the main **[README.md](../README.md)** for usage examples
3. Check **[CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md)** to understand checkpoints

### For Developers

1. Read **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** for architecture
2. Check **[PACKAGING_SUMMARY.md](PACKAGING_SUMMARY.md)** for API details
3. See **[CLI_REFACTORING.md](CLI_REFACTORING.md)** for CLI patterns

### For Migration

1. Read **[JSON_CHECKPOINT_MIGRATION.md](JSON_CHECKPOINT_MIGRATION.md)**
2. Use the conversion script: `python convert_legacy_checkpoints.py --all`
3. See **[CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md)** for details

---

## 🎯 Documentation by Topic

### Installation & Setup

- [INSTALL.md](INSTALL.md) - Installation instructions
- [PACKAGING_SUMMARY.md](PACKAGING_SUMMARY.md) - Package structure

### Usage & Features

- Main [README.md](../README.md) - Quick start and examples
- [CLI_REFACTORING.md](CLI_REFACTORING.md) - CLI usage
- [CONFIG_PRESETS.md](CONFIG_PRESETS.md) - Configuration presets
- [TRAINING_CONFIG.md](TRAINING_CONFIG.md) - Training configuration

### Checkpoints & Data

- [CHECKPOINT_FORMAT.md](CHECKPOINT_FORMAT.md) - Checkpoint system
- [JSON_CHECKPOINT_MIGRATION.md](JSON_CHECKPOINT_MIGRATION.md) - Migration
- [CONFIGURATION_STORAGE.md](CONFIGURATION_STORAGE.md) - Configuration storage

### Architecture & Design

- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Initial refactoring
- [FINAL_SUMMARY.md](FINAL_SUMMARY.md) - Complete overview

### Technical & Security

- [PYTORCH_SECURITY_FIX.md](PYTORCH_SECURITY_FIX.md) - Security fix
- [VERIFICATION.md](VERIFICATION.md) - Testing & QA

---

## 📝 Document Status

| Document                     | Lines | Status      | Last Updated |
| ---------------------------- | ----- | ----------- | ------------ |
| INSTALL.md                   | ~370  | ✅ Complete | Nov 2025     |
| CHECKPOINT_FORMAT.md         | ~490  | ✅ Complete | Nov 2025     |
| JSON_CHECKPOINT_MIGRATION.md | ~370  | ✅ Complete | Nov 2025     |
| PACKAGING_SUMMARY.md         | ~450  | ✅ Complete | Nov 2025     |
| CLI_REFACTORING.md           | ~390  | ✅ Complete | Nov 2025     |
| REFACTORING_SUMMARY.md       | ~280  | ✅ Complete | Nov 2025     |
| PYTORCH_SECURITY_FIX.md      | ~280  | ✅ Complete | Nov 2025     |
| VERIFICATION.md              | ~325  | ✅ Complete | Nov 2025     |
| FINAL_SUMMARY.md             | ~500  | ✅ Complete | Nov 2025     |

**Total**: ~3,500 lines of comprehensive documentation

---

## 🔗 External Links

- **Main README**: [../README.md](../README.md)
- **Example Usage**: [../example_usage.py](../example_usage.py)
- **PyTorch Docs**: https://pytorch.org/docs/
- **tiktoken**: https://github.com/openai/tiktoken

---

## 💡 Contributing to Documentation

When adding new documentation:

1. **Place in docs/ folder**: Keep all documentation here
2. **Update this index**: Add your document to the appropriate section
3. **Link from README**: Add relevant links to the main README
4. **Use clear titles**: Make it easy to find information
5. **Add examples**: Show don't just tell

---

## 📧 Need Help?

- **Issues**: Check existing documentation first
- **Questions**: See [FINAL_SUMMARY.md](FINAL_SUMMARY.md) for overview
- **Installation**: See [INSTALL.md](INSTALL.md)
- **Migration**: See [JSON_CHECKPOINT_MIGRATION.md](JSON_CHECKPOINT_MIGRATION.md)

---

**Happy coding with MyPT!** 🚀
