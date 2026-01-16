# MyPT Documentation

## Quick Start

| Guide                                                                  | Description                                              |
| ---------------------------------------------------------------------- | -------------------------------------------------------- |
| [**guides/GETTING_STARTED.md**](guides/GETTING_STARTED.md)             | 🚀 Beginner walkthrough - your first model in 30 minutes |
| [guides/DOMAIN_ADAPTATION_GUIDE.md](guides/DOMAIN_ADAPTATION_GUIDE.md) | Phase 1 & 2 training reproduction guide                  |
| [guides/QUICK_REFERENCE.md](guides/QUICK_REFERENCE.md)                 | Command cheat sheet                                      |
| [setup/INSTALL.md](setup/INSTALL.md)                                   | Detailed installation instructions                       |

---

## Documentation Structure

```
docs/
├── README.md                    # This file
│
├── guides/                      # Step-by-Step Guides
│   ├── GETTING_STARTED.md           # Beginner walkthrough
│   ├── DOMAIN_ADAPTATION_GUIDE.md   # Phase 1 & 2 training
│   ├── MODEL_SELECTION_GUIDE.md     # Choosing the right model
│   ├── QUICK_REFERENCE.md           # Command cheat sheet
│   └── TROUBLESHOOTING.md           # Common issues & solutions
│
├── setup/                       # Getting Started
│   ├── INSTALL.md              # Installation guide
│   ├── DEPENDENCIES.md         # Python dependencies
│   ├── DOCKER.md               # Docker deployment
│   └── PROJECT_STRUCTURE.md    # Project layout
│
├── training/                    # Training & Data
│   ├── LARGE_DATASET_TRAINING.md    # Sharded dataset training
│   ├── TRAINING_CONFIG.md           # Config storage & options
│   ├── phase2_domain_corpus.md      # Domain corpus building
│   ├── DATA_SOURCES_CONFIG.md       # Data source configuration
│   ├── PARAMETER_CALCULATION.md     # Model sizing
│   └── ...
│
├── sft/                         # Supervised Fine-Tuning
│   ├── PHASE3A_CHAT_SFT_GUIDE.md    # Chat SFT guide
│   ├── SFT_LOSS_MASKING.md          # Loss masking explained
│   ├── toolcall_sft.md              # Tool-calling SFT
│   ├── EPISODE_INDEXED_SFT.md       # Episode-indexed loader
│   └── ...
│
├── model/                       # Model & Architecture
│   ├── CHECKPOINT_FORMAT.md         # JSON checkpoint system
│   ├── GENERATION_GUIDE.md          # Text generation
│   ├── SPECIAL_TOKENS.md            # Special tokens
│   ├── TOKENIZATION_COMPARISON.md   # GPT-2 vs char tokenization
│   └── ...
│
├── webapp/                      # Web Application
│   ├── WEBAPP_GUIDE.md              # Web app guide
│   ├── AUTHENTICATION.md            # Auth system
│   ├── workspace_api.md             # Workspace API
│   └── ...
│
├── compliance/                  # Security & Compliance
│   ├── AUDIT_COMPLIANCE.md          # Audit logging
│   └── PYTORCH_SECURITY_FIX.md      # Security considerations
│
├── reference/                   # Reference Docs
│   ├── CONFIG_PRESETS.md            # Configuration presets
│   └── WHERE_TO_SEE_PARAMETERS.md   # Parameter inspection
│
├── specs/                       # Design Specifications
│   ├── spec_domain_datagrabber.md   # Domain corpus spec
│   ├── spec_gitinterface.md         # Git interface spec
│   └── ...
│
└── archive/                     # Historical Docs
    ├── FINAL_SUMMARY.md             # Project overview
    ├── REFACTORING_SUMMARY.md       # Refactoring history
    └── ...
```

---

## By Topic

### 🚀 Getting Started

- [Installation Guide](setup/INSTALL.md) - System requirements, install methods
- [Dependencies](setup/DEPENDENCIES.md) - Python packages, CUDA setup
- [Docker Guide](setup/DOCKER.md) - Container deployment
- [Project Structure](setup/PROJECT_STRUCTURE.md) - Codebase layout

### 📖 Guides

- [**Getting Started**](guides/GETTING_STARTED.md) - Beginner walkthrough (zero to first model)
- [**Domain Adaptation Guide**](guides/DOMAIN_ADAPTATION_GUIDE.md) - Phase 1 & 2 training
- [Model Selection Guide](guides/MODEL_SELECTION_GUIDE.md) - Choosing the right model size
- [Quick Reference](guides/QUICK_REFERENCE.md) - Command cheat sheet
- [Troubleshooting](guides/TROUBLESHOOTING.md) - Common issues & solutions

### 📊 Training

- [Large Dataset Training](training/LARGE_DATASET_TRAINING.md) - Sharded datasets
- [Training Config](training/TRAINING_CONFIG.md) - Configuration options
- [Phase 2 Domain Corpus](training/phase2_domain_corpus.md) - Building domain data
- [Data Sources Config](training/DATA_SOURCES_CONFIG.md) - JSON source files
- [Dataset Coverage](training/DATASET_COVERAGE_ANALYSIS.md) - Epoch calculations
- [Parameter Calculation](training/PARAMETER_CALCULATION.md) - Model sizing

### 💬 Supervised Fine-Tuning (SFT)

- [Chat SFT Guide](sft/PHASE3A_CHAT_SFT_GUIDE.md) - Conversation training
- [Loss Masking](sft/SFT_LOSS_MASKING.md) - Assistant-only training
- [Tool-calling SFT](sft/toolcall_sft.md) - Agentic RAG training
- [Episode-Indexed Loader](sft/EPISODE_INDEXED_SFT.md) - Conversation loader
- [Gold Episodes](sft/GOLDEPISODES_REFERENCE.md) - Episode structure

### 🧠 Model & Architecture

- [Checkpoint Format](model/CHECKPOINT_FORMAT.md) - JSON-based checkpoints
- [Generation Guide](model/GENERATION_GUIDE.md) - Text generation
- [Special Tokens](model/SPECIAL_TOKENS.md) - Custom tokens
- [Tokenization](model/TOKENIZATION_COMPARISON.md) - GPT-2 vs char
- [Sharded Datasets](model/SHARDED_DATASET_IMPLEMENTATION.md) - Binary shards

### 🌐 Web Application

- [Web App Guide](webapp/WEBAPP_GUIDE.md) - Browser interface
- [Authentication](webapp/AUTHENTICATION.md) - Login system
- [Workspace API](webapp/workspace_api.md) - Tool interface
- [Document Formats](webapp/DOCUMENT_FORMATS.md) - PDF, DOCX support

### 🔒 Security & Compliance

- [Audit & Compliance](compliance/AUDIT_COMPLIANCE.md) - Logging
- [PyTorch Security](compliance/PYTORCH_SECURITY_FIX.md) - weights_only

### 📖 Reference

- [Config Presets](reference/CONFIG_PRESETS.md) - Model configurations
- [Parameter Inspection](reference/WHERE_TO_SEE_PARAMETERS.md) - View params

### 📝 Design Specs

- [Domain Datagrabber](specs/spec_domain_datagrabber.md)
- [Episode SFT Loader](specs/spec_episode_index_sft_dataloader.md)
- [Git Interface](specs/spec_gitinterface.md)
- [Observability](specs/spec_obsinterface.md)

---

_Last updated: January 2026_
