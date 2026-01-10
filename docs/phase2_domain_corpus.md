# Phase 2 Domain Corpus Builder

Build domain-specific training data for IT security, protocols, Bash/Unix fundamentals, and programming language documentation.

## Overview

The Phase 2 Domain Corpus pipeline automates the creation of a high-quality training dataset from authoritative technical sources. It handles the complete workflow from source acquisition through tokenization:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FETCH     │───▶│  TRANSFORM  │───▶│   CLEAN     │───▶│   DEDUPE    │───▶│  TOKENIZE   │
│  git clone  │    │  md/rst/xml │    │  normalize  │    │ exact+near  │    │  GPT-2 BPE  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Features

### Multi-Format Transformers
- **Markdown/RST** - Preserves code blocks, headings, and lists
- **Man Pages** - Uses groff when available, fallback parser otherwise
- **RFC XML** - Extracts structured content from IETF specifications
- **HTML** - Removes navigation/boilerplate, keeps content structure
- **JSON** - Extracts MITRE ATT&CK threat intelligence data

### Intelligent Deduplication
- **Exact Deduplication** - SHA-256 hash matching
- **Near-Duplicate Detection** - Simhash with configurable Hamming distance threshold

### Deterministic & Auditable
- Configurable random seed for reproducibility
- Comprehensive file logging with timestamps
- Audit trail integration (`core.compliance.audit`)
- Build metadata JSON for complete provenance

### Cross-Platform
- Pure Python implementation (no bash dependencies)
- Works on Windows, macOS, and Linux

## Quick Start

### Option 1: Full Pipeline (Recommended)

```powershell
# Windows
python scripts/fetch_and_prepare_phase2_domain.py `
    --out_dir data/phase2_domain `
    --total_tokens 100000000

# Linux/macOS
python scripts/fetch_and_prepare_phase2_domain.py \
    --out_dir data/phase2_domain \
    --total_tokens 100000000
```

This will:
1. Clone all source repositories to `./sources/`
2. Build the corpus with deduplication
3. Tokenize into training shards

### Option 2: Step-by-Step

```powershell
# Step 1: Fetch sources
python tools/build_phase2_corpus.py --fetch --sources_dir sources

# Step 2: Build corpus
python tools/build_phase2_corpus.py `
    --sources_dir sources `
    --out_dir out/corpus `
    --dedupe exact,simhash

# Step 3: Tokenize
python scripts/prepare_weighted_dataset.py `
    --source domain:out/corpus/corpus_shards/*.txt `
    --total_tokens 100000000 `
    --out_dir data/phase2_tokenized
```

## Configuration

### Command Line Options

#### `scripts/fetch_and_prepare_phase2_domain.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--sources_dir` | `sources` | Directory for cloned repositories |
| `--work_dir` | `work` | Working directory for logs and intermediate files |
| `--out_dir` | `data/phase2_domain` | Output directory for final dataset |
| `--config_file` | `data/sources/phase2_domain.json` | JSON configuration file |
| `--total_tokens` | `100000000` | Target token count |
| `--tokenization` | `gpt2` | Tokenizer type (`gpt2` or `char`) |
| `--min_chars` | `400` | Minimum characters per document |
| `--dedupe` | `exact,simhash` | Deduplication methods |
| `--seed` | `42` | Random seed for reproducibility |
| `--skip_fetch` | - | Skip repository cloning |
| `--skip_build` | - | Skip corpus building |
| `--corpus_only` | - | Build corpus only, skip tokenization |
| `--sources` | all | Filter which sources to process |

#### `tools/build_phase2_corpus.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--fetch` | - | Clone repositories before building |
| `--shard_mb` | `25` | Target shard size in MB |
| `--simhash_threshold` | `4` | Hamming distance threshold for near-duplicates |
| `--include_ext` | `.md,.rst,.txt,.html,.xml,.json` | File extensions to process |
| `--max_docs_per_repo` | `0` | Max documents per repo (0 = unlimited) |
| `--run_tokenizer` | - | Run tokenizer after corpus build |

### JSON Configuration

The pipeline can be configured via `data/sources/phase2_domain.json`:

```json
{
  "name": "Phase 2 Domain Corpus",
  "default_weights": {
    "owasp-cheatsheets": 0.12,
    "python-docs": 0.12,
    "mdn-content": 0.10
  },
  "sources": {
    "owasp-cheatsheets": {
      "name": "OWASP Cheat Sheets",
      "path": "owasp-cheatsheets",
      "repo_url": "https://github.com/OWASP/CheatSheetSeries.git",
      "weight": 1.5,
      "include_patterns": ["cheatsheets/*.md"],
      "exclude_patterns": ["README.md"],
      "license_hint": "CC-BY-SA-4.0"
    }
  }
}
```

## Included Sources

### Security
| Source | Description | License |
|--------|-------------|---------|
| OWASP Top 10 | Security vulnerabilities documentation | CC-BY-SA-4.0 |
| OWASP Cheat Sheets | Security best practices | CC-BY-SA-4.0 |
| OWASP WSTG | Web security testing methodology | CC-BY-SA-4.0 |
| MITRE ATT&CK | Threat intelligence framework | Apache-2.0 |

### Protocols
| Source | Description | License |
|--------|-------------|---------|
| RFC XML | IETF RFC specifications | BSD-3-Clause |

### Unix/Bash
| Source | Description | License |
|--------|-------------|---------|
| Linux Man Pages | Official manual pages | GPL/BSD/MIT |
| GNU Bash | Shell documentation | GPL-3.0 |

### Programming Languages
| Source | Description | License |
|--------|-------------|---------|
| Python Docs | Official Python documentation | PSF-2.0 |
| Python PEPs | Python Enhancement Proposals | PSF-2.0 |
| Node.js Docs | Node.js API documentation | MIT |
| MDN Web Docs | Mozilla web documentation | CC-BY-SA-2.5 |
| OpenJDK | Java source documentation | GPL-2.0-classpath |

> **Note**: You are responsible for verifying licenses of included sources before use.

## Output Structure

```
data/phase2_domain/
├── corpus/
│   ├── corpus_shards/
│   │   ├── shard_0000.txt
│   │   ├── shard_0001.txt
│   │   └── ...
│   ├── manifest.jsonl
│   └── build_metadata.json
├── train/
│   ├── shard_00000.bin
│   ├── shard_00001.bin
│   └── ...
├── val/
│   ├── shard_00000.bin
│   └── ...
├── dataset_metadata.json
├── tokenizer_state.json
└── pipeline_metadata.json

work/
├── logs/
│   ├── build_corpus_20260110_143022.log
│   └── phase2_pipeline_20260110_143022.log
└── dedupe_report.json
```

### Key Files

| File | Description |
|------|-------------|
| `corpus_shards/*.txt` | Cleaned plaintext documents with provenance |
| `manifest.jsonl` | Document-level metadata (source, path, SHA256, bytes) |
| `build_metadata.json` | Complete build parameters and statistics |
| `pipeline_metadata.json` | Full pipeline execution record |
| `dedupe_report.json` | Deduplication statistics |
| `train/*.bin` | Tokenized training shards |
| `val/*.bin` | Tokenized validation shards |

### Shard Format

Each corpus shard contains documents with provenance markers:

```
==== DOC START ====
SOURCE: owasp-cheatsheets | PATH: cheatsheets/SQL_Injection.md
====
# SQL Injection Prevention Cheat Sheet

SQL injection is a code injection technique...

==== DOC END ====
```

## Logging & Audit

### Log Files

Logs are written to `work/logs/` with timestamps:

```
2026-01-10 14:30:22 | INFO     | Build started at 2026-01-10T14:30:22
2026-01-10 14:30:22 | INFO     | Random seed: 42
2026-01-10 14:30:23 | INFO     | Processing: OWASP Cheat Sheets
2026-01-10 14:30:45 | INFO     |   OWASP Cheat Sheets: 98 docs, 1.2 MB, 22.3s
```

### Audit Events

When `core.compliance.audit` is available, the pipeline logs:

- `phase2_pipeline_start` - Pipeline initialization
- `corpus_fetch_complete` - Repository cloning results
- `corpus_build_complete` - Corpus statistics
- `corpus_tokenize_complete` - Tokenization results
- `phase2_pipeline_complete` - Final summary

## Reproducibility

To ensure reproducible builds:

1. **Use the same seed**: `--seed 42`
2. **Pin source versions**: Clone with specific commits if needed
3. **Save metadata**: The `build_metadata.json` captures all parameters
4. **Check logs**: Compare log files between runs

Example metadata for verification:

```json
{
  "build_timestamp": "2026-01-10T14:30:22",
  "seed": 42,
  "min_chars": 400,
  "dedupe_methods": ["exact", "simhash"],
  "statistics": {
    "total_documents": 12847,
    "total_bytes": 45623891,
    "exact_dupes": 234,
    "near_dupes": 156
  }
}
```

## Training

After building the dataset:

```powershell
python train.py `
    --dataset_dir data/phase2_domain `
    --config_file configs/750M_2048.json `
    --model_name phase2_domain `
    --max_iters 50000
```

## Troubleshooting

### Git not found

```
[ERROR] Git not found - please install git and add to PATH
```

**Solution**: Install Git from https://git-scm.com/ and ensure it's in your PATH.

### Clone timeout

Large repositories (MDN, OpenJDK) may timeout. Solutions:
- Increase network timeout
- Clone problematic repos manually
- Use `--sources` to exclude large repos initially

### Out of disk space

The full pipeline requires ~15-20 GB:
- Sources: ~8 GB
- Corpus shards: ~2 GB
- Tokenized data: ~4 GB

Use `--sources` to process a subset, or `--corpus_only` to skip tokenization.

### No sources found

```
[ERROR] No source directories found!
```

**Solution**: Run with `--fetch` to clone repositories, or verify `--sources_dir` path.

## API Reference

### Transformer Classes

```python
from tools.transformers import (
    ManPageConverter,
    RFCXMLConverter,
    HTMLConverter,
    MarkdownRSTConverter,
)

# Convert a man page
converter = ManPageConverter()
content, status = converter.convert("man1/ls.1")

# Convert Markdown
md_converter = MarkdownRSTConverter()
content, status = md_converter.convert("README.md")
```

### Simhash Deduplication

```python
from tools.build_phase2_corpus import compute_simhash, hamming_distance

# Compute fingerprints
fp1 = compute_simhash("This is some text about security")
fp2 = compute_simhash("This is some text about security vulnerabilities")

# Check similarity (lower = more similar)
distance = hamming_distance(fp1, fp2)
is_duplicate = distance <= 4  # threshold
```

## See Also

- [prepare_weighted_dataset.py](../scripts/README.md) - Tokenization pipeline
- [spec_domain_datagrabber.md](spec_domain_datagrabber.md) - Original specification
- [multilingual_de_en.json](../data/sources/multilingual_de_en.json) - Example source config

