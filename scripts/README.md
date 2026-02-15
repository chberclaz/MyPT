# MyPT Scripts

Utility scripts for model management, dataset preparation, SFT, evaluation, and debugging.

## Directory Structure

```
scripts/
├── build_rag_index.py          # Build RAG embedding index from documents
├── workspace_chat.py           # Interactive workspace agent chat CLI
│
├── data_prep/                  # Dataset downloading, tokenization, mixing
│   ├── append_to_dataset.py
│   ├── convert_opensubtitles.py
│   ├── download_opensource_sft.py
│   ├── fetch_and_prepare_multilingual.py
│   ├── fetch_and_prepare_phase1_5.py
│   ├── fetch_and_prepare_phase2_domain.py
│   ├── mix_general_with_sft.py
│   ├── mix_tokenized_datasets.py
│   ├── prepare_dataset.py
│   └── prepare_weighted_dataset.py
│
├── sft/                        # SFT dataset generation, formatting, validation
│   ├── convert_hf_dataset.py          # NEW: Universal HuggingFace SFT dataset converter
│   ├── prepare_phase1_format_lock.py  # NEW: Automated Phase 1 pipeline (generate+mix+tokenize)
│   ├── generate_format_lock_dataset.py  # Phase 1 format lock Q&A (EN+DE)
│   ├── generate_echo_dataset.py         # Phase 1 echo/repeat instructions
│   ├── generate_operator_dataset.py     # Phase 2 COPY/WRAP/EXTRACT operators
│   ├── generate_rag_chat_sft.py          # NEW: Phase 3 RAG chat episodes (user_context+think+cite)
│   ├── generate_multiturn_sft.py        # NEW: Phase 4 multi-turn conversations
│   ├── generate_agent_sft.py            # Phase 5 tool-calling episodes (EN+DE, think+cite, NO_TOOL)
│   ├── generate_sft_tool_episodes.py    # NEW: Phase 6 multi-step agentic tool chains
│   ├── mix_sft_jsonl.py                 # Mix JSONL files with sampling ratios
│   ├── prepare_chat_sft.py              # Tokenize chat JSONL (Phase 1-4)
│   ├── prepare_tool_sft.py              # Tokenize tool JSONL (Phase 5-6)
│   ├── inspect_sft_dataset.py           # Inspect tokenized dataset
│   ├── validate_sft_dataset.py          # Validate dataset integrity
│   ├── validate_sft_episode_masks.py    # Validate loss masks
│   ├── verify_loss_mask_direction.py    # Verify mask direction
│   ├── verify_mask_alignment.py         # Token-level mask alignment
│   ├── augment_episodes_paraphrase.py   # Paraphrase augmentation
│   ├── diversify_user_messages.py       # User message variation
│   ├── deduplicate_episodes.py          # Episode deduplication
│   ├── deduplicate_by_user_message.py   # Deduplicate by user text
│   ├── analyze_episode_diversity.py     # Diversity analysis
│   └── archive/                         # Obsolete scripts
│       ├── generate_run2_minimal_qa.py  # (legacy) Superseded by Phase 1 format lock
│       └── generate_synthetic_sft.py    # (legacy) Requires external API, not offline
│
├── eval/                       # Evaluation and benchmarking
│   ├── eval_operator.py
│   └── sft_eval_suite.py
│
├── debug/                      # Debugging and diagnostics
│   ├── debug_inference_parity.py
│   ├── debug_train_eval_parity.py
│   ├── diagnose_operator_loss.py
│   ├── diagnose_sft_training.py
│   └── trace_training_step.py
│
├── model/                      # Model inspection, conversion, parameter tools
│   ├── calculate_params.py
│   ├── check_tie_weights.py
│   ├── compare_embeddings.py
│   ├── convert_legacy_checkpoints.py
│   ├── enable_weight_tying.py
│   ├── init_special_embeddings.py
│   ├── inspect_model.py
│   └── smoke_test_arch.py
│
├── translation/                # Multilingual and translation tooling
│   ├── extract_for_translation.py
│   ├── merge_bilingual_episodes.py
│   ├── recombine_translations.py
│   └── translate_deepl.py
│
├── utils/                      # General utilities
│   ├── manage_users.py
│   └── show_configs.py
│
└── unified_build/              # Unified from-scratch training pipeline
    ├── build_unified_dataset.py
    ├── download_nq_triviaqa.py
    ├── download_unified_sources.py
    ├── mix_multi_source.py
    └── pack_training_data.bat
```

## Quick Reference

### Data Preparation
```bash
python scripts/data_prep/prepare_dataset.py --input_files data.txt --out_dir data/my_dataset
python scripts/data_prep/prepare_weighted_dataset.py --config data/sources/config.json --out_dir data/weighted
python scripts/data_prep/mix_tokenized_datasets.py --base_dir data/base --replay_dir data/replay --out_dir data/mixed
```

### SFT Pipeline
```bash
# Phase 1 (automated: generate + mix + tokenize)
python scripts/sft/prepare_phase1_format_lock.py

# Convert HuggingFace datasets to myPT format
python scripts/sft/convert_hf_dataset.py --dataset OpenAssistant/oasst2 --output data/sft_hf/oasst2.jsonl

# Manual: tokenize, inspect, validate
python scripts/sft/prepare_chat_sft.py --input data/episodes.jsonl --output_dir data/sft_ready
python scripts/sft/inspect_sft_dataset.py --dataset_dir data/sft_ready --show_samples 3
python scripts/sft/validate_sft_dataset.py --dataset data/sft_ready
```

**Full SFT documentation:** See `docs/sft/SFT_PIPELINE_GUIDE.md`

### Model Tools
```bash
python scripts/model/inspect_model.py --model_name my_model
python scripts/model/calculate_params.py --config_file configs/base/750M_unified_v1.json
python scripts/utils/show_configs.py
```

### Evaluation
```bash
python scripts/eval/sft_eval_suite.py --model my_model -v
python scripts/eval/eval_operator.py --model my_model -v
```

### Unified Training Pipeline
```bash
python scripts/unified_build/build_unified_dataset.py
```

## See Also

- **Main scripts** (in root): `train.py`, `generate.py`
- **Documentation**: See `docs/` folder
