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
│   ├── analyze_episode_diversity.py
│   ├── augment_episodes_paraphrase.py
│   ├── deduplicate_by_user_message.py
│   ├── deduplicate_episodes.py
│   ├── diversify_user_messages.py
│   ├── generate_agent_sft.py
│   ├── generate_echo_dataset.py
│   ├── generate_format_lock_dataset.py
│   ├── generate_operator_dataset.py
│   ├── generate_run2_minimal_qa.py
│   ├── generate_synthetic_sft.py
│   ├── inspect_sft_dataset.py
│   ├── mix_sft_jsonl.py
│   ├── prepare_chat_sft.py
│   ├── prepare_tool_sft.py
│   ├── validate_sft_dataset.py
│   ├── validate_sft_episode_masks.py
│   ├── verify_loss_mask_direction.py
│   └── verify_mask_alignment.py
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
python scripts/sft/prepare_chat_sft.py --input data/episodes.jsonl --output_dir data/sft_ready
python scripts/sft/inspect_sft_dataset.py --dataset_dir data/sft_ready --show_samples 3
python scripts/sft/validate_sft_dataset.py --dataset data/sft_ready
```

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
