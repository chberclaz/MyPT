# Unified from-scratch dataset build pipeline (Phase 1 pre-training)
#
# Scripts in this folder:
#   build_unified_dataset.py   - Main orchestrator (download, holdback, tokenize, mix)
#   download_unified_sources.py - Downloads FineWeb-Edu + peS2o from HuggingFace
#   download_nq_triviaqa.py    - Downloads NQ + TriviaQA for retrieval head training
#   mix_multi_source.py        - Mixes N tokenized sources at specified proportions
