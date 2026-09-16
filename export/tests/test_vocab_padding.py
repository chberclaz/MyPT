"""Vocab length must equal embedding rows; unused pads sit above specials."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import get_special_token_ids
from export.paths import HF_TOKENIZER_DIR, MODEL_VOCAB_SIZE


@pytest.mark.skipif(not (HF_TOKENIZER_DIR / "vocab.json").exists(), reason="run tiktoken_to_hf.py first")
def test_vocab_padded_to_embedding_rows():
    vocab = json.loads((HF_TOKENIZER_DIR / "vocab.json").read_text(encoding="utf-8"))
    assert len(vocab) == MODEL_VOCAB_SIZE
    ids = get_special_token_ids()
    max_special = max(ids.values())
    for pad_id in range(max_special + 1, MODEL_VOCAB_SIZE):
        assert f"<|unused_{pad_id}|>" in vocab
        assert vocab[f"<|unused_{pad_id}|>"] == pad_id
    # Specials sit above base gpt2, below unused.
    assert min(ids.values()) == 50257
    assert max_special == 50275
    assert max_special < min(range(max_special + 1, MODEL_VOCAB_SIZE) or [MODEL_VOCAB_SIZE])
