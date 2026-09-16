"""GGUF special token types and IDs (requires converted F16)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from export.paths import ARTIFACTS_DIR, CHECKPOINT_GOLD, SPECIAL_TOKENS_JSON

# gguf TokenType: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6
CONTROL = 3
USER_DEFINED = 4
UNUSED = 5


def _gguf_candidates() -> list[Path]:
    out = []
    for folder in (ARTIFACTS_DIR, CHECKPOINT_GOLD):
        if folder.exists():
            out.extend(sorted(folder.glob("*.gguf")))
    return out


@pytest.mark.skipif(not _gguf_candidates(), reason="no GGUF artifact yet")
def test_specials_are_control_and_ids_match_packer():
    from gguf import GGUFReader

    path = _gguf_candidates()[0]
    reader = GGUFReader(str(path))
    tokens = reader.fields["tokenizer.ggml.tokens"].contents()
    tokens = [t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t) for t in tokens]
    types = [int(x) for x in reader.fields["tokenizer.ggml.token_type"].contents()]
    ids = get_special_token_ids()
    table = json.loads(SPECIAL_TOKENS_JSON.read_text(encoding="utf-8"))
    assert len(tokens) == table["model_vocab_size"]
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        tid = ids[name]
        assert tokens[tid] == surface, f"GGUF[{tid}]={tokens[tid]!r} want {surface!r}"
        assert int(types[tid]) in (CONTROL, USER_DEFINED), (
            f"{surface} type={types[tid]} must be CONTROL/USER_DEFINED"
        )
        assert int(types[tid]) != 1
