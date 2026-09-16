"""tiktoken (MyPT) vs reconstructed HF tokenizer — includes every myPT special tag."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from export.mypt_tok import make_mypt_tokenizer
from export.paths import FIXTURES_DIR, HF_TOKENIZER_DIR, SPECIAL_TOKENS_JSON


def _fixture_lines(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("##"):
            continue
        out.append(line)
    return out


def test_special_tokens_json_matches_packer():
    from export.generate_special_tokens import write_special_tokens_json

    write_special_tokens_json()
    table = json.loads(SPECIAL_TOKENS_JSON.read_text(encoding="utf-8"))
    ids = get_special_token_ids()
    assert table["source"] == "core.special_tokens"
    assert table["special_count"] == len(SPECIAL_TOKEN_STRINGS)
    names = {t["name"] for t in table["tokens"]}
    assert names == set(SPECIAL_TOKEN_STRINGS)
    for tok in table["tokens"]:
        assert tok["string"] == SPECIAL_TOKEN_STRINGS[tok["name"]]
        assert tok["id"] == ids[tok["name"]]
        assert tok["special"] is True
        assert tok["gguf_token_type"] == "CONTROL"


def test_every_special_tag_is_single_tiktoken_id():
    tok = make_mypt_tokenizer()
    ids = get_special_token_ids()
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        got = tok.encode(surface)
        assert got == [ids[name]], f"{surface!r} -> {got}, want [{ids[name]}]"
        assert tok.decode(got) == surface


@pytest.mark.skipif(not (HF_TOKENIZER_DIR / "tokenizer.json").exists(), reason="run tiktoken_to_hf.py first")
def test_tiktoken_hf_parity_fixtures():
    from transformers import GPT2TokenizerFast

    mypt = make_mypt_tokenizer()
    hf = GPT2TokenizerFast.from_pretrained(str(HF_TOKENIZER_DIR), add_prefix_space=False)
    ids = get_special_token_ids()
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        assert hf.convert_tokens_to_ids(surface) == ids[name]
        assert hf.encode(surface, add_special_tokens=False) == [ids[name]]

    texts = _fixture_lines(FIXTURES_DIR / "tokenizer_parity.txt")
    texts += _fixture_lines(FIXTURES_DIR / "nonascii_stress.txt")
    for text in texts:
        a = mypt.encode(text)
        b = hf.encode(text, add_special_tokens=False)
        assert a == b, f"mismatch on {text!r}\n tiktoken={a}\n hf={b}"
        assert mypt.decode(a) == text
        assert hf.decode(b, skip_special_tokens=False) == text
