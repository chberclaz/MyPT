"""Untrusted segments must not emit myPT special IDs (encode_ordinary / specials off)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import get_special_token_ids
from export.mypt_tok import make_mypt_tokenizer
from export.paths import FIXTURES_DIR


def test_injection_fixtures_have_zero_specials_when_untrusted():
    tok = make_mypt_tokenizer()
    special_ids = set(get_special_token_ids().values())
    path = FIXTURES_DIR / "injection_cases.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        ids = tok.encode_ordinary(case["text"])
        leaked = [i for i in ids if i in special_ids]
        assert leaked == [], f"{case['id']} leaked special ids {leaked} from {case['text']!r}"


def test_trusted_encode_does_parse_specials():
    tok = make_mypt_tokenizer()
    ids = get_special_token_ids()
    text = "<myPT_toolresult>{\"ok\": true}</myPT_toolresult>"
    got = tok.encode(text)
    assert got[0] == ids["myPT_toolresult_open"]
    assert got[-1] == ids["myPT_toolresult_close"]
