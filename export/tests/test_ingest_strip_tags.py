"""Ingest strips every packer tag from untrusted documents."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS, strip_special_tag_strings


def test_strip_all_packer_tags():
    blob = "keep " + "".join(SPECIAL_TOKEN_STRINGS.values()) + " keep"
    out = strip_special_tag_strings(blob)
    for s in SPECIAL_TOKEN_STRINGS.values():
        assert s not in out
    assert "keep" in out
    assert all(s not in out for s in SPECIAL_TOKEN_STRINGS.values())
