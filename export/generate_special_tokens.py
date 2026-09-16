#!/usr/bin/env python3
"""Generate export/special_tokens.json from core.special_tokens (never hand-edit)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import (
    BASE_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_STRINGS,
    get_special_token_ids,
)
from export.paths import MODEL_VOCAB_SIZE, SPECIAL_TOKENS_JSON


def special_token_table() -> dict:
    """Single export-side snapshot of the packer's special tag set."""
    ids = get_special_token_ids()
    tokens = []
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        tokens.append(
            {
                "name": name,
                "string": surface,
                "id": ids[name],
                "special": True,
                "gguf_token_type": "CONTROL",
            }
        )
    ids_sorted = [t["id"] for t in tokens]
    return {
        "source": "core.special_tokens",
        "base_vocab_size": BASE_VOCAB_SIZE,
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "special_count": len(tokens),
        "id_min": min(ids_sorted),
        "id_max": max(ids_sorted),
        "pad_id_start": max(ids_sorted) + 1,
        "pad_id_end": MODEL_VOCAB_SIZE - 1,
        "tokens": tokens,
        "eot": {
            "name": "myPT_eot",
            "string": SPECIAL_TOKEN_STRINGS["myPT_eot"],
            "id": ids["myPT_eot"],
        },
        "toolcall_close": {
            "name": "myPT_toolcall_close",
            "string": SPECIAL_TOKEN_STRINGS["myPT_toolcall_close"],
            "id": ids["myPT_toolcall_close"],
        },
    }


def write_special_tokens_json(path: Path | None = None) -> Path:
    path = path or SPECIAL_TOKENS_JSON
    table = special_token_table()
    # Guard: strings and IDs must stay aligned with the pinned map.
    for tok in table["tokens"]:
        if SPECIAL_TOKEN_STRINGS[tok["name"]] != tok["string"]:
            raise RuntimeError(f"string drift for {tok['name']}")
        if SPECIAL_TOKEN_IDS[tok["name"]] != tok["id"]:
            raise RuntimeError(f"id drift for {tok['name']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    out = write_special_tokens_json()
    print(f"Wrote {out} ({special_token_table()['special_count']} tags from core.special_tokens)")


if __name__ == "__main__":
    main()
