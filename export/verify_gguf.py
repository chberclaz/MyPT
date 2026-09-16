#!/usr/bin/env python3
"""Assert GGUF vocab: every core.special_tokens tag is CONTROL/USER_DEFINED at the packer ID."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from export.generate_special_tokens import write_special_tokens_json
from export.paths import CHECKPOINT_GOLD, MODEL_VOCAB_SIZE, SPECIAL_TOKENS_JSON

CONTROL, USER_DEFINED, UNUSED, NORMAL = 3, 4, 5, 1


def _strings(reader, key: str):
    raw = reader.fields[key].contents()
    out = []
    for t in raw:
        if isinstance(t, (bytes, bytearray)):
            out.append(t.decode("utf-8"))
        else:
            out.append(str(t) if not isinstance(t, str) else t)
    return out


def verify(gguf_path: Path) -> None:
    from gguf import GGUFReader

    write_special_tokens_json()
    table = json.loads(SPECIAL_TOKENS_JSON.read_text(encoding="utf-8"))
    reader = GGUFReader(str(gguf_path))
    tokens = _strings(reader, "tokenizer.ggml.tokens")
    types = [int(x) for x in reader.fields["tokenizer.ggml.token_type"].contents()]
    if len(tokens) != MODEL_VOCAB_SIZE:
        raise SystemExit(f"vocab {len(tokens)} != embedding rows {MODEL_VOCAB_SIZE}")
    ids = get_special_token_ids()
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        tid = ids[name]
        if tokens[tid] != surface:
            raise SystemExit(f"ID {tid} is {tokens[tid]!r}, packer wants {surface!r} ({name})")
        if types[tid] not in (CONTROL, USER_DEFINED):
            raise SystemExit(f"{surface} token_type={types[tid]} (must be CONTROL/USER_DEFINED, not NORMAL={NORMAL})")
        print(f"OK  {tid:5d}  type={types[tid]}  {surface}")
    pad_types = {types[i] for i in range(max(ids.values()) + 1, MODEL_VOCAB_SIZE)}
    print(f"pad token_types={sorted(pad_types)} (want UNUSED={UNUSED}; CONTROL also ok if unused strings never appear)")
    print(f"verify_gguf OK  tags={table['special_count']}  vocab={len(tokens)}  file={gguf_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("gguf", nargs="?", type=Path, default=CHECKPOINT_GOLD / "mypt-f16.gguf")
    args = p.parse_args()
    verify(args.gguf)


if __name__ == "__main__":
    main()
