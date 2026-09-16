"""In-process MyPT tokenizer (tiktoken gpt2 + specials from core.special_tokens)."""
from __future__ import annotations

from types import SimpleNamespace

from core.special_tokens import get_special_token_ids
from core.tokenizer import Tokenizer
from export.paths import MODEL_VOCAB_SIZE


def make_mypt_tokenizer() -> Tokenizer:
    cfg = SimpleNamespace(vocab_size=MODEL_VOCAB_SIZE)
    tok = Tokenizer(cfg, "gpt2")
    # Silence is already printed by Tokenizer; IDs must match packer.
    ids = get_special_token_ids()
    for name, tid in ids.items():
        if tok.special_tokens.get(name) != tid:
            raise RuntimeError(f"runtime special id drift: {name}")
    return tok
