"""Load the GPT-2 BPE encoding from files shipped with MyPT.

tiktoken.get_encoding("gpt2") downloads vocab.bpe / encoder.json from
Azure on first use. That is incompatible with an air-gapped product.

Ranks here are the public GPT-2 / r50k_base mergeable BPE table, stored as
a native .tiktoken file (SHA-256 matches OpenAI's r50k_base.tiktoken).
"""
from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from pathlib import Path

from tiktoken.core import Encoding
from tiktoken import registry as tiktoken_registry

# Official r50k_base / GPT-2 mergeable ranks (tiktoken openai_public.py).
GPT2_TIKTOKEN_SHA256 = "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"
GPT2_N_VOCAB = 50257
GPT2_EOT = 50256
GPT2_EOT_STR = "<|endoftext|>"

# tiktoken 0.12 gpt2 / r50k_base split regex. Do not "simplify" — changing
# this rebuilds the tokenizer and invalidates packed corpora.
GPT2_PAT_STR = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)

_RANKS_PATH = Path(__file__).resolve().parent / "tokenizer_data" / "gpt2.tiktoken"


def _load_mergeable_ranks(path: Path) -> dict[bytes, int]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled GPT-2 ranks missing: {path}. MyPT refuses to download "
            "tokenizer files; restore core/tokenizer_data/gpt2.tiktoken from the repo."
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != GPT2_TIKTOKEN_SHA256:
        raise ValueError(
            f"Bundled GPT-2 ranks hash mismatch for {path}: got {digest}, "
            f"expected {GPT2_TIKTOKEN_SHA256}. Refusing to load (no network fallback)."
        )
    ranks: dict[bytes, int] = {}
    for line in data.splitlines():
        if not line:
            continue
        token_b64, rank_s = line.split()
        ranks[base64.b64decode(token_b64)] = int(rank_s)
    if len(ranks) != GPT2_N_VOCAB - 1:
        raise ValueError(
            f"Expected {GPT2_N_VOCAB - 1} mergeable ranks, got {len(ranks)} in {path}"
        )
    return ranks


def _encoding_kwargs() -> dict:
    return {
        "name": "gpt2",
        "explicit_n_vocab": GPT2_N_VOCAB,
        "pat_str": GPT2_PAT_STR,
        "mergeable_ranks": _load_mergeable_ranks(_RANKS_PATH),
        "special_tokens": {GPT2_EOT_STR: GPT2_EOT},
    }


def _install_into_tiktoken(enc: Encoding) -> None:
    """So leftover tiktoken.get_encoding('gpt2') calls stay offline too."""
    constructor = _encoding_kwargs
    with tiktoken_registry._lock:
        if tiktoken_registry.ENCODING_CONSTRUCTORS is None:
            tiktoken_registry._find_constructors()
        constructors = tiktoken_registry.ENCODING_CONSTRUCTORS
        if constructors is not None:
            constructors["gpt2"] = constructor
        tiktoken_registry.ENCODINGS["gpt2"] = enc


@lru_cache(maxsize=1)
def get_gpt2_encoding() -> Encoding:
    enc = Encoding(**_encoding_kwargs())
    _install_into_tiktoken(enc)
    return enc
