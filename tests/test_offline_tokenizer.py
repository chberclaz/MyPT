"""GPT-2 encoding must load from bundled ranks with no network."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.gpt2_encoding import GPT2_N_VOCAB, GPT2_TIKTOKEN_SHA256, get_gpt2_encoding
from core.tokenizer import Tokenizer


def test_bundled_ranks_hash_is_openai_r50k_base():
    # Same digest tiktoken publishes for r50k_base.tiktoken.
    assert GPT2_TIKTOKEN_SHA256 == "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"


def test_gpt2_encoding_known_ids():
    enc = get_gpt2_encoding()
    assert enc.n_vocab == GPT2_N_VOCAB
    assert enc.encode_ordinary("Hello!") == [15496, 0]
    assert enc.encode_ordinary("hello world") == [31373, 995]
    assert enc.decode([15496, 0]) == "Hello!"


def test_tokenizer_and_tiktoken_registry_stay_offline(monkeypatch):
    import socket

    import requests
    import tiktoken

    def _no_net(*_a, **_k):
        raise AssertionError("tokenizer attempted a network call")

    monkeypatch.setattr(socket, "getaddrinfo", _no_net)
    monkeypatch.setattr(requests, "get", _no_net)

    get_gpt2_encoding.cache_clear()
    enc = get_gpt2_encoding()
    assert enc.encode_ordinary("Hello!") == [15496, 0]

    tok = Tokenizer(SimpleNamespace(vocab_size=50304), "gpt2")
    assert tok.encode("Hello!") == [15496, 0]
    assert tok.decode([15496, 0]) == "Hello!"

    # Leftover get_encoding("gpt2") must hit the in-process registry, not Azure.
    assert tiktoken.get_encoding("gpt2") is enc


def test_missing_ranks_file_does_not_download(tmp_path, monkeypatch):
    import core.gpt2_encoding as ge

    monkeypatch.setattr(ge, "_RANKS_PATH", tmp_path / "missing.tiktoken")
    ge.get_gpt2_encoding.cache_clear()
    with pytest.raises(FileNotFoundError, match="refuses to download"):
        ge.get_gpt2_encoding()
    ge.get_gpt2_encoding.cache_clear()
