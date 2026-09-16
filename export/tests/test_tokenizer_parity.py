"""HF vs llama.cpp tokenizer IDs (needs GGUF). Specials from core.special_tokens."""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids
from export.mypt_tok import make_mypt_tokenizer
from export.paths import CHECKPOINT_GOLD, HF_TOKENIZER_DIR, llama_tokenize


@pytest.mark.skipif(not (HF_TOKENIZER_DIR / "tokenizer.json").exists(), reason="run tiktoken_to_hf.py")
def test_hf_specials_present():
    from transformers import GPT2TokenizerFast

    hf = GPT2TokenizerFast.from_pretrained(str(HF_TOKENIZER_DIR), add_prefix_space=False)
    ids = get_special_token_ids()
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        assert hf.convert_tokens_to_ids(surface) == ids[name]


def _gguf() -> Path | None:
    for p in (
        CHECKPOINT_GOLD / "mypt-q4_k_m.gguf",
        CHECKPOINT_GOLD / "mypt-f16.gguf",
    ):
        if p.exists():
            return p
    return None


def _tokenize_ids(gguf: Path, text: str, parse_special: bool) -> list[int]:
    exe = llama_tokenize()
    if exe is None:
        pytest.skip("llama-tokenize missing")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False, newline="\n") as fh:
        fh.write(text)
        path = fh.name
    cmd = [
        str(exe),
        "-m",
        str(gguf),
        "--no-bos",
        "--no-escape",
        "--ids",
        "-f",
        path,
    ]
    if not parse_special:
        cmd.append("--no-parse-special")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    Path(path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    last = ""
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            last = line
    if not last:
        raise RuntimeError(f"no id list in llama-tokenize output:\n{proc.stdout}\n{proc.stderr}")
    return list(ast.literal_eval(last))


@pytest.mark.skipif(_gguf() is None or llama_tokenize() is None, reason="GGUF or llama-tokenize missing")
def test_llamacpp_specials_are_packer_ids():
    gguf = _gguf()
    ids = get_special_token_ids()
    for name, surface in SPECIAL_TOKEN_STRINGS.items():
        got = _tokenize_ids(gguf, surface, parse_special=True)
        assert got == [ids[name]], f"{surface!r} -> {got}, want [{ids[name]}]"


@pytest.mark.skipif(_gguf() is None or llama_tokenize() is None, reason="GGUF or llama-tokenize missing")
def test_llamacpp_untrusted_does_not_emit_special_ids():
    gguf = _gguf()
    special = set(get_special_token_ids().values())
    blob = "keep " + "".join(SPECIAL_TOKEN_STRINGS.values()) + " keep"
    got = _tokenize_ids(gguf, blob, parse_special=False)
    leaked = [i for i in got if i in special]
    assert leaked == []


@pytest.mark.skipif(_gguf() is None or llama_tokenize() is None, reason="GGUF or llama-tokenize missing")
def test_tiktoken_equals_llamacpp_on_specials_and_ascii():
    gguf = _gguf()
    tok = make_mypt_tokenizer()
    samples = list(SPECIAL_TOKEN_STRINGS.values()) + [
        "hello world",
        "Straße-Größe-Bär",
        "café naïve 12,50 €",
    ]
    for text in samples:
        a = tok.encode(text)
        b = _tokenize_ids(gguf, text, parse_special=True)
        assert a == b, f"mismatch on {text!r}\n tiktoken={a}\n llama.cpp={b}"
