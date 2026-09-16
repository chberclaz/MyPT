"""EOG halt and tool-loop tests (need Q4/F16 GGUF + llama-server)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS
from export.paths import CHECKPOINT_GOLD, llama_server


def _gguf():
    for p in (
        CHECKPOINT_GOLD / "mypt-q4_k_m.gguf",
        CHECKPOINT_GOLD / "mypt-f16.gguf",
    ):
        if p.exists():
            return p
    return None


def _have_server() -> bool:
    try:
        llama_server()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(_gguf() is None, reason="GGUF not exported yet")
@pytest.mark.skipif(not _have_server(), reason="llama-server binary missing")
def test_eog_halt_on_toolcall_close():
    from core.inference.gguf_backend import GGUFModel

    model = GGUFModel(_gguf())
    try:
        prompt = (
            f"{SPECIAL_TOKEN_STRINGS['myPT_system_open']}You are MyPT.{SPECIAL_TOKEN_STRINGS['myPT_system_close']}\n"
            f"{SPECIAL_TOKEN_STRINGS['myPT_user_open']}Search workspace.{SPECIAL_TOKEN_STRINGS['myPT_user_close']}\n"
            f"{SPECIAL_TOKEN_STRINGS['myPT_assistant_open']}"
        )
        out = model.generate(prompt, max_new_tokens=128, temperature=0.0)
        gen = out[len(prompt) :] if out.startswith(prompt) else out
        if SPECIAL_TOKEN_STRINGS["myPT_toolcall_close"] in gen:
            after = gen.split(SPECIAL_TOKEN_STRINGS["myPT_toolcall_close"], 1)[1]
            assert SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"] not in after
    finally:
        model.close()


def test_error_toolresult_shape_in_training_contract():
    from core.agent.parsing import render_toolresult

    block = render_toolresult({"error": "unknown_tool", "name": "foo.bar"})
    assert SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"] in block
    assert SPECIAL_TOKEN_STRINGS["myPT_toolresult_close"] in block
