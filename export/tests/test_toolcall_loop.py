from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS
from export.paths import CHECKPOINT_GOLD


@pytest.mark.skipif(
    not (CHECKPOINT_GOLD / "mypt-q4_k_m.gguf").exists()
    and not (CHECKPOINT_GOLD / "mypt-f16.gguf").exists(),
    reason="GGUF not exported",
)
def test_toolcall_loop_unknown_tool_error_contract():
    from core.agent.parsing import render_toolresult

    rendered = render_toolresult({"error": "unknown_tool", "name": "not.a.tool"})
    assert rendered.startswith(SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"])
    assert rendered.endswith(SPECIAL_TOKEN_STRINGS["myPT_toolresult_close"])
    inner = json.loads(
        rendered[
            len(SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"]) : -len(
                SPECIAL_TOKEN_STRINGS["myPT_toolresult_close"]
            )
        ]
    )
    assert inner["error"] == "unknown_tool"
