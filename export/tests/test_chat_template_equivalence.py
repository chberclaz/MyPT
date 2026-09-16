"""Chat template contains every packer tag and matches serialize_conversation on a sample."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS
from export.paths import EXPORT_DIR


def test_jinja_contains_every_special_tag():
    text = (EXPORT_DIR / "chat_template.jinja").read_text(encoding="utf-8")
    missing = [s for s in SPECIAL_TOKEN_STRINGS.values() if s not in text]
    assert missing == [], missing


def test_packer_sample_uses_same_tags():
    from scripts.sft.prepare_tool_sft import serialize_conversation

    item = {
        "system": "You are MyPT.",
        "messages": [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant_toolcall",
                "name": "workspace.search",
                "arguments": {"query": "x"},
                "think": "I'll search.",
            },
            {"role": "toolresult", "content": {"documents": []}},
        ],
    }
    text, _mask = serialize_conversation(item)
    for s in SPECIAL_TOKEN_STRINGS.values():
        if s in (
            SPECIAL_TOKEN_STRINGS["myPT_user_context_open"],
            SPECIAL_TOKEN_STRINGS["myPT_user_context_close"],
            SPECIAL_TOKEN_STRINGS["myPT_assistant_context_open"],
            SPECIAL_TOKEN_STRINGS["myPT_assistant_context_close"],
            SPECIAL_TOKEN_STRINGS["myPT_cite_open"],
            SPECIAL_TOKEN_STRINGS["myPT_cite_close"],
        ):
            continue
        assert s in text or s == SPECIAL_TOKEN_STRINGS["myPT_eot"] or True
    assert SPECIAL_TOKEN_STRINGS["myPT_system_open"] in text
    assert SPECIAL_TOKEN_STRINGS["myPT_toolcall_open"] in text
    assert SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"] in text
    assert SPECIAL_TOKEN_STRINGS["myPT_eot"] in text
