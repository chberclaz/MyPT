"""workspace_ask result shape (no model load)."""
from __future__ import annotations

from core.agent.workspace_ask import _conversation_generate
from core.special_tokens import SPECIAL_TOKEN_STRINGS


class _Echo:
    def generate(self, prompt, max_new_tokens, **kwargs):
        return prompt + "hello" + SPECIAL_TOKEN_STRINGS["myPT_assistant_close"]


def test_conversation_generate_strips_close_tag():
    hist = [{"role": "user", "content": "Hi"}]
    out = _conversation_generate(_Echo(), hist, "You are MyPT.", 16)
    assert out == "hello"
    assert SPECIAL_TOKEN_STRINGS["myPT_assistant_close"] not in out
