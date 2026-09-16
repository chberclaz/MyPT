"""Phase split for system prompts: chat vs agentic, no copied strings."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.system_prompts import (
    AGENTIC_STANDARD_PROMPT,
    CHAT_SYSTEM_PROMPT,
    CHAT_SYSTEM_PROMPTS,
    CONVERSATION_SYSTEM_PROMPT,
    DEFAULT_AGENTIC_PROMPT,
    SYSTEM_PROMPT_PRESETS,
    resolve_system_prompt,
)


def _has_tool_signal(text: str) -> bool:
    return "workspace." in text or "myPT_toolcall" in text


def test_chat_prompts_have_no_tools():
    assert not _has_tool_signal(CHAT_SYSTEM_PROMPT)
    assert not _has_tool_signal(CONVERSATION_SYSTEM_PROMPT)
    for variant in CHAT_SYSTEM_PROMPTS:
        assert not _has_tool_signal(variant), variant


def test_agentic_prompt_has_tools_and_differs_from_chat():
    assert "workspace." in AGENTIC_STANDARD_PROMPT
    assert "myPT_toolcall" in AGENTIC_STANDARD_PROMPT
    assert CHAT_SYSTEM_PROMPT != AGENTIC_STANDARD_PROMPT
    assert DEFAULT_AGENTIC_PROMPT == AGENTIC_STANDARD_PROMPT


def test_presets_resolve():
    assert resolve_system_prompt() == CONVERSATION_SYSTEM_PROMPT
    assert resolve_system_prompt(preset="conversation") == CONVERSATION_SYSTEM_PROMPT
    assert resolve_system_prompt(preset="chat") == CHAT_SYSTEM_PROMPT
    assert resolve_system_prompt(preset="agentic") == AGENTIC_STANDARD_PROMPT
    assert resolve_system_prompt(preset="chat", override="custom") == "custom"
    assert SYSTEM_PROMPT_PRESETS["chat"] is CHAT_SYSTEM_PROMPT


def test_prepare_chat_sft_exposes_preset():
    from scripts.sft import prepare_chat_sft

    src = Path(prepare_chat_sft.__file__).read_text(encoding="utf-8")
    assert "--system_prompt_preset" in src
    assert "choices=[\"conversation\", \"chat\", \"agentic\"]" in src or "conversation" in src


def test_eval_suite_exposes_preset():
    src = (PROJECT_ROOT / "scripts" / "eval" / "sft_eval_suite.py").read_text(encoding="utf-8")
    assert "--system_prompt_preset" in src
    assert "resolve_system_prompt" in src
    assert "rebuild_standard_eval_prompts" in src


def test_eval_rebuild_uses_chat_preset():
    from scripts.eval import sft_eval_suite as suite

    suite.SYSTEM_PROMPT = suite.resolve_system_prompt(preset="chat")
    suite.rebuild_standard_eval_prompts()
    _, prompt, _ = suite.ECHO_PROMPTS[0]
    assert CHAT_SYSTEM_PROMPT in prompt
    assert "workspace.search" not in prompt
    suite.SYSTEM_PROMPT = suite.resolve_system_prompt(preset="conversation")
    suite.rebuild_standard_eval_prompts()

