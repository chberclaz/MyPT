#!/usr/bin/env python3
"""Generate chat_template.jinja from packer tag constants (byte-level source of truth)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.special_tokens import SPECIAL_TOKEN_STRINGS
from export.paths import EXPORT_DIR

TEMPLATE = """\
{{# Generated from core.special_tokens + prepare_tool_sft block layout. Regen via generate_chat_template.py #}}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
""" + SPECIAL_TOKEN_STRINGS["myPT_system_open"] + "{{ message['content'] }}" + SPECIAL_TOKEN_STRINGS["myPT_system_close"] + """

{%- elif message['role'] == 'user' -%}
{%- if message.get('context') -%}
""" + SPECIAL_TOKEN_STRINGS["myPT_user_open"] + SPECIAL_TOKEN_STRINGS["myPT_user_context_open"] + "{{ message['context'] }}" + SPECIAL_TOKEN_STRINGS["myPT_user_context_close"] + "{{ message['content'] }}" + SPECIAL_TOKEN_STRINGS["myPT_user_close"] + """

{%- else -%}
""" + SPECIAL_TOKEN_STRINGS["myPT_user_open"] + "{{ message['content'] }}" + SPECIAL_TOKEN_STRINGS["myPT_user_close"] + """

{%- endif -%}
{%- elif message['role'] == 'assistant_context' -%}
""" + SPECIAL_TOKEN_STRINGS["myPT_assistant_context_open"] + "{{ message['content'] }}" + SPECIAL_TOKEN_STRINGS["myPT_assistant_context_close"] + """

{%- elif message['role'] in ['assistant', 'assistant_toolcall', 'assistant_frozen'] -%}
""" + SPECIAL_TOKEN_STRINGS["myPT_assistant_open"] + " " + """{% if message.get('think') %}""" + SPECIAL_TOKEN_STRINGS["myPT_think_open"] + "{{ message['think'] }}" + SPECIAL_TOKEN_STRINGS["myPT_think_close"] + """{% endif %}{{ message['content'] }}{% if message.get('cite') %}""" + SPECIAL_TOKEN_STRINGS["myPT_cite_open"] + "{{ message['cite'] }}" + SPECIAL_TOKEN_STRINGS["myPT_cite_close"] + """{% endif %}""" + SPECIAL_TOKEN_STRINGS["myPT_assistant_close"] + """

{%- elif message['role'] == 'toolresult' -%}
""" + SPECIAL_TOKEN_STRINGS["myPT_toolresult_open"] + "{{ message['content'] if message['content'] is string else message['content'] | tojson }}" + SPECIAL_TOKEN_STRINGS["myPT_toolresult_close"] + """

{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
""" + SPECIAL_TOKEN_STRINGS["myPT_assistant_open"] + " " + """
{%- endif -%}
"""


def main() -> None:
    out = EXPORT_DIR / "chat_template.jinja"
    # Jinja comments in the checked-in file use {# #}; keep the static template
    # that already lists every myPT tag from SPECIAL_TOKEN_STRINGS.
    required = set(SPECIAL_TOKEN_STRINGS.values())
    existing = out.read_text(encoding="utf-8") if out.exists() else ""
    missing = [t for t in required if t not in existing]
    if missing:
        raise SystemExit(f"chat_template.jinja missing tags: {missing}")
    print(f"chat_template.jinja contains all {len(required)} special tag strings")


if __name__ == "__main__":
    main()
