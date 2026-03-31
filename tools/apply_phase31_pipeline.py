#!/usr/bin/env python3
"""
Idempotent maintainer tool: apply Phase 3.1 restart pipeline patches documented in
docs/sft/SFT_PIPELINE_GUIDE.md.

Re-running when already applied is a no-op (sentinel checks).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONVERT_PATH = ROOT / "scripts" / "sft" / "convert_hf_dataset.py"
BUILD_PATH = ROOT / "scripts" / "sft" / "build_phase3_dataset.py"
GEN_JSON_PATH = ROOT / "scripts" / "sft" / "generate_phase3_json_sft.py"
CORRECTIVE_CFG = ROOT / "configs" / "sft" / "phase3_1_corrective.json"
RESTART_CFG = ROOT / "configs" / "sft" / "phase3_1_restart.json"
PS1_PATH = ROOT / "scripts" / "sft" / "run_phase31_prepare.ps1"

PHASE31_CONVERT_INSERT = r'''

def _strip_leading_think_block(text: str) -> str:
    """Remove a single leading fenced/backtick reasoning block if present."""
    return re.sub(r"(?is)^\s*`\s*.*?\s*`", "", text, count=1)


def _compact_json_if_parsable(text: str) -> str:
    """Extract JSON from ```json fences or fall back after think-strip; return compact JSON literal."""
    t = (text or "").strip()
    t = _strip_leading_think_block(t)
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    blob = m.group(1).strip() if m else t
    try:
        obj = json.loads(blob)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return t.strip()


def parse_no_robots(dataset, languages: List[str], max_examples: int, add_think: bool) -> List[dict]:
    """Parse HuggingFaceH4/no_robots (messages format, train+test)."""
    episodes: List[dict] = []
    for split_name in ("train", "test"):
        if split_name not in dataset:
            continue
        for row in dataset[split_name]:
            if len(episodes) >= max_examples:
                return episodes
            messages_raw = row.get("messages", [])
            if not messages_raw:
                continue
            messages: List[Dict[str, Any]] = []
            system_text = CONVERSATION_SYSTEM_PROMPT
            first_user_text = ""
            for msg in messages_raw:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role == "system":
                    system_text = content
                elif role == "user":
                    if not first_user_text:
                        first_user_text = content
                    messages.append(_build_user_message(content, raw_msg=msg, raw_row=row))
                elif role == "assistant":
                    messages.append(
                        _build_assistant_message(content, raw_msg=msg, raw_row=row, add_think=add_think)
                    )
            if len(messages) < 2:
                continue
            lang = _detect_language(first_user_text or messages[0].get("content", ""))
            if languages and lang not in languages:
                continue
            episodes.append(
                {
                    "system": system_text,
                    "messages": messages,
                    "language": lang,
                    "source": "no_robots",
                }
            )
    return episodes


def _coerce_ast_list(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw  # type: ignore[return-value]
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            if isinstance(v, list):
                return v  # type: ignore[return-value]
        except (ValueError, SyntaxError, TypeError):
            return None
    return None


def parse_amans_json_structuring(
    dataset, languages: List[str], max_examples: int, add_think: bool
) -> List[dict]:
    """Parse AmanPriyanshu/reasoning-sft-JSON-structuring-and-correcting (input list + response)."""
    episodes: List[dict] = []
    for split_name in ("train", "test"):
        if split_name not in dataset:
            continue
        for row in dataset[split_name]:
            if len(episodes) >= max_examples:
                return episodes
            messages_raw = _coerce_ast_list(row.get("input"))
            if not messages_raw:
                continue
            resp_raw = row.get("response", "")
            if not isinstance(resp_raw, str) or not resp_raw.strip():
                continue
            assistant_text = _compact_json_if_parsable(resp_raw)
            try:
                json.loads(assistant_text)
            except Exception:
                continue

            messages: List[Dict[str, Any]] = []
            system_text = CONVERSATION_SYSTEM_PROMPT
            first_user_text = ""
            for msg in messages_raw:
                role = str(msg.get("role", "")).strip().lower()
                content = msg.get("content", "")
                if isinstance(content, str):
                    content = content.strip()
                else:
                    content = str(content).strip()
                if role == "system":
                    if content:
                        system_text = content
                    continue
                if role == "user":
                    if content:
                        if not first_user_text:
                            first_user_text = content
                        messages.append(_build_user_message(content, raw_msg=msg, raw_row=row))
                elif role == "assistant":
                    if content:
                        messages.append(
                            _build_assistant_message(content, raw_msg=msg, raw_row=row, add_think=add_think)
                        )
                elif role == "tool":
                    if content:
                        messages.append(
                            {
                                "role": "toolresult",
                                "name": str(msg.get("name", "tool")),
                                "content": content,
                            }
                        )

            if messages and messages[-1].get("role") == "assistant":
                messages.pop()
            if not messages:
                continue
            messages.append(_build_assistant_message(assistant_text, raw_row=row, add_think=False))

            if len(messages) < 2:
                continue
            lang = _detect_language(first_user_text or "")
            if languages and lang not in languages:
                continue
            episodes.append(
                {
                    "system": system_text,
                    "messages": messages,
                    "language": lang,
                    "source": "amans_json_structuring",
                }
            )
    return episodes


# =============================================================================
'''

RUN_PS1_CONTENT = r'''# Phase 3.1 restart: HF converts + synthetic JSON/control + mixed build + prepare_chat_sft
# Requires: py -3, datasets, repo root as cwd (or set MYPT_ROOT).

$ErrorActionPreference = "Stop"
$ROOT = if ($env:MYPT_ROOT) { $env:MYPT_ROOT } else { Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }
Set-Location $ROOT

$py = "py"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { $py = "python" }

$dataHF = "data/sft_hf"
$dataMid = "data/sft_phase3_intermediate"
$outChat = "data/sft_phase3_1_restart_chat"

New-Item -ItemType Directory -Force -Path $dataHF,$dataMid | Out-Null

& $py -3 scripts/sft/convert_hf_dataset.py `
  --dataset HuggingFaceH4/no_robots `
  --output "$dataHF/no_robots.jsonl" `
  --languages en de `
  --max_examples 50000

& $py -3 scripts/sft/convert_hf_dataset.py `
  --dataset AmanPriyanshu/reasoning-sft-JSON-structuring-and-correcting `
  --output "$dataHF/amans_json_structuring.jsonl" `
  --languages en de `
  --max_examples 40000

& $py -3 scripts/sft/generate_phase3_phase31_control_sft.py `
  --output "$dataMid/phase3_phase31_control.jsonl"

& $py -3 scripts/sft/generate_phase3_json_sft.py `
  --output "$dataMid/phase3_json_strict.jsonl" `
  --num_examples 6000

& $py -3 scripts/sft/build_phase3_dataset.py `
  --output "$dataMid/phase3_1_restart_mixed.jsonl" `
  --meta_output "$dataMid/phase3_1_restart_mixed.meta.json" `
  --target_size 80000 `
  --precision_file data/sft_phase3_intermediate/phase3_precision_ref.jsonl `
  --grounded_file data/sft_phase3_intermediate/phase3_grounded_ref.jsonl `
  --remix_ratio 0.18 `
  --json_ratio 0.04 `
  --json_file "$dataMid/phase3_json_strict.jsonl" `
  --json_hf_file "$dataHF/amans_json_structuring.jsonl" `
  --json_hf_ratio 0.03 `
  --phase31_control_file "$dataMid/phase3_phase31_control.jsonl" `
  --phase31_control_ratio 0.04 `
  --injection_file data/sft_phase3_intermediate/phase3_injection_hierarchy_strict.jsonl `
  --injection_ratio 0.06 `
  --abstention_file data/sft_phase3_intermediate/phase3_abstention_strict.jsonl `
  --abstention_ratio 0.05 `
  --grounded_ratio 0.14 `
  --open_chat_cap_ratio 0.20 `
  --open_chat_files `
    "$dataHF/no_robots.jsonl" `
    data/sft_hf/oasst2.jsonl `
    data/sft_hf/dolci_instruct.jsonl `
    data/sft_hf/slimorca.jsonl `
    data/sft_hf/dolly.jsonl

if (Test-Path scripts/sft/audit_phase3_mix.py) {
  & $py -3 scripts/sft/audit_phase3_mix.py --input "$dataMid/phase3_1_restart_mixed.jsonl"
}
if (Test-Path scripts/sft/normalize_chat_jsonl.py) {
  & $py -3 scripts/sft/normalize_chat_jsonl.py --input "$dataMid/phase3_1_restart_mixed.jsonl" --in-place
}

& $py -3 scripts/sft/prepare_chat_sft.py `
  --input "$dataMid/phase3_1_restart_mixed.jsonl" `
  --output $outChat

Write-Host "Done. Packed chat: $outChat"
'''


def write_if_changed(path: Path, new_content: str) -> bool:
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == new_content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")
    return True


def patch_convert(content: str) -> tuple[str, bool]:
    changed = False
    if re.search(r"^import ast\s*$", content, re.M) is None:
        content = content.replace("import argparse\nimport json\n", "import argparse\nimport ast\nimport json\n", 1)
        changed = True
    marker = "# =============================================================================\n# UTILITY FUNCTIONS\n# ============================================================================="
    if "def parse_no_robots(" not in content:
        if marker not in content:
            raise RuntimeError("convert_hf_dataset.py: UTILITY FUNCTIONS marker not found")
        content = content.replace(marker, PHASE31_CONVERT_INSERT.strip() + "\n" + marker, 1)
        changed = True
    reg = (
        r"\n    load_kwargs\s*=\s*\{[^\}]*trust_remote_code[^\}]*\}\s*\n"
        r"(\s*if args\.subset:\n\s*load_kwargs\[\"name\"\] = args\.subset\s*\n)"
    )
    if re.search(reg, content):
        content = re.sub(
            reg,
            r"\n    load_kwargs: Dict[str, Any] = {}\n\1",
            content,
            count=1,
        )
        changed = True
    anchor = (
        '    "DiscoResearch/germanrag": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "germanrag"),\n'
        "}\n"
    )
    if '"HuggingFaceH4/no_robots"' not in content and anchor in content:
        content = content.replace(
            anchor,
            anchor[:-2]
            + '\n    "HuggingFaceH4/no_robots": parse_no_robots,\n'
            '    "AmanPriyanshu/reasoning-sft-JSON-structuring-and-correcting": '
            "parse_amans_json_structuring,\n}\n",
            1,
        )
        changed = True
    return content, changed


def patch_build(content: str) -> tuple[str, bool]:
    changed = False
    if "--json_hf_file" not in content:
        content = content.replace(
            '    p.add_argument("--json_file", type=str, default=None, help="Strict JSON-output dataset JSONL")\n',
            '    p.add_argument("--json_file", type=str, default=None, help="Strict JSON-output dataset JSONL")\n'
            '    p.add_argument("--json_hf_file", type=str, default=None, help="HF-converted strict JSON chat JSONL")\n',
            1,
        )
        changed = True
    if "--json_hf_ratio" not in content:
        content = content.replace(
            '    p.add_argument("--json_ratio", type=float, default=0.0)\n',
            '    p.add_argument("--json_ratio", type=float, default=0.0)\n'
            '    p.add_argument("--json_hf_ratio", type=float, default=0.0)\n',
            1,
        )
        changed = True
    if "--phase31_control_file" not in content:
        content = content.replace(
            '    p.add_argument("--abstention_file", type=str, default=None, help="Abstention corrective JSONL")\n',
            '    p.add_argument("--abstention_file", type=str, default=None, help="Abstention corrective JSONL")\n'
            '    p.add_argument("--phase31_control_file", type=str, default=None, '
            'help="Phase 3.1 eval-aligned control JSONL")\n',
            1,
        )
        changed = True
    if "--phase31_control_ratio" not in content:
        content = content.replace(
            '    p.add_argument("--abstention_ratio", type=float, default=0.0)\n',
            '    p.add_argument("--abstention_ratio", type=float, default=0.0)\n'
            '    p.add_argument("--phase31_control_ratio", type=float, default=0.0)\n',
            1,
        )
        changed = True
    if "json_hf_ratio > 0 and not args.json_hf_file" not in content:
        content = content.replace(
            "    if args.json_ratio > 0 and not args.json_file:\n"
            '        raise ValueError("json_ratio > 0 requires --json_file")\n',
            "    if args.json_ratio > 0 and not args.json_file:\n"
            '        raise ValueError("json_ratio > 0 requires --json_file")\n'
            "    if args.json_hf_ratio > 0 and not args.json_hf_file:\n"
            '        raise ValueError("json_hf_ratio > 0 requires --json_hf_file")\n'
            "    if args.phase31_control_ratio > 0 and not args.phase31_control_file:\n"
            '        raise ValueError("phase31_control_ratio > 0 requires --phase31_control_file")\n',
            1,
        )
        changed = True
    if "json_hf_rows = _add_source" not in content:
        content = content.replace(
            "    json_rows: List[Dict[str, Any]] = []\n"
            "    if args.json_file and args.json_ratio > 0:\n"
            '        json_rows = _add_source(_read_jsonl(Path(args.json_file)), "phase3_json_strict")\n',
            "    json_rows: List[Dict[str, Any]] = []\n"
            "    if args.json_file and args.json_ratio > 0:\n"
            '        json_rows = _add_source(_read_jsonl(Path(args.json_file)), "phase3_json_strict")\n'
            "    json_hf_rows: List[Dict[str, Any]] = []\n"
            "    if args.json_hf_file and args.json_hf_ratio > 0:\n"
            '        json_hf_rows = _add_source(_read_jsonl(Path(args.json_hf_file)), "phase3_json_hf")\n'
            "    phase31_rows: List[Dict[str, Any]] = []\n"
            "    if args.phase31_control_file and args.phase31_control_ratio > 0:\n"
            '        phase31_rows = _add_source(_read_jsonl(Path(args.phase31_control_file)), "phase3_phase31_control")\n',
            1,
        )
        changed = True
    if "n_json_hf = " not in content:
        content = content.replace(
            "    n_json = round(n_total * args.json_ratio) if (args.json_file and args.json_ratio > 0) else 0\n",
            "    n_json = round(n_total * args.json_ratio) if (args.json_file and args.json_ratio > 0) else 0\n"
            "    n_json_hf = round(n_total * args.json_hf_ratio) if (args.json_hf_file and args.json_hf_ratio > 0) else 0\n"
            "    n_phase31 = round(n_total * args.phase31_control_ratio) if (args.phase31_control_file and args.phase31_control_ratio > 0) else 0\n",
            1,
        )
        content = content.replace(
            "    fixed = n_remix + n_op + n_anti + n_json + n_injection + n_abstention + n_grounded\n",
            "    fixed = n_remix + n_op + n_anti + n_json + n_json_hf + n_phase31 + n_injection + n_abstention + n_grounded\n",
            1,
        )
        content = content.replace(
            "            \"remix_ratio + operators_ratio + anti_echo_ratio + json_ratio + injection_ratio + abstention_ratio + grounded_ratio exceed 100% \"\n",
            "            \"remix_ratio + operators_ratio + anti_echo_ratio + json_ratio + json_hf_ratio + phase31_control_ratio + injection_ratio + abstention_ratio + grounded_ratio exceed 100% \"\n",
            1,
        )
        changed = True
    if "_sample(json_hf_rows" not in content:
        content = content.replace(
            "    mixed.extend(_sample(json_rows, n_json, args.seed + 19))\n",
            "    mixed.extend(_sample(json_rows, n_json, args.seed + 19))\n"
            "    mixed.extend(_sample(json_hf_rows, n_json_hf, args.seed + 20))\n"
            "    mixed.extend(_sample(phase31_rows, n_phase31, args.seed + 20))\n",
            1,
        )
        changed = True
    if '"json_hf": args.json_hf_ratio' not in content:
        content = content.replace(
            '            "json_strict": args.json_ratio,\n',
            '            "json_strict": args.json_ratio,\n'
            '            "json_hf": args.json_hf_ratio,\n'
            '            "phase31_control": args.phase31_control_ratio,\n',
            1,
        )
        content = content.replace(
            '            "json_strict": n_json,\n',
            '            "json_strict": n_json,\n'
            '            "json_hf": n_json_hf,\n'
            '            "phase31_control": n_phase31,\n',
            1,
        )
        content = content.replace(
            '            "json_file": args.json_file,\n',
            '            "json_file": args.json_file,\n'
            '            "json_hf_file": args.json_hf_file,\n'
            '            "phase31_control_file": args.phase31_control_file,\n',
            1,
        )
        changed = True
    if 'source_counts.get("phase3_json_hf"' not in content:
        block = """    if args.json_file and args.json_ratio > 0:
        c = source_counts.get("phase3_json_strict", 0)
        lineage_inputs.append({
            "path": str(Path(args.json_file).resolve()),
            "sampled_rows": int(c),
            "effective_ratio": c / max(1, len(mixed)),
        })"""
        idx = content.find(block)
        if idx != -1 and "phase3_json_hf" not in content[idx : idx + 600]:
            content = content.replace(
                block,
                block
                + """
    if args.json_hf_file and args.json_hf_ratio > 0:
        c = source_counts.get("phase3_json_hf", 0)
        lineage_inputs.append({
            "path": str(Path(args.json_hf_file).resolve()),
            "sampled_rows": int(c),
            "effective_ratio": c / max(1, len(mixed)),
        })
    if args.phase31_control_file and args.phase31_control_ratio > 0:
        c = source_counts.get("phase3_phase31_control", 0)
        lineage_inputs.append({
            "path": str(Path(args.phase31_control_file).resolve()),
            "sampled_rows": int(c),
            "effective_ratio": c / max(1, len(mixed)),
        })""",
                1,
            )
            changed = True
    return content, changed


def patch_gen_json(content: str) -> tuple[str, bool]:
    changed = False
    if "STRICT_JSON_EVAL_USER" not in content:
        content = content.replace(
            "from core.system_prompts import CHAT_SYSTEM_PROMPTS\n\n\n",
            'from core.system_prompts import CHAT_SYSTEM_PROMPTS\n\n\nSTRICT_JSON_EVAL_USER = (\n'
            '    \'Return JSON only with schema {"label": "A|B", "score": number}. \''
            '\n    "Use label A and score 1."\n)\n'
            'STRICT_JSON_EVAL_ASSISTANT = \'{"label":"A","score":1}\'\n\n\n',
            1,
        )
        changed = True
    if "def _sample_eval_json_mirror" not in content:
        content = content.replace(
            "def _sample_label_score(rng: random.Random, language: str) -> Tuple[str, str, str]:",
            "def _sample_eval_json_mirror(rng: random.Random, language: str) -> Tuple[str, str, str]:\n"
            "    _ = rng\n"
            "    _ = language\n"
            "    return STRICT_JSON_EVAL_USER, STRICT_JSON_EVAL_ASSISTANT, \"json_strict_eval_mirror\"\n\n\n"
            "def _sample_label_score(rng: random.Random, language: str) -> Tuple[str, str, str]:",
            1,
        )
        changed = True
    if 'f"Create a JSON object with keys label (string) and score (number). "' in content:
        content = content.replace(
            "f\"Create a JSON object with keys label (string) and score (number). \"",
            'f"Create a JSON object with keys label (string) and score (integer). "',
            1,
        )
        changed = True
    if 'score (number), reason' in content:
        content = content.replace(
            '"Return exactly one JSON object with keys label (string), score (number), reason (string). "',
            '"Return exactly one JSON object with keys label (string), score (integer), reason (string). "',
            1,
        )
        changed = True
    if "_sample_eval_json_mirror," not in content.split("builders = [", 1)[1][:220]:
        content = content.replace(
            "    builders = [\n        _sample_label_score,\n",
            "    builders = [\n        _sample_eval_json_mirror,\n        _sample_label_score,\n",
            1,
        )
        changed = True
    if '"json_strict_eval_mirror": sum' not in content:
        content = content.replace(
            '"json_label_score": sum(1 for r in rows if r["_meta"]["category"] == "json_label_score"),\n',
            '"json_strict_eval_mirror": sum(1 for r in rows if r["_meta"]["category"] == "json_strict_eval_mirror"),\n'
            '            "json_label_score": sum(1 for r in rows if r["_meta"]["category"] == "json_label_score"),\n',
            1,
        )
        changed = True
    return content, changed


def ensure_restart_config() -> bool:
    if not CORRECTIVE_CFG.exists():
        return False
    base = json.loads(CORRECTIVE_CFG.read_text(encoding="utf-8"))
    base["name"] = "Phase 3.1: Restart SFT"
    base["description"] = (
        "Restart run after Phase 3 chat: mixed HF + strict JSON + Phase 3.1 control. Continue from phase3_chat_110k_gold."
    )
    base["log_file"] = "logs/train/phase3_1_restart.jsonl"
    text = json.dumps(base, indent=2) + "\n"
    return write_if_changed(RESTART_CFG, text)


def main() -> None:
    reports: list[str] = []

    c, _ = patch_convert(CONVERT_PATH.read_text(encoding="utf-8"))
    if write_if_changed(CONVERT_PATH, c):
        reports.append("convert_hf_dataset.py")

    b, _ = patch_build(BUILD_PATH.read_text(encoding="utf-8"))
    if write_if_changed(BUILD_PATH, b):
        reports.append("build_phase3_dataset.py")

    g, _ = patch_gen_json(GEN_JSON_PATH.read_text(encoding="utf-8"))
    if write_if_changed(GEN_JSON_PATH, g):
        reports.append("generate_phase3_json_sft.py")

    if ensure_restart_config():
        reports.append("phase3_1_restart.json")

    if write_if_changed(PS1_PATH, RUN_PS1_CONTENT):
        reports.append("run_phase31_prepare.ps1")

    print("apply_phase31_pipeline:", ", ".join(reports) if reports else "no changes")


if __name__ == "__main__":
    main()
