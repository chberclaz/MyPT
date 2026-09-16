#!/usr/bin/env python3
"""
Evaluate Phase 2.5 bridge quality:
- WRAP exact match across varied delimiters/styles
- Anti-echo behavior on gibberish prompts
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core import load_model
from core.system_prompts import CONVERSATION_SYSTEM_PROMPT


WRAP_CASES: List[Tuple[str, str]] = [
    ("Wrap this in []: horizon", "[horizon]"),
    ("Surround with [] brackets: blue ocean", "[blue ocean]"),
    ("Wrap this in (): marble", "(marble)"),
    ("Return cold winter night wrapped by ( and ).", "(cold winter night)"),
    ("Wrap this in {}: galaxy", "{galaxy}"),
    ("Enclose exactly: dark forest using {}", "{dark forest}"),
    ("Wrap this in <>: nebula", "<nebula>"),
    ("Return red mountain peak wrapped by < and >.", "<red mountain peak>"),
    ("Wrap this in [[]]: phoenix", "[[phoenix]]"),
    ("Return silver moonlight wrapped by [[ and ]].", "[[silver moonlight]]"),
    ("Wrap this in ((): quartz", "((quartz))"),
    ("Return golden sunset sky wrapped by (( and )).", "((golden sunset sky))"),
    ("Wrap this in {{}}: dragon", "{{dragon}}"),
    ("Return ancient stone bridge wrapped by {{ and }}.", "{{ancient stone bridge}}"),
    ("Wrap this in '' : cascade", "'cascade'"),
    ('Wrap this in "" : twilight', '"twilight"'),
]

ANTI_ECHO_CASES: List[Tuple[str, str]] = [
    ('What is "Blurpix"?', "blurpix"),
    ('Is "Zanthor" a real word?', "zanthor"),
    ('Does "Quexling" have a meaning?', "quexling"),
    ('Define "Flimzap".', "flimzap"),
    ('Translate "Norbex" to German.', "norbex"),
    ('What means "Xylotran"?', "xylotran"),
]


def normalize(text: str) -> str:
    text = text.strip()
    # Remove generated assistant closure and eot if present
    text = text.replace("</myPT_assistant>", "").replace("<myPT_eot>", "").strip()
    # Strip one trailing period for lenient exact-match
    if text.endswith("."):
        text = text[:-1].strip()
    return text


def make_prompt(user_msg: str, no_system_prompt: bool) -> str:
    if no_system_prompt:
        return f"<myPT_user>{user_msg}</myPT_user><myPT_assistant>"
    return (
        f"<myPT_system>{CONVERSATION_SYSTEM_PROMPT}</myPT_system>"
        f"<myPT_user>{user_msg}</myPT_user><myPT_assistant>"
    )


def generate(model, prompt: str, max_new_tokens: int) -> str:
    out = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    gen = out[len(prompt):] if out.startswith(prompt) else out
    return normalize(gen)


def evaluate_phase2_5(
    model_name: str,
    max_new_tokens: int = 48,
    no_system_prompt: bool = True,
    verbose: bool = False,
) -> dict:
    model = load_model(model_name)
    model.eval()

    print("=" * 60)
    print("  PHASE 2.5 EVAL: WRAP + ANTI-ECHO")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"System prompt: {'OMITTED' if no_system_prompt else 'INCLUDED'}")
    print()

    wrap_pass = 0
    for idx, (q, expected) in enumerate(WRAP_CASES, 1):
        prompt = make_prompt(q, no_system_prompt)
        got = generate(model, prompt, max_new_tokens)
        ok = got == expected
        wrap_pass += int(ok)
        if verbose or not ok:
            mark = "PASS" if ok else "FAIL"
            print(f"[WRAP {idx:02d}] {mark} | q='{q}'")
            if not ok:
                print(f"  expected: {expected}")
                print(f"  got:      {got}")

    anti_pass = 0
    for idx, (q, forbidden) in enumerate(ANTI_ECHO_CASES, 1):
        prompt = make_prompt(q, no_system_prompt)
        got = generate(model, prompt, max_new_tokens)
        # Fail if forbidden token appears as a whole word (case-insensitive)
        pat = rf"\b{re.escape(forbidden)}\b"
        ok = re.search(pat, got, flags=re.IGNORECASE) is None
        anti_pass += int(ok)
        if verbose or not ok:
            mark = "PASS" if ok else "FAIL"
            print(f"[ANTI {idx:02d}] {mark} | q='{q}'")
            if not ok:
                print(f"  forbidden: {forbidden}")
                print(f"  got:       {got}")

    wrap_total = len(WRAP_CASES)
    anti_total = len(ANTI_ECHO_CASES)
    wrap_pct = 100.0 * wrap_pass / wrap_total
    anti_pct = 100.0 * anti_pass / anti_total

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"WRAP exact-match: {wrap_pass}/{wrap_total} ({wrap_pct:.1f}%)")
    print(f"Anti-echo gate:   {anti_pass}/{anti_total} ({anti_pct:.1f}%)")

    wrap_ok = wrap_pct >= 90.0
    anti_ok = anti_pct >= 75.0
    print(f"WRAP target >= 90%: {'PASS' if wrap_ok else 'FAIL'}")
    print(f"ANTI target >= 75%: {'PASS' if anti_ok else 'FAIL'}")

    overall_ok = wrap_ok and anti_ok
    print(f"OVERALL: {'PASS' if overall_ok else 'FAIL'}")

    return {
        "wrap_passed": wrap_pass,
        "wrap_total": wrap_total,
        "wrap_pass_rate": round(wrap_pct, 1),
        "anti_passed": anti_pass,
        "anti_total": anti_total,
        "anti_pass_rate": round(anti_pct, 1),
        "wrap_threshold": 90.0,
        "anti_threshold": 75.0,
        "passed": overall_ok,
    }


def run_eval(model_name: str, max_new_tokens: int, no_system_prompt: bool, verbose: bool) -> int:
    result = evaluate_phase2_5(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        no_system_prompt=no_system_prompt,
        verbose=verbose,
    )
    return 0 if result["passed"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Phase 2.5 WRAP + anti-echo")
    parser.add_argument("--model", required=True, help="Model name in checkpoints/")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--no_system_prompt", action="store_true", default=True)
    parser.add_argument("--with_system_prompt", action="store_false", dest="no_system_prompt")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    raise SystemExit(
        run_eval(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            no_system_prompt=args.no_system_prompt,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
