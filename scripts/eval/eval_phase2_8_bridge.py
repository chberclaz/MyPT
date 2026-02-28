#!/usr/bin/env python3
"""
Phase 2.8 bridge gate.

Hard requirements:
- format_strict: 100%
- echo_basic: 100%
- anti_echo: 100%
- operators: 100%

Report only:
- regression_basic
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.sft_eval_suite import run_evaluation


HARD_BUCKETS = ["format_strict", "echo_basic", "anti_echo", "operators"]
REPORT_ONLY = ["regression_basic"]


def _bucket_rate(results: Dict[str, Any], name: str) -> float:
    b = results.get("buckets", {}).get(name)
    if not b:
        return 0.0
    total = int(b.get("passed", 0)) + int(b.get("failed", 0))
    return (float(b.get("passed", 0)) * 100.0 / total) if total > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2.8 ABCE bridge gate")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    eval_results = run_evaluation(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
        operators_only=False,
    )

    gate = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "requirements": {},
        "report_only": {},
        "gate_passed": True,
        "eval_results": eval_results,
    }

    print("=" * 64)
    print("Phase 2.8 Bridge Gate")
    print("=" * 64)
    for name in HARD_BUCKETS:
        rate = _bucket_rate(eval_results, name)
        ok = rate >= 100.0
        gate["requirements"][name] = {"required": 100.0, "actual": round(rate, 2), "status": "PASS" if ok else "FAIL"}
        print(f"{name:18s}: {rate:6.2f}%  {'PASS' if ok else 'FAIL'} (required 100%)")
        if not ok:
            gate["gate_passed"] = False

    print("-" * 64)
    for name in REPORT_ONLY:
        rate = _bucket_rate(eval_results, name)
        gate["report_only"][name] = {"actual": round(rate, 2)}
        print(f"{name:18s}: {rate:6.2f}%  REPORT-ONLY")

    print("=" * 64)
    print(f"OVERALL: {'PASS' if gate['gate_passed'] else 'FAIL'}")
    print("=" * 64)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
        print(f"Saved: {out}")

    raise SystemExit(0 if gate["gate_passed"] else 1)


if __name__ == "__main__":
    main()

