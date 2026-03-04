#!/usr/bin/env python3
"""
Inspect dataset lineage for unified-build outputs.

Usage:
    py.exe scripts/unified_build/inspect_lineage.py
    py.exe scripts/unified_build/inspect_lineage.py --inputs data/unified_6B data/unified_phase1_circuit
    py.exe scripts/unified_build/inspect_lineage.py --top 12 --output data/lineage_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset_lineage import load_lineage_for_input, iso_now


DEFAULT_INPUTS = [
    "data/unified_6B",
    "data/unified_phase1_circuit",
    "data/code_eval_tokenized",
    "data/retrieval_eval_tokenized",
]


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x):6.2f}%"
    except Exception:
        return "  n/a  "


def summarize_lineage(path: Path, top: int) -> Dict[str, Any]:
    lineage = load_lineage_for_input(path)
    direct = lineage.get("direct_inputs", [])
    flat = lineage.get("flattened_contributions", [])
    top_flat = flat[:top] if isinstance(flat, list) else []
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "creation_context": lineage.get("creation_context", {}),
        "direct_inputs_count": len(direct) if isinstance(direct, list) else 0,
        "recursive_origins_count": len(lineage.get("recursive_origins", [])) if isinstance(lineage.get("recursive_origins", []), list) else 0,
        "flattened_contributions_count": len(flat) if isinstance(flat, list) else 0,
        "top_flattened_contributions": top_flat,
        "lineage": lineage,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect lineage of unified-build datasets")
    p.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="Dataset dirs/files to inspect")
    p.add_argument("--top", type=int, default=8, help="Top flattened contributions to show")
    p.add_argument("--output", type=str, default=None, help="Optional JSON report output path")
    args = p.parse_args()

    resolved_inputs = []
    for x in args.inputs:
        px = Path(x)
        if not px.is_absolute():
            px = PROJECT_ROOT / px
        resolved_inputs.append(px)

    report = {
        "created_at": iso_now(),
        "script": "scripts/unified_build/inspect_lineage.py",
        "inputs": [str(p) for p in resolved_inputs],
        "results": [],
    }

    print("=" * 78)
    print("Unified Build Lineage Inspector")
    print("=" * 78)
    for path in resolved_inputs:
        r = summarize_lineage(path, args.top)
        report["results"].append(r)

        print(f"\nDataset: {r['path']}")
        print(f"  Exists: {r['exists']}")
        print(f"  Direct inputs: {r['direct_inputs_count']}")
        print(f"  Recursive origins: {r['recursive_origins_count']}")
        print(f"  Flattened contributions: {r['flattened_contributions_count']}")

        cc = r.get("creation_context", {})
        if isinstance(cc, dict) and cc:
            print(f"  Created by: {cc.get('script', 'unknown')}")
            print(f"  Timestamp:  {cc.get('timestamp', 'unknown')}")

        top_flat = r.get("top_flattened_contributions", [])
        if top_flat:
            print("  Top contributions:")
            for item in top_flat:
                origin = str(item.get("origin_path", "unknown"))
                rows = item.get("effective_rows", "n/a")
                pct = _fmt_pct(item.get("effective_percent", 0))
                print(f"    - {pct} | {rows:>10} | {origin}")
        else:
            print("  Top contributions: none")

    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nSaved report: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()

