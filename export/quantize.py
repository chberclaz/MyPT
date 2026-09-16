#!/usr/bin/env python3
"""Quantize F16 GGUF into the product matrix."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from export.paths import CHECKPOINT_GOLD, llama_quantize

QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=CHECKPOINT_GOLD / "mypt-f16.gguf")
    p.add_argument("--outdir", type=Path, default=CHECKPOINT_GOLD)
    args = p.parse_args()
    exe = llama_quantize()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for q in QUANTS:
        out = args.outdir / f"mypt-{q.lower()}.gguf"
        cmd = [str(exe), str(args.src), str(out), q]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
        lineage = args.src.with_suffix(".lineage.json")
        if lineage.exists():
            import json
            meta = json.loads(lineage.read_text(encoding="utf-8"))
            meta["mypt.lineage.quant_type"] = q
            out.with_suffix(".lineage.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )
    print("quantize OK")


if __name__ == "__main__":
    main()
