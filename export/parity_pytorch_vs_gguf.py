#!/usr/bin/env python3
"""Greedy token-sequence parity: PyTorch bf16 vs llama.cpp F16 GGUF.

Holds only one backend in VRAM at a time (2060 6GB). Specials come from
core.special_tokens via each model's tokenizer.encode / llama-server token ids.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from export.paths import CHECKPOINT_GOLD, FIXTURES_DIR, RESULTS_DIR


def _load_prompts(n_need: int) -> list[dict]:
    prompts = []
    for line in (FIXTURES_DIR / "greedy_prompts.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.append(json.loads(line))
    if len(prompts) < n_need:
        raise SystemExit(f"need {n_need} greedy prompts, got {len(prompts)}")
    return prompts[:n_need]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=CHECKPOINT_GOLD)
    p.add_argument("--gguf", type=Path, default=CHECKPOINT_GOLD / "mypt-f16.gguf")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    prompts = _load_prompts(args.limit)

    from core import load_model

    pt = load_model(args.ckpt.name)
    pt_rows = []
    try:
        for item in prompts:
            prompt = item["prompt"]
            text, ids = pt.generate(
                prompt,
                max_new_tokens=args.n,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                use_default_stop_tokens=True,
                return_ids=True,
            )
            n_prompt = len(pt.encode(prompt))
            gen = ids[n_prompt : n_prompt + args.n]
            pt_rows.append({"id": item["id"], "gen_ids": gen, "n": len(gen)})
            print(f"PT   {item['id']} ({len(gen)} toks) {gen[:8]}")
    finally:
        del pt
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    from core.inference.gguf_backend import GGUFModel

    gg = GGUFModel(args.gguf)
    fails = []
    try:
        for item, row in zip(prompts, pt_rows):
            prompt = item["prompt"]
            _text, ids = gg.generate(
                prompt,
                max_new_tokens=args.n,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
                use_default_stop_tokens=True,
                return_ids=True,
            )
            n_prompt = len(gg.encode(prompt))
            gen = ids[n_prompt : n_prompt + args.n]
            a = row["gen_ids"]
            n = min(len(a), len(gen))
            if a[:n] != gen[:n] or len(a) != len(gen):
                fails.append(item["id"])
                print(f"FAIL {item['id']} n_pt={len(a)} n_gg={len(gen)} pt={a[:16]} gg={gen[:16]}")
            else:
                print(f"OK   {item['id']} ({n} toks)")
    finally:
        gg.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "greedy_parity.json"
    out.write_text(
        json.dumps({"fails": fails, "n": args.n, "limit": args.limit, "gguf": str(args.gguf)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if fails:
        raise SystemExit(f"greedy mismatch: {fails}")
    print("PARITY_OK")


if __name__ == "__main__":
    main()
