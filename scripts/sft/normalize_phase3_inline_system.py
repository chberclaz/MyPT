#!/usr/bin/env python3
"""Hoist messages[0] with role=system into episode['system'] for prepare_chat_sft schema."""
import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--backup", action="store_true")
    args = p.parse_args()
    path = Path(args.input)
    if args.backup:
        bak = path.with_suffix(path.suffix + ".bak_pre_system_hoist")
        shutil.copy2(path, bak)
        print("Backup:", bak)
    tmp = path.with_suffix(".jsonl.tmp")
    n_fix = 0
    with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            ep = json.loads(line)
            msgs = ep.get("messages") or []
            if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
                sys_c = str(msgs[0].get("content", "")).strip()
                if not ep.get("system"):
                    ep["system"] = sys_c if sys_c else "You are MyPT."
                ep["messages"] = msgs[1:]
                n_fix += 1
            fout.write(json.dumps(ep, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print("Hoisted system from messages in", n_fix, "episodes ->", path)


if __name__ == "__main__":
    main()