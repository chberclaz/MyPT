#!/usr/bin/env python3
"""Convert HF Llama dir -> GGUF F16 using the pinned llama.cpp converter."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from export.paths import (
    ARTIFACTS_DIR,
    CHECKPOINT_GOLD,
    LLAMA_SRC,
    PIN_FILE,
    REPO_ROOT,
)
from core.special_tokens import SPECIAL_TOKEN_STRINGS, get_special_token_ids


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _pin_sha() -> str:
    if PIN_FILE.exists():
        for line in PIN_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("sha="):
                return line.split("=", 1)[1].strip()
    return "unpinned"


def _inject_hash_mapping(chkhsh: str, pre: str = "gpt-2") -> None:
    """Add MyPT tokenizer fingerprint -> known pre-tokenizer (expected first convert abort)."""
    base = LLAMA_SRC / "conversion" / "base.py"
    text = base.read_text(encoding="utf-8")
    needle = f'if chkhsh == "{chkhsh}":'
    if needle in text:
        print(f"hash {chkhsh[:12]} already registered")
        return
    marker = "        if chkhsh == \"3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454\":"
    insert = (
        f'        if chkhsh == "{chkhsh}":\n'
        f'            # MyPT gpt2 + {len(SPECIAL_TOKEN_STRINGS)} special tags from core.special_tokens\n'
        f'            res = "{pre}"\n'
    )
    if marker not in text:
        raise RuntimeError("could not find gpt-2 stock hash insertion point in conversion/base.py")
    base.write_text(text.replace(marker, insert + marker, 1), encoding="utf-8")
    print(f"registered chkhsh {chkhsh} -> {pre}")


def convert(hf_dir: Path, outfile: Path, outtype: str = "f16") -> None:
    converter = LLAMA_SRC / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"missing {converter}; clone llama.cpp first")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(LLAMA_SRC / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(converter),
        str(hf_dir),
        "--outfile",
        str(outfile),
        "--outtype",
        outtype,
    ]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(LLAMA_SRC), env=env, capture_output=True, text=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sys.stdout.write(proc.stdout or "")
    sys.stderr.write(proc.stderr or "")
    if proc.returncode != 0:
        if "BPE pre-tokenizer was not recognized" in combined or "not recognized" in combined.lower():
            # Compute hash the same way as conversion/base.py using our HF tokenizer.
            from hashlib import sha256
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(hf_dir), use_fast=True)
            chktxt = "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''``````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL"
            chktok = tok.encode(chktxt)
            chkhsh = sha256(str(chktok).encode()).hexdigest()
            print(f"unrecognized pre-tokenizer; registering {chkhsh} -> gpt-2")
            (REPO_ROOT / "export" / "pretokenizer_hash.json").write_text(
                json.dumps({"chkhsh": chkhsh, "pre": "gpt-2"}, indent=2) + "\n",
                encoding="utf-8",
            )
            _inject_hash_mapping(chkhsh, "gpt-2")
            proc2 = subprocess.run(cmd, cwd=str(LLAMA_SRC), env=env)
            if proc2.returncode != 0:
                raise SystemExit(proc2.returncode)
        else:
            raise SystemExit(proc.returncode)

    _write_sidecar_lineage(outfile)


def _write_sidecar_lineage(gguf_path: Path) -> None:
    ids = get_special_token_ids()
    meta = {
        "mypt.lineage.checkpoint_id": "phase6_3_ground_gold",
        "mypt.lineage.phase_chain": "1>1b>2>3>4>5>6>6.2>6.3",
        "mypt.lineage.training_commit": _git_sha(),
        "mypt.lineage.export_commit": _git_sha(),
        "mypt.lineage.llamacpp_commit": _pin_sha(),
        "mypt.lineage.export_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mypt.lineage.quant_type": gguf_path.stem,
        "mypt.tools.trained_tool_names": "workspace.search,workspace.list_docs,workspace.get_doc,workspace.summarize",
        "mypt.special_token_ids": json.dumps(ids, sort_keys=True),
        "mypt.special_token_strings": json.dumps(SPECIAL_TOKEN_STRINGS, sort_keys=True),
    }
    ckpt_state = CHECKPOINT_GOLD / "training_state.json"
    if ckpt_state.exists():
        state = json.loads(ckpt_state.read_text(encoding="utf-8"))
        ds = (state.get("training_config") or {}).get("dataset_dir")
        if ds:
            lineage_file = REPO_ROOT / ds / "dataset_lineage.json"
            if not lineage_file.exists():
                lineage_file = REPO_ROOT / ds / "dataset_metadata.json"
            if lineage_file.exists():
                digest = hashlib.sha256(lineage_file.read_bytes()).hexdigest()
                meta["mypt.lineage.manifest_hash"] = digest
    side = gguf_path.with_suffix(".lineage.json")
    side.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote lineage sidecar {side}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf", type=Path, default=ARTIFACTS_DIR / "hf_llama")
    p.add_argument("--out", type=Path, default=CHECKPOINT_GOLD / "mypt-f16.gguf")
    p.add_argument("--outtype", default="f16")
    args = p.parse_args()
    convert(args.hf, args.out, args.outtype)


if __name__ == "__main__":
    main()
