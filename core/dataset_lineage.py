#!/usr/bin/env python3
"""
Dataset lineage utilities for the full data pipeline.

This module is intentionally phase-agnostic and can be used by:
- pretraining/unified-build pipelines
- SFT data generators/mixers/preparers
- audit and validation scripts
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def candidate_metadata_paths(dataset_path: Path) -> List[Path]:
    candidates: List[Path] = []
    if dataset_path.suffix.lower() == ".jsonl":
        candidates.append(dataset_path.with_suffix(".mix_meta.json"))
        candidates.append(dataset_path.with_suffix(".meta.json"))
        candidates.append(dataset_path.with_suffix(".lineage.json"))
    if dataset_path.is_dir():
        candidates.append(dataset_path / "dataset_metadata.json")
        candidates.append(dataset_path / "dataset_lineage.json")
    return candidates


def _fallback_leaf_lineage(path: Path) -> Dict[str, Any]:
    rows = 0
    if path.suffix.lower() == ".jsonl":
        rows = count_jsonl_rows(path)
    elif path.is_dir():
        meta = _safe_read_json(path / "dataset_metadata.json") or {}
        rows = int(meta.get("total_tokens", 0) or meta.get("num_train_tokens", 0) or 0)

    return {
        "direct_inputs": [],
        "recursive_origins": [{
            "origin_path": str(path),
            "rows": rows,
            "notes": "leaf_source_no_lineage_metadata",
        }],
        "flattened_contributions": [{
            "origin_path": str(path),
            "effective_rows": rows,
            "effective_percent": 100.0 if rows > 0 else 0.0,
        }],
        "creation_context": {},
        "upstream_configs": [],
    }


def load_lineage_for_input(dataset_path: Path) -> Dict[str, Any]:
    path = dataset_path.resolve()
    for meta_path in candidate_metadata_paths(path):
        meta = _safe_read_json(meta_path)
        if not meta:
            continue
        lineage = meta.get("lineage")
        if isinstance(lineage, dict):
            return lineage
        if "flattened_contributions" in meta and "recursive_origins" in meta:
            return {
                "direct_inputs": meta.get("direct_inputs", []),
                "recursive_origins": meta.get("recursive_origins", []),
                "flattened_contributions": meta.get("flattened_contributions", []),
                "creation_context": meta.get("creation_context", {}),
                "upstream_configs": meta.get("upstream_configs", []),
            }
    return _fallback_leaf_lineage(path)


def merge_lineage(
    inputs: List[Dict[str, Any]],
    output_rows: int,
    creation_context: Dict[str, Any],
) -> Dict[str, Any]:
    direct_inputs: List[Dict[str, Any]] = []
    merged_flat: Dict[str, float] = {}
    recursive_origins: List[Dict[str, Any]] = []
    upstream_configs: List[Dict[str, Any]] = []

    for inp in inputs:
        in_path = Path(inp["path"]).resolve()
        sampled_rows = int(inp.get("sampled_rows", 0))
        eff_ratio = float(inp.get("effective_ratio", 0.0)) if output_rows > 0 else 0.0
        in_lineage = load_lineage_for_input(in_path)

        direct_inputs.append({
            "path": str(in_path),
            "sampled_rows": sampled_rows,
            "effective_ratio": eff_ratio,
        })

        for origin in in_lineage.get("recursive_origins", []):
            if isinstance(origin, dict):
                recursive_origins.append(origin)

        cc = in_lineage.get("creation_context")
        if isinstance(cc, dict) and cc:
            upstream_configs.append(cc)
        uc = in_lineage.get("upstream_configs", [])
        if isinstance(uc, list):
            upstream_configs.extend([x for x in uc if isinstance(x, dict)])

        flattened = [x for x in in_lineage.get("flattened_contributions", []) if isinstance(x, dict)]
        if flattened:
            total_upstream = sum(float(x.get("effective_rows", 0.0)) for x in flattened)
            if total_upstream <= 0:
                total_upstream = 1.0
            for item in flattened:
                origin_path = str(item.get("origin_path", ""))
                if not origin_path:
                    continue
                origin_rows = float(item.get("effective_rows", 0.0))
                ratio = max(0.0, origin_rows / total_upstream)
                merged_flat[origin_path] = merged_flat.get(origin_path, 0.0) + (sampled_rows * ratio)
        else:
            key = str(in_path)
            merged_flat[key] = merged_flat.get(key, 0.0) + sampled_rows

    if not merged_flat:
        for d in direct_inputs:
            merged_flat[d["path"]] = merged_flat.get(d["path"], 0.0) + float(d["sampled_rows"])

    total_rows = sum(merged_flat.values())
    flat_list = []
    for origin_path, eff_rows in merged_flat.items():
        flat_list.append({
            "origin_path": origin_path,
            "effective_rows": int(round(eff_rows)),
            "effective_percent": round((eff_rows * 100.0 / total_rows), 6) if total_rows > 0 else 0.0,
        })
    flat_list.sort(key=lambda x: x["effective_rows"], reverse=True)

    return {
        "direct_inputs": direct_inputs,
        "recursive_origins": recursive_origins,
        "flattened_contributions": flat_list,
        "creation_context": creation_context,
        "upstream_configs": upstream_configs,
    }


def write_lineage_sidecar(output_dataset_path: Path, lineage: Dict[str, Any]) -> Path:
    out = output_dataset_path.resolve()
    sidecar = out.with_suffix(".lineage.json") if out.suffix.lower() == ".jsonl" else out / "dataset_lineage.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({"lineage": lineage}, f, indent=2, ensure_ascii=False)
    return sidecar

