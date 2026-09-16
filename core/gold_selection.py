#!/usr/bin/env python3
"""
Configurable GOLD checkpoint selection and external gate evaluation utilities.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def deep_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in (path or "").split("."):
        if not part:
            continue
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


@dataclass
class GoldGuardConfig:
    overfit_ratio: float = 5.0
    consecutive_rises: int = 2
    eval_regression: float = 0.20


@dataclass
class ExternalGateConfig:
    enabled: bool = False
    script: str = ""
    args: List[str] | None = None
    timeout_s: int = 1200
    run_interval_evals: int = 1
    run_on_final_eval: bool = True
    metric_path: str = ""
    metric_goal: str = "max"  # max|min
    pass_path: str = "gate_passed"
    require_pass: bool = False
    strict: bool = False


@dataclass
class GoldSelectionConfig:
    strategy: str = "val_loss"  # val_loss|hybrid|external_gate
    apply_loss_guards: bool = True
    guards: GoldGuardConfig = field(default_factory=GoldGuardConfig)
    external_gate: ExternalGateConfig = field(default_factory=ExternalGateConfig)


def parse_gold_selection_config(raw: Optional[Dict[str, Any]]) -> GoldSelectionConfig:
    raw = raw or {}
    guards_raw = raw.get("guards", {}) if isinstance(raw.get("guards", {}), dict) else {}
    ext_raw = raw.get("external_gate", {}) if isinstance(raw.get("external_gate", {}), dict) else {}

    guards = GoldGuardConfig(
        overfit_ratio=float(guards_raw.get("overfit_ratio", 5.0)),
        consecutive_rises=int(guards_raw.get("consecutive_rises", 2)),
        eval_regression=float(guards_raw.get("eval_regression", 0.20)),
    )
    ext = ExternalGateConfig(
        enabled=bool(ext_raw.get("enabled", False)),
        script=str(ext_raw.get("script", "")),
        args=list(ext_raw.get("args", [])) if isinstance(ext_raw.get("args", []), list) else [],
        timeout_s=int(ext_raw.get("timeout_s", 1200)),
        run_interval_evals=int(ext_raw.get("run_interval_evals", 1)),
        run_on_final_eval=bool(ext_raw.get("run_on_final_eval", True)),
        metric_path=str(ext_raw.get("metric_path", "")),
        metric_goal=str(ext_raw.get("metric_goal", "max")).lower(),
        pass_path=str(ext_raw.get("pass_path", "gate_passed")),
        require_pass=bool(ext_raw.get("require_pass", False)),
        strict=bool(ext_raw.get("strict", False)),
    )

    strategy = str(raw.get("strategy", "val_loss")).lower()
    if strategy not in {"val_loss", "hybrid", "external_gate"}:
        strategy = "val_loss"

    return GoldSelectionConfig(
        strategy=strategy,
        apply_loss_guards=bool(raw.get("apply_loss_guards", True)),
        guards=guards,
        external_gate=ext,
    )


def should_run_gate(ext: ExternalGateConfig, eval_count: int, is_final_eval: bool) -> bool:
    if not ext.enabled:
        return False
    if is_final_eval:
        return ext.run_on_final_eval
    interval = max(1, int(ext.run_interval_evals))
    return (eval_count % interval) == 0


def run_external_gate(
    ext: ExternalGateConfig,
    project_root: Path,
    checkpoint_dir: Optional[str],
    model_name: Optional[str],
    iter_step: int,
    is_final_eval: bool = False,
) -> Dict[str, Any]:
    if not ext.enabled:
        return {"ran": False}
    if not ext.script:
        return {"ran": True, "ok": False, "error": "external_gate.script missing"}

    ckpt = Path(checkpoint_dir).resolve() if checkpoint_dir else None
    script_path = Path(ext.script)
    if not script_path.is_absolute():
        script_path = (project_root / script_path).resolve()
    if not script_path.exists():
        return {"ran": True, "ok": False, "error": f"script not found: {script_path}"}

    model_name_eff = model_name or (ckpt.name if ckpt else None) or ""
    output_json = None
    if ckpt is not None:
        out_name = f"gold_gate_final.json" if is_final_eval else f"gold_gate_step_{iter_step}.json"
        output_json = str((ckpt / out_name).resolve())

    # Template args with placeholders
    args = []
    for tok in (ext.args or []):
        tok_s = str(tok)
        tok_s = tok_s.replace("{model_name}", model_name_eff)
        if output_json:
            tok_s = tok_s.replace("{output_json}", output_json)
        args.append(tok_s)

    # Ensure output exists when possible for robust parsing.
    if output_json and "{output_json}" not in " ".join(ext.args or []):
        args.extend(["--output", output_json])

    cmd = [sys.executable, str(script_path)] + args
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=max(1, int(ext.timeout_s)),
        )
    except Exception as e:
        return {"ran": True, "ok": False, "error": str(e), "cmd": cmd}

    result: Dict[str, Any] = {
        "ran": True,
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "output_json": output_json,
    }

    gate_json: Dict[str, Any] = {}
    if output_json and Path(output_json).exists():
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                gate_json = json.load(f)
        except Exception as e:
            result["json_error"] = str(e)

    gate_pass = bool(deep_get(gate_json, ext.pass_path, proc.returncode == 0))
    metric_val = deep_get(gate_json, ext.metric_path, None) if ext.metric_path else None
    metric_num = None
    if metric_val is not None:
        try:
            metric_num = float(metric_val)
        except Exception:
            metric_num = None

    result["gate_pass"] = gate_pass
    result["metric_path"] = ext.metric_path
    result["metric_value"] = metric_num
    result["raw_metric_value"] = metric_val
    result["gate_json"] = gate_json
    return result


def compare_metric(candidate: float, best: Optional[float], goal: str) -> bool:
    if best is None:
        return True
    if goal == "min":
        return candidate < best
    return candidate > best

