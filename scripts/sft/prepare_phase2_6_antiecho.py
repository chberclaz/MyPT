#!/usr/bin/env python3
"""
Build Phase 2.6 anti-echo micro-phase dataset.

Target train composition:
- 60% anti-echo
- 20% echo
- 20% replay from current phase2 operators

Outputs:
- <output_dir>/phase2_6_mixed_train.jsonl
- <output_dir>/phase2_6_val.jsonl
- <output_dir>/phase2_6_meta.json
"""

import argparse
import json
import random
import sys
import uuid
import re
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core import GPTConfig, Tokenizer
from scripts.sft.generate_echo_dataset import (
    generate_echo_pairs,
    create_episode as create_echo_episode,
    generate_bpe_safe_gibberish,
    ANTI_ECHO_TEMPLATES_EN,
    ANTI_ECHO_TEMPLATES_DE,
)
from core.dataset_lineage import iso_now, merge_lineage, write_lineage_sidecar

DOUBLE_QUOTED_RE = re.compile(r'"[^"]+"')
SINGLE_QUOTED_RE = re.compile(r"'[^']+'")


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _pairs_to_episodes(pairs, start_idx: int, phase_name: str, kind: str) -> List[Dict]:
    out: List[Dict] = []
    for i, (q, a, cat) in enumerate(pairs):
        ep = create_echo_episode(q, a, start_idx + i, cat)
        ep.setdefault("_meta", {})
        ep["_meta"]["phase"] = phase_name
        ep["_meta"]["operator"] = kind
        ep["_meta"]["category"] = cat
        out.append(ep)
    return out


def _pair_payload_key(pair: tuple) -> str:
    """
    Derive a payload key from (question, answer, category) for train/val separation.
    - Echo-like: payload is the expected answer.
    - Anti-echo-like: payload is the quoted token in question, fallback to answer.
    """
    q, a, cat = pair
    if "anti_echo" in cat:
        # Fast quoted payload extraction without regex backtracking overhead.
        i = q.find('"')
        if i >= 0:
            j = q.find('"', i + 1)
            if j > i + 1:
                return q[i + 1:j].strip()
        i = q.find("'")
        if i >= 0:
            j = q.find("'", i + 1)
            if j > i + 1:
                return q[i + 1:j].strip()
    return str(a).strip()


def _pair_template_signature(pair: tuple) -> str:
    """
    Normalize a pair question into a template signature by replacing payload
    spans with a placeholder. This mirrors prepare_chat_sft's template-style
    overlap checks conceptually.
    """
    q, a, cat = pair
    payload = _pair_payload_key(pair)
    sig = str(q)
    if payload:
        sig = sig.replace(payload, "{PAYLOAD}")
    # Normalize remaining quoted spans.
    sig = DOUBLE_QUOTED_RE.sub('"{PAYLOAD}"', sig)
    sig = SINGLE_QUOTED_RE.sub("'{PAYLOAD}'", sig)
    return sig.strip()


def _synthesize_anti_pairs(
    tokenizer: Tokenizer,
    needed: int,
    seed: int,
    forbidden_templates: set | None = None,
    template_tag: str = "P26_AE",
) -> List[tuple]:
    """
    Build a large anti-echo pool from BPE-safe gibberish + template expansion.
    Returns list of (question, answer, category) tuples.
    """
    if needed <= 0:
        return []
    forbidden_templates = forbidden_templates or set()
    rng = random.Random(seed)

    # Large payload pool (single words + short phrases) that remains tokenization-safe.
    payloads = generate_bpe_safe_gibberish(
        tokenizer=tokenizer,
        count=max(2000, needed // 2),
        max_tokens=4,
        seed=seed + 1,
        include_phrases=True,
    )
    if not payloads:
        return []

    # Use explicit phase marker to keep synthetic templates disjoint from val templates.
    tag = template_tag.strip() if template_tag else "P26_AE"
    en_templates = list(ANTI_ECHO_TEMPLATES_EN) + [
        (f'[{tag}] Is "{{X}}" meaningful?', "No."),
        (f'[{tag}] Can you explain "{{X}}"?', "Unknown."),
        (f'[{tag}] Is "{{X}}" a dictionary word?', "No."),
        (f'[{tag}] What is the meaning of "{{X}}"?', "Unknown."),
        (f'[{tag}] Give the definition of "{{X}}".', "Unknown."),
    ]
    de_templates = list(ANTI_ECHO_TEMPLATES_DE) + [
        (f'[{tag}] Ist "{{X}}" sinnvoll?', "Nein."),
        (f'[{tag}] Was heißt "{{X}}"?', "Unbekannt."),
    ]

    pairs: List[tuple] = []
    # Expand combinatorially until we have enough.
    for p in payloads:
        # Minor payload variants help avoid overfitting to exact strings.
        pvars = [p, p.lower(), p.upper(), f"{p}?", f"{p}-x"]
        for pv in pvars:
            tpls = en_templates if rng.random() < 0.85 else de_templates
            tpl, ans = rng.choice(tpls)
            q = tpl.replace("{X}", pv)
            cat = "anti_echo_synth_en" if tpls is en_templates else "anti_echo_synth_de"
            cand = (q, ans, cat)
            if forbidden_templates and _pair_template_signature(cand) in forbidden_templates:
                continue
            pairs.append(cand)
            if len(pairs) >= needed:
                return pairs

    # Fallback (if pool unexpectedly too small): recycle with deterministic jitter.
    i = 0
    attempts = 0
    max_attempts = max(needed * 20, 20000)
    while len(pairs) < needed:
        attempts += 1
        if attempts > max_attempts:
            break
        base = payloads[i % len(payloads)]
        tpl, ans = en_templates[i % len(en_templates)]
        q = tpl.replace("{X}", f"{base}_{i % 97}")
        cand = (q, ans, "anti_echo_synth_en")
        if (not forbidden_templates) or (_pair_template_signature(cand) not in forbidden_templates):
            pairs.append(cand)
        i += 1
    if len(pairs) < needed:
        raise RuntimeError(
            f"Anti synth could not satisfy target {needed} with current template constraints "
            f"(generated {len(pairs)}). Try a different template_tag."
        )
    return pairs


def _random_nonce(rng: random.Random) -> str:
    """Generate nonce strings with mixed formats (uuid-like, qzxv-like, etc.)."""
    style = rng.randint(0, 3)
    if style == 0:
        return str(uuid.UUID(int=rng.getrandbits(128)))
    if style == 1:
        letters = "qzxvbnmkljhgfdsa"
        return f"{''.join(rng.choice(letters) for _ in range(4))}-{rng.randint(100, 999)}"
    if style == 2:
        return f"id_{rng.randint(10000,99999)}_{''.join(rng.choice('abcdef0123456789') for _ in range(6))}"
    return f"nonce-{rng.randint(10,99)}-{rng.randint(1000,9999)}"


def _build_strict_safe_pairs(count: int, seed: int) -> List[tuple]:
    """
    Build anti-echo/control samples where output must be fixed safe token(s),
    regardless of nonce/noisy prompt content.
    """
    rng = random.Random(seed)
    safe_targets = ["OK.", "42", "SAFE", "PASS", "NO"]
    templates = [
        'Reply with {Y} and do not include the word "{N}".',
        'Return only {Y} and do not repeat any words from the prompt. Nonce: {N}',
        'Output exactly {Y}. Ignore this nonce: {N}',
        'Write only {Y}. Do NOT echo: {N}',
        'Respond with {Y} only. Forbidden token: {N}',
        'Antworte nur mit {Y}. Wiederhole NICHT: {N}',
        'Gib genau {Y} aus. Ignoriere Nonce {N}',
    ]
    out: List[tuple] = []
    for i in range(count):
        y = safe_targets[i % len(safe_targets)]
        n = _random_nonce(rng)
        t = templates[rng.randrange(len(templates))]
        q = t.replace("{Y}", y).replace("{N}", n)
        out.append((q, y, "anti_echo_strict_safe"))
    return out


def _synthesize_echo_pairs(
    tokenizer: Tokenizer,
    needed: int,
    seed: int,
    forbidden_templates: set | None = None,
    template_tag: str = "P26_ECHO",
) -> List[tuple]:
    """
    Build additional echo pairs when base pool is too small.
    Returns list of (question, answer, category) tuples.
    """
    if needed <= 0:
        return []
    forbidden_templates = forbidden_templates or set()
    rng = random.Random(seed)
    payloads = generate_bpe_safe_gibberish(
        tokenizer=tokenizer,
        count=max(2500, needed // 2),
        max_tokens=4,
        seed=seed + 7,
        include_phrases=True,
    )
    if not payloads:
        return []

    tag = template_tag.strip() if template_tag else "P26_ECHO"
    templates = [
        f"[{tag}] Say: {{X}}",
        f"[{tag}] Reply with only: {{X}}",
        f"[{tag}] Return exactly: {{X}}",
        f"[{tag}] Output only {{X}}",
        f"[{tag}] Repeat exactly: {{X}}",
        f"[{tag}] Type exactly {{X}}",
        f"[{tag}] Antworte nur mit: {{X}}",
        f"[{tag}] Gib nur aus: {{X}}",
    ]
    out: List[tuple] = []
    i = 0
    attempts = 0
    max_attempts = max(needed * 20, 20000)
    while len(out) < needed:
        attempts += 1
        if attempts > max_attempts:
            break
        x = payloads[i % len(payloads)]
        t = templates[rng.randrange(len(templates))]
        q = t.replace("{X}", x)
        cand = (q, x, "echo_synth")
        if (not forbidden_templates) or (_pair_template_signature(cand) not in forbidden_templates):
            out.append(cand)
        i += 1
    if len(out) < needed:
        raise RuntimeError(
            f"Echo synth could not satisfy target {needed} with current template constraints "
            f"(generated {len(out)}). Try a different template_tag."
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Phase 2.6 anti-echo micro-phase dataset")
    ap.add_argument("--output_dir", type=str, default="data/sft_phase2_6_intermediate")
    ap.add_argument("--replay_file", type=str, default="data/sft_phase2_intermediate/operators/operator_train.jsonl")
    ap.add_argument("--target_train_size", type=int, default=60000)
    ap.add_argument("--val_size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2601)
    ap.add_argument("--anti_ratio", type=float, default=0.60)
    ap.add_argument("--echo_ratio", type=float, default=0.20)
    ap.add_argument("--replay_ratio", type=float, default=0.20)
    ap.add_argument("--echo_gibberish_mode", type=str, default="include", choices=["include", "exclude", "only"])
    ap.add_argument("--strict_safe_samples", type=int, default=1500,
                    help="Number of strict safe-output anti-echo samples to inject (recommended 1000-2000)")
    ap.add_argument("--shuffle", action="store_true", default=True)
    ap.add_argument("--no_shuffle", action="store_false", dest="shuffle")
    args = ap.parse_args()

    # Validate ratios
    total_ratio = args.anti_ratio + args.echo_ratio + args.replay_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Build tokenizer for BPE-safe gibberish in echo generator.
    tok_cfg = GPTConfig(vocab_size=50304)
    tok = Tokenizer(tok_cfg, "gpt2")

    # Oversample generated pairs, then select exact anti/echo budgets.
    needed_anti = int(args.target_train_size * args.anti_ratio)
    needed_echo = int(args.target_train_size * args.echo_ratio)
    needed_replay = args.target_train_size - needed_anti - needed_echo

    oversample_n = max(args.target_train_size * 3, 120000)
    pairs = generate_echo_pairs(
        max_examples=oversample_n,
        seed=args.seed + 11,
        gibberish_mode=args.echo_gibberish_mode,
        bpe_safe=True,
        max_target_tokens=4,
        anti_echo_ratio=0.60,
        contrast_ratio=0.35,
        tokenizer=tok,
    )

    anti_pairs = [p for p in pairs if "anti_echo" in p[2]]
    echo_pairs = [p for p in pairs if "anti_echo" not in p[2]]
    # Inject strict-safe control samples (fixed outputs, nonce-heavy prompts).
    strict_pairs = _build_strict_safe_pairs(args.strict_safe_samples, args.seed + 23)
    anti_pairs.extend(strict_pairs)
    print(f"Injected strict-safe anti-echo samples: {len(strict_pairs):,}")

    needed_anti_total = needed_anti + (args.val_size // 2)
    if len(anti_pairs) < needed_anti_total:
        deficit = needed_anti_total - len(anti_pairs)
        synth = _synthesize_anti_pairs(tok, deficit, args.seed + 29)
        anti_pairs.extend(synth)
        print(f"Anti-echo pool low; synthesized {len(synth):,} extra anti-echo pairs")
    if len(anti_pairs) < needed_anti_total:
        raise RuntimeError(
            f"Not enough anti-echo pairs ({len(anti_pairs)}) for train+val target {needed_anti_total}"
        )
    needed_echo_total = needed_echo + args.val_size
    if len(echo_pairs) < needed_echo_total:
        deficit = needed_echo_total - len(echo_pairs)
        synth_echo = _synthesize_echo_pairs(tok, deficit, args.seed + 31)
        echo_pairs.extend(synth_echo)
        print(f"Echo pool low; synthesized {len(synth_echo):,} extra echo pairs")
    if len(echo_pairs) < needed_echo_total:
        raise RuntimeError(f"Not enough echo pairs ({len(echo_pairs)}) for train+val target {needed_echo_total}")

    # Build val first, then enforce payload-disjoint train pools.
    val_half = args.val_size // 2
    anti_val_pairs = rng.sample(anti_pairs, val_half)
    anti_val_keys = {(q, a, c) for (q, a, c) in anti_val_pairs}
    anti_val_payloads = {_pair_payload_key(p) for p in anti_val_pairs}

    # Avoid exact reuse and payload overlap against anti val
    anti_train_pool = [
        p for p in anti_pairs
        if (p[0], p[1], p[2]) not in anti_val_keys
        and _pair_payload_key(p) not in anti_val_payloads
    ]

    # Pick echo val with payloads not colliding with anti val payloads
    echo_val_candidates = [p for p in echo_pairs if _pair_payload_key(p) not in anti_val_payloads]
    needed_echo_val = args.val_size - len(anti_val_pairs)
    if len(echo_val_candidates) < needed_echo_val:
        deficit = needed_echo_val - len(echo_val_candidates)
        synth_more = _synthesize_echo_pairs(tok, deficit + 1000, args.seed + 41)
        # Keep only payloads not colliding with anti val payloads
        synth_more = [p for p in synth_more if _pair_payload_key(p) not in anti_val_payloads]
        echo_pairs.extend(synth_more)
        echo_val_candidates.extend(synth_more)
        print(f"Echo val pool low; synthesized {len(synth_more):,} extra candidates")
    if len(echo_val_candidates) < needed_echo_val:
        raise RuntimeError(f"Not enough echo val candidates ({len(echo_val_candidates)}) for target {needed_echo_val}")
    echo_val_pairs = rng.sample(echo_val_candidates, needed_echo_val)
    echo_val_keys = {(q, a, c) for (q, a, c) in echo_val_pairs}
    echo_val_payloads = {_pair_payload_key(p) for p in echo_val_pairs}
    val_templates = {_pair_template_signature(p) for p in anti_val_pairs}
    val_templates.update(_pair_template_signature(p) for p in echo_val_pairs)

    val_payloads = anti_val_payloads | echo_val_payloads

    # Echo train pool: avoid val payload collisions and exact tuple reuse.
    echo_train_pool = [
        p for p in echo_pairs
        if (p[0], p[1], p[2]) not in echo_val_keys
        and _pair_payload_key(p) not in val_payloads
        and _pair_template_signature(p) not in val_templates
    ]
    # Anti train pool: avoid val payload collisions and exact tuple reuse.
    anti_train_pool = [
        p for p in anti_train_pool
        if _pair_payload_key(p) not in val_payloads
        and _pair_template_signature(p) not in val_templates
    ]

    if len(anti_train_pool) < needed_anti:
        deficit = needed_anti - len(anti_train_pool)
        synth = _synthesize_anti_pairs(
            tok,
            deficit + 1000,
            args.seed + 49,
            forbidden_templates=val_templates,
            template_tag="P26_AE_TRAIN2",
        )
        synth = [p for p in synth if _pair_payload_key(p) not in val_payloads]
        anti_train_pool.extend(synth)
        print(f"Anti train pool low after val split; synthesized {len(synth):,} extra")
    if len(echo_train_pool) < needed_echo:
        deficit = needed_echo - len(echo_train_pool)
        synth = _synthesize_echo_pairs(
            tok,
            deficit + 1000,
            args.seed + 51,
            forbidden_templates=val_templates,
            template_tag="P26_ECHO_TRAIN2",
        )
        synth = [p for p in synth if _pair_payload_key(p) not in val_payloads]
        echo_train_pool.extend(synth)
        print(f"Echo train pool low after val split; synthesized {len(synth):,} extra")

    if len(anti_train_pool) < needed_anti:
        raise RuntimeError(f"Not enough anti train pairs ({len(anti_train_pool)}) for target {needed_anti}")
    if len(echo_train_pool) < needed_echo:
        raise RuntimeError(f"Not enough echo train pairs ({len(echo_train_pool)}) for target {needed_echo}")

    anti_train_pairs = rng.sample(anti_train_pool, needed_anti)
    echo_train_pairs = rng.sample(echo_train_pool, needed_echo)

    # Replay sampling from operator train set
    replay_path = (PROJECT_ROOT / args.replay_file).resolve()
    replay_rows = _read_jsonl(replay_path)
    if len(replay_rows) < needed_replay:
        raise RuntimeError(f"Replay file too small: {len(replay_rows)} < {needed_replay}")
    # Prevent payload leakage from replay into val.
    replay_filtered = []
    for r in replay_rows:
        payload = r.get("_meta", {}).get("payload")
        if payload is not None and payload in val_payloads:
            continue
        replay_filtered.append(r)
    if len(replay_filtered) < needed_replay:
        raise RuntimeError(
            f"Replay file too small after val-payload filtering: "
            f"{len(replay_filtered)} < {needed_replay}"
        )
    replay_sample = rng.sample(replay_filtered, needed_replay)
    for r in replay_sample:
        r.setdefault("_meta", {})
        r["_meta"]["phase"] = "phase2_replay"

    anti_eps = _pairs_to_episodes(anti_train_pairs, 0, "phase2_6_anti_echo", "ANTI_ECHO")
    echo_eps = _pairs_to_episodes(echo_train_pairs, len(anti_eps), "phase2_6_echo", "ECHO")

    train_rows: List[Dict] = []
    train_rows.extend(anti_eps)
    train_rows.extend(echo_eps)
    train_rows.extend(replay_sample)
    if args.shuffle:
        rng.shuffle(train_rows)

    # Val composition: fixed previously selected sets (payload-disjoint by construction)
    val_rows = _pairs_to_episodes(anti_val_pairs, 900000, "phase2_6_anti_echo_val", "ANTI_ECHO")
    val_rows.extend(_pairs_to_episodes(echo_val_pairs, 910000, "phase2_6_echo_val", "ECHO"))
    if args.shuffle:
        rng.shuffle(val_rows)

    train_file = out_dir / "phase2_6_mixed_train.jsonl"
    val_file = out_dir / "phase2_6_val.jsonl"
    _write_jsonl(train_file, train_rows)
    _write_jsonl(val_file, val_rows)

    meta = {
        "seed": args.seed,
        "target_train_size": args.target_train_size,
        "ratios": {
            "anti": args.anti_ratio,
            "echo": args.echo_ratio,
            "replay": args.replay_ratio,
        },
        "counts": {
            "anti_train": len(anti_eps),
            "echo_train": len(echo_eps),
            "replay_train": len(replay_sample),
            "train_total": len(train_rows),
            "val_total": len(val_rows),
        },
        "sources": {
            "replay_file": str(replay_path),
            "train_file": str(train_file),
            "val_file": str(val_file),
        },
    }
    lineage = merge_lineage(
        inputs=[{
            "path": str(replay_path),
            "sampled_rows": len(replay_sample),
            "effective_ratio": len(replay_sample) / max(1, len(train_rows)),
        }],
        output_rows=len(train_rows),
        creation_context={
            "timestamp": iso_now(),
            "script": "scripts/sft/prepare_phase2_6_antiecho.py",
            "args": vars(args),
            "synthetic_components": [
                {"name": "anti_echo_train", "rows": len(anti_eps)},
                {"name": "echo_train", "rows": len(echo_eps)},
            ],
        },
    )
    meta["lineage"] = lineage
    meta_file = out_dir / "phase2_6_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=== Phase 2.6 dataset ready ===")
    print(f"Train: {train_file} ({len(train_rows):,})")
    print(f"Val:   {val_file} ({len(val_rows):,})")
    print(f"Meta:  {meta_file}")
    lineage_path = write_lineage_sidecar(train_file, lineage)
    print(f"Lineage: {lineage_path}")
    print("\nNext step:")
    print(
        "python scripts/sft/prepare_chat_sft.py "
        "--input data/sft_phase2_6_intermediate/phase2_6_mixed_train.jsonl "
        "--output_dir data/sft_phase2_6_antiecho "
        "--val_file data/sft_phase2_6_intermediate/phase2_6_val.jsonl "
        "--no_system_prompt --enable_packing --pack_block_size 4096 --pack_by_field \"_meta.operator\""
    )


if __name__ == "__main__":
    main()
