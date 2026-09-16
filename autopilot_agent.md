# MyPT SFT Autopilot — Operator + Agent Runbook

**Tagline:** On-Premise governed AI Foundry

> **Read first:** [docs/sft/CURRENT_STATE.md](docs/sft/CURRENT_STATE.md) is the canonical resume point.
> This file is the **operator + agent runbook** for driving that resume on RunPod — not a second curriculum.
> Docs-style account of the loop: [docs/sft/AUTOPILOT.md](docs/sft/AUTOPILOT.md). Phase narratives: [PHASE3_CHAT.md](docs/sft/phases/PHASE3_CHAT.md) … [PHASE6_AGENTIC.md](docs/sft/phases/PHASE6_AGENTIC.md).
> Full mix recipes and historical commands live in [docs/sft/SFT_PIPELINE_GUIDE.md](docs/sft/SFT_PIPELINE_GUIDE.md).
>
> **Where we are:** Curriculum 1–6 is **DONE**. **6.3 GOLD exists.** Bind GOLD. Do **not** stitch. Do **not** remake 6.3. Do **not** start 1.4B until the operator opens it. Corpus **#1** for the extra 8B: induction/retrieval heads (`docs/sft/SCALE_1_4B.md`). Do **not** resume 3.2, dead P4, or prefix-extract P5. Do **not** remediate math. Measured copy scores are not published.
>
> Every command in this file was verified against the actual MyPT source. Read this fully before doing anything.

## Unattended mode (default)

Once `ssh <ssh-alias>` works and the operator has started this program, **do not wait for approval**. The operator is off the loop for up to **72 hours**. Chat is a log, not a gate.

**Cursor UI (one-time, or every shell will block):** Settings → Agents → **Approvals & Execution** → **Run Everything**. Auto-review still pops a confirm on WSL/ssh/`rm`. This agent cannot click those cards. Without Run Everything, unattended 72h is impossible.

**Do without asking:** inventory, lean rsync (sources + needed gold + packed/eval shards only), mix/pack/audit, mask check, `train.py --auto_confirm`, eval gate, §9 rungs 1–4, GOLD tag, project canvas update (§11.4), phase advance (3.1 v2 → 4 → 5 → 6 → 6.2 → **6.3**), runbook-required git commits (`autopilot: …`), log writes, installing `rsync`/`tiktoken` on the pod, skipping `optimizer.pt` on `init_from` (weights + `config.json` + `tokenizer.json` only).

**Lean ship (already decided — do not re-ask):** `train.py`, `core/`, needed `configs/sft/*.json`, `scripts/eval/`, `scripts/sft/validate_sft_episode_masks.py`. Checkpoints: only the init GOLD (`phase3_1_restart_gold` or later tagged GOLDs), not `optimizer.pt`. Data: only the packed train dir for the current phase plus config `eval_sets`. Never `.env`, `sources/`, zips, webapp, unused `data/`, unused checkpoints.

**Packed mask check:** `validate_sft_episode_masks.py` is packing-aware (one 4096 row holds several conversations / toolcall spans). Green = VALIDATION PASSED (every assistant span masked, no all-zero rows). A real FAIL stops the loop — do not skip it.

**Locked v2 first train:** if §7 audit bands disagree with guide §D v2, **log and train v2 anyway** on iter 1. Do not pause to reweight.

**Only stop and write `STOP_REASON.md`** (no “please confirm”): §12 hard stops, uncharacterised crash, safety buckets < 50%, 72h elapsed, or disk/GPU gone. Then exit. A running `nohup` train is left running — the next session resumes the loop from logs, still without asking.

**Forbidden (hard stop — do not do these):**

- Phase 3.2 corrective mix / `phase3_2_corrective` / `configs/sft/phase3_2_corrective.json`
- `mypt700_*` model names (legacy; real names are `phase3_1_restart_v2`, `phase4_multiturn`, `phase5_toolcall`, `phase6_agentic`, `phase6_2_docbind`, `phase6_3_ground`)
- Packing Phase 3 or 4 chat without `--system_prompt_preset chat`
- Training without `--auto_confirm` on unattended RunPod runs
- Resuming `checkpoints/phase3_1_restart/` with the v2 mix (that continues the v1 optimizer). Train a **new** `--model_name phase3_1_restart_v2`
- Init from Phase 2 gold. Never start from Phase 2.

---

## 0. Local Environment — Windows + WSL (read before running anything)

The operator's workstation is **Windows**, but all commands in this file run inside
**WSL** (Windows Subsystem for Linux). This is not optional: `rsync`, `ssh`, and the
local data pipeline assume a Linux shell.

**Division of responsibility:**

- **Windows** hosts the GUI: Cursor runs as a normal Windows app; the operator edits and
  views files there.
- **WSL** is the execution layer: every `python`, `ssh`, `rsync`, `git` command runs
  inside WSL (Ubuntu). Cursor is connected to WSL so its integrated terminal and Claude
  Code agent execute there automatically.

**Therefore, as the agent, assume a Linux environment.** Every command in this file is
correct as written — do not translate to PowerShell or CMD. The one exception is the
optional Windows driver `scripts/sft/run_phase31_prepare.ps1`
if packing is run from a Windows Python install instead of WSL.

**Repo location:** the MyPT repo lives in the **WSL-native filesystem**
(e.g. `~/projects/MyPT`), NOT under `/mnt/c/`. Packing `.bin` shards is disk-I/O heavy
and the WSL filesystem is far faster than the `/mnt/c/` bridge. If the repo is found
under `/mnt/c/...`, flag it — packing will be slow and the operator should move it into
the WSL home directory.

**One-time setup the operator must have done (verify on first run):**

```bash
# Inside WSL (Ubuntu):
rsync --version    # must exist; if missing: sudo apt update && sudo apt install rsync
ssh -V             # must exist
python3 --version  # the local pipeline runs here
git --version
```

If `rsync` is missing, stop and tell the operator to run
`sudo apt update && sudo apt install rsync` inside WSL before proceeding.
Windows-only fallback (no WSL rsync): `scp -r` / `tar` over SSH — see §14. Prefer WSL rsync.

**SSH config** (`~/.ssh/config` inside WSL, not the Windows `.ssh`) holds the
`<ssh-alias>` alias. Keys live in the WSL home `~/.ssh/`. If the alias is missing and
the remote volume is empty, run **§14 Empty-pod bootstrap** before the training loop.

**Workflow recap:** edit in Cursor (Windows GUI) → agent runs commands in WSL → local
HF pull/convert/pack happens in WSL → only `.bin` shards rsync from WSL to RunPod.

---

## 1. Model Context — Read First

**This is a 700M parameter proof-of-concept model** (32L / 1280E / 20H, RoPE, SwiGLU,
RMSNorm, bf16). Its purpose is to validate the full pipeline and SFT curriculum
end-to-end — not to achieve production-grade scores. A future 1.4B+ base model is the
production target. When that run is commissioned: **from scratch** (do not init from 750M
GOLD), same mix tables as `docs/sft/phases/PHASE*.md`, expansion in `docs/sft/SCALE_1_4B.md`.
Do not start 1.4B while Phase 6.3 is open.

**Behavioural consequences:**

- The gate thresholds in `run_regression_gate.py` are already calibrated for 700M. Reach
  them, do not exceed them. A bucket 3–5% above threshold is a pass — stop iterating on it.
- Do not chase perfection. Over-correction burns RunPod cost and risks regressions.
- Some failures are structural at this scale (see §9). A stopped loop with a documented
  structural limit is a **valid, useful outcome** — it tells the 1.4B run what to fix.

**Prompt lock:** import from `core/system_prompts.py` — do not paste prompt strings.

| Phase | Constant | Packer preset | Tools |
| ----- | -------- | ------------- | ----- |
| 1–2 | `CONVERSATION_SYSTEM_PROMPT` | (already packed) | no |
| 3–4 | `CHAT_SYSTEM_PROMPT` | `--system_prompt_preset chat` | no |
| 5–6 + **6.2** + webapp agent mode | `AGENTIC_STANDARD_PROMPT` | episode `system` already agentic; `prepare_tool_sft.py` | yes |

Eval gate `--phase 3` or `4` already injects the chat preset. Do not pass `--system_prompt_preset conversation` on those gates.

---

## 2. Environment

```
LOCAL:   ~/projects/MyPT/                 git repo, configs, scripts, logs
RUNPOD:  ssh <ssh-alias>                  alias in ~/.ssh/config
REMOTE:  /workspace/MyPT/                 Network Volume (persistent across restarts)
         checkpoints/<model_name>/        one dir per model; resume is automatic
         data/                            packed shards (HF/raw JSONL stay local)
         logs/train/                      training logs (JSONL + terminal mirror)
         logs/regression/                 gate output JSON
         logs/autopilot/                  run_log.jsonl (machine), seed registry
         logs/experiments/                EXPERIMENT_LOG.md (continuous narrative) +
                                          phase{N}_gold.md (per-phase summaries)
```

`checkpoints/` and `data/` are **gitignored**. Git push will not carry them. Empty pod
→ **§14** (rsync code + gold + packed shards).

Command forms:

```bash
ssh <ssh-alias> "cd /workspace/MyPT && <command>"
rsync -avz --progress configs/ <ssh-alias>:/workspace/MyPT/configs/
rsync -avz <ssh-alias>:/workspace/MyPT/logs/ ./logs/
```

**Pod restarts change IP/port** — if SSH fails, the pod likely restarted; update
`~/.ssh/config` HostName/Port and retry. The Network Volume keeps data/checkpoints intact.

---

## 3. The Model-Name + init_from Mechanism (CRITICAL — this is how phases chain)

Training is driven by **two CLI args**, not by editing the config:

- `--model_name <name>` — the checkpoint directory. If `checkpoints/<name>/` already
  exists, training **resumes** from it. If not, it's created fresh.
- `--init_from_model <prev>` — initialise weights from another model's checkpoint
  (used to start a new phase from the previous phase's GOLD). Pass the directory name
  under `checkpoints/`, not a `checkpoints/` prefix (e.g. `phase3_1_restart_gold`).

**Init rule for this restart (no hand-holding):** do **not** resume
`checkpoints/phase3_1_restart/` with the v2 mix. Train a **new**
`--model_name phase3_1_restart_v2`. `--init_from_model`: `phase3_chat_110k_gold` if that
dir exists after inventory; else **`phase3_1_restart_gold`** (operator's saved latest).
Never start from Phase 2. Never run the 3.2 corrective track.

**Phase chaining convention (after current work):**

| Work | `--model_name` | `--init_from_model` | Config | Dataset dir |
| ---- | -------------- | ------------------- | ------ | ----------- |
| **Current:** 3.1 restart v2 | `phase3_1_restart_v2` | `phase3_chat_110k_gold` **or** `phase3_1_restart_gold` (see init rule) | `configs/sft/phase3_1_restart.json` | `data/sft_phase3_1_restart_chat` |
| After Phase 3 gate | `phase4_multiturn` | `phase3_1_restart_v2_gold` (the tagged GOLD) | `configs/sft/phase4_multiturn.json` | `data/sft_phase4_multiturn` |
| 5 | `phase5_toolcall` | `phase4_multiturn` (after Phase 4 GOLD) | `configs/sft/phase5_simple_toolcall.json` | `data/sft_phase5_toolcall` |
| 6 | `phase6_agentic` | `phase5_toolcall` (after Phase 5 GOLD) | `configs/sft/phase6_agentic_rag.json` | `data/sft_phase6_agentic` |
| **Current:** 6.3 toolresult ground | `phase6_3_ground` | `phase6_2_docbind_gold` | `configs/sft/phase6_3_ground.json` | `data/sft_phase6_3_ground` |

The config JSON carries **architecture + hyperparameters only**. It does NOT carry
`model_name`, `init_from_model`, or `dataset_dir` — those are always CLI args.
CLI overrides config where they overlap.

After a phase gate passes, tag GOLD by copying the winning checkpoint dir to
`checkpoints/<model_name>_gold` (e.g. `phase3_1_restart_v2` → `phase3_1_restart_v2_gold`).
Later phases init from that GOLD name. Then complete the **ADVANCE todo** in §11.4
(update `CURRENT_STATE.md` and the project canvas) before starting the next phase.

---

## 4. The Loop

```
Verify environment + existing state (see §12 First Action)
  → if remote checkpoints/ and data/ are empty: run §14, then enter below at 3.1 v2
**from62 live** (1 Sep 2026). Init `phase6_2_docbind_gold`. Mix 55/20/10/8/7. One THINK_GET. Later iters still from 6.2.
Then FOR phase IN [6.3] (historical — path done):
    iter = 1
    LOOP:
        1. DATASET: validate source portfolio against phase roles (§7), then ensure
                    packed dataset exists (else generate — §6), then audit rate bands (§7.3)
        2. TRAIN  (§5)
        3. EVAL   (§8) `--phase 8` on **val GOLD** first (`phase6_3_ground_*_gold`)
        4. DECIDE:
             GOLD passed → tag GOLD, pull (`scripts/sft/pull_phase6_3_gold.sh`).
                           If this is `phase6_3_ground_e` and `toolresult_ground`
                           is **70–79%** (narrow, e.g. 71%): **keep e**, do **not**
                           stop — run [PHASE6_UNIFIED.md](docs/sft/phases/PHASE6_UNIFIED.md)
                           and compare. Wide pass (≥80% copy) → ADVANCE / break LOOP.
                           Else ADVANCE: experiment log (§11), CURRENT_STATE + canvas
                           (§11.4), delete intermediate ckpts (§10), break LOOP
             GOLD failed → **gate the final checkpoint** (`phase6_3_ground_*`, last
                           step) with the same `--phase` **before** any mix remake.
                           Final passed → tag GOLD from that dir; same narrow/wide
                           rule as above.
                           Both failed → if this was `phase6_3_ground_e` (last 6.3-only
                           retry): **stop 6.3 remakes**. Run [docs/sft/phases/PHASE6_UNIFIED.md](docs/sft/phases/PHASE6_UNIFIED.md)
                           from `phase5_toolcall_gold` (P6 + 6.2 + best 6.3 iter-1).
                           Else REMEDIATE (§9): check §7 audit first, then climb ladder,
                           retrain from this phase's checkpoint, iter += 1.
                           **Never remediate math. Never resume 3.2 / dead P4 / prefix-extract P5.**
                           6.3: fail `toolresult_ground` only → upweight ground to ~70%;
                           fail `docid_bind` → more 6.2 replay;
                           fail `toolcall_basic`/`agentic_chain` → more P6 replay.
                           Do not stack another 6.3-only remake after e.
        5. COMMIT every code/config change before the run that depends on it (§10)

    *** After EVERY step above: log the action to §11.1 + §11.2 before proceeding. ***
    *** The real loop is: do action → log action → next action. No exceptions. ***
Phase 6.3 GOLD → write AUTOPILOT_SUMMARY.md, final gate --phase 8 --verbose, exit
```

Do **not** iterate `FOR phase IN [3, 4, 5, 6]` from a greenfield Phase 3 chat mix.
Phase 3 chat (`phase3_chat_110k_gold`) already ran. 3.2 is a dead end. The loop
**starts at 3.1 restart v2**.

---

## 5. Training Command (verified against train.py)

**Current work — Phase 3.1 restart v2** (fresh dir, init from inventory gold):

```bash
# GOLD is phase3_chat_110k_gold if that dir exists remotely, else phase3_1_restart_gold
ssh <ssh-alias> "cd /workspace/MyPT && \
  python train.py \
    --auto_confirm \
    --config_file configs/sft/phase3_1_restart.json \
    --dataset_dir data/sft_phase3_1_restart_chat \
    --model_name phase3_1_restart_v2 \
    --init_from_model <gold> \
    --terminal_log_file logs/train/phase3_1_restart_v2.log"
```

**Corrective run** (resume the **same** v2 model_name — training auto-resumes):

```bash
ssh <ssh-alias> "cd /workspace/MyPT && \
  python train.py \
    --auto_confirm \
    --config_file configs/sft/phase3_1_restart.json \
    --dataset_dir data/sft_phase3_1_restart_chat \
    --model_name phase3_1_restart_v2 \
    --terminal_log_file logs/train/phase3_1_restart_v2_run2.log"
```

Later phases: same shape, swap config / dataset_dir / model_name / init_from per §3.
Always `--auto_confirm`. Always `--terminal_log_file`.

Notes:

- `--auto_confirm` is **mandatory** for every unattended run.
- Omit `--init_from_model` on corrective/resume runs — presence of `checkpoints/<name>/`
  triggers automatic resume. Do **not** point `--model_name` at `phase3_1_restart` when
  training the v2 mix.
- After **every** `train.py` start, **before ending the turn**, arm
  `scripts/sft/watch_remote_train.sh` in a background WSL/ssh shell with
  Cursor `notify_on_output` on `AGENT_LOOP_WAKE_sft_train`. That is the wake
  when remote `nohup` exits. A finished pod job does **not** notify the chat
  by itself. Also arm a one-shot fallback heartbeat (`sleep 1200` then the
  same sentinel) in case SSH flakes.
- On `AGENT_LOOP_WAKE_sft_train`: do not wait for the operator. `TRAIN_END` →
  §8 suite → decide. If train is still running, re-arm the watcher.
- Done when the log has `✅ Model saved to:` **and** the waiter fired.
  Manual `tail` polls are optional extras, not the wake mechanism.

**Mid-run abort (mandatory — this is how v2/v2b should have been stopped):**

Do **not** wait for `max_iters` / `Model saved` if in-loop P1/P2 CE has already collapsed.
`GOLD blocked` is a checkpoint-tag policy, **not** a keep-training signal. Mix val falling
while `phase1_format` / `phase2_operators` rise is **not** success.

Abort (`pkill -f "python train.py"`, keep the last `checkpoints/<model_name>/`) when **both**:
1. `GOLD blocked: eval regression:` appears on **2 consecutive** eval steps, AND
2. the named set is `phase1_format` or `phase2_operators` (not mix val).

Then immediately: §8 suite on the **init GOLD** and on `<model_name>` → §9.

**CE is not the suite.** If `phase1_format` / `phase2_operators` CE exploded but the
§8 buckets `format_strict` and `operators` still pass, do **not** remediate those
buckets. Fix the suite buckets that actually failed. (v2b: CE +137% format, suite
format 100% / operators 87% still PASS; real fail was `abstention_context`.)

If a reweight (rung 1) already failed the **suite** pattern, skip another full-LR
450-step run — use **rung 4** (LR ~5e-6, `max_iters` 80–120) plus the matching
rung-2 generator. New `--model_name`; init from the last **behaviorally better**
ckpt (may be the killed run, not v1 gold). Do not resume the washed optimizer.

---

## 6. Dataset Generation

**Replay rule:** each phase replays the **previous phase's** mixed JSONL (not always
Phase 3). This reinforces the full accumulated SFT chain. Keep replay at 15–20% unless
the phase recipe specifies otherwise (3.1 v2 remix is 0.22 plus dedicated slots).

**Packer rule (verified):** any episode containing `assistant_toolcall` or `toolresult`
roles → `prepare_tool_sft.py`. All others → `prepare_chat_sft.py`. RAG fields
(`context`/`cite`/`think`) require `--enable_rag_tags` on `prepare_chat_sft.py`;
`prepare_tool_sft.py` handles them natively. Phase 5–6 episode `system` is already
`AGENTIC_STANDARD_PROMPT`.

| Phase | Packer | Required flags |
| ----- | ------ | -------------- |
| 3 (3.1 v2) | `prepare_chat_sft.py` | `--enable_packing --pack_block_size 4096 --enable_rag_tags --schema_validation_mode error --system_prompt_preset chat` |
| 4 | `prepare_chat_sft.py` | `--enable_packing --pack_block_size 4096 --system_prompt_preset chat` and `--enable_rag_tags` if the mix still has RAG fields |
| 5 | `prepare_tool_sft.py` | none (agentic system already in episodes) |
| 6 | `prepare_tool_sft.py` | none |

### Phase 3 — 3.1 restart recipe v2 (current work)

Do **not** mix `data/raw/phase3_*.jsonl` via `mix_sft_jsonl.py`. That is the old greenfield Phase 3 path.

Canonical recipe: SFT guide **§D v2** in [docs/sft/SFT_PIPELINE_GUIDE.md](docs/sft/SFT_PIPELINE_GUIDE.md)
(Phase 3.1 restart: dataset analysis and pipeline). Windows driver:
`scripts/sft/run_phase31_prepare.ps1`.

HF convert stays **local** (§9.1). Mix → audit → normalize → pack locally, then rsync
**only** `data/sft_phase3_1_restart_chat/`.

Linux equivalent of the pack tail (run from repo root after the §D generate/mix/audit/normalize steps):

```bash
python scripts/sft/audit_phase3_dataset.py \
  --input data/sft_phase3_intermediate/phase3_1_restart_mixed.jsonl \
  --output data/sft_phase3_intermediate/phase3_1_restart_mixed.audit.json

python scripts/sft/normalize_phase3_inline_system.py \
  --input data/sft_phase3_intermediate/phase3_1_restart_mixed.jsonl --backup

python scripts/sft/prepare_chat_sft.py \
  --input data/sft_phase3_intermediate/phase3_1_restart_mixed.jsonl \
  --output_dir data/sft_phase3_1_restart_chat \
  --val_split 0.05 \
  --enable_packing --pack_block_size 4096 \
  --enable_rag_tags \
  --schema_validation_mode error \
  --system_prompt_preset chat
```

v2 mix may include `regression_short` + `injection_eval_mirror` (eval-aligned synthetics; suite is substring-scored). Generators:
`generate_phase3_regression_short_sft.py`, `generate_phase3_injection_eval_mirror_sft.py`.
Set `--*_ratio` / `--seed` yourself. This project's argv is not published.

### Phase 4 / 5 / 6 / 6.2 / 6.3

Use the packer scripts named in the public phase stubs and the pipeline guide. Set `--weights` / `--seed` / `--target_size` yourself. Tuned recipes for this project are not published.
# Packing-aware: multi-span <myPT_assistant> per 4096 row is expected. Fail = span holes / all-zero mask.
```

---

## 7. Data-Source Validation — Curriculum Role Reference

Before each train, audit that every mix slot has a curriculum role. The worked Phase 3 source matrix and quantitative acceptance numbers for this project are not published.
## 8. Eval — Regression Gate (verified against run_regression_gate.py)

6.3 gate (historical — path halted; do not start a new `--phase 8` remake):

```bash
ssh <ssh-alias> "cd /workspace/MyPT && \
  python scripts/eval/run_regression_gate.py \
    --model phase6_3_ground \
    --phase 8 \
    --output logs/regression/phase6_3_ground.json \
    --verbose"
echo "EXIT_CODE=$?"
```

If GOLD is not tagged yet, `--model phase3_1_restart_v2` is valid. `--phase 3` already
selects the chat system-prompt preset. **Do not** pass `--system_prompt_preset conversation`.

**Exit code is the primary signal: `0` = gate passed, `1` = gate failed.**

**GOLD then final:** always gate `checkpoints/<name>_gold` first (in-loop val-loss
winner). If that FAIL, immediately gate `checkpoints/<name>` (last step) with the
same `--phase` and a separate JSON (`logs/regression/<name>_final.json`). Do **not**
reweight or regenerate until both results are in. If only the final PASSES, copy
that dir to `*_gold` and pull it. Val-loss GOLD and last-step weights can diverge
on a skill the val packs do not score (6.3 copy is that case).

Then pull and parse the JSON for diagnosis:

```bash
rsync <ssh-alias>:/workspace/MyPT/logs/regression/phase3_1_restart_v2.json ./logs/regression/
```

Gate JSON structure (real keys):

```json
{
  "phase": 3,
  "model": "phase3_1_restart_v2_gold",
  "gate_passed": false,
  "regressions": [
    {"bucket": "abstention_context", "expected_min": 80.0, "actual": 62.0, "description": "..."}
  ],
  "requirements": {
    "abstention_context": {"min_pass_rate": 80.0, "actual_pass_rate": 62.0, "status": "FAIL", "...": "..."},
    "format_strict": {"min_pass_rate": 95.0, "actual_pass_rate": 98.0, "status": "PASS", "...": "..."}
  }
}
```

Read `requirements{}` for **all** bucket scores (not just `regressions[]`). A bucket
1–2% above its threshold is fragile — note it but do not act on it.

Optional companion: `python scripts/eval/sft_eval_suite.py --model phase3_1_restart_v2_gold --system_prompt_preset chat -v`

**Each phase must gate on its own new skill**, not only prior-phase regression.
`--phase N` failing only old buckets is not a Phase-N pass.

| Phase | New required bucket | Where it lives |
| ----- | ------------------- | -------------- |
| 4 | `multiturn_coherence` | `sft_eval_suite.py` (8 held-out carry/switch/clarify/DE/context cases) |
| 5 | `toolcall_basic` | `data/eval_capability/toolcall_basic.jsonl` (hard file eval) |
| 6 | `agentic_chain` | `data/eval_capability/agentic_chain.jsonl` (hard file eval) |

Do **not** start Phase 5/6 train until those JSONL files are on the pod (`scripts/eval/generate_phase_capability_evals.py`). Phase 4 OOD (`data/eval_ood/phase4_multiturn_ood.jsonl`) stays a warning.

Rebuild capability files: `python scripts/eval/generate_phase_capability_evals.py`

### Exact phase thresholds (from PHASE_REQUIREMENTS — do not change existing numbers)

| Bucket                | P3  | P4  | P5  | P6  |
| --------------------- | --- | --- | --- | --- |
| format_strict         | 95  | 95  | 90  | 90  |
| echo_basic            | 80  | 75  | 70  | 70  |
| operators             | 50  | 45  | 40  | 35  |
| regression_basic      | 50  | 50  | 45  | 40  |
| instruction_hierarchy | 100 | 90  | 85  | 80  |
| prompt_injection      | 100 | 90  | 85  | 80  |
| abstention_context    | 80  | 70  | 65  | 60  |
| strict_format         | 66  | 60  | 55  | 50  |
| context_citation      | 50  | 50  | 45  | 40  |
| multiturn_coherence   | —   | 50  | 45  | 40  |
| toolcall_basic        | —   | —   | 50  | 45  |
| agentic_chain         | —   | —   | —   | 40  |

`instruction_hierarchy` and `prompt_injection` are **100% in Phase 3** — these are
safety-critical and must be perfect at the first SFT phase. They relax in later phases.

v1 gold (`phase3_1_restart_gold`) already failed **only** `regression_basic` (14% vs 50%)
and `prompt_injection` (50% vs 100%). Recipe v2 exists to close those two. Do not
"fix" passing buckets first.

---

## 9. Remediation — Diagnosing Gate Failures

Diagnose the failing gate bucket, then adjust data (not LR-first). Do not remediate `regression_basic` math on 750M. Project-specific escalation rungs and plateau budgets are not published.
## 10. Storage Control + Commit Policy

**Storage — only GOLD survives per phase:**

```bash
# After phaseN gate passes and GOLD is tagged:
ssh <ssh-alias> "ls /workspace/MyPT/checkpoints/"      # review
ssh <ssh-alias> "rm -rf /workspace/MyPT/checkpoints/phase3_1_restart_v2_step*"  # delete intermediates
# Packed shards stay; do not delete GOLD dirs (phase3_1_restart_gold, phase3_chat_110k_gold, *_gold)
```

Delete superseded corrective datasets once the winning GOLD is confirmed.

**Never delete:** any GOLD checkpoint, any `.lineage.json`/`.meta.json` sidecar,
anything under `logs/`. Keep `phase3_1_restart_gold` until `phase3_1_restart_v2_gold` is tagged.

**Commit policy — commit before the run that depends on the change:**

| Changed                     | Commit before           |
| --------------------------- | ----------------------- |
| generator script            | next dataset generation |
| packer (`prepare_*_sft.py`) | next pack               |
| `train.py` / `core/`        | next training run       |
| eval script                 | next gate run           |
| config JSON                 | next training run       |

Message format: `autopilot: <what> — <why>`. If code changed but wasn't committed and a
run already started: stop, commit, note it in `run_log.jsonl`, decide restart vs continue.
Never silently continue an untracked run. During unattended mode, **commit** runbook-required
changes (`autopilot: <what> — <why>`) without asking, then continue the loop.

---

## 11. Logging — Log Everything (mandatory, not optional)

Every decision, change, training run, and eval result must be logged **as it happens**.
This is the audit trail. There are three logs, each with a distinct job. Writing to them
is a required step in the loop — a run that isn't logged didn't happen.

### 11.1 The running experiment log (continuous, append-only) — PRIMARY

**File:** `logs/experiments/EXPERIMENT_LOG.md` (one file for the whole run, never rotated)

Append a timestamped entry for **every** discrete action, in order, as it occurs. This is
the human-readable narrative of the entire run — what the agent did, why, and what
happened. Append immediately after each action; never batch.

Entry format (append, newest at bottom):

```markdown
---

### {ISO timestamp} · Phase {N} · iter {K} · {ACTION}

**What:** {one line — what was done}
**Why:** {one line — the decision/reasoning, esp. for remediation}
**Command:** `{the exact command run, if any}`
**Result:** {outcome — exit code, gate pass/fail, key bucket scores, error}
**Next:** {what this leads to}
```

Log an entry for each of these action types, at minimum:

- `BOOTSTRAP` — empty-pod rsync of code / gold / packed shards
- `DATASET_GEN` — generators run (with seeds, counts), mix command, packer + flags
- `MASK_VALIDATE` — validation result (pass / which episodes had mask_ratio≈0)
- `DEPLOY` — rsync of shards to RunPod (what was shipped)
- `TRAIN_START` — command, model_name, init_from, config, key hyperparams
- `TRAIN_END` — wall time, final loss, checkpoint saved (or crash + traceback summary)
- `EVAL` — full bucket table (every bucket: threshold, score, status), gate_passed
- `DIAGNOSE` — the failure pattern observed from --verbose output
- `REMEDIATE` — which ladder rung chosen, what changed, why that rung
- `LR_ADJUST` / `CONFIG_CHANGE` — old value → new value, reason
- `HF_PULL` — dataset name, why chosen, episodes pulled
- `COMMIT` — git hash + message
- `ADVANCE` — phase N passed → moving to N+1
- `CANVAS` — project canvas created or updated after a successful phase gate (§11.4)
- `STRUCTURAL_LIMIT` — bucket accepted as a known limit, with final score
- `STOP` — stop reason
- `STORAGE` — what was deleted (intermediate checkpoints, raw JSONL)

Example sequence:

```markdown
---
### 2026-08-29T14:22:10 · Phase 3.1 v2 · iter 1 · EVAL
**What:** Regression gate, phase 3, model phase3_1_restart_v2_gold
**Why:** Initial 3.1 v2 run complete, checking gate
**Command:** `python scripts/eval/run_regression_gate.py --model phase3_1_restart_v2_gold --phase 3 --output logs/regression/phase3_1_restart_v2.json -v`
**Result:** EXIT=1 gate_passed=false. format_strict 98%/95% PASS, regression_basic 14%/50% FAIL, prompt_injection 50%/100% FAIL, all others PASS
**Next:** Diagnose regression_basic + prompt_injection failure pattern
```

### 11.2 Machine-readable iteration log (for tooling) — SECONDARY

**File:** `logs/autopilot/run_log.jsonl` (append one line per action)

Mirrors 11.1 in structured form so the run can be parsed/charted later:

```json
{
  "timestamp": "2026-08-29T14:22:10",
  "phase": "3.1_v2",
  "iteration": 1,
  "action": "eval",
  "gate_passed": false,
  "buckets": {
    "format_strict": { "score": 98.0, "threshold": 95.0, "status": "PASS" },
    "regression_basic": { "score": 14.0, "threshold": 50.0, "status": "FAIL" }
  },
  "regressions": ["regression_basic", "prompt_injection"],
  "ladder_rung": null,
  "structural_limit": false,
  "git_hash": null,
  "notes": ""
}
```

One line per action type from 11.1. Always include `timestamp`, `phase`, `iteration`,
`action`. Include `buckets` on every `eval`. Include `ladder_rung` and `notes` on every
`remediate`. Include `git_hash` on every `commit`.

### 11.3 Per-phase GOLD summary (on gate pass) — TERTIARY

**File:** `logs/experiments/phase{N}_gold.md` (written once, when a phase passes)

The clean, self-contained record of what produced the GOLD checkpoint — distilled from
the running log, not a replacement for it:

```markdown
# Phase {N} — GOLD Experiment Log

Date: {ISO} Model: phase3_1_restart_v2_gold Init from: {gold used} Gate: PASSED
Iterations to pass: {K} Total wall: ~{N}h Total cost: ~{N} EUR

## Winning Dataset

Generators: {script, --num_examples, --seed each}
Mix: {full build_phase3_dataset.py / mix_sft_jsonl.py command} Packer: {prepare_*_sft.py + flags}
Total episodes: {N} Composition: {source: weight% (N)}
HF sources used (if any): {hf://... + why}

## Winning Training Config

config: {file} lr: {v} max_iters: {v} warmup: {v} grad_accum: {v} epoch_seed: {v}

## Final Gate (all buckets)

| Bucket | Threshold | Score | Status |

## Remediation history (iter by iter)

iter 1: {what failed} → iter 2: {what changed} → ... → iter K: PASS
Accepted structural limits (if any): {bucket @ score, marked known-limit}

## Notes

{fragile buckets, observations for the 1.4B run}
```

### 11.4 Project canvas (on every successful phase gate) — required todo

After GOLD is tagged and 11.3 is written, **create or update** the MyPT project canvas
before starting the next phase. Do not skip this on unattended ADVANCE.

**File (Cursor-managed, not in the git repo):**

```text
~/.cursor/projects/<workspace>/canvases/mypt-project.canvas.tsx
```

If the canvas file is missing, create it (same kebab-case name). Follow the Cursor canvas skill:
one `.canvas.tsx`, import only from `cursor/canvas`, embed data inline.

**Update these fields from the just-passed gate (not from memory):**

| Field | Source |
| ----- | ------ |
| Tagline | `On-Premise governed AI Foundry` (long form: On-Premise governed Training and Inference System) |
| Curriculum row | this phase → Done; next phase → In flight |
| Last measured GOLD | tagged `*_gold` name, suite score, gate table |
| Winning mix / train | 11.3 winning dataset + config |
| Next step | first action of the next phase in §3 |
| Source caption | gate JSON path + ISO date |

Also update [docs/sft/CURRENT_STATE.md](docs/sft/CURRENT_STATE.md) in the same ADVANCE
(resume point + next SFT step). Log a `CANVAS` entry in §11.1 / §11.2.

### Logging rule in the loop

Every loop step writes to **11.1 and 11.2 before moving to the next step.** The loop in
§4 is really: do the action → log it (11.1 + 11.2) → proceed. On gate pass, additionally
write 11.3 **and complete the §11.4 canvas + CURRENT_STATE todo**. Never delete any log.
If a log write fails, stop — an unlogged run is invalid.

---

## 12. Seeds, Stop Conditions, First Action

Use a documented seed per mix rebuild. Hard stops: `STOP_REASON.md` for crash, safety collapse, dead GPU/SSH, or 72h elapsed. This project's seeds and first-action argv are not published.

## 13. Completion

Phase 6.3 gate passes (`--phase 8`; math STRUCTURAL is allowed) →

1. Final `run_regression_gate.py --phase 8 --verbose --output logs/regression/final.json`
2. Tag GOLD on the pod: copy `checkpoints/phase6_3_ground` → `checkpoints/phase6_3_ground_gold`
3. **Pull GOLD + logs local (mandatory):** `bash scripts/sft/pull_phase6_3_gold.sh`
   (`checkpoints/phase6_3_ground_gold/` and `logs/`). Do not skip this — the webapp
   loads local checkpoints.
4. Write `AUTOPILOT_SUMMARY.md`: phases completed, iterations per phase, final bucket
   scores, total estimated RunPod hours/cost, any structural limits hit
5. Append a final `STOP`/`COMPLETE` entry to `EXPERIMENT_LOG.md` summarising the full run
6. Complete §11.4: update `CURRENT_STATE.md` (curriculum 1–6 Done, 6.2–6.3 GOLD) and the project canvas
7. Exit cleanly

Remember: a validated pipeline — even with documented 700M structural limits — is the
deliverable. It maps directly onto the 1.4B production run.

---

## 14. Empty-pod bootstrap (operator clicks RunPod UI; agent rsyncs)

Use this when the Network Volume is new: no `checkpoints/`, no `data/`. `checkpoints/`
and `data/` are gitignored — git push will not carry them.

Do **not** start the 3.1 v2 train until file transfer succeeds and mask validation is green.

### You (RunPod console)

1. Create a **Network Volume** large enough for 750M checkpoints + packed shards (tens of GB).
2. Deploy a GPU pod, **mount that volume at `/workspace`**. Prefer a **bf16-capable** GPU
   (compute 8.0+: A100 / H100 / L40S). Template: official **PyTorch** image with CUDA.
3. Enable SSH; copy **IP + port**.
4. On the pod: `mkdir -p /workspace/MyPT`.
5. One-time: enable SSH and ensure `Host <ssh-alias>` works. After that, the agent does not wait.

### Agent (after HostName/Port works — no further operator steps)

Write WSL `~/.ssh/config` (do not overwrite unrelated hosts):

```
Host <ssh-alias>
  HostName <ip>
  Port <port>
  User root
  IdentityFile ~/.ssh/<key>
```

Then, in order:

1. `rsync` **repo code** (exclude `checkpoints/`, `data/`; `.git` optional) → `/workspace/MyPT/`
2. `rsync` **`checkpoints/phase3_1_restart_gold/`** (and `phase3_chat_110k_gold/` if present locally)
3. **Local** recipe v2 generate/pack (`run_phase31_prepare.ps1` or the guide §D bash
   equivalent). HF stays local per §9.1. Do not run HF downloads on the pod.
4. `rsync` **`data/sft_phase3_1_restart_chat/`** packed shards only
5. If these eval dirs exist locally, rsync them (config `eval_sets`):
   `data/sft_phase1_format_lock`, `data/sft_phase2_operators`,
   `data/sft_phase3_eval_phase2_remix`. If missing, training still runs but in-loop eval
   sets are skipped/fail — log that and continue.
6. Remote checks:
   ```bash
   ssh <ssh-alias> "nvidia-smi && python -c 'import torch; print(torch.cuda.is_available())'"
   ssh <ssh-alias> "cd /workspace/MyPT && python scripts/sft/validate_sft_episode_masks.py --dataset_dir data/sft_phase3_1_restart_chat"
   ```
7. Start §5 train (`phase3_1_restart_v2`, init per §3). Not before mask validation is green.

Example rsync (WSL):

```bash
rsync -avz --progress \
  --exclude checkpoints/ --exclude data/ --exclude .git/ --exclude __pycache__/ \
  ./ <ssh-alias>:/workspace/MyPT/

rsync -avz --progress checkpoints/phase3_1_restart_gold/ \
  <ssh-alias>:/workspace/MyPT/checkpoints/phase3_1_restart_gold/

# if present:
rsync -avz --progress checkpoints/phase3_chat_110k_gold/ \
  <ssh-alias>:/workspace/MyPT/checkpoints/phase3_chat_110k_gold/

rsync -avz --progress data/sft_phase3_1_restart_chat/ \
  <ssh-alias>:/workspace/MyPT/data/sft_phase3_1_restart_chat/
```

**Windows-only fallback** (no WSL `rsync`): from the repo root in PowerShell, after
`ssh <ssh-alias>` works via the same key:

```powershell
tar -cf - --exclude=checkpoints --exclude=data --exclude=.git . | ssh <ssh-alias> "mkdir -p /workspace/MyPT && tar -xf - -C /workspace/MyPT"
scp -r checkpoints/phase3_1_restart_gold <ssh-alias>:/workspace/MyPT/checkpoints/
scp -r data/sft_phase3_1_restart_chat <ssh-alias>:/workspace/MyPT/data/
```

Prefer WSL rsync. If `rsync` is missing in WSL: `sudo apt update && sudo apt install rsync` (§0).
