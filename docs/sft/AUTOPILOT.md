# Autopilot — unattended SFT loop

**Status:** Operator + agent contract (canonical in-repo description)  
**Updated:** 31 August 2026  
**Resume:** [CURRENT_STATE.md](CURRENT_STATE.md)  
**Runnable runbook:** [autopilot_agent.md](../../autopilot_agent.md) (repo root — commands, SSH, mix recipes)  
**Window summary:** [AUTOPILOT_SUMMARY.md](../../AUTOPILOT_SUMMARY.md)  
**Action log:** run history is not published.

This page is the documentation-style account of how Phases 3–6 (and 6.2 / 6.3) were driven. The root runbook stays the command source of truth.

---

## What it is

Once SSH to the training host works, the agent runs pack → lean rsync → `train.py --auto_confirm` → eval gate → remediate or tag GOLD **without waiting for the operator** (up to 72 hours). Chat is a status log, not a confirm loop.

Hard stops only: `STOP_REASON.md` (crash, safety collapse, dead GPU/SSH, 72h elapsed).

---

## Host split

| Side | Role |
| --- | --- |
| Windows + Cursor | Edit, docs, this agent |
| WSL | `ssh`, `rsync`, unixify CRLF scripts |
| Remote GPU host | `python train.py`, `run_regression_gate.py` |
| Alias | `<ssh-alias>` in WSL `~/.ssh/config` |

Do not ship `.env`, `sources/`, dataset zips, or unused checkpoints. Lean rsync: `train.py`, `core/`, current `configs/sft/*.json`, `scripts/eval/`, `scripts/sft/`, packed train dir, `data/eval_capability/`, init GOLD **without** `optimizer.pt`.

---

## Loop

```
inventory → generate / mix / pack → mask check → rsync lean
  → train.py --auto_confirm
  → arm watch_remote_train.sh (notify AGENT_LOOP_WAKE_sft_train)
  → TRAIN_END → run_regression_gate.py --phase N on **val GOLD** (`*_gold`)
  → PASS: tag/promote that dir → pull_phase*_gold.sh local → advance
  → GOLD FAIL: **gate the final checkpoint** (`checkpoints/<model_name>/`, last step)
      before any mix remake. If final PASSES, treat it as the winner (tag GOLD from
      that dir). If both FAIL, then §9 remediations only
      (never math, never 3.2 / dead P4 / prefix-extract P5)
  → if the phase path is closed, stop. Do not mix-remake a closed GOLD. Measured copy scores are not published.
```

In-loop CE (format/operator val in the train log) is **advisory**. Decisions use `run_regression_gate.py`.

---

## Watcher

A finished `nohup` on the pod does not wake Cursor. After every train start:

```bash
# WSL, unix line endings
bash scripts/sft/watch_remote_train.sh <ssh-alias> 45
```

Notify on `AGENT_LOOP_WAKE_sft_train`. If the watcher exits while train is still up, re-arm it.

SSH `nohup python train.py … &` must redirect stdin (`</dev/null`) or the session holds until TRAIN_END.

---

## Packed mask check

`validate_sft_episode_masks.py` is **packing-aware** (31 Aug 2026). One 4096 row holds several conversations and many `<myPT_assistant>` spans (toolcall chains). Green = VALIDATION PASSED (every span masked, no all-zero rows).

The old “exactly one assistant tag” check false-failed every packed row (e.g. 1335/1335 on 6.3). That is not a data bug. A real FAIL still stops the loop.

---

## Gate phase numbers vs curriculum

| `run_regression_gate.py --phase` | Curriculum | Extra buckets |
| ---: | --- | --- |
| 3 | Phase 3 chat | (base chat gate) |
| 4 | Phase 4 multi-turn | `multiturn_coherence` |
| 5 | Phase 5 toolcall | `toolcall_basic` |
| 6 | Phase 6 agentic | `agentic_chain` |
| 7 | Phase **6.2** bind | `docid_bind` |
| 8 | Phase **6.3** ground | `toolresult_ground`, `search_miss` |

`--phase 7` / `8` keep prior thresholds so `phase6_agentic_gold` still grades as `--phase 6`. Math is STRUCTURAL / advisory on 7–8.

---

## GOLD pull (mandatory)

The webapp loads **local** `checkpoints/`. After a gate pass, tag on the pod then:

```bash
bash scripts/sft/pull_phase6_2_gold.sh   # 6.2
bash scripts/sft/pull_phase6_3_gold.sh   # 6.3
```

Do not leave a new GOLD only on RunPod.

---

## Locked “do not”

- Phase 3.2 corrective / `phase3_2_corrective`
- Resume `phase4_multiturn` (killed 3000-step) or prefix-extract P5
- Remediate `regression_basic` math on 750M
- Stitch the final answer in the RAG controller (6.3 is SFT)
- `mypt700_*` model names
- Pack Phase 3 or 4 without `--system_prompt_preset chat`
- Train without `--auto_confirm` on unattended RunPod

---

## Runtime contracts (not SFT, but decided in this window)

- Inference: strict greedy (`temperature=0`, `top_k=0`, `repetition_penalty=1.0`).
- Rank-1 binder stays **on** in `core/agent/controller.py` (doc_id only).
- `workspace.search` returns `{"documents": [], "total": 0}` when no distinctive query term overlaps title/filename/text (dense top-k is never empty by itself). Restart the local webapp to pick this up.
- `core/workspace/tools.py` is MyPT’s in-process tool registry (MCP-like role, not MCP).

---

## Cost note

Measured GPU-hours and billing for this project's unattended window are not published.

---

## Later: 1.4B from scratch

Same loop, new weights. Mix tables and “do not” lists in the PHASE*.md files are the starting SFT recipe. **First 1.4B session:** Phase 1 **≥ 14B unique tokens** — keep 6B + add ~8B different, **or** rebuild 14B. Not re-epoching 6B. [SCALE_1_4B.md](SCALE_1_4B.md). Do not run that while 6.3 is open.
