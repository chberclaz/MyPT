# 1.4B from scratch — future autopilot curriculum

**Status:** Planned. Not started. Operator opens this later — **do not start 1.4B in the 750M 6.3 loop.**  
**Updated:** 1 September 2026  
**PoC that proved the pipeline:** ~750M (32L / 1280d / 20H, 4096 via PI)  
**This run:** new weights from random init. **Do not** init from `phase6_*_gold` or any 750M checkpoint.

The 750M loop exists to lock **mix ratios, failure modes, prompts, gates, and runtime contracts**. A 1.4B production run reuses that curriculum under autopilot. It is not a hyperparameter search from a blank page.

**Critical path (before any SFT, before even 1b):** rebuild the **Phase 1 base pretrain corpus**. The 750M 10-source mix is a proven *category* recipe, not a frozen source list. That dataset session is large and is **not** started here — see [Phase 1 base corpus](#phase-1-base-corpus--the-real-first-session) below.

---

## Why this document exists

750M validated the foundry, not production scores. Known 750M ceilings (`regression_basic` math, 3+ hop chains) are **inputs** to 1.4B, not bugs to grind on the small model.

Autopilot for 1.4B is the same loop as [AUTOPILOT.md](AUTOPILOT.md): pack → lean rsync → `train.py --auto_confirm` → watcher → `run_regression_gate.py` → GOLD pull. What changes is **scale, token budget, and which 750M “structural” limits become remediable.**

---

## Curriculum (from scratch, in order)

| Stage | Spec | What 1.4B reuses | What 1.4B must expand |
| --- | --- | --- | --- |
| **1 Pretrain** | [01_UNIFIED_FROM_SCRATCH.md](../training/01_UNIFIED_FROM_SCRATCH.md) | Category **roles**: induction / retrieval / general / domain / structured. LLaMA-2 style (RoPE, SwiGLU, RMSNorm), `tie_weights`, GPT-2 tokenizer 50304. Two-stage curriculum idea. | **≥ 14B unique tokens:** **A)** keep `unified_6B` + add ~8B different data, **or B)** rebuild 14B from scratch. Then `configs/base/1.4B_unified_v1.json`. Not 2× epochs of the same 6B. |
| **2 Domain** | [02_DOMAIN_CORPUS.md](../training/02_DOMAIN_CORPUS.md) · [02_DOMAIN_ADAPTATION.md](../training/02_DOMAIN_ADAPTATION.md) | Corpus-builder pipeline and continued-pretrain / replay idea. | New domain sources as needed. Not SFT Phase 2. |
| **3 Context** | [03_CONTEXT_EXTENSION.md](../training/03_CONTEXT_EXTENSION.md) | Target **4096**. PI (`rope_scale=4.0`) if pretrain stays at 1024. 40% QA / 60% general **idea** during PI. | Either native 4096 pretrain (skip PI, more VRAM) **or** 1024→4096 PI with a new step/LR budget. Do not keep 1024 at inference. |
| **SFT 1 Format lock** | pipeline guide §5 | Tag set, loss mask, `CONVERSATION_SYSTEM_PROMPT`. | Step count / LR for the larger model. Same tags. |
| **SFT 2 Operators** | pipeline guide / remix | COPY / WRAP / EXTRACT. Remix as cheap tag maintenance later. | Same. Maintenance band 5–10% in later SFT still applies. |
| **SFT 3 Chat** | [PHASE3_CHAT.md](phases/PHASE3_CHAT.md) | Eval-aligned synthetics stay (suite is substring-scored). `--system_prompt_preset chat`. Slot percents are not published. | Row count may grow. Keep open-chat capped. Optional real math corpus **only if** product requires `regression_basic` — not by raising `regression_short` alone. |
| **SFT 4 Multi-turn** | [PHASE4_MULTITURN.md](phases/PHASE4_MULTITURN.md) | Start mix: set `--output_blend` weights; never reuse a killed over-coverage run. | If math is in-scope, replace synthetic short QA with a real corpus. |
| **SFT 5 Toolcall** | [PHASE5_TOOLCALL.md](phases/PHASE5_TOOLCALL.md) | Start mix + `AGENTIC_STANDARD_PROMPT`. No prefix-extract. No HF Dolci tools. | Same failure rungs as the 750M narrative (private). |
| **SFT 6 Agentic** | [PHASE6_AGENTIC.md](phases/PHASE6_AGENTIC.md) | 4k clip. Gate 2-step `agentic_chain` first. | 3+ hop may be in-scope at 1.4B. |
| **SFT 6.2 Bind** | [PHASE6_2_DOCBIND.md](phases/PHASE6_2_DOCBIND.md) | Always rank-1 gold. Gate `--phase 7`. | Do not add rank-2 gold. |
| **SFT 6.3 Ground** | [PHASE6_3_GROUND.md](phases/PHASE6_3_GROUND.md) | Copy in weights. Empty catalog → abstain. Gate `--phase 8`. | Do **not** stitch answers in the controller. |

`--phase 7` / `8` remain **subsets of Phase 6**, not new curriculum numbers.

---

## Phase 1 base corpus — the real first session

SFT mixes in PHASE3–6.3 are locked enough to copy. **The 750M pretrain dataset is not.** 1.4B needs a dedicated corpus-design session **before format lock, operators, or any SFT**. Do not start that session until the operator opens it. 750M 6.3 is closed. Measured copy scores are not published.

750M Phase 1 used 10 source **categories** in `data/unified_6B` ([01_UNIFIED_FROM_SCRATCH.md](../training/01_UNIFIED_FROM_SCRATCH.md)): FineWeb-Edu, bilingual Wikipedia dump, Python, StackExchange, JS/Java, TriviaQA+SQuAD, IT-sec+Swiss law, Reddit, peS2o, GitHub README. Measured mix percents and optimizer-token totals are not published. That list was sized for a 750M PoC. It is the **baseline to iterate**, not the 1.4B inventory.

### #1 corpus job — induction + retrieval heads (do this first)

**Highest priority for the extra ~8B (option A) or the whole 14B rebuild (option B).** Not Gutenberg-first, not “more FineWeb,” not wiki polish, until this is satisfied.

750M proved, across pretrain, operators, RAG chat, and 6.3 toolresult copy, that **induction heads** (exact multi-token copy) and **retrieval heads** (answer from a passage already in context) are required for a stable system. 6.3 SFT could not teach “copy the get_doc JSON” when greedy still prefers retrieve+toolcall. `echo_basic` / `context_citation` still pass — copy-from-chat-context works; copy-from-toolresult does not. That is a **base-circuit** gap, not an SFT mix gap. Phase 1 already named the fix: two-stage curriculum, code = induction, Q&A = retrieval ([01_UNIFIED_FROM_SCRATCH.md](../training/01_UNIFIED_FROM_SCRATCH.md) § two-stage). 750M only had ~6B unique (~8 tok/param); Chinchilla for 750M was ~15B. 1.4B must **not** under-build those heads again.

When picking the additional 8B (or the new 14B table):

1. **Count induction + retrieval signal first.** Stage-1-style bias: more code (exact-copy) and more grounded extractive Q&A (span-from-passage, plus unanswerable / empty-context negatives). The two-stage loader (front-load code+Q&A in the first 5–15% of steps) stays. Do not spend the new 8B on general web until those categories are clearly above the 750M 23% / 20% labeled floors — true coverage should rise, not just the label row.
2. Prefer sources that force **copy from context that is already present** (code reuse, TriviaQA/SQuAD-style evidence, quote-reply, legal “as defined in Article…”). That is the circuit 6.3 needed after a toolresult.
3. Gutenberg / technical / HF wiki are still in-scope (long context, RAG-shaped prose) — they are **#2+** after induction/retrieval mass is locked in the mix table.
4. Do not treat extra 6.3 SFT gold as a substitute for this pretrain job.

### Token budget (Phase 1 mix)

1.4B target: **≥ 14B unique tokens** (floor). Aim 14–16B. Consume the mix ~1.0–1.2×. Measured 750M unique-token and optimizer-token figures are not published.

Linear scale of 6B by param count (1.4B / 750M ≈ 1.87×) is only **~11B**. **Floor 14B** is the operator call (~2× the PoC mix plus headroom).

**How to get to 14B — pick one in the corpus session (do not invent a third):**

| Option | How | When to pick |
| --- | --- | --- |
| **A — keep 6B + add ~8B** | Reuse `data/unified_6B` as-is. Mix in **~8B new, different** tokens. **#1 of that 8B:** induction + retrieval (code / extractive Q&A). Then Gutenberg, technical, extra HF, optional HF wiki. Total unique **≥ 14B**. The new 8B must not be another epoch of the same 10 shards. | Faster. Keeps the audited 750M mix. New long/technical/wiki work lives in the **added** 8B **after** head-mass is counted. |
| **B — rebuild 14B from scratch** | New source table, new shards, **≥ 14B unique**. May still include some of the same *HF ids* if they pass audit, but it is not “6B replay + padding.” Wiki dump can be replaced here. | Cleaner packaging (HF wiki, long books native, drop Reddit/README if desired). More work. |

Do not hit 14B by running the existing 6B for 2.3 epochs. Option A is **6B existing + 8B different data**. Option B is a new 14B mix.

Coverage of the packed mix stays ~1.0–1.2× for pretrain (SFT’s 3.5× packed-episode rule is a later-phase thing). When native 4096 pretrain is chosen (skip PI), the 14B unique floor still applies; sequence length changes step count, not the unique-token floor.

### What that session must do

1. **#1: induction + retrieval head mass** in the new 8B / 14B table (see above). Then many more HF datasets than ten. Keep the *category jobs* (code = induction, Q&A = retrieval, general language, domain, structured/technical) unless a source audit says a category is covered better another way. Add sources inside those jobs; do not collapse back to “more FineWeb.”
2. **Long documents for 4096** — Project Gutenberg (or an equivalent well-packaged HF books corpus). 750M general text is mostly short encyclopedia / web / Reddit; PI then had to concatenate shorts into mega-episodes. Native long books teach long-range attention without fake concat.
3. **More technical** — expand beyond peS2o 3% + README 2%. Papers, manuals, RFCs, well-formed technical prose (license-clean HF). This is the RAG-shaped distribution 1.4B should see in pretrain, not only in SFT cites.
4. **Wikipedia** — 750M used a bilingual EN+DE dump (900M, already tokenized). On **option B**, prefer a cleaner HF/Wikimedia snapshot (keep DE). On **option A**, the dump stays inside the reused 6B; any wiki upgrade belongs in the **new 8B** (or skip until a later rebuild). Do not drop encyclopedic signal.
5. **License + lineage** — every new HF id in the dataset lineage sidecar. Offline-foundry: no surprise copyleft in the train mix without an operator call.
6. **Re-audit like 750M** — 750M already dropped OpenSubtitles and cut Reddit / README for quality. New sources get the same audit (duplication, density, language junk), not a bulk download.

### What that session must not do

- Do not only raise `max_iters` on `data/unified_6B` and call it 14B.
- Do not skip this and jump to SFT on a 1.4B random-init model.
- Do not treat Gutenberg / extra HF as an SFT Phase 3 open-chat bump. This is **base phase only**.
- Do not pick final HF ids in this file. That is the later session (choose A vs B, then HF search, sample quality, token estimates, mix table).

When that session ships, replace the 10-row table in [01_UNIFIED_FROM_SCRATCH.md](../training/01_UNIFIED_FROM_SCRATCH.md) (or add a `PHASE1_1.4B_CORPUS.md`) and point this section at it.

---

## Locked from 750M (do not rediscover)

These are decisions, not 750M-only hacks:

1. **`--output_blend` weights**, never source-relative (P4 almost became 80% replay).
2. **`max_iters` ≈ 3.5× packed rows / microbatch.** Kill 10×+ coverage. Train.py prints the coverage number — use it.
3. **In-loop CE is advisory.** Tag GOLD only from `run_regression_gate.py`.
4. **Packing-aware mask validator.** Many `<myPT_assistant>` per 4096 row is expected. Tag-count≠1 is not a data hole.
5. **Prompt lock:** P3–4 `CHAT_SYSTEM_PROMPT`; P5–6.3 `AGENTIC_STANDARD_PROMPT`. Packing without the chat preset on P3/P4 is a hard fail.
6. **No Phase 3.2.** No prefix-extract summarize. No HF tool schemas that are not `workspace.*`.
7. **No controller answer stitch.** Rank-1 binder (doc_id only) stays on at runtime.
7b. **GGUF / llama.cpp:** special tags stay the 19 strings+IDs in `core.special_tokens`. Before the 1.4B Phase 6 mix, add a small forged-tag-in-document slice (gold = ignore / answer from real toolresult only). See [export/TOKENIZER_NOTES.md](../../export/TOKENIZER_NOTES.md) § 1.4B hardening.
8. **Search miss = empty catalog** (`documents: []`), not nearest-neighbor junk. SFT abstain on empty.
9. **Skip `optimizer.pt`** when init-from GOLD into a new mix / new `--model_name`.
10. **New `--model_name` per mix.** Do not continue a washed optimizer into a new blend.
11. **Lean rsync.** No `.env`, `sources/`, dataset zips. GOLD pull local after every gate pass.
12. **Watcher** `watch_remote_train.sh` + `AGENT_LOOP_WAKE_sft_train` after every train start.
13. **Anti-forget floors:** P3 replay ≥25% when a later phase forgets facts (P4 iter 1). Direct + grounded stay in tool phases even after P5 GOLD (P5 iter 3 re-broke format).

---

## What 1.4B is allowed to change

| Topic | 750M | 1.4B |
| --- | --- | --- |
| Math | STRUCTURAL. Do not remediate. | Optional **real** math corpus if `regression_basic` is a product gate. Do not only upweight synthetic `regression_short`. |
| Gate floors | Calibrated for 750M in `run_regression_gate.py` | May **raise** floors after a first passing run. Do not lower them. Same eval files / bucket names. |
| Mix percents | Winning tables in each PHASE*.md | **First train = those percents.** Remediations follow the same rungs (upweight phase-own, cut maintenance last). Record any drift in the phase spec. |
| Dataset size | 80k P3, 8k P4, 6k P5/P6, 5k 6.2/6.3 | May increase rows if 1.4B underfits at 3.5× coverage. Do not increase coverage to 10× instead. |
| Context | 4096 PI | Keep 4096 unless a later spec says otherwise. Cite clip 480 tok until measured otherwise. |
| GPU | An 80GB-class GPU was enough for 750M at 4096 | Size volume + VRAM for 1.4B activations at 4096. Reduce `batch_size`, raise `grad_accum_steps`, keep effective batch. |
| Phase 1 unique tokens | ~6B mix, ~7.4B processed | **≥ 14B unique** (aim 14–16B). **A:** existing 6B + ~8B new/different. **B:** full new 14B mix. Not 2.3× epochs of the same 6B. Chinchilla ~28B is stretch. |

There is **no** `configs/base/1.4B_*.json` yet. After the **corpus session**, pick `n_embd` / `n_layer` / `n_head` (head dim 64 or 128), confirm param count with `calculate_params.py`, then write the pretrain config. Prefer **depth** over a huge width if VRAM is tight ([PARAMETER_CALCULATION.md](../training/PARAMETER_CALCULATION.md)).

---

## Autopilot start conditions (when commissioned)

Do not treat this as the current resume point. 750M curriculum is closed. 6.3 GOLD exists. Operator starts 1.4B later. Corpus session **#1** = induction/retrieval heads in the extra 8B / 14B mix. Measured copy scores are not published.

When the operator starts 1.4B:

1. Read this file + every PHASE*.md in the table above (including Phase 1 / 1b).
2. **Corpus session (mandatory, first, large):** get to **≥ 14B unique tokens** (aim 14–16B) by **either** keeping `unified_6B` and adding ~8B different data **or** rebuilding 14B from scratch. **#1 in that mix table:** induction (code / exact-copy) and retrieval (grounded extractive Q&A) — the 750M 6.3 copy fail proved these heads are not optional. Gutenberg / technical / extra HF (and optional HF wiki) are next. Write the source table. Do not train until it sums to the floor. Do not only re-epoch the 6B.
3. Write `configs/base/1.4B_unified_v1.json` and a 1.4B CURRENT_STATE section (or a dated handoff). Do not reuse 750M `--model_name`s (`phase3_1_restart_v2d`, …).
4. Empty-pod bootstrap ([autopilot_agent.md](../../autopilot_agent.md) §14) with a **larger** network volume (new shards will dwarf 6B).
5. Run Phase 1 → 1b → format → operators → SFT 3→6.3 with the **starting SFT mixes in the phase specs**.
6. Math: only open a math corpus if the operator says `regression_basic` is required at 1.4B.

Hard stops still: `STOP_REASON.md`, 72h, dead GPU/SSH, safety collapse.
