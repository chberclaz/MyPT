Phase 3a0 — Chat protocol lock-in (hard syntax)

Goal: The model always emits the correct role/tags and never leaks other roles.

Strict <myPT_user> … <myPT_assistant> … structure

No tools yet, no JSON requirements yet

Include a small % malformed prompts only if you want robustness, but keep it low

Pass criteria: Near-zero tag violations on an eval set of adversarial prompts.

Phase 3a1 — Ultra-short compliance (your current run)

Goal: “Obey instruction → respond minimally → stay in assistant tag”.

One-word answers, “YES/NO”, single token labels, etc.

Trains response length control and stop discipline

Great place to teach: “If unknown → say I don't know.” (or your preferred canonical)

Pass criteria: Response length + format is stable even under prompt variation.

Phase 3a2 — Minimal Q&A (short, 1 sentence answers)

Goal: “Normal user question → short correct answer” (no verbosity).

Small questions, 1 sentence answers (your suggestion is exactly right)

Mix styles: statements, imperatives, incomplete prompts, typos, “give me X in 5 words”, etc.

Include “refuse / safe-complete” behaviors if you want them later

Pass criteria: Doesn’t ramble; obeys constraints like “one sentence”, “max 10 words”.

Phase 3a3 — Controlled verbosity ladder (short ↔ medium)

Goal: Teach the model to choose the right length on command.

Same intent asked with different constraints:

“Answer in 1 sentence”

“Answer in 3 bullet points”

“Explain like I’m in a hurry”

“Give steps 1–3 only”

This is where many models break if not trained: they default to one “house style”.

Pass criteria: Constraint-following generalizes (new topics, same instruction types).

Phase 3a4 — Text-only task following (multi-step, no tools)

Goal: Get the “agent core” without tools: plans, transforms, structured outputs.

Examples:

Extract → normalize → reformat

Classify + give reason (short)

Write a checklist from a paragraph

Summarize with specific schema (but still assistant text)

This phase is the bridge to tool calling because it trains:

step discipline

schema discipline

“do X, then Y, then stop”

Pass criteria: Executes multi-step instructions without inventing extra steps or formats.

Phase 3a5 — Structured outputs (schemas, but still no tool execution)

Goal: Output that is machine-parseable consistently.

You can pick one or more:

Strict JSON (with escaping rules)

YAML

“tagged blocks” (your MyPT tags + internal sections)

Function-call-like stubs (but not real tool calls yet)

This is where you prepare the exact shape you want for 3b:

arguments formatting

deterministic keys ordering (if you care)

no trailing commentary

Pass criteria: 99%+ parse success on held-out prompts.

Phase 3a6 — Tool-readiness simulation (fake tools, text-only)

Goal: Teach: decide when a tool is needed and compose the call, without actually calling.

Pattern:

User asks something that clearly requires a tool (search DB / calculate / retrieve file)

Assistant responds with a “tool request block” in your intended format

Then you include the “tool result” (as if <myPT_tool> …) and train the assistant to finish.

Even if you don’t enable <myPT_tool> until 3b, you can still simulate it with placeholder tags or a “TOOL_REQUEST:” block.

Pass criteria: Correct “call vs no-call” decision + correct argument filling + clean finalization.

Recommended “minimal set” if you want to move fast

If speed matters and you want to reach 3b ASAP, you can compress to:

3a0 protocol

3a1 ultra-short control

3a2 minimal Q&A

3a4 multi-step text tasks

3a5 schema outputs
→ then Phase 3b real tool calls

(3a3 is very valuable, but it can be partially covered by making 3a2/3a4 include explicit length constraints.)

One critical warning (based on your earlier pain)

Avoid mixing too many styles inside one run early on (that was the “gateway” feeling you described). Keep each sub-phase “single-purpose”, then merge later with small weights.

A typical merge strategy:

current focus dataset: 70–85%

replay of previous 3a phases: 10–25%

tiny general replay (optional): 0–10% (only if you see language degradation)

What Phase 3b needs from 3a to be easy

By the time you start toolcalls, the model should already:

always produce valid MyPT chat tags

obey output constraints (short, schema, stop)

handle multi-step instructions

emit strict structured blocks reliably

decide “tool needed?” vs “answer directly?”

If those are true, 3b becomes mostly: “new output head format + supervised examples of tool selection.”

If you want, paste your planned 3b tool-call envelope (how you want <myPT_tool> to look, how args are encoded), and I’ll map 3a5/3a6 exactly onto that so your transition is frictionless.

How to train summarizing (Phase 3a)
1) Put summarization in the right place

Summarization is Phase 3a4/3a5-ish: it’s task-following + controlled output, not just “short answers”.

If you try to teach summarization during 3a1/3a2, you’ll get:

over-short “summaries” (too lossy)

bad constraint following (“1 sentence” but rambles)

style collapse (“generic summary voice”)

So: teach it after the model is already obedient.

2) Build summarization as a ladder (recommended datasets)

You want the model to learn two things:

compression skill (keep meaning, drop fluff)

constraint obedience (length, format, focus)

Dataset S1 — “Hard length constraint” summarization

Same input text, different constraints:

“Summarize in 10 words.”

“Summarize in 1 sentence.”

“Summarize in 3 bullet points.”

This forces length control, not just “summarize vaguely”.

Gold rule: The assistant output must contain no new facts not present in input.

Dataset S2 — “Extractive-first” summarization (very effective)

Teach a 2-step pattern:

Extract key facts (bullets, short)

Produce a summary only from those facts

Example target format:

Key facts: (3–6 bullets)

Summary: (1–3 sentences)

This dramatically reduces hallucinations because you’re training an internal “grounding step”.

Later you can remove the “Key facts” section once it’s learned.

Dataset S3 — “Schema summaries”

Pick 1–2 stable schemas you’ll want later for tool use:

TL;DR: one line

Key points: bullets

Action items: numbered list

Risks: bullets

Open questions: bullets

This is basically “structured output discipline,” and it transfers directly into tool-call argument discipline.

Dataset S4 — “Query-focused summarization”

This is what makes summarization useful:

Input: long text + user asks:

“Summarize only security-relevant parts.”

“Summarize legal implications.”

“Summarize for a manager (non-technical).”

“Summarize with a focus on costs and deadlines.”

This trains selective attention and reduces generic summaries.

3) Training recipe (simple)

A very practical way to add summarizing without blowing up Phase 3a:

Option A (clean + fast)

Do one dedicated “summarization run” after your minimal Q&A run.

70–85% summarization dataset (S1–S4 mixed)

15–30% replay of your existing “format/tag obedience” set (3a0–3a2)

Option B (integrated)

If you don’t want a separate run:

Add summarization samples as 10–20% of your 3a4/3a5 run

Keep the rest task-following + schema

4) What to avoid

Only one style of summary (you’ll get “house voice” lock-in)

No constraints (“summarize this”) → model learns vague compression

Training only short summaries → model can’t do “detailed brief”

Letting it invent (“background context” not in the text)

5) Minimal dataset size

You can get decent summarization behavior surprisingly fast if your data is clean:

100–300 high-quality episodes already moves the needle

500–1500 gives robustness across styles/constraints

More helps, but quality and constraint diversity matter more than volume here

Ground rule (now explicit)

Random-fed phases (1–2): replay = fine

Sequential-fed phases (3a+): replay only with other sequential datasets
→ no sharded random text, no concat soup, no leakage of phase-1/2 data

This is correct and important. You’re training behavior, not language.

How many Phase-3a runs do you actually need?

Not 7.

You need 4 sequential runs, each with a clear behavioral target and limited replay.

✅ Recommended Phase-3a execution plan (sequential-safe)
Run 1 — Protocol & stop discipline

(Phase 3a0 + 3a1)

Primary dataset (≈80%)

MyPT tag correctness

One-word / ultra-short answers

Hard stops, no leakage

Replay (≈20%)

Earlier protocol samples only

What this locks in

Role obedience

Output termination

Length discipline

👉 Do not proceed until this is rock-solid.

Run 2 — Minimal Q&A obedience

(Phase 3a2)

Primary dataset (≈70–80%)

Short questions → 1 sentence answers

Length-bounded answers

Unknown → canonical fallback (“I don’t know.”)

Replay (≈20–30%)

Run-1 sequential data (protocol + short answers)

What this locks in

Natural language instruction following

No verbosity creep

Stable assistant “voice”

Run 3 — Task execution + summarization

(Phase 3a4 + summarization)

This is where summarizing belongs.

Primary dataset (≈70%)

Multi-step text tasks

Summarization (S1–S4 ladder):

length-constrained summaries

extract-then-summarize

schema summaries

query-focused summaries

Replay (≈30%)

Run-2 minimal Q&A

Run-1 protocol samples

What this locks in

Compression without hallucination

Step discipline

Selective attention

“Do exactly what was asked, nothing more”

Run 4 — Structured output & tool-readiness

(Phase 3a5 + 3a6)

Primary dataset (≈70–80%)

Strict schemas (JSON / tagged blocks)

Fake tool calls (text-only)

Decide: tool vs no tool

Argument filling

Finalization after “tool result”

Replay (≈20–30%)

Summarization + task execution samples

Minimal protocol data (small dose)

What this locks in

Parse-safe outputs

Tool call muscle memory

Zero commentary outside schema

Why summarization is not its own run

Summarization is:

a task

with constraints

and schemas

So it belongs with other task-execution behaviors.
If you isolate it, you risk:

a “summary-voice” style lock-in

poor transfer to tool usage

Embedding it in Run 3 is the sweet spot.

Sanity checks (important)

After each run, test for:

Regression in earlier behavior (tags, length)

Over-compression (summaries losing key facts)

Schema drift (extra text sneaking in)

If you see drift → increase replay by ~5–10%, but only from sequential data.






You don’t need multi-turn for “extract key facts → summary”. Do it as a single assistant response with two sections (same turn), so it stays perfectly compatible with your sequential, single-episode pipeline.

Option A (recommended): Single-turn, two-block output

User provides text. Assistant outputs:

KEY_FACTS: bullets (pure extraction; no new info)

SUMMARY: 1–3 sentences (must be derivable only from the bullets)

Example gold episode (adapt to your tags):

<myPT_user>
Extract key facts and then summarize. Text:
"ACME reported a data breach on 12 Jan 2026. An exposed API key allowed access to 3,200 customer records. Password hashes were not accessed. ACME rotated keys and notified regulators."
</myPT_user>

<myPT_assistant>
KEY_FACTS:
- ACME reported a data breach on 12 Jan 2026.
- An exposed API key enabled access.
- 3,200 customer records were accessed.
- Password hashes were not accessed.
- Keys were rotated and regulators notified.

SUMMARY:
ACME reported a breach on 12 Jan 2026 caused by an exposed API key, affecting 3,200 customer records. The company rotated keys and notified regulators, and password hashes were not accessed.
</myPT_assistant>


This trains the “internal grounding step” without tool tags, multi-turn, or unclear formatting.

Option B: Schema version (better for later toolcalls)

Same idea, but in strict JSON (or your preferred schema):

<myPT_assistant>
{
  "key_facts": [
    "…",
    "…"
  ],
  "summary": "…"
}
</myPT_assistant>


This transfers directly into Phase 3b argument discipline.

The three rules that make it work

Key facts must be extractive: only what’s explicitly stated.

Summary must be supported by key facts: no extra claims.

Hard constraints: number of bullets + max sentences.

Good defaults:

key_facts: 3–7 bullets

summary: 1–2 sentences (or “<= 25 words” for harsh control)

How to vary it (so it generalizes)

For each input text, create 2–4 variants:

“Summarize in 10 words.”

“Summarize in 1 sentence.”

“Summarize for a manager.”

“Summarize only risks and actions.”

All still using the same two-block structure.