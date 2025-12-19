CURSOR SPEC — SFT Tool-Episode Generator (Offline, Deterministic)

Objective: Generate 20k–50k SFT episodes for toolcalling behavior using:

your handcrafted gold set as behavioral spec

template-driven diverse prompts

real tool execution (WorkspaceTools now; repo/obs later)

strict validators

output in JSONL matching your roles/tags

Constraints

Offline only, no external APIs

Deterministic outputs (seeded RNG)

Tool outputs must come from actual tool execution (or controlled mocks only for not-yet-implemented tools)

Must support your loss_mask training format (assistant-only loss)

Deliverables

Create:

scripts/
generate_sft_tool_episodes.py
data/
gold/
gold_toolcalls.jsonl
synthetic/
sft_tool_episodes.jsonl

3.1 Input: Gold set format

Read data/gold/gold_toolcalls.jsonl (schema in section 4 below)

Gold entries serve as:

canonical toolcall formatting examples

pattern definitions (“search→answer”, “list→get→summarize→answer”, “no tool needed”, error recovery)

3.2 Template library

Inside generator, define templates per “pattern”:

Patterns (MVP):

SEARCH_ANSWER

LIST_SELECT_SUMMARIZE

GETDOC_SUMMARIZE

NO_TOOL

ERROR_RECOVERY

(later) REPO_SEARCH_ANSWER

(later) OBS_COUNT_AGG_DRILLDOWN

Each pattern defines:

user prompt templates (with slots)

toolcall plan (which tools to call in which order)

how to construct args

how to turn toolresult into a grounded answer (extractive/templated)

3.3 Episode generation steps

For each episode:

Pick pattern (weighted)

Generate user prompt (template + fuzz: synonyms, reorder phrasing)

Execute toolcalls using your tool dispatcher:

for workspace tools use WorkspaceTools.execute()

Build assistant turns:

assistant emits toolcall JSON (must validate)

toolresult injected as JSON (must truncate if huge)

assistant final answer created deterministically from toolresult:

extract top snippets

cite doc_id/title/filename

Validate episode:

toolcall JSON parseable

tool exists

required args present

answer references at least one field from toolresult (simple overlap)

Write JSONL episode

3.4 Diversity controls

Seeded RNG: --seed

Minimum unique user prompts ratio (e.g. 0.8)

Vary:

top_k in [3..8]

query terms from doc titles/keywords

summarization max_length

3.5 CLI

python scripts/generate_sft_tool_episodes.py --workspace_dir workspace/ --index_dir workspace/index/latest --gold data/gold/gold_toolcalls.jsonl --out data/synthetic/sft_tool_episodes.jsonl --n 30000 --seed 42

3.6 Output schema

Use the “gold template” schema described in docs/GOLDEPISODES_JSON_OUTLINE.md and docs/GOLDEPISODES_REFERENCE.md

Acceptance tests

Generate 1k episodes quickly, all pass JSON validation

No unknown tools in output

At least:

60% tool-using episodes

30% no-tool episodes

10% multi-step episodes

Toolresults truncated to max chars

Prompt renders correctly in your training pipeline
