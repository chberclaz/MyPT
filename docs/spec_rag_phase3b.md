================================
📘 CURSOR SPEC — MyPT Phase 3B
Workspace + Toolcall / Agentic RAG
================================
🎯 Goal

Extend MyPT from “minimal RAG with <context>” → agentic workspace RAG:

The model can call tools (e.g., workspace.search, workspace.summarize).

A WorkspaceEngine manages documents, chunks, and summaries.

An AgentController runs a loop: model → toolcall → tool_result → final answer.

A toolcall SFT pipeline teaches the model the tool API and grammar.

Constraints:

Offline only (local files, local embeddings).

Compatible with Phase 3A tagging & loss-masking.

Code kept modular and readable, minimal dependencies.

📁 Target structure (new / extended)
core/
rag/
retriever.py # already exists (Phase 3A)
pipeline.py # already exists (Phase 3A)
workspace/
engine.py # NEW: WorkspaceEngine + models
tools.py # NEW: Python tool implementations
**init**.py
agent/
controller.py # NEW: AgentController for toolcalls
parsing.py # NEW: toolcall parsing utilities
**init**.py
scripts/
prepare_tool_sft.py # NEW: SFT dataset builder for toolcalls
workspace_chat.py # NEW: CLI interactive workspace+agent chat
docs/
toolcall_sft.md # NEW: doc describing toolcall SFT format
workspace_api.md # NEW: doc describing workspace tools & behavior

Reuse:

core/embeddings/local_embedder.py

core/rag/retriever.py

core/generator.py

scripts/build_rag_index.py

scripts/prepare_chat_sft.py

scripts/rag_chat.py

🧱 Core Concepts

1. Tool universe (Phase 3B)

We define a small, stable set of tools under workspace.\*:

workspace.search

Inputs: {"query": str, "top_k": int}

Returns: top document chunks (text + metadata).

workspace.list_docs

Inputs: {}

Returns: list of documents (ids, titles, paths).

workspace.get_doc

Inputs: {"doc_id": str}

Returns: full doc or main text.

workspace.summarize

Inputs: {"doc_id": str} or {"text": str}

Returns: textual summary (will internally use the model with <context> RAG).

These tools are implemented in Python (no external services), and the model learns to call them via SFT.

2. Text grammar / tags for tools

We extend the tag set with:

<toolcall name="workspace.search"> ...JSON... </toolcall>
<toolresult name="workspace.search"> ...JSON... </toolresult>

Same for other tools:

<toolcall name="workspace.summarize">...</toolcall>
<toolresult name="workspace.summarize">...</toolresult>

We keep existing tags:

<system>...</system>
<context>...</context>
<user>...</user>
<assistant>...</assistant>

So a full conversation with tools looks like:

<system>You are MyPT workspace assistant...</system>

<user>Find relevant docs about X and summarize them.</user>

<assistant>
<toolcall name="workspace.search">
{"query": "X", "top_k": 5}
</toolcall>
</assistant>

<toolresult name="workspace.search">
{"documents": [...], "metadata": {...}}
</toolresult>

<assistant>
<toolcall name="workspace.summarize">
{"doc_id": "doc_1"}
</toolcall>
</assistant>

<toolresult name="workspace.summarize">
{"summary": "..." }
</toolresult>

<assistant>Here is an overview of the main points...</assistant>

3. Loss-mask rules for tool SFT

For toolcall SFT, use token-level loss mask:

<assistant> and its content (including toolcall JSON) → mask = 1

<assistant> final natural language answer → mask = 1

<user>, <system>, <context> → mask = 0

<toolresult> blocks (content comes from tools, not model) → mask = 0

Tag tokens themselves inside <assistant>...</assistant> are included in mask = 1 (model must learn grammar).

This matches Phase 3A: only the model’s own tokens are supervised.

## 🔧 Component specs

## A. core/workspace/engine.py

Purpose:
Central workspace abstraction managing documents, indexes, and metadata.

Key concepts:

@dataclass
class Document:
doc_id: str
title: str
path: str
created_at: float
updated_at: float

@dataclass
class Chunk:
chunk_id: str
doc_id: str
text: str
position: int # chunk index within doc

Class: WorkspaceEngine

Minimal API:

class WorkspaceEngine:
def **init**(self, base_dir: str, index_dir: str):
"""
base_dir: root of workspace (docs, metadata)
index_dir: directory where RAG index lives (embeddings + meta)
"""

    # Document management
    def list_docs(self) -> list[Document]: ...
    def get_doc(self, doc_id: str) -> Document | None: ...
    def get_doc_text(self, doc_id: str) -> str | None: ...

    # RAG / search
    def search(self, query: str, top_k: int = 5) -> list[Chunk]: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    # (optional) summaries, notes, tasks later

Implementation details:

Under the hood, use Phase 3A RAG index:

load embeddings.npy + meta.jsonl

meta.jsonl should include doc_id and title if possible

search():

reuse Retriever (Phase 3A)

wrap results into Chunk objects

The engine should be purely local (filesystem-based) and stateless except cached index in memory.

---

## B. core/workspace/tools.py

Purpose:
Expose Python functions implement the workspace.\* tools.

Each tool:

Takes a dict as input.

Returns a dict that’s JSON-serializable (no complex types).

Is deterministic / pure (for now).

Example:

class WorkspaceTools:
def **init**(self, engine: WorkspaceEngine):
self.engine = engine

    def workspace_search(self, args: dict) -> dict:
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))
        chunks = self.engine.search(query, top_k=top_k)
        return {
            "documents": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "position": c.position,
                }
                for c in chunks
            ]
        }

    def workspace_list_docs(self, args: dict) -> dict:
        docs = self.engine.list_docs()
        return {
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "path": d.path,
                }
                for d in docs
            ]
        }

    def workspace_get_doc(self, args: dict) -> dict:
        doc_id = args.get("doc_id")
        text = self.engine.get_doc_text(doc_id)
        return {"doc_id": doc_id, "text": text or ""}

    def workspace_summarize(self, args: dict) -> dict:
        # Phase 3B minimal version:
        #  - if doc_id is given, fetch text
        #  - call RAG or generator directly with that text as <context>
        ...

Also define a registry:

TOOL_REGISTRY = {
"workspace.search": WorkspaceTools.workspace_search,
"workspace.list_docs": WorkspaceTools.workspace_list_docs,
"workspace.get_doc": WorkspaceTools.workspace_get_doc,
"workspace.summarize": WorkspaceTools.workspace_summarize,
}

AgentController will use this registry to execute tools by name.

---

## C. core/agent/parsing.py

Purpose:
Detect and parse <toolcall> blocks in model output, and render <toolresult> blocks.

Requirements:

A simple, robust parser that can:

scan an output string

find the last well-formed <toolcall ...>...</toolcall>

extract name attribute

extract JSON body safely (brace matching)

Avoid heavy dependencies: use regex + simple JSON parsing.

Functions:

@dataclass
class ToolCall:
name: str
arguments: dict
raw_block: str

def find_last_toolcall(text: str) -> ToolCall | None: ...
def render_toolresult(name: str, result: dict) -> str: # returns a string like: # <toolresult name="workspace.search"> # {...json...} # </toolresult>
...

This file does not know about Workspace; it’s generic tool parsing/rendering.

---

## D. core/agent/controller.py

Purpose:
Orchestrate model + tools until a final answer is produced.

Dependencies:

core.generator.Generator (Phase 3A)

core.workspace.engine.WorkspaceEngine

core.workspace.tools.WorkspaceTools

core.agent.parsing

Class:

class AgentController:
def **init**(self, generator: Generator, workspace_tools: WorkspaceTools):
self.generator = generator
self.workspace_tools = workspace_tools

    def run(self, history: list[dict], max_steps: int = 3) -> dict:
        """
        history: list of messages:
          {"role": "user" | "assistant" | "system" | "tool" | "tool_result", "content": str, "name"?: str}
        Returns:
          final assistant message dict.
        """

Algorithm (high-level):

Build prompt from history using the same serialization logic as SFT:

<system>...</system>

sequences of <user>, <assistant>, <toolcall>, <toolresult>

Call generator.generate(prompt, stop_strings=[ "</assistant>" ]) to get next assistant turn.

Inspect the generated text:

If it contains a <toolcall> block:

Parse with find_last_toolcall.

Execute the corresponding Python tool via WorkspaceTools.

Render <toolresult> string.

Append both:

{"role": "assistant_toolcall", "name": ..., "content": raw_block}

{"role": "tool_result", "name": ..., "content": rendered_result}

Loop again (next step).

Else:

Treat this as the final answer:

{"role": "assistant", "content": answer_text}

Return.

Stop after max_steps to avoid infinite loops.

The AgentController is the heart of Phase 3B.

---

## E. scripts/prepare_tool_sft.py

Purpose:
Build SFT dataset with toolcall patterns, tokens.bin + \*\_mask.bin.

Input JSONL schema (per line):

{
"system": "You are MyPT workspace assistant...",
"messages": [
{"role": "user", "content": "Find docs about X and summarize them."},
{
"role": "assistant_toolcall",
"name": "workspace.search",
"arguments": {"query": "X", "top_k": 5}
},
{
"role": "tool_result",
"name": "workspace.search",
"content": {"documents": [/* truncated for SFT */]}
},
{
"role": "assistant",
"content": "Here is a summary of the docs..."
}
]
}

Serialization rules:

<system>...</system> if system present.

For each message:

<user>...</user>
<assistant>...</assistant>
<toolcall name="..."> {...json...} </toolcall>
<toolresult name="..."> {...json...} </toolresult>

Mask rules (token-level, same as earlier):

assistant & assistant_toolcall segments → mask = 1

user, system, tool_result segments → mask = 0

The script:

Loads JSONL.

For each example:

Builds serialized text string + char-level mask list.

Uses tokenizer to get tokens.

Maps char-level mask → token-level mask.

Shards into:

train/shard_XXXXX.bin

train/shard_XXXXX_mask.bin

val/...

Writes dataset_metadata.json with has_loss_mask: true and schema: "toolcall_sft_v1".

---

## F. scripts/workspace_chat.py

Purpose:
Interactive demo CLI for full workspace agent.

Behavior:

CLI args:

python scripts/workspace_chat.py \
 --model_ckpt checkpoints/mypt_toolchat.pt \
 --workspace_dir workspace/ \
 --index_dir workspace/index/latest

Initialize:

Load model + tokenizer.

Construct Generator.

Build WorkspaceEngine(base_dir, index_dir).

Build WorkspaceTools(engine).

Build AgentController(generator, workspace_tools).

Loop:

user> <input>
assistant> <output>

Internally:

Maintain a history list of message dicts.

On each user input:

Append {"role": "user", "content": user_text} to history.

Call AgentController.run(history).

Append returned assistant message to history.

Print answer text only (not toolcalls/results).

This is your “Glean-like” demo.

---

## G. Docs: docs/toolcall_sft.md & docs/workspace_api.md

toolcall_sft.md should describe:

JSONL format.

Tagging and serialization rules.

Loss mask policy.

Example full conversation with toolcalls.

How to build dataset:

python scripts/prepare_tool_sft.py \
 --input_jsonl data/tool_sft.jsonl \
 --out_dir data/tool_sft \
 --tokens_per_shard 10_000_000

Training example (config configs/sft2/toolchat_150M.json):

python train.py \
 --config_file configs/sft2/toolchat_150M.json \
 --dataset_dir data/tool_sft \
 --model_name mypt_toolchat

workspace_api.md should describe:

WorkspaceEngine concepts (Document, Chunk).

workspace.\* tools (names, arguments, return JSON).

How they map to SFT toolcalls.

How AgentController interprets them.

🧪 Training story recap

Typical training path:

Phase 1 / 2: base model pretrain.

Phase 3A: chat + <context> SFT → mypt_chat.

Phase 3B: toolcall SFT → mypt_toolchat.

Phase 3B SFT uses:

relatively small curated dataset with:

search → summarize flows

list_docs → get_doc → summarize

simple multi-step tool sequences.

✅ What Cursor should not change

Do not change core model architecture (core/model.py etc.).

Do not modify existing loss-mask semantics; only add new datasets using them.

Do not hard-code tool logic into the model; keep it in workspace.tools + agent.controller.
user core/special_tokey.ps where neccessary or adapt this files to match the new needs.

Do not add external services.
