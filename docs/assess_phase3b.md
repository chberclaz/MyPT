CURSOR CHECKLIST — MyPT Phase-3B (Agentic RAG) Readiness Evaluation

This checklist is designed for Cursor to automatically check whether the MyPT codebase contains the necessary components for a working agentic RAG "mini-GPT" system.

Cursor:
For each item, search the repository and answer YES (implemented) or NO (missing/partial) and point to file paths.

1. Special Token Consistency Check

Verify all of these exist with matching names:

1.1 Check for defined special tokens

myPT_system_open, myPT_system_close

myPT_user_open, myPT_user_close

myPT_assistant_open, myPT_assistant_close

myPT_context_open, myPT_context_close

myPT_toolcall_open, myPT_toolcall_close

myPT_toolresult_open, myPT_toolresult_close

myPT_eot

Cursor should verify they exist in:

core/special_tokens.py

1.2 Check that all code references the same keys

Search for conflicting spellings:

tool_call vs toolcall

tool_result vs toolresult

snake_case vs camelCase

hyphens or underscores mismatches

Make sure all usage across:

core/rag/tags.py

core/agent/parsing.py

scripts/prepare_chat_sft.py

scripts/prepare_tool_sft.py

all refer to the same token names.

Cursor response must list mismatches if any.

2. Context + Tool SFT Pipeline Check
   2.1 Verify existence of Chat-SFT builder

Required file:

scripts/prepare_chat_sft.py

Cursor: confirm that this script:

Accepts a JSONL with "messages" and "context"

Emits:

\*.bin

\*\_mask.bin

dataset_metadata.json

Uses use_loss_mask = True

Serializes special tags correctly.

2.2 Verify existence of Tool-SFT builder

Required file:

scripts/prepare_tool_sft.py

Cursor: confirm that this script:

Accepts roles:

"assistant_toolcall"

"toolresult"

Serializes:

<toolcall ...>

<toolresult ...>

Applies:

assistant + assistant_toolcall mask = 1

all other masks = 0.

2.3 Verify training configs exist

Search for configs under:

configs/sft1/\* (chat SFT)

configs/sft2/\* (tool SFT)

If missing, mark as NO.

3. RAG Functionality Check
   3.1 Check RAG index builder

Required file:

scripts/build_rag_index.py

Cursor: confirm it:

Loads documents from workspace

Uses DocumentLoader + TextChunker

Uses a local embedding model

Saves:

embeddings.npy

meta.jsonl

index config JSON

3.2 Check that RAG index is used by retriever

Required file:

core/rag/retriever.py

Cursor: confirm:

Loads saved index

Computes cosine similarity

Implements:

.retrieve(query)

.reload_index(path)

3.3 Confirm RAG prompt wrapper exists

Required file:

core/rag/tags.py

Cursor: confirm:

wrap_context

wrap_system

wrap_user

wrap_assistant

build_rag_prompt(question, context_chunks, system)

4. Workspace Engine Check
   4.1 Verify WorkspaceEngine implementation

Required file:

core/workspace/engine.py

Cursor: confirm it implements:

list_docs()

get_doc(doc_id)

get_doc_text(doc_id)

search(query, top_k)

reload_index(path)

Uses Retriever internally.

4.2 Document + Chunk structures

Check file:

core/document/\*

Cursor: confirm:

loader.py loads files

chunker.py chunks text

Documents have:

filename

path

text

metadata

Chunks have:

chunk_id

text

metadata (filename, position)

5. Workspace Tools Check
   5.1 Required file:

core/workspace/tools.py

Cursor: confirm it provides:

workspace.search

workspace.list_docs

workspace.get_doc

workspace.summarize

And that TOOL_REGISTRY maps these names correctly.

6. Agent / Tool-call Check
   6.1 Verify Toolcall parser

Required file:

core/agent/parsing.py

Cursor: confirm:

find_toolcall

find_all_toolcalls

ToolCall dataclass

Renders <toolcall> and <toolresult> blocks

6.2 Verify Agent Controller

Required file:

core/agent/controller.py

Cursor: confirm:

Builds full prompt from history (system, user, assistant, toolcalls)

Detects toolcalls via parser

Executes correct Python tool via registry

Inserts <toolresult> into history

Continues loop until final assistant answer

6.3 Check for history trimming

Cursor: Confirm whether there is any logic to prevent prompt overflow against block_size.

If missing, mark as NO.

7. Interactive CLI Check
   7.1 RAG chat CLI

Required file:

scripts/rag_chat.py

Cursor: confirm:

Loads model + retriever

Provides interactive Q&A with RAG context

7.2 Workspace agent chat CLI

Required file:

scripts/workspace_chat.py

Cursor: confirm:

Loads model + WorkspaceEngine + AgentController

Supports:

/docs

/reload

/verbose

normal chat with toolcalls

8. Final “Mini-GPT” Capability Check

Cursor: Evaluate whether the full chain exists:

Drop documents into workspace/docs/

Build index with build_rag_index.py

Model uses WorkspaceEngine + tools

AgentController processes toolcalls

Tool-SFT pipeline allows training a tool-using model

Interactive chat demonstrates tool use and RAG reasoning

If any part is missing, list missing components.

📌 Cursor Output Format

Cursor, produce your evaluation as:

{
"special_tokens_consistent": true/false,
"sft_chat_pipeline": true/false,
"sft_tool_pipeline": true/false,
"rag_index_builder": true/false,
"rag_retriever": true/false,
"workspace_engine": true/false,
"workspace_tools": true/false,
"agent_controller": true/false,
"toolcall_parser": true/false,
"cli_rag_chat": true/false,
"cli_workspace_chat": true/false,
"history_truncation": true/false,
"all_components_present": true/false,
"missing_components": [
"name1", "name2", ...
],
"notes": "Cursor may include additional findings here."
}
