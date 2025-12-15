# Workspace API Reference

This document describes the workspace tools and agent system for agentic RAG.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   AgentController                            │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  1. Build prompt from history                           ││
│  │  2. Generate with model                                 ││
│  │  3. Parse toolcall (if any)                            ││
│  │  4. Execute tool → inject result → loop                ││
│  │  5. Return final answer                                 ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
            │                           │
            ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐
│   WorkspaceTools      │   │      Model            │
│   - search            │   │   - generate()        │
│   - list_docs         │   │                       │
│   - get_doc           │   │                       │
│   - summarize         │   │                       │
└───────────────────────┘   └───────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│                    WorkspaceEngine                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │   Documents     │  │    Chunks       │  │   Retriever   │ │
│  │   (metadata)    │  │   (indexed)     │  │   (search)    │ │
│  └─────────────────┘  └─────────────────┘  └───────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

## Core Components

### WorkspaceEngine

Central abstraction for document and chunk management.

```python
from core.workspace import WorkspaceEngine

engine = WorkspaceEngine(
    base_dir="workspace/",      # Root workspace directory
    index_dir="workspace/index/latest"  # RAG index location
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_docs` | int | Number of documents |
| `num_chunks` | int | Number of indexed chunks |
| `has_index` | bool | Whether RAG index is loaded |

#### Methods

**`list_docs() -> List[Document]`**

List all documents in the workspace.

```python
docs = engine.list_docs()
for doc in docs:
    print(f"{doc.title} ({doc.doc_id})")
```

**`get_doc(doc_id: str) -> Document | None`**

Get document metadata by ID.

```python
doc = engine.get_doc("abc123")
if doc:
    print(f"Title: {doc.title}, Path: {doc.path}")
```

**`get_doc_text(doc_id: str) -> str | None`**

Get full document text content.

```python
text = engine.get_doc_text("abc123")
print(text[:500])
```

**`search(query: str, top_k: int = 5) -> List[Chunk]`**

Search for relevant chunks using semantic similarity.

```python
chunks = engine.search("machine learning", top_k=3)
for chunk in chunks:
    print(f"[{chunk.score:.2f}] {chunk.text[:100]}...")
```

**`reload_index(index_dir: str = None)`**

Hot-reload the RAG index.

```python
engine.reload_index("workspace/index/v2")
```

### Data Models

**Document**

```python
@dataclass
class Document:
    doc_id: str      # Unique ID (hash of path)
    title: str       # Filename without extension
    path: str        # Full file path
    created_at: float
    updated_at: float
```

**Chunk**

```python
@dataclass
class Chunk:
    chunk_id: str    # Unique ID
    doc_id: str      # Parent document ID
    text: str        # Chunk content
    position: int    # Position in document
    score: float     # Similarity score (from search)
    metadata: dict   # Source info (filename, lines, etc.)
```

### WorkspaceTools

Tool implementations that the agent can call.

```python
from core.workspace import WorkspaceEngine, WorkspaceTools

engine = WorkspaceEngine("workspace/", "workspace/index/latest")
tools = WorkspaceTools(engine)

# Execute a tool
result = tools.execute("workspace.search", {"query": "python", "top_k": 5})
```

#### Available Tools

**`workspace.search`**

Search documents by semantic similarity.

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `query` | str | Yes | - | Search query |
| `top_k` | int | No | 5 | Number of results |

Returns:
```json
{
  "documents": [
    {"chunk_id": "0", "doc_id": "abc", "text": "...", "score": 0.85, "filename": "doc.md"}
  ],
  "total": 5
}
```

**`workspace.list_docs`**

List all documents in workspace.

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| (none) | - | - | - | - |

Returns:
```json
{
  "documents": [
    {"doc_id": "abc", "title": "Python Guide", "path": "workspace/docs/python.md"}
  ],
  "total": 10
}
```

**`workspace.get_doc`**

Get full document text.

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `doc_id` | str | No* | - | Document ID |
| `title` | str | No* | - | Document title |

*One of `doc_id` or `title` is required.

Returns:
```json
{
  "doc_id": "abc",
  "title": "Python Guide",
  "text": "Full document content...",
  "length": 5000
}
```

**`workspace.summarize`**

Summarize a document or text.

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `doc_id` | str | No* | - | Document ID to summarize |
| `text` | str | No* | - | Text to summarize |
| `max_length` | int | No | 500 | Maximum summary length |

*One of `doc_id` or `text` is required.

Returns:
```json
{
  "summary": "This document describes...",
  "source": "doc_id",
  "original_length": 5000
}
```

### AgentController

Orchestrates model generation with tool execution.

```python
from core.agent import AgentController
from core.workspace import WorkspaceEngine, WorkspaceTools

engine = WorkspaceEngine("workspace/", "workspace/index/latest")
tools = WorkspaceTools(engine)
controller = AgentController(model, tools)
```

#### Constructor Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model` | - | Required | Model with `generate()` method |
| `tools` | WorkspaceTools | Required | Tool executor |
| `system_prompt` | str | DEFAULT | System prompt for agent |
| `max_result_chars` | int | 2000 | Max chars for tool results |

#### Methods

**`run(history, max_steps=5, max_new_tokens=512, verbose=False) -> dict`**

Run the agent loop.

```python
result = controller.run([
    {"role": "user", "content": "Find docs about Python"}
], max_steps=5)

print(result["content"])  # Final answer
print(result["tool_calls"])  # List of tool calls made
print(result["steps"])  # Number of steps taken
```

**Return Value:**
```python
{
    "role": "assistant",
    "content": "Based on my search...",
    "tool_calls": [
        {"name": "workspace.search", "arguments": {...}, "result": {...}}
    ],
    "steps": 2
}
```

**`run_single(user_input, system=None, **kwargs) -> dict`**

Convenience method for single-turn interaction.

```python
result = controller.run_single("What documents do we have?")
print(result["content"])
```

## Toolcall Format

The agent uses a specific format for tool calls:

```
<myPT_toolcall>{"name": "workspace.search", "query": "python", "top_k": 5}</myPT_toolcall>
```

Tool results are injected as:

```
<myPT_toolresult>{"documents": [...], "total": 5}</myPT_toolresult>
```

### Parsing Utilities

```python
from core.agent import find_toolcall, render_toolcall, render_toolresult

# Parse toolcall from text
tc = find_toolcall(model_output)
if tc:
    print(f"Tool: {tc.name}, Args: {tc.arguments}")

# Render toolcall
block = render_toolcall("workspace.search", {"query": "python"})
# -> <myPT_toolcall>{"name": "workspace.search", "query": "python"}</myPT_toolcall>

# Render tool result
result_block = render_toolresult({"documents": [...], "total": 5})
# -> <myPT_toolresult>{"documents": [...], "total": 5}</myPT_toolresult>
```

## CLI Usage

### workspace_chat.py

Interactive agent chat:

```bash
python scripts/workspace_chat.py \
    --model_name my_agent \
    --workspace_dir workspace/ \
    --index_dir workspace/index/latest \
    --verbose
```

Commands:
- `/reload` - Reload workspace and index
- `/docs` - List documents
- `/tools` - Show available tools
- `/history` - Show conversation history
- `/clear` - Clear history
- `/verbose` - Toggle verbose mode
- `/quit` - Exit

## Extending Tools

### Adding a New Tool

1. Add implementation in `core/workspace/tools.py`:

```python
def workspace_new_tool(self, args: Dict[str, Any]) -> Dict[str, Any]:
    """New tool implementation."""
    param = args.get("param", "default")
    # ... implementation ...
    return {"result": "..."}
```

2. Register in `__init__`:

```python
self._tools: Dict[str, Callable] = {
    # ... existing tools ...
    "workspace.new_tool": self.workspace_new_tool,
}
```

3. Add to `TOOL_REGISTRY`:

```python
TOOL_REGISTRY = {
    # ... existing ...
    "workspace.new_tool": {
        "description": "Description of new tool",
        "args": {
            "param": {"type": "str", "required": False, "default": "default"}
        }
    }
}
```

4. Update system prompt if needed.

5. Create training data with the new tool.

## See Also

- [Toolcall SFT Training](toolcall_sft.md) - Training for tool use
- [Chat SFT with Context](chat_sft_with_context.md) - Basic RAG training
- [Special Tokens](SPECIAL_TOKENS.md) - Token definitions



