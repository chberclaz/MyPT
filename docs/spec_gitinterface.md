CURSOR SPEC — Git Repository Interface (Code-Aware RAG)

Objective: High-performance RAG over local Git repos with incremental indexing + code-aware chunking (Python/JS/Java) + hybrid retrieval (vector + lexical). No GitHub API, no internet.

Constraints

Must work on local filesystem repos only

Offline-only: no network calls

Deterministic output

Skip unchanged blobs (incremental indexing by blob SHA)

Deliverables

Create module tree:

core/repo/
**init**.py
git_repo.py
repo_indexer.py
repo_search.py
repo_metadata.py
repo_chunkers/
base.py
python_chunker.py
js_chunker.py
java_chunker.py

1.1 GitRepo adapter

File: core/repo/git_repo.py

Implement:

current_commit() -> str

list_tracked_files() -> list[dict] returning:

path, blob_sha, size_bytes, language

changed_files_since(commit_sha: str) -> list[dict] (only changed blobs)

read_file(path: str) -> str (safe, text-only)

Implementation notes:

Use Git plumbing via subprocess (offline):

git rev-parse HEAD

git ls-tree -r -l HEAD

git diff --name-only <sha>..HEAD

git cat-file -p <blob>

File filtering:

ignore .git/, binaries, node_modules/, dist/, build/, .venv/

skip files > configurable max size (default 2–5MB)

Language detection by extension map.

1.2 Code chunkers (Python / JS / Java)

File: core/repo/repo_chunkers/base.py

Define:

@dataclass
class CodeChunk:
text: str
file_path: str
language: str
symbol_kind: str # function|class|method|module
symbol_name: str
signature: str
line_start: int
line_end: int
extra: dict

BaseRepoChunker.chunk(code: str, file_path: str) -> list[CodeChunk]

Python chunker

Use Python ast module

Chunk per function + class (include docstring + signature + body)

Track line ranges

JS/TS chunker

Use a simple regex-based fallback initially (offline)

Chunk:

function foo(...)

class Foo

export function

export class

Capture JSDoc block if directly above symbol

(Later you can add a proper parser, but keep MVP pure + offline.)

Java chunker

Regex-based MVP:

class X

method signatures within classes

Include annotations above methods/classes

1.3 RepoIndexer

File: core/repo/repo_indexer.py

Responsibilities:

For each tracked file:

determine if changed via blob_sha

chunk via language chunker

embed chunks (reuse workspace embedding pipeline)

write to workspace vector index with metadata:

repo_path, file_path, language, symbol_name, line_start/end, blob_sha, commit_sha

Store repo index state:

workspace/index/<repo_index_name>/repo_state.json containing:

last indexed commit

mapping of file_path -> blob_sha

chunk ids

1.4 Hybrid retrieval

File: core/repo/repo_search.py

Implement:

lexical_search(query, k, path_glob?, language?) using local ripgrep (preferred) or Python fallback

vector_search(query, k, filters)

fused_search(query, k, filters) that merges + reranks

Rerank strategy (simple):

normalize scores

boost lexical exact identifier matches

return final top-k chunks with provenance

Acceptance tests

Index repo once → second run should reindex only changed files

fused_search("loss_mask") returns correct code/document location

Chunk metadata includes symbol name and line ranges

No internet access required
