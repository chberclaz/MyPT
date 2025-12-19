CURSOR SPEC — Generic Observability Backend Tool Interface

Objective: Add an observability tool adapter layer so MyPT can query Splunk-like systems (or Loki/OpenSearch) on demand via toolcalls. This is not a SIEM replacement (no streaming ingestion).

Constraints

100% offline / self-hosted

Pull-based queries only

Deterministic JSON outputs

Strict allow-list + args validation

Do NOT build ingestion/indexing pipelines

Deliverables

Create:

core/tools/observability/
**init**.py
base.py
registry.py
loki_backend.py # optional first implementation
opensearch_backend.py # optional first implementation
splunk_backend.py # optional depending on your environment

2.1 Base interface

File: core/tools/observability/base.py

Define ObservabilityBackend:

search_logs(query: str, start: str, end: str, limit: int, fields: list[str]|None=None, order: str="desc") -> dict

count_logs(query: str, start: str, end: str) -> dict

aggregate_logs(query: str, start: str, end: str, group_by: list[str], limit: int) -> dict

Optional:

get_alert(alert_id: str) -> dict

list_alerts(start: str, end: str, limit: int=100) -> dict

Normalize outputs to a common schema:

events: list of normalized event objects:

ts, source, host, service, level, message, trace_id, raw

summary: optional counts/buckets

next_cursor: optional pagination cursor

2.2 Tool registry

File: core/tools/observability/registry.py

Expose allow-listed tools:

obs.search_logs

obs.count_logs

obs.aggregate_logs

obs.get_alert (optional)

obs.list_alerts (optional)

All tools must:

validate args types

cap limit (e.g., max 500)

require time windows (start/end)

2.3 Agent integration

Add observability tools to your main tool dispatcher (like WorkspaceTools.execute)

Ensure AgentController can call obs.\* like any other tool

Toolresults must be truncatable via existing truncate_toolresult

Acceptance tests

Toolcall JSON → backend query executed → normalized toolresult returned

Invalid args rejected with helpful error

Large results truncated cleanly

No background ingestion; each call is explicit
