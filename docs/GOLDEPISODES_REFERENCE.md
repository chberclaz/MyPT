# MyPT Gold Episode Reference (Agentic RAG)

This document defines the **gold episode structure and coverage plan** for
agentic RAG + toolcalling behavior in MyPT.

Gold episodes are **behavioral specifications**, not knowledge storage.
They teach the model:

- when to call tools
- which tool to use
- how to chain tools
- how to ask clarifying questions
- how to recover from errors
- how to respond in English or German

They are written against a **staging workspace**, not customer data.

---

## Categories

- Workspace (docs / RAG)
- Repo (code search)
- Observability (logs / events)
- Cross-family (logs → code / docs)
- No-tool-needed
- Error recovery

---

## Workspace Episodes (GOLD-WS-001 … GOLD-WS-080)

### A) Search → Answer (core pattern)

- GOLD-WS-001 (EN): search loss_mask → explain
- GOLD-WS-002 (DE): suche loss_mask → erklären
- GOLD-WS-003 (EN): search toolcall format → explain
- GOLD-WS-004 (DE): suche toolcall format → erklären
- …
- GOLD-WS-035

### B1) List → Select → Summarize (INTERACTIVE)

Model asks the user to choose.

- GOLD-WS-036 (EN): list_docs → ask user which → summarize
- GOLD-WS-037 (DE): list_docs → rückfrage → zusammenfassen
- …
- GOLD-WS-045

### B2) List → Select → Summarize (AUTONOMOUS)

Model selects based on evidence (search).

- GOLD-WS-046 (EN): list_docs → search → pick → summarize
- GOLD-WS-047 (DE): liste → suche → auswählen → zusammenfassen
- …
- GOLD-WS-060

### C) Get_doc → Answer

- GOLD-WS-061 (EN): get_doc by doc_id → explain
- GOLD-WS-062 (DE): get_doc per titel → erklären
- …
- GOLD-WS-070

### D) Summarize

- GOLD-WS-071 (EN): summarize doc_id
- GOLD-WS-072 (DE): fasse text zusammen
- …
- GOLD-WS-075

### E) Workspace Errors & Recovery

- GOLD-WS-076 (EN): search without query → recover
- GOLD-WS-077 (DE): search ohne query → rückfrage
- GOLD-WS-078 (EN): doc not found → list_docs fallback
- GOLD-WS-079 (DE): dokument nicht gefunden → fallback
- GOLD-WS-080

---

## Repo Episodes (GOLD-REPO-081 … GOLD-REPO-110)

### F) Repo Search → Explain

- GOLD-REPO-081 (EN): find loss_mask implementation
- GOLD-REPO-082 (DE): finde loss_mask im code
- …
- GOLD-REPO-095

### G) Repo Search → Get File → Explain

- GOLD-REPO-096 (EN): search symbol → get_file → explain
- GOLD-REPO-097 (DE): suche symbol → datei lesen → erklären
- …
- GOLD-REPO-105

### H) Repo Errors

- GOLD-REPO-106 (EN): no results → broaden search
- GOLD-REPO-107 (DE): keine treffer → suchbegriff anpassen
- …
- GOLD-REPO-110

---

## Observability Episodes (GOLD-OBS-111 … GOLD-OBS-140)

### I) Count / Aggregate / Drilldown

- GOLD-OBS-111 (EN): count errors last hour
- GOLD-OBS-112 (DE): zähle errors letzte stunde
- GOLD-OBS-113 (EN): aggregate by service
- GOLD-OBS-114 (DE): gruppiere nach service
- GOLD-OBS-115 (EN): drilldown logs
- GOLD-OBS-116 (DE): drilldown logs
- …
- GOLD-OBS-125

### J) Trace / Correlation

- GOLD-OBS-126 (EN): search logs by trace_id
- GOLD-OBS-127 (DE): logs nach trace_id
- …
- GOLD-OBS-132

### K) Observability Errors

- GOLD-OBS-133 (EN): invalid time window → fix
- GOLD-OBS-134 (DE): falsches zeitformat → korrigieren
- …
- GOLD-OBS-140

---

## Cross-Family Episodes (GOLD-X-141 … GOLD-X-150)

### L) Logs → Code

- GOLD-X-141 (EN): logs show exception → repo search → explain
- GOLD-X-142 (DE): logs → code
- GOLD-X-143 (EN): logs → runbook docs
- GOLD-X-144 (DE): logs → dokumentation
- GOLD-X-145 … GOLD-X-150 (mixed)

---

## No-Tool Episodes (distributed)

~10–15% across all ranges.

Examples:

- definitions
- conceptual explanations
- language switching
- clarification-only responses

---

## Language Distribution

- ~50–65% English
- ~35–50% German
- Respond in the user’s language
- Toolcalls always remain English/JSON

---

## Important Rules

- Toolcalls must be valid JSON
- Toolresults must be real or auto-filled later
- Final answers must reference toolresults
- Assistant tokens are the only ones trained (loss masking)

Gold episodes define **how to behave**, not **what to know**.
