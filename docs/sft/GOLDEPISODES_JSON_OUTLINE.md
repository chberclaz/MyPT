# MyPT Gold Episode JSON Outline

This document defines the **canonical JSONL structure**
for gold and synthetic SFT tool episodes.

Gold episodes may contain placeholders.
Toolresults and final answers can be auto-filled later.

---

## Canonical Episode Schema

Each line in `gold_toolcalls.jsonl` represents ONE episode.

```json
{
  "episode_id": "GOLD-WS-001",
  "meta": {
    "category": "workspace | repo | observability | cross | no_tool | error",
    "pattern": "SEARCH_ANSWER | LIST_SELECT_INTERACTIVE | LIST_SELECT_AUTONOMOUS | ...",
    "language": "en | de",
    "source": "gold"
  },
  "messages": [
    { "role": "user", "content": "..." },

    {
      "role": "assistant",
      "content": "<myPT_toolcall>{\"name\":\"tool.name\",\"arg\":\"value\"}</myPT_toolcall>"
    },

    {
      "role": "toolresult",
      "name": "tool.name",
      "content": "__TOOLRESULT_FILL__"
    },

    {
      "role": "assistant",
      "content": "__FINAL_ANSWER_FILL__"
    }
  ]
}
```
