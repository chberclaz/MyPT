# GGUF export and how to run the exported model

**Tagline:** On-Premise governed AI Foundry

This directory converts a MyPT PyTorch GOLD checkpoint into **GGUF** (llama.cpp) and is the supported way to serve that file from the **MyPT RAG web UI** or from the **console**.

The testcase checkpoint is **`phase6_3_ground_gold`** (750M, curriculum 6.3 iter1). Tokenizer internals: [`TOKENIZER_NOTES.md`](TOKENIZER_NOTES.md). Tag source of truth: [`core/special_tokens.py`](../core/special_tokens.py) — never hand-edit `special_tokens.json`.

---

## Contents

1. [What this is](#what-this-is)
2. [What you need](#what-you-need)
3. [Artifacts](#artifacts)
4. [Special tags (required reading)](#special-tags-required-reading)
5. [Export the checkpoint](#export-the-checkpoint)
6. [Verify](#verify)
7. [Use in the MyPT RAG web UI](#use-in-the-mypt-rag-web-ui)
8. [CLI: workspace chat and RAG](#cli-workspace-chat-and-rag)
9. [llama-cli / llama-server (no tools)](#llama-cli--llama-server-no-tools)
10. [Third-party UIs (LM Studio / Ollama)](#third-party-uis-lm-studio--ollama)
11. [Which quant to ship](#which-quant-to-ship)
12. [Known limits](#known-limits)
13. [Troubleshooting](#troubleshooting)
14. [File map](#file-map)

---

## What this is

MyPT trains in PyTorch (`checkpoints/<name>/model.pt`). Deployment is a **single GGUF file** plus the pinned llama.cpp binaries:

- No CUDA driver matching, no multi-GB torch install on the customer box.
- Q4 is the “runs on ~4 GB VRAM / Pi 5” claim.
- Tool execution is **ours**. llama.cpp only generates tokens. The MyPT `AgentController` parses `<myPT_toolcall>`, runs workspace tools, and injects `<myPT_toolresult>`.

Path A: remap MyPT weights to `LlamaForCausalLM`, then the stock `convert_hf_to_gguf.py` from the **pinned** llama.cpp tree.

---

## What you need

| Piece | Where | Notes |
| --- | --- | --- |
| GOLD checkpoint | `checkpoints/phase6_3_ground_gold/` | `model.pt` + `config.json` + tokenizer state |
| llama.cpp **source** | `third_party/llama.cpp` | SHA in [`LLAMACPP_PIN.txt`](LLAMACPP_PIN.txt) (`sha=1b89a43e…`). Used for `convert_hf_to_gguf.py` and `gguf-py`. |
| llama.cpp **binaries** | `third_party/llama.cpp-bin/` | GitHub release **b10731** (Vulkan on Windows). `llama-cli`, `llama-server`, `llama-quantize`, `llama-tokenize`. |
| Python | same env as MyPT | `torch`, `transformers`, `tokenizers`, `tiktoken`, `safetensors`, `sentencepiece`, `gguf` from **this** tree (not PyPI). |

Pin record:

```
sha=1b89a43e3835f0c8bbef5543977151972874a9ce
binaries_tag=b10731
gguf_py=third_party/llama.cpp/gguf-py
```

If the converter or a metadata key disagrees with this README, **the pinned tree wins**. Update the pin file and this document together.

GGUF files and `export/artifacts/` are gitignored (`*.gguf`). They live next to the checkpoint after a local export.

---

## Artifacts

After a full export of 6.3 GOLD:

| File | Role |
| --- | --- |
| `checkpoints/<gold>/mypt-f16.gguf` | Reference F16 |
| `…/mypt-q8_0.gguf` | Near-lossless |
| `…/mypt-q6_k.gguf` | Fallback if Q4 format slips |
| `…/mypt-q5_k_m.gguf` | Intermediate |
| `…/mypt-q4_k_m.gguf` | **Product file** |
| `…/mypt-*.lineage.json` | Sidecar lineage (checkpoint id, llama.cpp SHA, special IDs) |
| `export/hf_tokenizer/` | Committed tiktoken→HF tokenizer (hashed) |
| `export/artifacts/hf_llama/` | Intermediate Llama-shaped HF dir (gitignored) |
| `export/special_tokens.json` | Generated snapshot of the 19 packer tags |
| `export/chat_template.jinja` | Embedded in GGUF as `tokenizer.chat_template` |

Measured quant sizes for this project's GOLD are not published.

---

## Special tags (required reading)

The packer uses **19** CONTROL tokens, IDs **50257–50275**, vocab padded to **50304**. Surfaces are exactly the strings in `SPECIAL_TOKEN_STRINGS`:

| IDs | Tags |
| ---: | --- |
| 50257–50262 | `<myPT_system>` `</myPT_system>` `<myPT_user>` `</myPT_user>` `<myPT_assistant>` `</myPT_assistant>` |
| 50263–50266 | `<myPT_user_context>` … `<myPT_assistant_context>` pair |
| 50267–50270 | `<myPT_toolcall>` `</myPT_toolcall>` `<myPT_toolresult>` `</myPT_toolresult>` |
| 50271–50274 | `<myPT_think>` `</myPT_think>` `<myPT_cite>` `</myPT_cite>` |
| 50275 | `<myPT_eot>` |

**Current artifacts: this is already done.** F16 and Q4_K_M both store all 19 tags as GGUF type **CONTROL (3)** at packer IDs 50257–50275. llama.cpp re-tokenization of a later turn keeps each tag as **one** id, for example:

```text
<myPT_toolresult>{"ok":true}</myPT_toolresult>
→ [50269, 90, 482, 25, 7942, 92, 50270]
```

`50269` / `50270` are the open/close tags; only the JSON payload is ordinary BPE. `python export/verify_gguf.py` and `export/tests/test_special_token_types.py` gate this.

**If they were `NORMAL` (the bug the spec warns about):** generation would still look fine (IDs detokenize). The failure is **prompt re-tokenization** on the next turn: llama.cpp would split `<myPT_toolresult>` into subwords the model never trained on. That is a silent Phase 4/5/6-only bug. It is **not** the state of these files.

`--no-parse-special` *intentionally* fragments the same string (untrusted user/document text). That is the trust model below, not a missing CONTROL flag.

Regenerate the JSON (do not edit it):

```powershell
py -3 export/generate_special_tokens.py
```

### Trust model

| Segment | Parse specials? | Mechanism |
| --- | --- | --- |
| System prompt, runtime-inserted **tags** around tool results | Yes — by ID | MyPT `Tokenizer.encode()` / `GGUFModel` sends token IDs to llama-server |
| User text, retrieved documents, tool **payload** JSON | No | `encode_ordinary()`; ingest `strip_special_tag_strings()` |

llama.cpp **does** parse specials out of raw prompt strings by default. A document that contains the literal `<myPT_toolresult>` becomes a real CONTROL token unless you use the MyPT wrapper (IDs) or `--no-parse-special`. The web UI and `workspace_chat` / `workspace_rag` use the wrapper.

---

## Export the checkpoint

Run from the repo root. On Windows use `py -3` (or `python`) as below; `export/convert.sh` / `quantize.sh` are the same steps for bash.

Set `PYTHONPATH` so `gguf-py` comes from the pin, not PyPI:

```powershell
$env:PYTHONPATH = "$(Get-Location);$(Get-Location)\third_party\llama.cpp\gguf-py"
```

### 1. Tokenizer (once per tokenizer version)

Rebuilds `export/hf_tokenizer/` from tiktoken `gpt2` + `core.special_tokens`. Output is committed; a tokenizer change is a visible HASH diff.

```powershell
py -3 export/tiktoken_to_hf.py
```

HASH is written to `export/hf_tokenizer/HASH.txt`.

### 2. Path A: MyPT → Llama-shaped HF

```powershell
py -3 export/to_hf_llama.py --ckpt checkpoints/phase6_3_ground_gold --out export/artifacts/hf_llama
```

Copies the HF tokenizer into that directory. Logs `lm_head.bias` (dropped; HF Llama has none). A non-trivial bias is expected to hurt exact greedy match vs PyTorch.

### 3. HF → GGUF F16

First convert may abort with “BPE pre-tokenizer was not recognized”. That is expected. `convert_to_gguf.py` registers the fingerprint → `gpt-2` in the cloned `conversion/base.py` and retries.

```powershell
py -3 export/convert_to_gguf.py --hf export/artifacts/hf_llama --out checkpoints/phase6_3_ground_gold/mypt-f16.gguf
```

Writes `mypt-f16.lineage.json` next to the GGUF.

### 4. Quantize the product matrix

```powershell
py -3 export/quantize.py --src checkpoints/phase6_3_ground_gold/mypt-f16.gguf --outdir checkpoints/phase6_3_ground_gold
```

Produces Q8_0, Q6_K, Q5_K_M, Q4_K_M and copies lineage sidecars with `mypt.lineage.quant_type` set.

One-shot bash: `bash export/convert.sh` then `bash export/quantize.sh`.

---

## Verify

```powershell
$env:PYTHONPATH = "$(Get-Location);$(Get-Location)\third_party\llama.cpp\gguf-py"

py -3 export/verify_gguf.py checkpoints/phase6_3_ground_gold/mypt-f16.gguf
py -3 export/verify_lineage.py checkpoints/phase6_3_ground_gold/mypt-q4_k_m.gguf
py -3 -m pytest export/tests -q
```

`verify_gguf.py` asserts all 19 tags at packer IDs with token type CONTROL (or USER_DEFINED). `verify_lineage.py` is customer-runnable (JSON sidecar, no torch).

Spot-check llama.cpp IDs (must be a **single** id):

```powershell
.\third_party\llama.cpp-bin\llama-tokenize.exe -m checkpoints\phase6_3_ground_gold\mypt-q4_k_m.gguf --no-bos --ids -p "<myPT_toolcall>"
```

Expect `[50267]`. With `--no-parse-special` the same string fragments.

Optional remap gate (slow; loads PyTorch then llama-server):

```powershell
py -3 export/parity_pytorch_vs_gguf.py --n 128 --limit 20
```

Exact 128-token identity vs PyTorch is **not** currently a pass (see [Known limits](#known-limits)). Tokenizer / CONTROL registration is.

---

## Use in the MyPT RAG web UI

This is the **product** path: chat page + workspace tools + `AgentController`. GGUF is loaded through `GGUFModel`, which starts `llama-server` and **tokenizes with the MyPT packer tokenizer** (specials are IDs, not re-split strings).

### 1. Files in place

- `checkpoints/phase6_3_ground_gold/mypt-q4_k_m.gguf` (or F16 / other quants)
- `third_party/llama.cpp-bin/llama-server.exe` (or `llama-server` on Unix)

The picker lists every `checkpoints/<dir>/*.gguf` as `<dir>/<filename>`, e.g.

```text
phase6_3_ground_gold/mypt-q4_k_m.gguf
```

PyTorch GOLD still appears as `phase6_3_ground_gold` (the folder, if `model.pt` exists). Pick the **`.gguf` row** for llama.cpp.

### 2. Workspace index (agentic RAG)

Put documents under `workspace/docs/` and build the index (or use **Rebuild Index** on the chat page):

```powershell
py -3 scripts/build_rag_index.py
```

Without an index, `workspace.search` returns empty; conversation mode still works.

Ingest strips forged myPT tag strings from documents (`strip_special_tag_strings`).

### 3. Start the app

```powershell
pip install -e ".[webapp]"
py -3 -m webapp.main
```

Open http://localhost:8000/chat

### 4. Chat settings

| Control | What to choose |
| --- | --- |
| **Model** | `phase6_3_ground_gold/mypt-q4_k_m.gguf` (product) or `…/mypt-f16.gguf` (debug) |
| **Mode** | **Agentic (RAG)** — tools (`workspace.search`, `list_docs`, `get_doc`, `summarize`). **Conversation** — chat SFT prompt, no tools. |

First GGUF load starts `llama-server` (Vulkan GPU layers `-ngl 99`, context 4096). Wait until the first reply; a 503 during warmup means the server was not healthy yet (the backend now waits on `/health`).

The controller:

1. Builds a packer-shaped prompt (`<myPT_system>` … `<myPT_assistant>` ).
2. Generates until `</myPT_assistant>`, `<myPT_eot>`, or `</myPT_toolcall>`.
3. If a toolcall is present, executes the allow-listed tool, appends `<myPT_toolresult>…</myPT_toolresult>`, and continues.

Ollama/LM Studio **cannot** do step 3. Stay in this UI or the **CLI below** for RAG/tools.

---

## CLI: workspace chat and RAG

Same stack as the web UI: `load_runtime_model` → MyPT packer encode → `AgentController` → workspace tools. Works with a PyTorch GOLD **folder** or a **`.gguf`** picker id.

`generate.py` is still PyTorch-only (plain completion, no tools).

### Model name

| You pass | Loads |
| --- | --- |
| `phase6_3_ground_gold` | PyTorch `checkpoints/phase6_3_ground_gold/model.pt` |
| `phase6_3_ground_gold/mypt-q4_k_m.gguf` | GGUF next to that GOLD (starts `llama-server`) |
| `<path>/mypt-q4_k_m.gguf` | That file |
| `gguf/mypt-q4_k_m.gguf` | `export/artifacts/` |

List everything the picker knows:

```powershell
py -3 scripts/workspace_chat.py --list-models
# or:  mypt-workspace-chat --list-models
```

Need `third_party/llama.cpp-bin/llama-server` on PATH-equivalent (repo tree) for GGUF. `--ngl 0` is CPU-only llama.cpp.

### Interactive: `workspace_chat`

```powershell
py -3 scripts/workspace_chat.py --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf
py -3 scripts/workspace_chat.py --model_name phase6_3_ground_gold --mode conversation
```

Installed entry point: `mypt-workspace-chat`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model_name` | required | Folder, `dir/file.gguf`, or path |
| `--workspace_dir` | `workspace/` | Docs + index root |
| `--index_dir` | `<workspace>/index/latest` | RAG index |
| `--mode` | `agentic` | `agentic` (tools) or `conversation` (no tools) |
| `--max_steps` | 5 | Tool-loop cap |
| `--max_tokens` | 512 | Tokens per generate |
| `--system` | packer default | Override system prompt |
| `--ngl` | 99 | GGUF GPU layers (`0` = CPU) |
| `--port` | first free ≥8765 | llama-server port |
| `--n_ctx` | 4096 | Context |
| `--verbose` / `-v` | off | Print prompts and tool calls |
| `--query` | unset | One-shot; no REPL. `-` = stdin |
| `--json` | off | With `--query`: one JSON object on stdout |

Slash commands in the REPL:

| Command | Action |
| --- | --- |
| `/docs` | List workspace documents |
| `/tools` | List allow-listed tools |
| `/reload` | Refresh docs + index |
| `/history` | Show turns |
| `/clear` | Drop history |
| `/verbose` | Toggle debug |
| `/quit` | Exit (also closes llama-server) |

Ctrl+C / EOF also close the GGUF server so the port is not left bound.

### One-shot JSON for other applications: `workspace_rag`

Always prints **one JSON object** to stdout. Logs on stderr. Exit `0` if `"ok": true`.

```powershell
py -3 scripts/workspace_rag.py --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf --query "Which documents do we have?"
```

Installed entry point: `mypt-workspace-rag`.

Pipe a question:

```powershell
"What is in the handbook?" | py -3 scripts/workspace_rag.py --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf --query -
```

Same payload from the interactive CLI:

```powershell
py -3 scripts/workspace_chat.py --model_name phase6_3_ground_gold/mypt-q4_k_m.gguf --query "List the documents" --json
```

JSON shape:

```json
{
  "ok": true,
  "content": "assistant text",
  "tool_calls": [
    {"name": "workspace.search", "arguments": {"query": "..."}, "result": {}}
  ],
  "steps": 2,
  "error": null,
  "model": "phase6_3_ground_gold/mypt-q4_k_m.gguf",
  "backend": "gguf",
  "mode": "agentic"
}
```

`backend` is `gguf` or `pytorch`. Flags match `workspace_chat` (`--workspace_dir`, `--index_dir`, `--mode`, `--max_steps`, `--max_tokens`, `--ngl`, `--port`, `--n_ctx`, `--system`, `--verbose`).

### Python (embed in another app)

```python
from core.agent import workspace_ask
from core.inference import load_runtime_model, close_runtime_model, list_runtime_models

print(list_runtime_models())

result = workspace_ask(
    "phase6_3_ground_gold/mypt-q4_k_m.gguf",
    "List the documents",
    workspace_dir="workspace",
    mode="agentic",
)
print(result["content"], result["tool_calls"])
```

For many turns without restarting llama-server, load once and pass `keep_model=`:

```python
model = load_runtime_model("phase6_3_ground_gold/mypt-q4_k_m.gguf")
try:
    hist = []
    r = workspace_ask(
        "phase6_3_ground_gold/mypt-q4_k_m.gguf",
        "Hello",
        history=hist,
        keep_model=model,
    )
    hist.append({"role": "user", "content": "Hello"})
    hist.append({"role": "assistant", "content": r["content"]})
    r2 = workspace_ask(
        "phase6_3_ground_gold/mypt-q4_k_m.gguf",
        "What did I just say?",
        history=hist,
        keep_model=model,
    )
    print(r2["content"])
finally:
    close_runtime_model(model)
```

Index first (`py -3 scripts/build_rag_index.py` or **Rebuild Index** in the UI) or `workspace.search` returns empty.

---

## llama-cli / llama-server (no tools)

Raw llama.cpp **re-tokenizes the `-p` string**. Tags in the prompt become CONTROL (good). Tags inside *user-copied document text* also become CONTROL (the injection surface). Fine for a hello-world; not the RAG product.

PowerShell:

```powershell
$m = "checkpoints\phase6_3_ground_gold\mypt-q4_k_m.gguf"
$cli = "third_party\llama.cpp-bin\llama-cli.exe"
$prompt = "<myPT_system>You are MyPT.</myPT_system>`n<myPT_user>Say hello in one word.</myPT_user>`n<myPT_assistant> "
& $cli -m $m -ngl 99 -c 4096 -n 128 --temp 0.7 --repeat-penalty 1.1 --no-warmup `
  --reverse-prompt "</myPT_assistant>" --reverse-prompt "<myPT_eot>" `
  -p $prompt
```

`--jinja` uses the embedded chat template (for generic `messages` UIs). Our SFT packer layout is the `<myPT_*>` blocks above, not ChatML.

### llama-server only

The web UI / `GGUFModel` / `workspace_chat` already spawn this. To attach a debugger or a custom client:

```powershell
.\third_party\llama.cpp-bin\llama-server.exe -m checkpoints\phase6_3_ground_gold\mypt-q4_k_m.gguf --port 8765 --ctx-size 4096 -ngl 99
```

POST `/completion` with `"prompt": [token ids, …]` (not a string) if you want packer IDs. String prompts will parse specials.

---

## Third-party UIs (LM Studio / Ollama)

Point them at `mypt-q4_k_m.gguf`. The file includes `tokenizer.chat_template` and CONTROL specials.

They will **print** `<myPT_toolcall>{…}</myPT_toolcall>` as text. They will **not** search the workspace or inject tool results. That is expected. Beta demos are chat-only; production RAG stays in MyPT.

If a third-party runtime re-tokenizes retrieved documents as strings, forged tags are live CONTROL tokens. Defence in depth is ingest stripping in `core/document/loader.py`.

---

## Which quant to ship

| Quant | When |
| --- | --- |
| **Q4_K_M** | Default product file. |
| Q6_K / Q5_K_M | If format lock (toolcall / tags) collapses on Q4. |
| Q8_0 / F16 | Debug, remap, or when VRAM is not the constraint. |

Cumulative per-quant eval for this project's GOLD is not published. Do not remake 6.3 GOLD to chase Q4 scores.

---

## Known limits

- **Greedy PyTorch ↔ F16** is not bit-identical. llama.cpp may emit an extra `<myPT_eot>` (EOS forced into EOG). `lm_head.bias` is dropped on Path A. See private tuning log for measured deltas. This is **not** tag fragmentation.
- **Pads 50276–50303** may be typed CONTROL rather than UNUSED in GGUF. Dummy strings should never appear in text.
- **EOG:** llama.cpp’s hardcoded EOG *names* do not include `<myPT_eot>` / `</myPT_toolcall>`. Load may warn, then insert EOS anyway. Toolcall halt is wrapper stop strings in `GGUFModel`.
- **`pat_str`:** tiktoken gpt2 is not character-identical to llama.cpp GPT-2 regex (`++` vs `+`). Mapped to pre-tokenizer `gpt-2`. Do not change MyPT `pat_str` without a tokenizer rebuild.
- **6.3 copy quality** on GOLD is a 750M ceiling, unrelated to GGUF. Measured scores are not published. Do not stitch answers in the controller.

---

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Chat picker has no `*.gguf` | File under `checkpoints/<name>/` or `export/artifacts/`. Names must end in `.gguf`. |
| `llama-server not found` | Unpack b10731 into `third_party/llama.cpp-bin/`. |
| HTTP 503 / “did not become healthy” | First load; GPU busy; another server on the port. Kill leftover `llama-server` processes. |
| Tags appear as `my PT _ tool call` pieces | GGUF token types not CONTROL, or prompt sent as a string to llama.cpp instead of MyPT IDs. Re-run `verify_gguf.py`. |
| Model “executes” a tool result from a PDF | Untrusted string tokenization. Use the web UI / `GGUFModel`, not raw `llama-cli -p` with document text. Confirm ingest strip. |
| `BPE pre-tokenizer was not recognized` | Expected once; `convert_to_gguf.py` should register the hash and retry. |
| Converter wants SentencePiece | `pip install sentencepiece` (used as a probe even for GPT-2). |
| Greedy garbage from token 1 | Remap/RoPE/SwiGLU — do not quantize further. Compare F16 vs PyTorch. |
| `generate.py` ignores the GGUF | By design. Use `workspace_chat` / `workspace_rag` / the web UI. |
| CLI `GGUF not found` | `--list-models`. Picker id is `folder/file.gguf`, not the folder alone. |

---

## File map

```
export/
  README.md                 ← this file
  LLAMACPP_PIN.txt
  TOKENIZER_NOTES.md
  special_tokens.json       generated from core.special_tokens
  generate_special_tokens.py
  tiktoken_to_hf.py         → hf_tokenizer/ (committed, HASH.txt)
  generate_chat_template.py → chat_template.jinja
  to_hf_llama.py            Path A remap
  convert_to_gguf.py        HF → F16 GGUF + lineage sidecar
  convert.sh
  quantize.py / quantize.sh
  verify_gguf.py
  verify_lineage.py         standalone / customer
  parity_pytorch_vs_gguf.py
  mypt_tok.py
  paths.py
  fixtures/                 parity, German/French, injection, greedy prompts
  tests/                    pytest export/tests
  results/                   measured quant tables are not published
  hf_tokenizer/             committed HF tokenizer
  artifacts/hf_llama/       gitignored intermediate

core/inference/gguf_backend.py   GGUFModel (llama-server + MyPT encode)
core/inference/runtime.py        load_runtime_model / list_runtime_models
core/agent/workspace_ask.py      one-shot RAG used by CLI and other apps
scripts/workspace_chat.py        mypt-workspace-chat (REPL + --query/--json)
scripts/workspace_rag.py         mypt-workspace-rag (JSON stdout)
webapp/routers/chat.py           same loader; lists checkpoints/**/*.gguf
```
