# Tokenizer notes (GGUF export)

**Source of special tags:** [`core/special_tokens.py`](../core/special_tokens.py) — 19 tags, pinned IDs 50257–50275. Regenerated into [`special_tokens.json`](special_tokens.json) by `python export/generate_special_tokens.py`. Do not hand-edit the JSON.

## Base encoding

- tiktoken encoding **name:** `gpt2` (`tiktoken.get_encoding("gpt2")`)
- `n_vocab` base: **50257** (ranks 0–50256; `<|endoftext|>` = 50256)
- Model embedding rows: **50304** (19 specials + 28 UNUSED pads 50276–50303)

## `pat_str` (extracted, not inferred from the name)

tiktoken 0.12 `gpt2._pat_str`:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
```

Pinned llama.cpp `LLAMA_VOCAB_PRE_TYPE_GPT2` (`src/llama-vocab.cpp`):

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)
```

**Not a character-identical match.** Differences: possessive `++` vs greedy `+`, compact `'s|'t|…` class, extra trailing whitespace alts in tiktoken.

**Resolution:** map the HF fingerprint to llama.cpp pre-tokenizer **`gpt-2`** (same family llama.cpp already implements). Do **not** change MyPT `pat_str` (that would rebuild the tokenizer and re-pack the corpus).

Acceptance: `tiktoken == HF` ID sequences on the parity fixtures, including German/French non-ASCII. If that test fails, this is a **pre-1.4B escalate**, not an export patch.

Stock openai-community/gpt2 converter hash (for reference):
`3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454` → `gpt-2`.
MyPT's hash is written to `export/pretokenizer_hash.json` on first convert abort and injected into the cloned `conversion/base.py`.

## Special tokens in GGUF

Every string in `SPECIAL_TOKEN_STRINGS` is:

- a single token ID identical to `SPECIAL_TOKEN_IDS`
- typed `CONTROL` or `USER_DEFINED`, never `NORMAL`
- contiguous 50257–50275, above gpt2 base, below UNUSED pad

This is required so llama.cpp **re-tokenization** of `<myPT_toolresult>` on later turns does not fragment the tag (Phase 4/5/6 silent fail).

Verified on `mypt-f16.gguf` / `mypt-q4_k_m.gguf`: `llama-tokenize --no-bos --ids` of `<myPT_toolcall>` is `[50267]`. With `--no-parse-special` the same string fragments into subword IDs (the untrusted path).

## EOG / stop

Pinned llama.cpp builds the EOG set from a **hardcoded list of surface strings** (`<|eot_id|>`, `</s>`, …), not from GGUF token types. `<myPT_eot>` is not on that list, so load logs `special_eos_id is not in special_eog_ids` and then **inserts EOS (50275) into EOG anyway**. `</myPT_toolcall>` (50268) is CONTROL but not EOG. The product wrapper therefore passes stop strings for `</myPT_assistant>`, `<myPT_eot>`, and `</myPT_toolcall>` and re-appends the stopping word if llama-server strips it (controller `find_toolcall` needs the close tag).

## Segment trust model

| Segment | Special parsing | How |
| --- | --- | --- |
| System prompt, runtime-inserted toolresult **tags** | Trusted — insert by ID / MyPT `encode()` | Controller + GGUF backend encode with packer tokenizer |
| User turns, retrieved documents, tool **payload** JSON | Untrusted — `encode_ordinary()` / ingest strip | [`strip_special_tag_strings`](../core/special_tokens.py) at RAG load; injection tests |

A document containing a forged `<myPT_toolresult>` must not become a CONTROL token in untrusted tokenize, and is stripped at ingest.

## 1.4B training-side hardening

Do not remake 750M GOLD. Before the 1.4B Phase 6 mix, add a small slice of episodes where retrieved-document text contains forged myPT tags and the gold assistant **ignores** them. See [SCALE_1_4B.md](../docs/sft/SCALE_1_4B.md).
