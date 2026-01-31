Fix SFT obedience not showing up at inference (Tokenizer + Generate + Cache + Load parity)
Goal

Make sure a Phase3a1 SFT model that learned “echo/format” during training can actually express it during inference, with no hidden drift from:

tokenizer special-token mapping changes

cache decode mismatch

“temperature=0” not being truly greedy

fp16/bf16 autocast/caching inconsistencies

hardcoded stop token IDs that don’t match tokenizer

silent “special token regex” partial matches

This task must:

harden tokenizer serialization/restoration so token IDs are stable

harden generation to be deterministic when requested

add deep parity tests: (train forward) == (eval forward) == (inference forward) == (cached decode)

0) Add a single “self-check” command that answers PASS/FAIL

Create a script scripts/debug_inference_parity.py that runs:

Loads model checkpoint via your normal load_model()

Prints: model dtype, tokenizer kind, config vocab_size, base_vocab_size

Runs:

tokenizer_self_test()

special_token_id_stability_test()

kv_cache_parity_steps_test()

greedy_is_greedy_test()

single_batch_memorize_test() (optional but recommended)

Exit code:

0 if all passed

1 on first failure with a clear error message

This script is what you run after each fix.

1) Tokenizer: persist exact special-token mapping and restore it exactly
Problem

Right now tokenizer.get_state() only saves:

{"token_kind","chars","base_vocab_size"}


But special token IDs are derived from current SPECIAL_TOKEN_STRINGS.items() order at runtime. If:

dict order changes (file edit, python version differences, accidental reorder)

you insert a new token in the middle

you run a different branch/commit than training
…then IDs shift silently. Your model weights still treat e.g. 50263 as <myPT_assistant>, but your tokenizer may encode <myPT_assistant> as some other ID. That produces “model learned it but cannot be prompted correctly”.

Fix

Update tokenizer serialization to include the exact mapping at training time.

Implement in Tokenizer.get_state():

Include:

special_token_strings: the full dict from SPECIAL_TOKEN_STRINGS (as used at training)

special_token_encoder: mapping {token_string: id}

special_tokens_by_name: mapping {name: id}

special_token_version: int you bump when you change the set

Example:

def get_state(self):
    state = {
        "token_kind": self.token_kind,
        "chars": self.chars,
        "base_vocab_size": self.base_vocab_size,
        "model_vocab_size": self.model_vocab_size,
    }
    if self.token_kind == "gpt2":
        state.update({
            "special_token_strings": SPECIAL_TOKEN_STRINGS,  # snapshot
            "special_token_encoder": self.special_token_encoder,
            "special_tokens_by_name": self.special_tokens,
            "special_token_version": 1,
        })
    return state

Implement Tokenizer.set_state(state) to restore mappings:

Rules:

If special_token_encoder exists in saved state:

use it (do NOT regenerate from current SPECIAL_TOKEN_STRINGS)

Validate:

len(saved_encoder) == len(saved_strings)

all IDs are unique

all IDs are in [base_vocab_size, model_vocab_size)

Rebuild:

special_token_decoder as inverse map

_special_token_pattern from special_token_encoder.keys()

Also: store and restore model_vocab_size and assert it matches config.vocab_size.

Fix Tokenizer.from_state(...)

Currently:

tok = cls(config, state["token_kind"])
tok.chars = state.get("chars", None)
return tok


This does not call set_state(), so it ignores the saved mapping.

Change to:

@classmethod
def from_state(cls, config, state):
    tok = cls(config, state["token_kind"])
    tok.set_state(state)
    return tok

Add hard failure in model load if mapping mismatch

In GPT.load(...) after tokenizer restore:

verify that the restored tokenizer has expected special token IDs

if mismatch: raise and print both maps

Specifically:

ensure all token strings in saved state exist in runtime tokenizer encoder

ensure saved special_token_encoder exactly equals restored

This converts silent corruption into loud failure.

2) Tokenizer: fix regex ordering to avoid prefix-matching bugs
Problem

You build:

escaped_tokens = [re.escape(tok) for tok in self.special_token_encoder.keys()]
pattern = re.compile('|'.join(escaped_tokens))


Regex alternation matches left-to-right. If any special token is a prefix of another, the shorter one can match first and break encoding.

Fix

When building the token list for regex, sort by descending length:

tokens = sorted(self.special_token_encoder.keys(), key=len, reverse=True)
escaped_tokens = [re.escape(tok) for tok in tokens]
self._special_token_pattern = re.compile('|'.join(escaped_tokens))

3) Stop hardcoding stop token IDs in generate()
Problem

You hardcode:

DEFAULT_STOP_TOKENS = {50264, 50271}


If special token IDs shift (or even if you just change your token list), stopping will break.

Fix

Replace with tokenizer lookup:

DEFAULT_STOP_TOKENS = {
    self.tokenizer.special_token_encoder.get("</myPT_assistant>"),
    self.tokenizer.special_token_encoder.get("<myPT_eot>"),
}
DEFAULT_STOP_TOKENS = {t for t in DEFAULT_STOP_TOKENS if t is not None}


Or (preferred) use names:

DEFAULT_STOP_TOKENS = {
    self.tokenizer.special_tokens["myPT_assistant_close"],
    self.tokenizer.special_tokens["myPT_eot"],
}


(Use whatever your SPECIAL_TOKEN_STRINGS names are.)

Also print these IDs once at generation start in debug mode.

4) Make “temperature=0” truly greedy (skip ALL filters)
Problem

Even with temperature=0 you still apply:

repetition penalty

top-k / top-p masking

no-repeat-ngram bans

So “deterministic” is not deterministic.

Fix

Implement STRICT_GREEDY mode:

If temperature <= 0, do pure argmax on raw logits

No rep penalty, no topk/topp, no ngram bans

Add:

STRICT_GREEDY = (temperature is None) or (float(temperature) <= 0.0)


Wrap all filtering blocks with if not STRICT_GREEDY:.

5) Autocast dtype must follow model dtype (bf16 vs fp16)
Problem

You always do fp16 autocast on CUDA:

autocast(dtype=torch.float16)


Even when weights/cache are bf16.

Fix

Pick autocast dtype based on next(self.parameters()).dtype:

bf16 weights + bf16 supported -> bf16 autocast

else fp16 autocast

CPU -> no autocast

6) KV-cache parity: test multi-step decode parity (not just prefill)
Problem

Your current debug only compares cache vs non-cache on the prefill last token.
Bugs often appear during incremental decode.

Fix

Add a method:
debug_cache_parity_steps(prompt, steps=16)
It must:

prefill prompt with cache

compute logits for last prompt token with no-cache

for N steps:

compare argmax cache vs argmax no-cache

append the token

advance cache by feeding only the new token

recompute no-cache on full context window (trim to block_size)
Fail fast on first mismatch and print:

step index

both token IDs

decoded strings

max/mean logit diff

7) Add an “eval in inference mode” parity test (what user asked)
Purpose

Verify that inference path (autocast + eval + no_grad + caches off) matches training forward.

Implement debug_forward_parity(prompt_text)

Compute next-token logits in these modes:

train mode forward (no dropout if dropout=0, but you currently have dropout=0.05)

set model.train()

set dropout to 0 temporarily OR run with torch.no_grad() but still in train()

this is mostly for sanity; dropout will cause differences, so you can skip if dropout>0

eval forward no-cache:

model.eval()

autocast same dtype policy

use_cache=False

compare argmax token to (3)

eval forward cache prefill:

use_cache=True prefill

compare last-logit argmax to (2)

eval forward cache multi-step:

call debug_cache_parity_steps

If (2) vs (3) mismatch on prefill: cache path is wrong.
If (3) is ok but multi-step fails: decode update/caching logic is wrong.

8) Add tokenizer atomicity tests for chat tags
Implement in scripts/debug_inference_parity.py:

For each special token string in special_token_encoder:

ids = tok.encode(tok_str)

assert len(ids) == 1

assert ids[0] == tok.special_token_encoder[tok_str]

Also test mixed text:
"<myPT_user>Say OK.</myPT_user><myPT_assistant>"

ensure IDs contain special token IDs exactly at expected positions

decode(encode(x)) roundtrip must equal input string

This catches regex and mapping bugs immediately.

9) Add “micro-memorization” overfit test to prove SFT pipeline works end-to-end
Goal

If your inference stack is correct, the model must be able to overfit a tiny batch fast.

Implement:

Create a micro dataset of 16 episodes:

Say OK. → OK.

Repeat: Here → Here

What is 5+7 → 12

5–10 similar

Train for 200–500 iters at higher LR on that set

Immediately run greedy generation
Expected: 100% correct on those prompts.

If it fails, the issue is still in:

tokenizer mapping

masking alignment

generation path

checkpoint load mismatch

10) Improvements to loss-mask alignment (defensive)

In episode loader _get_episode_sample:

Assert lengths match:

len(mask) == len(tokens) for the episode slice

After padding, ensure:

padding mask is 0

Add a debug mode that prints first/last 10 tokens with mask values

Also: enforce that the last trained token is the last answer token (or EOT) consistently.

Deliverables checklist

 Tokenizer state contains special-token mapping + version

 Tokenizer.from_state uses set_state (not regenerate)

 Tokenizer regex sorted by length desc

 generate() uses tokenizer-derived stop tokens (no hardcoded IDs)

 STRICT_GREEDY bypasses all filtering when temperature<=0

 autocast dtype follows weight dtype

 debug_cache_parity_steps compares multi-step decode cache vs no-cache

 debug_forward_parity compares eval no-cache vs cache prefill

 tokenizer atomicity tests for each special token

 micro-memorization overfit test proves end-to-end obedience works