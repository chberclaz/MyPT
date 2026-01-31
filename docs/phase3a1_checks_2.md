# TASK: Fix Phase3a1 “format-lock” evaluation + verify SFT target learning

## A) Add training-time inference evaluation (MISSING)
Goal: during training eval steps, run the *real* generation path and log outputs.

1) Add CLI arg to train.py:
   --eval_prompts_file configs/sft_eval/phase3a1_eval_prompts.json
   --eval_max_new_tokens 64

2) Implement in GPT.fit() inside `if iter % eval_interval == 0:` AFTER estimate_loss():
   - temporarily self.eval()
   - for each prompt in eval_prompts_file:
       out = self.generate(
         prompt,
         max_new_tokens=eval_max_new_tokens,
         temperature=0.0,
         top_k=0,
         top_p=1.0,
         repetition_penalty=1.0,
         use_default_stop_tokens=True
       )
       store output string in log_entry["gen"][name] = out
   - append to JSONL log_file.

3) Also add a no-cache reference output (slow but only at eval):
   - implement `generate_nocache_greedy(prompt, max_new_tokens)` that does:
       idx = encode(prompt)
       for t in range(max_new_tokens):
         idx_cond = idx[-block_size:]
         logits,_,_ = self(torch.tensor([idx_cond],... ), use_cache=False)
         next = argmax(logits[-1])
         append; break on stop tokens
   - log both: gen_cache and gen_nocache. If they diverge -> bug.

## B) Add dataset-wide loss-mask validator
Create scripts/validate_sft_episode_masks.py that:
- loads tokens.bin, mask.bin, episodes.idx
- for each episode:
  - decode small window around <myPT_assistant> and </myPT_assistant>
  - assert exactly 1x <myPT_assistant>, 1x </myPT_assistant>, 1x <myPT_eot>
  - assert mask is 0 before assistant-open
  - assert mask is 1 for all assistant content tokens
  - assert mask is 1 for </myPT_assistant> and <myPT_eot> (if that is intended)
  - assert padded region mask is 0
- print summary stats:
  - trained_token_count min/median/max
  - count of episodes failing each rule
- exit non-zero on any failure

## C) Add mask alignment unit test (off-by-one protection)
In the validator:
- for N random episodes:
  - find first token position p in seq with mask[p]==1
  - ensure that in training alignment (x=seq[:-1], y=seq[1:], mask_y=mask[1:]):
       mask_y[p-1] == 1
  - if not, mask is shifted and training is wrong.

## D) Build an eval prompt suite (exact strings)
Create configs/sft_eval/phase3a1_eval_prompts.json with exact dataset-style prompts:
{
  "say_ok": "<myPT_system>...</myPT_system><myPT_user>Say OK.</myPT_user><myPT_assistant>",
  "say_hello": "...Say Hello.</myPT_user><myPT_assistant>",
  "math_5_7": "...What is 5 + 7?</myPT_user><myPT_assistant>",
  "capital_de": "...Capital of Germany?</myPT_user><myPT_assistant>",
  "copy_nonsense": "...Say Galigali.</myPT_user><myPT_assistant>"
}

Ensure punctuation/spacing/case match training episodes.

## E) If copy_nonsense fails: patch dataset for generalization
Add 50-200 new “copy tasks” episodes:
- user: "Say <random_token>."
- assistant: "<same token>."
Include:
- nonsense words, mixed case, digits, underscores, multi-word spans
Rebuild dataset and retrain Phase3a1 for ~1-2x coverage; confirm eval suite improves.

## F) Sanity check tie_weights and save/load
- Confirm tie_weights is stored in config.json and preserved on load.
- Add an assertion in load_model(): loaded config tie_weights matches expected.
- If tie_weights enabled after loading, do NOT re-average weights; just tie pointers.
  (Averaging should happen only once at the moment you switch regimes.)






# PATCH: If Phase3a1 trains ONLY closers (</myPT_assistant> + <myPT_eot>), enforce correctness

## 1) DataLoader: preserve end-of-episode when truncating
File: core/episode_data_loader.py
Function: _get_episode_sample()

Current behavior slices start:start+L. That can drop closing tags for long episodes.
Change to:
- If length <= L: keep start:start+length (current)
- If length > L: keep the LAST L tokens of the episode:
    start2 = start + (length - L)
    seq = tokens[start2:start2+L]
    and slice mask the same way.
This ensures closers always present in training window.

Also update mask slicing accordingly (same start2, same actual_len=L).

## 2) Add scripts/validate_closer_only_masks.py
Validate the dataset aligns with "train only closers":
For N random episodes:
- Decode episode text (or at least locate token IDs for </myPT_assistant> and <myPT_eot>)
- Compute x/y/mask_y = m[1:]
- Find i_close = index in seq where token==50264
- Find i_eot   = index in seq where token==50271
Assertions:
- Exactly one 50264 and one 50271 exist in seq window
- mask_y[i_close-1] == 1
- mask_y[i_eot-1] == 1
- sum(mask_y)==2  (or ==1 if you decide not to train eot; but your stated choice is both)
- All other positions are 0
Print failing episode ids + decoded snippet around closers.

## 3) Add training-time inference eval (greedy)
In GPT.fit() eval block:
- run self.generate(...) on 5 fixed prompts
- ALSO run generate_nocache_greedy(...) on same prompts
- log both strings
If cache and nocache differ -> stop and investigate.




