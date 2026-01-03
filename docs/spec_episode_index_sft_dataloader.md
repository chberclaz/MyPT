Cursor Spec: Episode-Indexed SFT Loader with Padding to block_size (Fixed-Shape Batches), Controlled via XML Config
Goal

Add a new episode-indexed SFT data loading mode that:

samples episodes (conversations) deterministically per epoch,

never crosses episode boundaries,

returns fixed-shape tensors padded to block_size so model.fit stays unchanged,

supports optional loss masking aligned to y (assistant-only training),

is controlled via new XML config flags.

This is a Phase-3 (SFT/tool) loader design that coexists with your current token-stream GPTDataLoader without breaking Phase-1/2.

Overview of the New Dataset Format

Episode-indexed datasets are stored as:

train/tokens.bin (uint32 token ids)

train/mask.bin (optional; uint8 or float32 aligned to tokens; 0/1)

train/episodes.idx (uint64 pairs: start, length) for each episode

same layout for val/

episodes.idx binary format (minimal)

Each record is exactly 16 bytes:

start: uint64 (token offset into tokens.bin)

length: uint64 (episode token length)

Total episodes = file_size / 16.

Note: “episode tokens” should already include special tokens you want (e.g., BOS/EOS). Padding is handled at load time.

XML Config Changes

Add under <training> (adapt path to your schema):

<training>
  ...
  <dataset_mode>token_stream</dataset_mode> <!-- token_stream | sft_episode -->
  <batch_sampling_mode>random</batch_sampling_mode> <!-- random | epoch (applies to sft_episode) -->
  <epoch_shuffle>true</epoch_shuffle>
  <epoch_drop_last>true</epoch_drop_last>
  <epoch_seed>1337</epoch_seed>

<pad_token_id>50256</pad_token_id> <!-- default: tokenizer.eos_id if not set -->
<episode_min_tokens>2</episode_min_tokens> <!-- drop episodes shorter than this -->
</training>

Defaults / Backward Compatibility

If dataset_mode absent: default token_stream (current behavior).

sft_episode mode requires tokens.bin + episodes.idx present.

batch_sampling_mode defaults to:

random for token_stream (unchanged)

epoch recommended for sft_episode (but still configurable)

Functional Requirements

1. Fixed-shape outputs (no changes to model.fit)

In sft_episode mode, get_batch(split) must always return:

x.shape = (batch_size, block_size)

y.shape = (batch_size, block_size)

if loss mask enabled and available:

mask.shape = (batch_size, block_size) float32

Episodes shorter than block_size+1 must be padded.
Episodes longer must be truncated.

2. Padding rule

Let L = block_size + 1 (we need L tokens to produce x and y).

For an episode token sequence t (length n):

if n >= L: take t = t[:L]

if n < L: pad t to length L using pad_token_id

Then:

x = t[0:block_size]

y = t[1:block_size+1]

Mask (if present):

mask_seq = episode_mask[1:block_size+1]

padded positions must have mask = 0

3. Loss mask behavior (assistant-only training)

If use_loss_mask=True and mask.bin exists:

return (x, y, mask)

mask aligns with y

If masks missing/incomplete:

behave like today: return (x, y) (and optionally warn once)

4. Sampling behavior

In sft_episode mode:

batch_sampling_mode=epoch:

epoch-shuffle a list of episode IDs (0..N-1)

each episode seen once per epoch (subject to drop_last)

batch_sampling_mode=random:

sample episode IDs uniformly at random (with replacement)

Phase-1/2 token_stream mode must remain unchanged.

5. Episode length sanity

Episodes must have at least episode_min_tokens (default 2) so that y has something meaningful.

Shorter episodes should be skipped or filtered out when building episode lists.

Implementation Plan
A) New loader class (recommended)

Add a new class GPTEpisodeDataLoader rather than overloading everything in GPTDataLoader:

GPTDataLoader (existing, token stream/sharded windows)

GPTEpisodeDataLoader (new, episode-indexed)

Both expose:

get_batch(split='train')

Then in training init:

if config.dataset_mode == "sft_episode" → instantiate GPTEpisodeDataLoader

else → instantiate current GPTDataLoader

This keeps code clean and avoids regressions.

GPTEpisodeDataLoader: Design Details
B) Files and loading

Expected directory layout:

dataset_dir/train/tokens.bin

dataset_dir/train/episodes.idx

optional: dataset_dir/train/mask.bin

same for val/

In **init**:

memmap tokens:

self.train_tokens = np.memmap(tokens.bin, dtype=np.uint32, mode='r')

memmap masks if enabled:

self.train_mask = np.memmap(mask.bin, dtype=np.uint8 or np.float32, mode='r')

load episode index:

either memmap episodes.idx as np.uint64 and reshape (-1,2)

OR load into RAM (recommended; it’s small compared to tokens)

Example:

idx = np.memmap(episodes_idx, dtype=np.uint64, mode='r').reshape(-1,2)

self.train_episodes = idx where idx[k] = (start, length)

Filter episodes shorter than episode_min_tokens:

create self.train_episode_ids = [k for k in range(N) if length >= episode_min_tokens]

C) Epoch state

Maintain per split:

self.\_epoch_train, self.\_pos_train, self.\_order_train

self.\_epoch_val, self.\_pos_val, self.\_order_val

If batch_sampling_mode == "epoch":

On epoch start:

order = np.array(train_episode_ids)

shuffle once if epoch_shuffle

determinism: seed_epoch = epoch_seed + epoch (if seed set)

Batch is next batch_size IDs from order[pos:pos+bs]

If drop_last=true and remaining < bs:

start next epoch and take from new epoch

If batch_sampling_mode == "random":

each batch selects batch_size IDs via RNG

D) Batch materialization with padding

For each selected episode ID:

Read (start, length) from episodes index

Slice episode tokens:

seq = tokens[start : start + min(length, block_size + 1)]

Pad to length block_size+1 using pad_token_id

Form x/y:

x = seq_padded[:-1] (len block_size)

y = seq_padded[1:] (len block_size)

Mask:

If use_loss_mask and mask file available:

slice mask aligned to the sliced token span:

m = mask[start : start + min(length, block_size + 1)]

pad mask with zeros to length block_size+1

mask_y = m_padded[1:] (len block_size)

Ensure mask_y is float32 tensor on device

Stack into batch tensors:

x_batch: int64

y_batch: int64

mask_batch: float32 (optional)

Move to device at end.

E) pad_token_id selection

Use config.pad_token_id if set

Else default to tokenizer’s EOS if available (tokenizer.eos_id)

Else fail with clear error

Logging

On loader init:

number of episodes (train/val), tokens length, mask availability
On epoch boundary:
[EpisodeLoader] split=train epoch=2 episodes=45000 batches=703 shuffle=true drop_last=true pad_id=50256 mask=true

Tests (must-have)

Padding correctness:

episode shorter than block_size+1 produces padded x/y of correct length

mask padded positions are 0

Truncation correctness:

episode longer than block_size+1 is truncated

Mask alignment:

verify mask.shape == y.shape

verify mask corresponds to y positions (shifted by +1)

Epoch determinism:

same seed produces same episode order per epoch

Epoch coverage:

with drop_last=false, every episode id appears exactly once per epoch

Acceptance Criteria

dataset_mode=token_stream works exactly as before (no regression).

dataset_mode=sft_episode:

get_batch always returns fixed shapes padded to block_size

episode boundary is respected (no cross-episode windows)

batch_sampling_mode=epoch provides deterministic epoch-shuffled coverage

optional loss mask returns (x, y, mask) aligned to y

Notes / Optional Future Enhancements (out of scope)

Episode packing (concatenate multiple short episodes into one block for efficiency)

Weighted sampling by episode type/domain

Storing episode metadata (type/tool flags) alongside idx records
