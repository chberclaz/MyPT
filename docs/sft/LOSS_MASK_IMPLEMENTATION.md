1. How can the model “learn conversation” if we only train on assistant tokens?

Key idea: the model still sees the whole conversation, it just gets punished only for the assistant side.

Take this toy example (tokenized sequence):

<User> Hi, who are you? </User>
<Assistant> I'm an AI assistant. </Assistant>

Imagine the token sequence is:

[ U, "Hi", ",", "who", "are", "you", "?", UA,
A, "I'm", "an", "AI", "assistant", ".", AA ]

During training:

At each time step t, the model takes all previous tokens [0..t-1] as context and predicts token t.

We compute logits for all positions.

With an assistant-only loss mask, we do:

positions where the target token is `<myPT_assistant>`, assistant content, or `</myPT_assistant>` → loss_mask[t] = 1

positions where the target is user/system content or user/system/EOT tags → loss_mask[t] = 0

**Note:** The `<myPT_assistant>` tag itself has mask=1 because the model must learn WHEN to generate it.

So for predicting "I'm":

Input context = [U, "Hi", ",", "who", "are", "you", "?", UA, A]

We do compute a loss (because "I'm" is an assistant token).

Gradient flows through:

token embeddings for user tokens

all attention layers that processed the user tokens

everything that turned that full context into a good prediction for "I'm".

Even though the user tokens themselves don’t have direct loss on their own next-token predictions, their representations are adjusted because they are part of the context that must be useful to predict assistant tokens.

So the model learns:

“Given a user message like this, the assistant typically responds like that.”

We don’t want the model to become good at predicting the user side (we don’t need the model to role-play the user). We care about:

𝑃
(
assistant tokens
∣
conversation history
)
P(assistant tokens∣conversation history)

Masking loss to assistant tokens is exactly that.

If we didn’t mask:

The model would try to also learn to generate the user questions.

It can start hallucinating <User> turns in its output.

Mixed-quality user text (typos, weird phrasing) would also be treated as “targets”, diluting the signal.

So: we use user tokens as input context, but we only supervise the assistant’s answers.

2. Implementing loss_mask as a config option (no CLI switch)

Cool, let’s do it your way:

Add a field like use_loss_mask: bool = False to GPTConfig.

This will then go into your JSON presets (configs/\*.json) and be saved/loaded automatically.

train.py won’t get a separate CLI flag; it will just read from config.use_loss_mask.

GPTDataLoader uses config.use_loss_mask to decide whether to return (X, Y) or (X, Y, loss_mask).

Below are concrete changes.

2.1. GPTConfig: add use_loss_mask

In model.py, modify the dataclass:

@dataclass
class GPTConfig:
batch_size: int = 32
block_size: int = 256
vocab_size: int = 50304
n_embd: int = 384
n_head: int = 6
n_layer: int = 6
dropout: float = 0.2
bias: bool = False
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # NEW: SFT / loss-masking behavior
    use_loss_mask: bool = False   # if True, expect loss_mask from data loader during SFT

You can also add it to **str** if you like, but not required. Because you use asdict, it will automatically be saved/loaded in JSON configs.

Then in a preset you could have:

{
"name": "200M_chat_sft",
"description": "200M model for chat SFT with assistant-only loss",
"batch_size": 32,
"block_size": 256,
"vocab_size": 50304,
"n_embd": 640,
"n_head": 10,
"n_layer": 12,
"dropout": 0.1,
"bias": false,
"use_loss_mask": true
}

2.2. GPT.forward: add optional loss_mask

Modify forward in GPT:

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            if loss_mask is not None:
                # loss_mask expected shape (B, T), values 0 or 1 (or floats)
                loss_mask = loss_mask.view(B * T).to(logits.device)

                per_token_loss = F.cross_entropy(
                    logits, targets, reduction='none'
                )
                denom = loss_mask.sum()
                if denom.item() > 0:
                    loss = (per_token_loss * loss_mask).sum() / denom
                else:
                    loss = per_token_loss.mean()
            else:
                loss = F.cross_entropy(logits, targets)

        return logits, loss

Note: we don’t check config.use_loss_mask here. The model stays generic; whether to pass a mask is a data/runner decision.

2.3. estimate_loss: support batches with or without mask

In GPT.estimate_loss:

    @torch.no_grad()
    def estimate_loss(self, data_loader, eval_iters, splits=['train', 'val']):
        out = {}
        self.eval()
        for split in splits:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                batch = data_loader.get_batch(split)
                if isinstance(batch, (tuple, list)) and len(batch) == 3:
                    X, Y, loss_mask = batch
                    _, loss = self(X, Y, loss_mask=loss_mask)
                else:
                    X, Y = batch
                    _, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out

2.4. fit: same idea for training loop

In GPT.fit, change the training step:

        for iter in range(start_step, max_iters):
            # Evaluation and checkpointing (unchanged)
            if iter % eval_interval == 0:
                ct = datetime.datetime.now()
                print(f"{iter} : {ct}")
                losses = self.estimate_loss(data_loader, eval_iters)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if checkpoint_dir:
                    self.save_checkpoint_bundle(
                        checkpoint_dir,
                        step=iter,
                        optimizer_state=optimizer.state_dict(),
                        training_config=training_config
                    )

            # === Training step ===
            batch = data_loader.get_batch('train')
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                xb, yb, loss_mask = batch
                _, loss = self(xb, yb, loss_mask=loss_mask)
            else:
                xb, yb = batch
                _, loss = self(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

No config check here either. The mask is present or not.

2.5. train.py: use config.use_loss_mask instead of CLI flag

We do not add a new CLI arg. Instead, we rely on config.

Your data loader creation currently:

    if args.dataset_dir:
        data_loader = GPTDataLoader(model.config, model.tokenizer, dataset_dir=args.dataset_dir)
    else:
        data_loader = GPTDataLoader(model.config, model.tokenizer)
        data_loader.prepare_data(text)

Change to:

    print("Preparing data...")
    if args.dataset_dir:
        # Sharded mode
        data_loader = GPTDataLoader(
            model.config,
            model.tokenizer,
            dataset_dir=args.dataset_dir,
            use_loss_mask=model.config.use_loss_mask
        )
        ...
    else:
        # In-memory mode
        data_loader = GPTDataLoader(
            model.config,
            model.tokenizer,
            use_loss_mask=model.config.use_loss_mask
        )
        data_loader.prepare_data(text)
        ...

So:

If use_loss_mask is false in the config → loader just ignores masks and returns (X, Y).

If use_loss_mask is true → loader will try to return (X, Y, loss_mask).

No CLI switch involved; everything is controlled by the preset (or by manually setting config.use_loss_mask = True before training).

2.6. GPTDataLoader: expected contract

You didn’t paste GPTDataLoader, but here’s the expected behavior:

Constructor:

class GPTDataLoader:
def **init**(self, config, tokenizer, dataset_dir=None, use_loss_mask=False, ...):
self.config = config
self.tokenizer = tokenizer
self.use_loss_mask = use_loss_mask
... # When loading from dataset_dir, if use_loss_mask: # load mask shards alongside token shards (e.g. train_mask.bin, val_mask.bin)

And get_batch:

    def get_batch(self, split):
        # existing code to produce xb, yb from train/val data
        xb = ...
        yb = ...

        if self.use_loss_mask:
            # only if you have masks for this dataset
            if split == 'train' and hasattr(self, "train_mask"):
                mb = ...  # (B, T)
                return xb, yb, mb
            elif split == 'val' and hasattr(self, "val_mask"):
                mb = ...
                return xb, yb, mb

        # default: no mask
        return xb, yb

So the switch lives in the config (preset), and the model just reacts to whether a mask is present.

3. How you’d use this in practice

Base LM run (no masking)

Config (e.g. configs/200M_base.json):

{ "name": "200M_base", "use_loss_mask": false, ... }

Run:

python train.py \
 --model_name 200M_base \
 --config_file configs/200M_base.json \
 --dataset_dir data/base_corpus

Phase-2 SFT run (assistant-only)

Config (e.g. configs/200M_chat_sft.json):

{
"name": "200M_chat_sft",
"description": "200M chat SFT with assistant-only loss",
"batch_size": 32,
"block_size": 256,
"n_embd": 640,
"n_head": 10,
"n_layer": 12,
"dropout": 0.1,
"bias": false,
"use_loss_mask": true
}

Run:

python train.py \
 --model_name 200M_chat_sft \
 --config_file configs/200M_chat_sft.json \
 --dataset_dir data/chat_sft \
 --init_from_model 200M_base \
 --learning_rate 3e-5 \
 --max_iters 20000

Everything about “this is an SFT run with assistant-only loss” is encoded in the preset, just how you wanted.
