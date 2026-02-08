#!/usr/bin/env python3
"""
Diagnose operator SFT: break down val loss into content vs structural tokens.

This script answers the question: "Is my val loss of 0.13 real, or is it
inflated by trivially-predictable structural tokens?"

It loads a trained model and operator dataset, then measures:
  1. Per-token cross-entropy loss on the val set
  2. Breakdown: content tokens vs structural tokens (</myPT_assistant>, <myPT_eot>)
  3. Per-operator breakdown (COPY, WRAP, EXTRACT)
  4. Per-word-count breakdown (1-word, 2-word, 3-word, 4-word payloads)

Usage:
    python scripts/diagnose_operator_loss.py --model phase3a_operator --dataset data/sft_operator_vs/packed
    python scripts/diagnose_operator_loss.py --model phase3a_operator --dataset data/sft_operator_vs/packed --num_batches 50
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import GPT, GPTConfig
from core.episode_data_loader import GPTEpisodeDataLoader
from core.tokenizer import Tokenizer
from core.checkpoint import CheckpointManager


def main():
    parser = argparse.ArgumentParser(description="Diagnose operator SFT loss breakdown")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name in checkpoints/ dir (e.g. phase3a_operator)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to prepared dataset dir (with train/ and val/ subdirs)")
    parser.add_argument("--num_batches", type=int, default=100,
                        help="Number of val batches to evaluate (default: 100)")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                        help="Which split to evaluate (default: val)")
    parser.add_argument("--show_tokens", type=int, default=3,
                        help="Number of episodes to show token-level detail (default: 3)")
    args = parser.parse_args()

    print("=" * 70)
    print("  OPERATOR LOSS DIAGNOSTIC")
    print("  Breaking down loss: content tokens vs structural tokens")
    print("=" * 70)

    # ---- Load model ----
    print(f"\n[1] Loading model '{args.model}'...")
    ckpt_dir = os.path.join("checkpoints", args.model)
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        print(f"    ERROR: No checkpoint found at {ckpt_dir}")
        print(f"    Available checkpoints:")
        ckpt_base = "checkpoints"
        if os.path.exists(ckpt_base):
            for d in sorted(os.listdir(ckpt_base)):
                full = os.path.join(ckpt_base, d)
                if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
                    print(f"      - {d}")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, step, _ = GPT.load(ckpt_dir, map_location=device, load_optimizer=False)
    model.eval()
    print(f"    Loaded: {model.config.n_layer}L/{model.config.n_embd}E, step={step}")
    print(f"    Device: {device}, dtype: {next(model.parameters()).dtype}")

    # ---- Load dataset ----
    print(f"\n[2] Loading dataset from '{args.dataset}'...")
    if not os.path.exists(args.dataset):
        print(f"    ERROR: Dataset dir not found: {args.dataset}")
        sys.exit(1)

    # Configure for eval
    config = model.config
    config.use_loss_mask = True
    config.batch_sampling_mode = "epoch"

    loader = GPTEpisodeDataLoader(
        config=config,
        tokenizer=model.tokenizer,
        dataset_dir=args.dataset,
    )
    print(f"    Dataset loaded OK")

    # ---- Get special token IDs ----
    ASSISTANT_CLOSE_ID = model.tokenizer.special_tokens.get('myPT_assistant_close')
    EOT_ID = model.tokenizer.special_tokens.get('myPT_eot')
    ASSISTANT_OPEN_ID = model.tokenizer.special_tokens.get('myPT_assistant_open')
    PAD_TOKEN_ID = model.tokenizer.special_tokens.get('myPT_eot', 50256)

    print(f"\n    Special token IDs:")
    print(f"      <myPT_assistant>:  {ASSISTANT_OPEN_ID}")
    print(f"      </myPT_assistant>: {ASSISTANT_CLOSE_ID}")
    print(f"      <myPT_eot>:        {EOT_ID}")

    # ---- Evaluate ----
    print(f"\n[3] Evaluating {args.num_batches} batches on '{args.split}' split...")

    # Accumulators
    content_losses = []
    structural_losses = []
    all_masked_losses = []

    # Per-position stats (position within assistant response)
    position_losses = defaultdict(list)  # pos_idx -> list of losses

    episodes_shown = 0

    # AMP context (match training)
    device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
    if device_type == 'cuda':
        weight_dtype = next(model.parameters()).dtype
        if weight_dtype == torch.bfloat16:
            ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        elif weight_dtype == torch.float16:
            ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        else:
            ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with torch.no_grad():
        for batch_idx in range(args.num_batches):
            batch = loader.get_batch(args.split)

            if not isinstance(batch, (tuple, list)) or len(batch) < 3:
                print(f"    ERROR: Expected (X, Y, mask[, segment_ids]) tuple, got {type(batch)} len={len(batch)}")
                sys.exit(1)

            X, Y, mask = batch[0], batch[1], batch[2]  # (B, block_size)

            # Forward pass
            with ctx:
                logits, _, _ = model(X, Y, loss_mask=None)  # Get logits without built-in loss

            # Per-token cross-entropy (no reduction)
            B, T, V = logits.shape
            per_token_loss = torch.nn.functional.cross_entropy(
                logits.view(B * T, V),
                Y.view(B * T),
                reduction='none'
            ).view(B, T)  # (B, T)

            # Process each sample in batch
            for b in range(B):
                y_seq = Y[b].cpu()          # (T,)
                m_seq = mask[b].cpu()        # (T,)
                l_seq = per_token_loss[b].cpu()  # (T,)

                # Track position within current assistant response
                in_assistant = False
                assistant_pos = 0

                for t in range(T):
                    tok_id = y_seq[t].item()
                    m_val = m_seq[t].item()
                    loss_val = l_seq[t].item()

                    if m_val == 0:
                        # Check if this is assistant_open (marks start of response)
                        if tok_id == ASSISTANT_OPEN_ID:
                            in_assistant = True
                            assistant_pos = 0
                        continue

                    # mask=1 token - this contributes to training loss
                    all_masked_losses.append(loss_val)

                    if tok_id == ASSISTANT_CLOSE_ID or tok_id == EOT_ID:
                        structural_losses.append(loss_val)
                    else:
                        content_losses.append(loss_val)
                        position_losses[assistant_pos].append(loss_val)
                        assistant_pos += 1

                # Show detailed token-level breakdown for first N episodes
                if episodes_shown < args.show_tokens and batch_idx == 0:
                    _show_episode_detail(
                        model.tokenizer, X[b].cpu(), y_seq, m_seq, l_seq,
                        ASSISTANT_OPEN_ID, ASSISTANT_CLOSE_ID, EOT_ID,
                        episodes_shown
                    )
                    episodes_shown += 1

            if (batch_idx + 1) % 25 == 0:
                print(f"    Batch {batch_idx + 1}/{args.num_batches}...")

    # ---- Report ----
    print(f"\n{'=' * 70}")
    print(f"  LOSS BREAKDOWN RESULTS")
    print(f"{'=' * 70}")

    n_content = len(content_losses)
    n_structural = len(structural_losses)
    n_total = len(all_masked_losses)

    if n_total == 0:
        print("  ERROR: No mask=1 tokens found! Check dataset.")
        sys.exit(1)

    mean_content = np.mean(content_losses) if content_losses else float('nan')
    mean_structural = np.mean(structural_losses) if structural_losses else float('nan')
    mean_overall = np.mean(all_masked_losses)

    print(f"\n  Token counts (mask=1 only):")
    print(f"    Content tokens:     {n_content:>8,} ({100*n_content/n_total:.1f}%)")
    print(f"    Structural tokens:  {n_structural:>8,} ({100*n_structural/n_total:.1f}%)")
    print(f"    Total mask=1:       {n_total:>8,}")

    print(f"\n  Average cross-entropy loss:")
    print(f"    Content tokens:     {mean_content:.4f}  (the hard part - actual payloads)")
    print(f"    Structural tokens:  {mean_structural:.4f}  (trivial - always same tokens)")
    print(f"    Overall (reported): {mean_overall:.4f}  (this is what val loss shows)")

    # Diagnosis
    print(f"\n  {'=' * 50}")
    print(f"  DIAGNOSIS")
    print(f"  {'=' * 50}")

    if mean_structural < 0.05 and mean_content > 0.2:
        ratio = mean_content / mean_overall
        print(f"  ** CONFIRMED: Loss mirage detected! **")
        print(f"")
        print(f"  Your reported val loss of {mean_overall:.4f} is misleading.")
        print(f"  Structural tokens (always {ASSISTANT_CLOSE_ID}/{EOT_ID}) have near-zero loss,")
        print(f"  dragging down the average.")
        print(f"")
        print(f"  Real content loss: {mean_content:.4f} ({ratio:.1f}x higher than reported)")
        print(f"  The model is {'poorly' if mean_content > 0.5 else 'weakly'} learning content tokens.")
    elif mean_content < 0.1:
        print(f"  Content loss is low ({mean_content:.4f}) - model may be generalizing!")
        print(f"  If generation still fails, the issue is likely autoregressive error")
        print(f"  compounding, not loss masking.")
    else:
        print(f"  Content loss ({mean_content:.4f}) is moderate.")
        print(f"  Structural token loss ({mean_structural:.4f}) is {'low' if mean_structural < 0.1 else 'moderate'}.")

    # Position analysis
    if position_losses:
        print(f"\n  Content loss by position in assistant response:")
        max_pos = min(max(position_losses.keys()), 15)  # Show up to 15 positions
        for pos in range(max_pos + 1):
            if pos in position_losses and len(position_losses[pos]) > 5:
                pos_mean = np.mean(position_losses[pos])
                pos_n = len(position_losses[pos])
                bar = "#" * min(int(pos_mean * 20), 40)
                print(f"    Position {pos:>2}: loss={pos_mean:.4f} (n={pos_n:>5}) {bar}")

    # Token-level accuracy estimate
    if content_losses:
        # Estimate: p(correct) ≈ exp(-loss) for each token
        content_acc = np.mean([np.exp(-l) for l in content_losses])
        structural_acc = np.mean([np.exp(-l) for l in structural_losses]) if structural_losses else 1.0

        print(f"\n  Estimated per-token accuracy (exp(-loss)):")
        print(f"    Content tokens:    {100*content_acc:.1f}%")
        print(f"    Structural tokens: {100*structural_acc:.1f}%")

        # Sequence-level accuracy estimate for different lengths
        print(f"\n  Estimated exact-match probability by payload length:")
        for n_tokens in [1, 2, 3, 5, 8]:
            seq_acc = content_acc ** n_tokens
            print(f"    {n_tokens}-token payload: {100*seq_acc:.1f}%")

    print(f"\n{'=' * 70}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'=' * 70}")


def _show_episode_detail(tokenizer, x_seq, y_seq, m_seq, l_seq,
                          asst_open_id, asst_close_id, eot_id, ep_num):
    """Show token-by-token breakdown for one episode."""
    print(f"\n    --- Episode {ep_num} token detail ---")

    in_assistant = False
    content_start = None

    for t in range(len(y_seq)):
        tok_id = y_seq[t].item()
        m_val = m_seq[t].item()
        loss_val = l_seq[t].item()

        # Skip padding
        if tok_id == 0 and m_val == 0 and t > 50:
            continue

        # Decode token
        try:
            tok_str = tokenizer.decode([tok_id])
        except Exception:
            tok_str = f"<id:{tok_id}>"

        # Identify token type
        if tok_id == asst_open_id:
            tok_type = "ASST_OPEN"
            in_assistant = True
            content_start = t + 1
        elif tok_id == asst_close_id:
            tok_type = "ASST_CLOSE"
            in_assistant = False
        elif tok_id == eot_id:
            tok_type = "EOT"
        elif tok_id >= 50257:
            tok_type = "SPECIAL"
        elif in_assistant:
            tok_type = "CONTENT"
        else:
            tok_type = "input"

        # Only show tokens near the assistant response
        if tok_type == "input" and (content_start is None or t < content_start - 5):
            continue

        mask_str = "TRAIN" if m_val > 0 else "skip "
        loss_str = f"{loss_val:.4f}" if m_val > 0 else "  -  "

        # Truncate token string for display
        tok_display = repr(tok_str)
        if len(tok_display) > 25:
            tok_display = tok_display[:22] + "..."

        print(f"      t={t:>3} | {mask_str} | loss={loss_str} | {tok_type:>10} | "
              f"id={tok_id:>5} | {tok_display}")

        # Stop after EOT
        if tok_type == "EOT":
            break

    print()


if __name__ == "__main__":
    main()
