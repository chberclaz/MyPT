# Token Accuracy Saturation Signal

## Purpose

This feature adds a train-time diagnostic to detect when SFT may be shifting from useful generalization toward memorization.

It tracks masked token accuracy (assistant-supervised tokens only) on both train and val at eval intervals, then raises a saturation signal when gains flatten and/or the train-vs-val gap widens.

v1 is intentionally non-intrusive: **log and flag only** (no automatic early stop or LR changes).

## Where It Is Implemented

- Training/eval loop and detector logic: [`core/model.py`](d:/coding/MyPT/core/model.py)
- Config plumbing from JSON config into fit loop: [`train.py`](d:/coding/MyPT/train.py)
- Example config usage: [`configs/sft/phase3_1_corrective.json`](d:/coding/MyPT/configs/sft/phase3_1_corrective.json)

## Metrics Computed

At each eval event (`iter % eval_interval == 0`), the model computes:

- `train_masked_token_acc`: top-1 token accuracy on train split where `loss_mask == 1`
- `val_masked_token_acc`: top-1 token accuracy on val split where `loss_mask == 1`
- `masked_acc_gap = train_masked_token_acc - val_masked_token_acc`
- `train_masked_token_count`, `val_masked_token_count` for denominator/context

If a split does not provide `loss_mask`, accuracy is skipped (`None`).

## Detector Logic (v1)

The detector uses a rolling window over eval points:

- `window` (default 4)
- `min_steps_before_check` (default 200)
- `min_delta` (default 0.002 = 0.2 pp per eval-point slope proxy)
- `gap_growth_threshold` (default 0.01 = 1.0 pp gap widening over window)

It can flag saturation when one or more conditions are met:

1. **Plateau**: train gain and val gain both below `min_delta`
2. **Memorization shift**: val gain below `min_delta`, train gain positive, and gap growth > 0
3. **Gap widening**: gap growth exceeds `gap_growth_threshold`

The first detected step is stored as `saturation_step`.

## Config

Add this block to a training config JSON:

```json
"token_accuracy_saturation": {
  "enabled": true,
  "window": 4,
  "min_steps_before_check": 200,
  "min_delta": 0.002,
  "gap_growth_threshold": 0.01,
  "action": "log_and_flag"
}
```

Notes:

- `enabled` is the toggle.
- `action` is currently informational (`log_and_flag`) in v1.
- Omitted block falls back to internal defaults.

## Logged Outputs

### JSONL Training Log (`log_file`)

Per eval event, the JSON line includes:

- `train_masked_token_acc`
- `val_masked_token_acc`
- `train_masked_token_count`
- `val_masked_token_count`
- `masked_acc_gap`
- `saturation_detected`
- `saturation_reason`
- `saturation_step`

### Checkpoint/Training State

`training_config.token_accuracy_saturation` is persisted with:

- detector params used
- `saturation_detected`
- `saturation_step`
- last observed masked-accuracy values and reason

## How To Read It

- Healthy: both train and val masked accuracy rise; gap roughly stable.
- Saturation: both curves flatten for several evals.
- Memorization risk: train rises while val stalls and the gap grows.

Use this signal with other gates (loss curves, eval suites, regression checks), not as a sole stop condition.

## Validation Checklist

1. Run a short train (100-200 steps) and confirm new log fields appear.
2. Verify no crash when datasets/splits have no loss mask (fields may be `null`).
3. Run a high-coverage small-dataset test to verify saturation is flagged.
4. Confirm `training_state.json` includes `token_accuracy_saturation` state.
