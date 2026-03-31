# Phase 3.1 restart: HF converts + synthetic JSON/control + mixed build + prepare_chat_sft
# Requires: py -3, datasets, repo root as cwd (or set MYPT_ROOT).

$ErrorActionPreference = "Stop"
$ROOT = if ($env:MYPT_ROOT) { $env:MYPT_ROOT } else { Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }
Set-Location $ROOT

$py = "py"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { $py = "python" }

$dataHF = "data/sft_hf"
$dataMid = "data/sft_phase3_intermediate"
$outChat = "data/sft_phase3_1_restart_chat"

New-Item -ItemType Directory -Force -Path $dataHF,$dataMid | Out-Null

& $py -3 scripts/sft/convert_hf_dataset.py `
  --dataset HuggingFaceH4/no_robots `
  --output "$dataHF/no_robots.jsonl" `
  --languages en de `
  --max_examples 50000

& $py -3 scripts/sft/convert_hf_dataset.py `
  --dataset AmanPriyanshu/reasoning-sft-JSON-structuring-and-correcting `
  --output "$dataHF/amans_json_structuring.jsonl" `
  --languages en de `
  --max_examples 40000

& $py -3 scripts/sft/generate_phase3_phase31_control_sft.py `
  --output "$dataMid/phase3_phase31_control.jsonl"

& $py -3 scripts/sft/generate_phase3_json_sft.py `
  --output "$dataMid/phase3_json_strict.jsonl" `
  --num_examples 6000

& $py -3 scripts/sft/build_phase3_dataset.py `
  --output "$dataMid/phase3_1_restart_mixed.jsonl" `
  --meta_output "$dataMid/phase3_1_restart_mixed.meta.json" `
  --target_size 80000 `
  --precision_file data/sft_phase3_intermediate/phase3_precision_ref.jsonl `
  --grounded_file data/sft_phase3_intermediate/phase3_grounded_ref.jsonl `
  --remix_ratio 0.18 `
  --json_ratio 0.04 `
  --json_file "$dataMid/phase3_json_strict.jsonl" `
  --json_hf_file "$dataHF/amans_json_structuring.jsonl" `
  --json_hf_ratio 0.03 `
  --phase31_control_file "$dataMid/phase3_phase31_control.jsonl" `
  --phase31_control_ratio 0.04 `
  --injection_file data/sft_phase3_intermediate/phase3_injection_hierarchy_strict.jsonl `
  --injection_ratio 0.06 `
  --abstention_file data/sft_phase3_intermediate/phase3_abstention_strict.jsonl `
  --abstention_ratio 0.05 `
  --grounded_ratio 0.14 `
  --open_chat_cap_ratio 0.20 `
  --open_chat_files `
    "$dataHF/no_robots.jsonl" `
    data/sft_hf/oasst2.jsonl `
    data/sft_hf/dolci_instruct.jsonl `
    data/sft_hf/slimorca.jsonl `
    data/sft_hf/dolly.jsonl

if (Test-Path scripts/sft/audit_phase3_mix.py) {
  & $py -3 scripts/sft/audit_phase3_mix.py --input "$dataMid/phase3_1_restart_mixed.jsonl"
}
if (Test-Path scripts/sft/normalize_chat_jsonl.py) {
  & $py -3 scripts/sft/normalize_chat_jsonl.py --input "$dataMid/phase3_1_restart_mixed.jsonl" --in-place
}

& $py -3 scripts/sft/prepare_chat_sft.py `
  --input "$dataMid/phase3_1_restart_mixed.jsonl" `
  --output $outChat

Write-Host "Done. Packed chat: $outChat"
