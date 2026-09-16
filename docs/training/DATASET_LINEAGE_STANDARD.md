# Dataset Lineage Standard (Pipeline-Core)

## Purpose

`core/dataset_lineage.py` defines a pipeline-wide lineage and audit standard for **all** dataset builders, not just SFT.

Use it to keep complete creation history for:
- unified pretraining build/mix pipelines
- SFT generators, mixers, and preparers
- downstream audit/report scripts

## Canonical Fields

Every lineage block should provide:
- `creation_context.timestamp`
- `creation_context.script`
- `creation_context.args`
- `direct_inputs[]` (`path`, `sampled_rows`, `effective_ratio`)
- `recursive_origins[]` (all reachable leaf origins)
- `flattened_contributions[]` (`origin_path`, `effective_rows`, `effective_percent`)
- `upstream_configs[]` (inherited creation contexts from parents)

## Storage Convention

- For JSONL outputs: sidecar `*.lineage.json`
- For dataset directories: `dataset_lineage.json`
- Metadata files may also embed lineage directly under `lineage`

The loader checks common metadata/sidecar locations recursively and falls back
to a leaf-origin record when no lineage is found.

## API

Core functions in `core/dataset_lineage.py`:
- `iso_now()`
- `count_jsonl_rows(path)`
- `load_lineage_for_input(path)`
- `merge_lineage(inputs, output_rows, creation_context)`
- `write_lineage_sidecar(output_dataset_path, lineage)`

## Multi-Hop Example

If:
- `A + B -> X`
- `X + Y -> Z`

then `Z` should contain lineage whose `flattened_contributions` includes
`A`, `B`, and `Y` with final effective percentages.

## Current Integration

- Unified build: `scripts/unified_build/mix_multi_source.py`, `scripts/unified_build/build_unified_dataset.py`
- SFT pipeline: all active phase builders/mixers/preparers and major generators

## Audit Usage

Lineage can be used as a primary audit source for:
- source provenance verification
- reproducibility/debugging
- train/val split chain validation
- compliance traceability

