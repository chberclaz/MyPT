# Data Preparation Scripts

Additional scripts for dataset conversion and augmentation.

---

## convert_opensubtitles.py

Converts OPUS OpenSubtitles parallel corpus from Moses format to TSV.

### Problem Solved

The OPUS OpenSubtitles download contains separate files for each language:
- `OpenSubtitles.de-en.de` — German sentences (one per line)
- `OpenSubtitles.de-en.en` — English sentences (one per line)

These need to be merged into a single TSV file for use with `prepare_weighted_dataset.py`.

### Usage

```bash
python scripts/convert_opensubtitles.py --input_dir data/raw/opensubtitles_de_en
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input_dir` | `data/raw/opensubtitles_de_en` | Directory with zip or extracted files |
| `--output_file` | `<input_dir>/opensubtitles_de_en.tsv` | Output TSV path |
| `--max_lines` | None | Limit lines (for testing) |
| `--skip_extract` | False | Skip zip extraction |

### Output Format

Tab-separated parallel sentences:
```
german_sentence<TAB>english_sentence
```

### Example Output

```
============================================================
  SUCCESS!
============================================================
  Output: data/raw/opensubtitles_de_en/opensubtitles_de_en.tsv
  Lines: 22,491,121
```

---

## append_to_dataset.py

Appends additional source data to an existing sharded dataset.

### Problem Solved

When you need to add more data to a dataset that's already been prepared:
- Source wasn't available during initial run
- Want to augment with additional data
- Incremental dataset building

### Usage

```bash
python scripts/append_to_dataset.py \
    --dataset_dir data/multilingual_1.5B_wiki90 \
    --source opensub:data/raw/opensubtitles_de_en/opensubtitles_de_en.tsv \
    --target_tokens 75000000 \
    --split_tsv
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset_dir` | Required | Existing dataset with train/val subdirs |
| `--source` | Required | `NAME:path1,path2` format |
| `--target_tokens` | Required | Tokens to add from this source |
| `--tokenization` | `gpt2` | Tokenizer type |
| `--tokens_per_shard` | `10000000` | Tokens per .bin shard |
| `--val_fraction` | `0.05` | Fraction of new shards for validation |
| `--split_tsv` | False | Split TSV lines into segments |
| `--no_normalize` | False | Skip text normalization |
| `--no_filter` | False | Skip line filtering |

### What It Does

1. Loads existing dataset metadata
2. Determines next shard index
3. Tokenizes new source data
4. Writes additional shards
5. Distributes to train/val directories
6. Updates `dataset_metadata.json` with combined totals

### Example Output

```
============================================================
  APPEND COMPLETE
============================================================
  New tokens added: 75,000,000
  New shards: 8 (train: 7, val: 1)

  Updated totals:
    Total tokens: 1,350,000,000 → 1,425,000,000
    Train shards: 128 → 135
    Val shards: 7 → 8
```

---

## Typical Workflow

### Scenario: Build multilingual dataset with OpenSubtitles

```bash
# 1. Run main pipeline (opensub might fail due to format)
python scripts/fetch_and_prepare_multilingual.py \
    --sources_file data/sources/multilingual_de_en.json \
    --total_tokens 1500000000 \
    --out_dir data/multilingual_1.5B

# 2. Convert OpenSubtitles to TSV
python scripts/convert_opensubtitles.py \
    --input_dir data/raw/opensubtitles_de_en

# 3. Append OpenSubtitles to dataset
python scripts/append_to_dataset.py \
    --dataset_dir data/multilingual_1.5B \
    --source opensub:data/raw/opensubtitles_de_en/opensubtitles_de_en.tsv \
    --target_tokens 75000000 \
    --split_tsv
```

---

## See Also

- [prepare_weighted_dataset.py](../scripts/README.md) — Main dataset preparation
- [fetch_and_prepare_multilingual.py](../scripts/README.md) — Full download + prepare pipeline


