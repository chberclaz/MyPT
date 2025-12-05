# Data Sources Configuration Guide

This guide explains how to configure data sources for the `fetch_and_prepare_multilingual.py` script using JSON configuration files.

**Location:** Data source JSON files are stored in `data/sources/`

---

## Table of Contents

1. [Available Source Files](#available-source-files)
2. [Usage](#usage)
3. [JSON File Format](#json-file-format)
4. [Field Descriptions](#field-descriptions)
5. [Creating Custom Sources](#creating-custom-sources)
6. [Weight Normalization](#weight-normalization)
7. [Tips](#tips)
8. [Troubleshooting](#troubleshooting)

---

## Available Source Files

| File | Description |
|------|-------------|
| `multilingual_de_en.json` | German-English multilingual dataset (Wikipedia, OpenSubtitles, Europarl, News) |

---

## Usage

```bash
# Use default sources file
python scripts/fetch_and_prepare_multilingual.py

# Use a specific sources file
python scripts/fetch_and_prepare_multilingual.py --sources_file data/sources/multilingual_de_en.json

# List available source files
python scripts/fetch_and_prepare_multilingual.py --list_sources

# Override weights
python scripts/fetch_and_prepare_multilingual.py --weight wiki_en:0.4 --weight wiki_de:0.4

# Use only specific sources from the file
python scripts/fetch_and_prepare_multilingual.py --sources wiki_en wiki_de
```

---

## JSON File Format

Each sources file should have this structure:

```json
{
  "name": "Dataset Name",
  "description": "Human-readable description",
  "version": "1.0",
  "default_weights": {
    "source_key": 0.30,
    "another_source": 0.70
  },
  "sources": {
    "source_key": {
      "name": "Human Readable Name",
      "subdir": "directory_for_downloads",
      "urls": [
        ["filename1", "https://example.com/file1.zip"],
        ["filename2", "https://example.com/file2.zip"]
      ],
      "final_pattern": "*.txt",
      "type": "zip",
      "est_size_mb": 1000,
      "description": "Brief description"
    }
  }
}
```

---

## Field Descriptions

### Top-level Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | No | Human-readable name for the dataset |
| `description` | No | Description of the dataset |
| `version` | No | Version string |
| `default_weights` | No | Default sampling weights for each source |
| `sources` | **Yes** | Dictionary of data source configurations |

### Source Configuration Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | **Yes** | Human-readable name |
| `subdir` | **Yes** | Subdirectory name within data_dir |
| `urls` | **Yes** | List of `[filename, url]` pairs to download |
| `final_pattern` | **Yes** | Glob pattern for final data files (e.g., `*.txt`) |
| `type` | **Yes** | One of: `zip`, `gzip`, `split_zip` |
| `est_size_mb` | No | Estimated download size in MB |
| `description` | No | Brief description |
| `combined_name` | Only for `split_zip` | Filename for combined archive |

### Source Types

| Type | Description |
|------|-------------|
| `zip` | Standard ZIP archive - will be extracted |
| `gzip` | GZIP compressed file - will be decompressed |
| `split_zip` | Multi-part archive - parts combined then extracted |

---

## Creating Custom Sources

### Example: Single Language Wikipedia

```json
{
  "name": "French Wikipedia",
  "description": "French Wikipedia corpus",
  "version": "1.0",
  "default_weights": {
    "wiki_fr": 1.0
  },
  "sources": {
    "wiki_fr": {
      "name": "French Wikipedia",
      "subdir": "wikipedia_fr",
      "urls": [
        ["frwiki-dump.txt.gz", "https://example.com/frwiki.txt.gz"]
      ],
      "final_pattern": "*.txt",
      "type": "gzip",
      "est_size_mb": 5000,
      "description": "French Wikipedia articles"
    }
  }
}
```

### Example: Code Dataset

```json
{
  "name": "Code Dataset",
  "description": "Programming code from various sources",
  "version": "1.0",
  "default_weights": {
    "github_python": 0.5,
    "github_js": 0.3,
    "stackoverflow": 0.2
  },
  "sources": {
    "github_python": {
      "name": "GitHub Python",
      "subdir": "github_python",
      "urls": [
        ["python_code.tar.gz", "https://example.com/python.tar.gz"]
      ],
      "final_pattern": "*.py",
      "type": "gzip",
      "est_size_mb": 10000,
      "description": "Python code from GitHub"
    },
    "github_js": {
      "name": "GitHub JavaScript",
      "subdir": "github_js",
      "urls": [
        ["js_code.tar.gz", "https://example.com/js.tar.gz"]
      ],
      "final_pattern": "*.js",
      "type": "gzip",
      "est_size_mb": 8000,
      "description": "JavaScript code from GitHub"
    },
    "stackoverflow": {
      "name": "StackOverflow",
      "subdir": "stackoverflow",
      "urls": [
        ["so_dump.zip", "https://example.com/stackoverflow.zip"]
      ],
      "final_pattern": "*.xml",
      "type": "zip",
      "est_size_mb": 5000,
      "description": "StackOverflow Q&A"
    }
  }
}
```

---

## Weight Normalization

Weights are automatically normalized to sum to 1.0. For example:

```json
"default_weights": {
  "source_a": 3,
  "source_b": 2,
  "source_c": 1
}
```

Will be normalized to:
- `source_a`: 0.50 (50%)
- `source_b`: 0.33 (33%)
- `source_c`: 0.17 (17%)

---

## Tips

1. **Start with existing files**: Copy `multilingual_de_en.json` and modify it
2. **Test with `--download_only`**: Download first, then prepare separately
3. **Use `--skip_download`**: Skip downloads if files already exist
4. **Override weights**: Use `--weight source:value` for experimentation
5. **Select sources**: Use `--sources source1 source2` to use subset

---

## Adding New Data Sources

1. Create a new JSON file in `data/sources/`
2. Define the sources with their download URLs
3. Set appropriate weights
4. Test with:
   ```bash
   python scripts/fetch_and_prepare_multilingual.py \
       --sources_file data/sources/your_sources.json \
       --download_only
   ```

---

## Troubleshooting

### "Sources file not found"

```bash
# Check available files
python scripts/fetch_and_prepare_multilingual.py --list_sources
```

### "No files found for source"

- Check that `final_pattern` matches the extracted files
- Verify extraction completed successfully
- Check the `subdir` path is correct

### "Invalid weight format"

```bash
# Correct format (colon separator)
--weight wiki_en:0.4

# Wrong (equals sign)
--weight wiki_en=0.4
```

---

## See Also

- [Large Dataset Training Guide](LARGE_DATASET_TRAINING.md)
- [Sharded Dataset Implementation](SHARDED_DATASET_IMPLEMENTATION.md)
- [scripts/prepare_weighted_dataset.py](../scripts/prepare_weighted_dataset.py)
- [scripts/fetch_and_prepare_multilingual.py](../scripts/fetch_and_prepare_multilingual.py)

