# Document Format Support

MyPT's workspace and RAG system can index various document formats. This guide covers installation and configuration for PDF and DOCX support.

---

## Supported Formats

| Format     | Extensions         | Library     | Status                   |
| ---------- | ------------------ | ----------- | ------------------------ |
| Plain Text | `.txt`, `.text`    | Built-in    | ✅ Always available      |
| Markdown   | `.md`, `.markdown` | Built-in    | ✅ Always available      |
| PDF        | `.pdf`             | PyMuPDF     | 📦 Requires installation |
| Word       | `.docx`            | python-docx | 📦 Requires installation |

---

## Installation

### Quick Install (Both Libraries)

```bash
pip install PyMuPDF python-docx
```

### Per-Library Installation

#### PyMuPDF (PDF Support)

PyMuPDF (also known as `fitz`) is a high-performance PDF library that works on **Windows**, **Linux**, and **macOS**.

```bash
pip install PyMuPDF
```

**Features:**

- Fast text extraction from all PDF pages
- Handles complex layouts (multi-column, tables)
- Works with embedded fonts
- No system dependencies required
- Pre-built wheels for all major platforms

**Package Size:** ~15-20 MB (includes MuPDF binaries)

#### python-docx (DOCX Support)

python-docx is a pure Python library for reading Word documents.

```bash
pip install python-docx
```

**Features:**

- Extracts text from paragraphs
- Extracts content from tables
- Pure Python - no system dependencies
- Cross-platform

**Package Size:** ~200 KB

---

## Platform-Specific Notes

### Windows

Both libraries install cleanly via pip with pre-built wheels:

```powershell
# PowerShell / Command Prompt
pip install PyMuPDF python-docx
```

No additional configuration needed.

### Linux (Ubuntu/Debian)

```bash
pip install PyMuPDF python-docx
```

PyMuPDF includes its own MuPDF binaries, so no system packages are required.

### Linux (RHEL/CentOS/Rocky)

```bash
pip install PyMuPDF python-docx
```

### macOS

```bash
pip install PyMuPDF python-docx
```

---

## Offline Installation

For air-gapped environments:

### 1. Download packages on online machine

```bash
# Create packages directory
mkdir packages

# Download PyMuPDF and dependencies
pip download PyMuPDF -d ./packages

# Download python-docx and dependencies
pip download python-docx -d ./packages
```

### 2. Transfer packages folder to offline machine

### 3. Install on offline machine

```bash
pip install --no-index --find-links=./packages PyMuPDF python-docx
```

---

## Verification

### Check Available Formats

```python
from core.document.loader import get_supported_formats

formats = get_supported_formats()
for ext, available in formats.items():
    status = "✅" if available else "❌"
    print(f"{status} {ext}")
```

**Expected output with all libraries installed:**

```
✅ .txt
✅ .text
✅ .md
✅ .markdown
✅ .pdf
✅ .docx
```

### Test PDF Loading

```python
from core.document.loader import DocumentLoader

loader = DocumentLoader()
doc = loader.load_file("path/to/document.pdf")

if doc:
    print(f"Loaded: {doc.filename}")
    print(f"Characters: {doc.num_chars}")
    print(f"First 500 chars:\n{doc.text[:500]}")
```

### Test DOCX Loading

```python
from core.document.loader import DocumentLoader

loader = DocumentLoader()
doc = loader.load_file("path/to/document.docx")

if doc:
    print(f"Loaded: {doc.filename}")
    print(f"Characters: {doc.num_chars}")
    print(f"First 500 chars:\n{doc.text[:500]}")
```

---

## Web UI Integration

Once installed, the web UI workspace will automatically detect and index PDF and DOCX files:

1. **Add documents** to your workspace directory
2. **Rebuild index** via the workspace panel
3. **Query** your documents using RAG

The workspace info panel will show the file types being indexed.

---

## Troubleshooting

### "PyMuPDF not installed" Error

```bash
# Verify installation
pip show PyMuPDF

# If not found, install
pip install PyMuPDF
```

### "python-docx not installed" Error

```bash
# Verify installation
pip show python-docx

# If not found, install
pip install python-docx
```

### PDF Extraction Issues

**Empty text from PDF:**

- The PDF might be image-based (scanned document)
- PyMuPDF extracts embedded text, not OCR
- For scanned PDFs, consider OCR tools like `pytesseract`

**Garbled text:**

- PDF might use custom encoding
- Try opening in a PDF reader to verify original text

### DOCX Extraction Issues

**"Package not found" error:**

- File might be corrupted
- File might be `.doc` (old format) instead of `.docx`
- python-docx only supports `.docx` (Office 2007+)

**Missing content:**

- Headers/footers are not extracted by default
- Some complex elements (charts, SmartArt) are not text-extractable

---

## Performance Considerations

### PDF Files

- Large PDFs (100+ pages) may take a few seconds to process
- Text extraction is memory-efficient (streaming)
- Typical speed: ~50-100 pages/second

### DOCX Files

- Very fast extraction (pure Python)
- Memory usage scales with document size
- Typical speed: ~1000 paragraphs/second

---

## API Reference

### DocumentLoader Class

```python
from core.document.loader import DocumentLoader, get_supported_formats

# Check what formats are available
formats = get_supported_formats()
# Returns: {'.txt': True, '.pdf': True, '.docx': True, ...}

# Create loader
loader = DocumentLoader()

# Load single file
doc = loader.load_file("report.pdf")

# Load directory (recursive)
docs = loader.load_directory("workspace/", recursive=True)

# Load specific extensions only
docs = loader.load_directory("workspace/", extensions={'.pdf', '.docx'})
```

### Document Dataclass

```python
@dataclass
class Document:
    text: str          # Full extracted text
    source: str        # Absolute file path
    filename: str      # Just the filename
    format: str        # Extension without dot (pdf, docx, txt)
    metadata: Dict     # size_bytes, modified_time, created_time

    # Properties
    num_chars: int     # len(text)
    num_lines: int     # text.count('\n') + 1
```

---

## See Also

- [WEBAPP_GUIDE.md](WEBAPP_GUIDE.md) - Web UI workspace usage
- [workspace_api.md](workspace_api.md) - Workspace API reference
- [spec_RAG.md](spec_RAG.md) - RAG pipeline specification
