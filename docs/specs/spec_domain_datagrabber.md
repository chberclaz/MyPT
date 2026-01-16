You are Cursor acting as my coding assistant. Implement a Phase-2 domain corpus builder for MyPT.

GOAL
Build a large offline plaintext corpus for a 750M model (Phase 2 / domain knowledge) focused on:

- IT security
- Internet protocols
- Bash + Unix fundamentals
- JavaScript/Node, Java, Python basics (docs + standards)
  Output: normalized plaintext shards ready for my tokenizer/sharder pipeline.

WORKFLOW (end-to-end)

1. Acquire sources (git clones + optional direct downloads) into ./sources/
2. Extract text candidates (md/rst/txt/manpages/xml/html where applicable) into ./work/raw/
3. Transform formats to clean plaintext (RFC XML -> txt, man pages -> txt, html -> txt)
4. Normalize/clean:
   - strip boilerplate/nav, collapse excessive whitespace
   - keep code blocks (but normalize indentation)
   - ensure UTF-8, remove control chars
5. Deduplicate aggressively:
   - exact dup via hash
   - near-dup via simhash/minhash (configurable threshold)
6. Chunk/shard:
   - write shards of ~25MB each (configurable), plus a manifest with provenance metadata
7. Produce final output:
   - ./out/corpus_shards/shard_0001.txt ...
   - ./out/manifest.jsonl with {source, path, license_hint, original_file, sha256, bytes}

REPOSITORIES / SOURCES TO COVER (clone/download)
Internet Protocols (canonical backbone)

- RFC XML: https://github.com/ietf-tools/rfcxml
- RFC Editor retrieval (optional): https://www.rfc-editor.org/retrieve/

Security (conceptual + practical)

- OWASP Top10: https://github.com/OWASP/Top10
- OWASP CheatSheetSeries: https://github.com/OWASP/CheatSheetSeries
- OWASP WSTG: https://github.com/OWASP/wstg
- MITRE CTI (ATT&CK): https://github.com/mitre/cti
- NIST SP publications (optional download): https://csrc.nist.gov/publications/sp

Bash / Unix fundamentals

- Linux man-pages: https://github.com/mkerrisk/man-pages
- GNU Bash: https://git.savannah.gnu.org/git/bash.git (docs/)

Languages

- Python docs: https://github.com/python/cpython (Doc/)
- Python PEPs: https://github.com/python/peps
- Node.js docs: https://github.com/nodejs/node (doc/)
- MDN content (large): https://github.com/mdn/content
- OpenJDK: https://github.com/openjdk/jdk (docs/)

DATA TRANSFORMATIONS (required)
A) Manpage -> txt

- Prefer `man -l` + `col -bx` if available, else parse roff with `groff -Tutf8 -man`.
  Examples:
  groff -Tutf8 -man <file> | col -bx > out.txt
  man -l <manfile> | col -bx > out.txt
  Handle _.1, _.2, _.3, _.5, _.7, _.8(.gz). Support gz by streaming gunzip.

B) RFC XML -> txt

- Parse XML; extract <title>, <abstract>, <section> text while preserving headings.
- Ignore boilerplate (front matter indexes).
- If needed, use existing xml2rfc tooling if installed; otherwise implement a simple extractor.

C) HTML -> txt (MDN/NIST)

- Use a robust HTML-to-text conversion:
  - remove nav/aside/footer
  - keep headings and code blocks
  - preserve lists reasonably
    Python: `readability-lxml` or `beautifulsoup4` + custom rules (prefer minimal deps; but ok if optional).
    If avoiding deps, implement a simple BeautifulSoup pass.

D) Markdown/RST -> txt

- Markdown: strip formatting but keep headings, code fences, inline code.
- RST: basic conversion (can be crude); keep code blocks and headings.

IMPLEMENTATION REQUIREMENTS

- Create a CLI tool: `python tools/build_phase2_corpus.py ...`
- Configurable via args + YAML/JSON config.
- Deterministic output with a fixed seed.
- Log progress and stats (files scanned, accepted, rejected, deduped, shard sizes).
- Write a manifest.jsonl for traceability (MyPT style).

SUGGESTED CLI
python tools/build_phase2_corpus.py \
 --sources_dir ./sources \
 --work_dir ./work \
 --out_dir ./out \
 --shard_mb 25 \
 --min_chars 400 \
 --languages en \
 --dedupe exact,simhash \
 --simhash_threshold 4 \
 --include_ext .md,.rst,.txt,.man,.1,.2,.3,.5,.7,.8,.xml,.html \
 --max_docs_per_repo 0 (0 = unlimited)

REPO ACQUISITION SCRIPT
Create `tools/fetch_phase2_sources.sh`:

- mkdir -p sources
- git clone --depth 1 <repo> sources/<name>
- For NIST SP PDFs/HTML: optional downloader stub (user can add)
- Print disk usage per repo

CLEANING RULES (KEEP IT SIMPLE + SAFE)

- Remove lines that look like site chrome (e.g., “Edit on GitHub”, “Previous/Next”, cookie banners)
- Drop empty/very short docs (< min_chars)
- Normalize whitespace; keep paragraph boundaries
- Keep code blocks; prefix with “CODE:” optionally for token clarity
- Force UTF-8, replace invalid chars

DEDUPLICATION

- Exact: sha256 of normalized text
- Near-dup: simhash over 5-grams; drop if Hamming distance <= threshold
- Save dedupe decisions in ./work/dedupe_report.jsonl

OUTPUT

- Write shards as plain UTF-8 text with separators:
  "==== DOC START ===="
  "SOURCE: <repo> | PATH: <path>"
  "===="
  <content>
  "==== DOC END ===="

DELIVERABLES

- tools/fetch_phase2_sources.sh
- tools/build_phase2_corpus.py
- tools/transformers/{man_to_txt.py,rfcxml_to_txt.py,html_to_txt.py,md_rst_to_txt.py}
- out/corpus_shards/\*.txt
- out/manifest.jsonl
- README snippet describing usage and licensing note: “User is responsible for verifying licenses of included sources.”

Start by scaffolding the repo structure and implementing the fetch + build pipeline with manpage and markdown support first, then add RFC XML and HTML.
