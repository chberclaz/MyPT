#!/usr/bin/env python
"""
Swiss Federal Law Downloader - Fedlex Assets (HTML)

Downloads Swiss federal legislation from the droid-f/fedlex-assets GitHub repository
which contains the actual HTML content of Swiss laws (not just metadata).

Output format (per document):
    <|doc|>
    source=fedlex
    level=federal
    lang=de
    type=statute
    id=SR-220
    title=Obligationenrecht
    <|text|>
    Art. 1 Begriff
    1 Zum Abschluss eines Vertrages ...
    <|enddoc|>

Source: https://github.com/droid-f/fedlex-assets
This repository contains the HTML manifestations of Swiss federal legislation.

Usage:
    # Default: DE + EN combined into one file
    python tools/swiss_law_fedlex_scraper.py --out data/swiss_law/fedlex_corpus.txt

    # Quick test with limit
    python tools/swiss_law_fedlex_scraper.py --out test.txt --limit 100

    # Resume interrupted download
    python tools/swiss_law_fedlex_scraper.py --out corpus.txt --resume

Author: MyPT Project
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, TextIO

# Add parent for banner
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.banner import print_banner
    HAS_BANNER = True
except ImportError:
    HAS_BANNER = False

# HTML parsing
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# GitHub repository with HTML content (actual law texts)
FEDLEX_ASSETS_REPO = "https://github.com/droid-f/fedlex-assets.git"

# Document markers (tokenizer-friendly)
DOC_START = "<|doc|>"
DOC_TEXT = "<|text|>"
DOC_END = "<|enddoc|>"

# Supported languages
SUPPORTED_LANGS = {"de", "en", "fr", "it", "rm"}


def clone_fedlex_assets_repo(target_dir: Path) -> bool:
    """Clone the Fedlex Assets GitHub repository (HTML content)."""
    if (target_dir / ".git").exists():
        logger.info("Fedlex-assets repo already cloned, pulling updates...")
        try:
            subprocess.run(["git", "pull"], cwd=target_dir, capture_output=True)
            return True
        except Exception:
            return True  # Already exists, good enough
    
    logger.info(f"Cloning Fedlex-assets repository to {target_dir}...")
    logger.info("(This may take 5-10 minutes - ~1.5GB repo with HTML content)")
    
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", FEDLEX_ASSETS_REPO, str(target_dir)],
            capture_output=True,
            text=True,
            timeout=1200  # 20 minutes for larger repo
        )
        
        if result.returncode == 0:
            logger.info("Clone successful!")
            return True
        else:
            logger.error(f"Clone failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Clone timed out after 20 minutes")
        return False
    except FileNotFoundError:
        logger.error("Git not found. Please install git.")
        return False


def extract_info_from_path(file_path: Path, repo_dir: Path) -> Dict:
    """Extract SR number, language, doc type, and VERSION DATE from HTML file path.
    
    Path format: /eli/{type}/{sr_parts...}/{date_YYYYMMDD}/{lang}/html/{filename}.html
    Example: /eli/cc/27/317_321_377/20250101/de/html/fedlex-...-de-html.html
    
    The DATE folder (8 digits like 20250101) indicates the version date.
    We want the MOST RECENT date for each SR+language.
    """
    info = {
        'sr_id': 'SR-unknown',
        'lang': None,
        'doc_type': 'statute',
        'version_date': '00000000',  # YYYYMMDD format for sorting
    }
    
    try:
        rel_path = file_path.relative_to(repo_dir)
        path_parts = rel_path.parts
        
        # Detect doc type from path
        if 'cc' in path_parts:
            info['doc_type'] = 'statute'
        elif 'oc' in path_parts:
            info['doc_type'] = 'official_compilation'
        elif 'fga' in path_parts:
            info['doc_type'] = 'federal_gazette'
        
        # Find the doc type marker index (cc, oc, or fga)
        doc_type_idx = None
        for i, part in enumerate(path_parts):
            if part in ('cc', 'oc', 'fga'):
                doc_type_idx = i
                break
        
        if doc_type_idx is None:
            return info
        
        # Collect SR parts and find the date
        # Path after doc_type: {sr_parts...}/{date}/{lang}/html/{file}
        sr_parts = []
        for i in range(doc_type_idx + 1, len(path_parts)):
            part = path_parts[i]
            
            # Check if this is the VERSION DATE (8 digits starting with 19 or 20)
            if re.match(r'^(19|20)\d{6}$', part):
                info['version_date'] = part
                # The next part should be the language
                if i + 1 < len(path_parts) and path_parts[i + 1] in SUPPORTED_LANGS:
                    info['lang'] = path_parts[i + 1]
                break
            
            # Check if this is a language (we've gone too far)
            if part in SUPPORTED_LANGS:
                info['lang'] = part
                break
            
            # Check if this is 'html' folder (we've gone too far)
            if part == 'html':
                break
            
            # This is an SR number part - collect it
            # SR parts can be: numbers, numbers with underscores (e.g., "317_321_377")
            if re.match(r'^[\d_]+$', part) or re.match(r'^[IVXLCDM]+$', part):
                sr_parts.append(part.replace('_', '.'))
        
        if sr_parts:
            info['sr_id'] = f"SR-{'.'.join(sr_parts)}"
        
    except Exception as e:
        logger.debug(f"Error extracting info from {file_path}: {e}")
    
    return info


def is_substring_of_existing(text: str, existing_texts: List[str]) -> bool:
    """Check if text is a substring of any existing text or vice versa."""
    for existing in existing_texts:
        # Skip if this text is contained in an existing one
        if text in existing:
            return True
        # If existing is contained in this text, we'll replace it later
    return False


def deduplicate_texts(texts: List[str]) -> List[str]:
    """Remove texts that are substrings of other texts."""
    if not texts:
        return []
    
    # Sort by length descending - keep longer texts
    sorted_texts = sorted(texts, key=len, reverse=True)
    
    result: List[str] = []
    for text in sorted_texts:
        # Check if this text is a substring of any already-added text
        is_substring = False
        for existing in result:
            if text in existing:
                is_substring = True
                break
        if not is_substring:
            result.append(text)
    
    # Return in original-ish order (by first appearance)
    text_order = {t: i for i, t in enumerate(texts)}
    result.sort(key=lambda t: text_order.get(t, len(texts)))
    return result


def extract_text_from_html(html_content: str) -> tuple[Optional[str], Optional[str]]:
    """Extract title and text content from HTML using BeautifulSoup.
    
    Returns: (title, text_content)
    """
    if not HAS_BS4:
        # Fallback: basic regex extraction
        return extract_text_from_html_regex(html_content)
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, nav elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'meta', 'link']):
            tag.decompose()
        
        # Try to extract title
        title = None
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
        if not title:
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text(strip=True)
        
        # Extract main content
        # Look for main content containers
        main_content = None
        for selector in ['main', 'article', '.content', '#content', '.law-text', '.document']:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.body if soup.body else soup
        
        # BETTER APPROACH: Get text from leaf nodes only (elements with no child elements)
        # This avoids the parent-containing-child duplication issue
        text_parts: List[str] = []
        
        def get_direct_text(element) -> Optional[str]:
            """Get text only from this element, not from children."""
            # Get only direct text content (not from children)
            direct_text = ''.join(
                child for child in element.children 
                if isinstance(child, str)
            ).strip()
            return direct_text if direct_text else None
        
        def is_leaf_or_text_container(element) -> bool:
            """Check if element is a leaf or primarily contains text."""
            # Elements that typically contain actual content
            text_tags = {'p', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
            if element.name in text_tags:
                return True
            # Check if it has no element children (only text)
            children = [c for c in element.children if hasattr(c, 'name')]
            return len(children) == 0
        
        # Process only leaf/text elements
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                                               'p', 'li', 'td', 'th']):
            text = element.get_text(separator=' ', strip=True)
            if text and len(text) > 10:
                # Clean up whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                text_parts.append(text)
        
        if not text_parts:
            # Fallback: get all text directly
            text = main_content.get_text(separator='\n', strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)
            return (title, text if len(text) > 100 else None)
        
        # Deduplicate - remove texts that are substrings of others
        unique_parts = deduplicate_texts(text_parts)
        
        content = '\n\n'.join(unique_parts)
        return (title, content if len(content) > 100 else None)
        
    except Exception as e:
        logger.debug(f"BeautifulSoup error: {e}")
        return extract_text_from_html_regex(html_content)


def extract_text_from_html_regex(html_content: str) -> tuple[Optional[str], Optional[str]]:
    """Fallback HTML extraction using regex (when BeautifulSoup not available)."""
    # Extract title
    title = None
    title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    
    # Remove script/style tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML tags but keep content
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</p>', '\n\n', text)
    text = re.sub(r'</div>', '\n', text)
    text = re.sub(r'</h[1-6]>', '\n\n', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Decode HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    
    # Clean whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return (title, text if len(text) > 100 else None)


def write_document(out_f: TextIO, sr_id: str, lang: str, doc_type: str, 
                   title: Optional[str], content: str):
    """Write a single document with structured markers."""
    out_f.write(DOC_START + "\n")
    out_f.write("source=fedlex\n")
    out_f.write("level=federal\n")
    out_f.write(f"lang={lang}\n")
    out_f.write(f"type={doc_type}\n")
    out_f.write(f"id={sr_id}\n")
    if title:
        # Clean title (remove newlines)
        clean_title = ' '.join(title.split())
        out_f.write(f"title={clean_title}\n")
    out_f.write(DOC_TEXT + "\n")
    out_f.write(content.strip() + "\n")
    out_f.write(DOC_END + "\n\n")


def load_done_set(out_path: Path) -> Set[str]:
    """Load already processed document IDs from existing output file."""
    done: Set[str] = set()
    if not out_path.exists():
        return done
    
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            current_lang = None
            current_id = None
            for line in f:
                if line.startswith("lang="):
                    current_lang = line.strip().split("=", 1)[1]
                elif line.startswith("id="):
                    current_id = line.strip().split("=", 1)[1]
                    if current_lang and current_id:
                        done.add(f"{current_lang}:{current_id}")
    except Exception as e:
        logger.warning(f"Could not read existing output for resume: {e}")
    
    return done


def process_fedlex_assets(repo_dir: Path, out_path: Path, languages: List[str],
                          limit: Optional[int], resume: bool) -> Dict:
    """Process the cloned Fedlex-assets repository (HTML files) and write to single output file.
    
    Uses a TWO-PASS approach:
    - Pass 1: Scan all files, find the MOST RECENT version date for each SR+language
    - Pass 2: Process only files from the most recent version date
    """
    stats = {
        'files_scanned': 0,
        'files_processed': 0,
        'docs_written': 0,
        'bytes_total': 0,
        'skipped_lang': 0,
        'skipped_done': 0,
        'skipped_older': 0,
        'skipped_short': 0,
        'errors': 0,
    }
    
    logger.info(f"Scanning {repo_dir} for HTML files...")
    html_files = list(repo_dir.rglob("*.html"))
    logger.info(f"Found {len(html_files)} HTML files total")
    
    # =========================================================================
    # PASS 1: Find the MOST RECENT version date for each SR+language
    # =========================================================================
    logger.info("Pass 1: Finding most recent version for each law...")
    
    # Dict: doc_key (lang:sr_id) -> (best_date, list of files with that date, info)
    best_versions: Dict[str, tuple] = {}
    
    for html_file in html_files:
        stats['files_scanned'] += 1
        
        info = extract_info_from_path(html_file, repo_dir)
        
        # Skip if language not detected or not requested
        if info['lang'] not in languages:
            stats['skipped_lang'] += 1
            continue
        
        doc_key = f"{info['lang']}:{info['sr_id']}"
        version_date = info['version_date']
        
        if doc_key not in best_versions:
            # First time seeing this SR+language
            best_versions[doc_key] = (version_date, [html_file], info)
        elif version_date > best_versions[doc_key][0]:
            # Found a MORE RECENT version - replace
            stats['skipped_older'] += len(best_versions[doc_key][1])
            best_versions[doc_key] = (version_date, [html_file], info)
        elif version_date == best_versions[doc_key][0]:
            # Same date - add to list (multiple HTML files for same version)
            best_versions[doc_key][1].append(html_file)
        else:
            # Older version - skip
            stats['skipped_older'] += 1
    
    logger.info(f"Pass 1 complete: {len(best_versions)} unique laws (SR+language) found")
    logger.info(f"  Skipped {stats['skipped_older']} files from older versions")
    logger.info(f"  Skipped {stats['skipped_lang']} files in other languages")
    
    # Apply limit if specified (limit on unique laws, not files)
    if limit and len(best_versions) > limit:
        keys_to_keep = list(best_versions.keys())[:limit]
        best_versions = {k: best_versions[k] for k in keys_to_keep}
        logger.info(f"Limited to {limit} unique laws")
    
    # Load already done documents if resuming
    done_set: Set[str] = set()
    if resume:
        done_set = load_done_set(out_path)
        logger.info(f"Resume mode: {len(done_set)} documents already done")
    
    # =========================================================================
    # PASS 2: Process the most recent files for each SR+language
    # =========================================================================
    logger.info("Pass 2: Extracting text from most recent versions...")
    
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output file
    mode = "a" if resume else "w"
    with open(out_path, mode, encoding="utf-8") as out_f:
        # Write header if starting fresh
        if not resume or not out_path.exists() or out_path.stat().st_size == 0:
            out_f.write("# Fedlex Swiss Law Corpus (Full Text - Most Recent Versions)\n")
            out_f.write(f"# Languages: {', '.join(languages)}\n")
            out_f.write(f"# Source: {FEDLEX_ASSETS_REPO}\n")
            out_f.write(f"# Generated: {datetime.now().isoformat()}\n")
            out_f.write("# Note: Only the most recent version of each law is included\n\n")
        
        processed_count = 0
        for doc_key, (version_date, files, info) in best_versions.items():
            processed_count += 1
            if processed_count % 500 == 0:
                logger.info(f"Progress: {processed_count}/{len(best_versions)} laws | "
                           f"{stats['docs_written']} written | "
                           f"{stats['bytes_total'] / (1024*1024):.1f} MB")
            
            # Skip if already done in a previous run (resume mode)
            if doc_key in done_set:
                stats['skipped_done'] += 1
                continue
            
            # Try each file for this SR+language until we get content
            best_content = None
            best_title = None
            
            for html_file in files:
                try:
                    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                        html_content = f.read()
                    
                    stats['files_processed'] += 1
                    title, content = extract_text_from_html(html_content)
                    
                    # Keep the longest content we find
                    if content and (best_content is None or len(content) > len(best_content)):
                        best_content = content
                        best_title = title
                        
                except Exception as e:
                    stats['errors'] += 1
                    logger.debug(f"Error reading {html_file}: {e}")
            
            # Skip if no content or too short
            if not best_content or len(best_content) < 200:
                stats['skipped_short'] += 1
                continue
            
            # Write document with version date in metadata
            out_f.write(DOC_START + "\n")
            out_f.write("source=fedlex\n")
            out_f.write("level=federal\n")
            out_f.write(f"lang={info['lang']}\n")
            out_f.write(f"type={info['doc_type']}\n")
            out_f.write(f"id={info['sr_id']}\n")
            out_f.write(f"version={version_date}\n")
            if best_title:
                clean_title = ' '.join(best_title.split())
                out_f.write(f"title={clean_title}\n")
            out_f.write(DOC_TEXT + "\n")
            out_f.write(best_content.strip() + "\n")
            out_f.write(DOC_END + "\n\n")
            
            stats['docs_written'] += 1
            stats['bytes_total'] += len(best_content.encode('utf-8'))
    
    return stats


def parse_args():
    ap = argparse.ArgumentParser(
        description="Export Swiss federal law (Fedlex) to structured plaintext for LLM training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    ap.add_argument("--out", default="data/swiss_law/fedlex_corpus.txt",
                    help="Output file path (all languages combined into this file).")
    ap.add_argument("--lang", nargs="+", default=["de", "en"],
                    help="Languages to export (default: de en). All go into one output file.")
    ap.add_argument("--repo_dir", default=None,
                    help="Directory for cloned Fedlex-assets repo (default: alongside output)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of HTML files to process (for testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume by skipping documents already in output file")
    ap.add_argument("--skip_clone", action="store_true",
                    help="Skip cloning if repo already exists")
    return ap.parse_args()


def main():
    args = parse_args()
    
    # Check for BeautifulSoup
    if not HAS_BS4:
        logger.warning("BeautifulSoup not installed. Using fallback regex extraction.")
        logger.warning("For better results, run: pip install beautifulsoup4")
    
    # Validate languages
    languages = [lang.lower() for lang in args.lang]
    for lang in languages:
        if lang not in SUPPORTED_LANGS:
            raise SystemExit(f"Unsupported language '{lang}'. Supported: {sorted(SUPPORTED_LANGS)}")
    
    # Print banner
    if HAS_BANNER:
        print_banner("MyPT", "Fedlex Swiss Law Corpus")
    else:
        print("=" * 60)
        print("  Fedlex Swiss Law Corpus Builder")
        print("  Swiss Federal Legislation (HTML) -> Plaintext")
        print("=" * 60)
    
    out_path = Path(args.out)
    repo_dir = Path(args.repo_dir) if args.repo_dir else out_path.parent / "fedlex-assets-repo"
    
    print(f"\n  Output file:  {out_path}")
    print(f"  Languages:    {', '.join(languages)}")
    print(f"  Repo cache:   {repo_dir}")
    if args.limit:
        print(f"  Limit:        {args.limit} files")
    if args.resume:
        print(f"  Mode:         Resume")
    print(f"\n  Source: github.com/droid-f/fedlex-assets")
    print(f"  (Contains actual HTML law texts, ~1.5GB)")
    print()
    
    start_time = time.time()
    
    # Step 1: Clone the repository
    if not args.skip_clone or not (repo_dir / ".git").exists():
        if not clone_fedlex_assets_repo(repo_dir):
            logger.error("Failed to clone Fedlex-assets repository")
            logger.info("\nAlternative: Clone manually with:")
            logger.info(f"  git clone --depth 1 {FEDLEX_ASSETS_REPO} {repo_dir}")
            sys.exit(1)
    
    # Step 2: Process and write output
    logger.info("\nExtracting text content from HTML files...")
    stats = process_fedlex_assets(repo_dir, out_path, languages, args.limit, args.resume)
    
    elapsed = time.time() - start_time
    
    # Save metadata
    meta_path = out_path.parent / "fedlex_metadata.json"
    metadata = {
        'download_date': datetime.now().isoformat(),
        'source_repo': FEDLEX_ASSETS_REPO,
        'languages': languages,
        'files_processed': stats['files_processed'],
        'docs_written': stats['docs_written'],
        'bytes_total': stats['bytes_total'],
        'output_file': str(out_path),
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  EXPORT COMPLETE")
    print("=" * 60)
    print(f"  Time elapsed:       {elapsed/60:.1f} minutes")
    print(f"  Files scanned:      {stats['files_scanned']}")
    print(f"  Files processed:    {stats['files_processed']}")
    print(f"  Unique laws found:  {stats['docs_written'] + stats['skipped_done'] + stats['skipped_short']}")
    print(f"  Documents written:  {stats['docs_written']}")
    print(f"  Total size:         {stats['bytes_total'] / (1024*1024):.1f} MB")
    print(f"  Skipped (lang):     {stats['skipped_lang']}")
    print(f"  Skipped (older):    {stats['skipped_older']}")
    print(f"  Skipped (done):     {stats['skipped_done']}")
    print(f"  Skipped (short):    {stats['skipped_short']}")
    print(f"  Errors:             {stats['errors']}")
    print("=" * 60)
    print(f"\n  Output file: {out_path}")
    print(f"  Metadata:    {meta_path}")
    print(f"\n  Document format (includes version date):")
    print(f"    {DOC_START}")
    print(f"    source=fedlex")
    print(f"    lang=de")
    print(f"    type=statute")
    print(f"    id=SR-220")
    print(f"    version=20250101")
    print(f"    title=Obligationenrecht")
    print(f"    {DOC_TEXT}")
    print(f"    Art. 1 Begriff des Vertrages")
    print(f"    1 Zum Abschlusse eines Vertrages ist...")
    print(f"    {DOC_END}")


if __name__ == "__main__":
    main()
