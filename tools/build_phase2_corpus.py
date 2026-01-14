#!/usr/bin/env python
"""
Build Phase 2 Domain Corpus for MyPT

This script builds a domain-specific text corpus for Phase 2 training,
focused on IT security, protocols, Bash/Unix, and programming languages.

Pipeline:
1. Optionally fetch sources (git clone repositories)
2. Scan sources for supported files (.md, .rst, .txt, .man, .xml, .html)
3. Transform files to clean plaintext using format-specific converters
4. Clean and normalize text (remove boilerplate, normalize whitespace)
5. Deduplicate (exact via SHA-256, near-dup via simhash)
6. Output clean text files organized by source
7. Optionally call prepare_weighted_dataset.py for tokenization/sharding

Usage:
    python tools/build_phase2_corpus.py --sources_dir ./sources --out_dir ./out

    # Full pipeline with fetching and tokenization
    python tools/build_phase2_corpus.py \\
        --sources_dir ./sources \\
        --out_dir ./out \\
        --fetch \\
        --run_tokenizer \\
        --total_tokens 100000000

Options:
    --sources_dir       Directory containing cloned source repos
    --work_dir          Working directory for intermediate files (default: ./work)
    --out_dir           Output directory for corpus (default: ./out)
    --fetch             Fetch/clone source repositories before building
    --config_file       Optional JSON/YAML config file
    --min_chars         Minimum characters per document (default: 400)
    --shard_mb          Target shard size in MB (default: 25)
    --dedupe            Deduplication methods: exact,simhash (default: exact)
    --simhash_threshold Simhash Hamming distance threshold (default: 4)
    --include_ext       File extensions to process (default: .md,.rst,.txt,.man,.xml,.html)
    --max_docs_per_repo Max docs per repo, 0 = unlimited (default: 0)
    --run_tokenizer     Run prepare_weighted_dataset.py after building corpus
    --total_tokens      Total tokens for tokenizer (default: 100000000)
    --seed              Random seed for reproducibility (default: 42)
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Iterator

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import banner
from core.banner import print_banner

# Import transformers
from tools.transformers import (
    ManPageConverter,
    RFCXMLConverter,
    HTMLConverter,
    MarkdownRSTConverter,
    JavaDocConverter,
    TexinfoConverter,
    CodeDocExtractor,
    RFCDownloader,
)

# Audit logging for compliance
try:
    from core.compliance import audit
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, log_level: int = logging.INFO) -> logging.Logger:
    """Set up logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"build_corpus_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("phase2_corpus")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler - info level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('  [%(levelname)s] %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    return logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SourceConfig:
    """Configuration for a source repository."""
    name: str
    path: str
    repo_url: str = ""
    weight: float = 1.0
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    description: str = ""
    license_hint: str = ""


DEFAULT_REPOS = {
    # Security sources - OWASP Projects (comprehensive)
    "owasp-top10": SourceConfig(
        name="OWASP Top 10",
        path="owasp-top10",
        repo_url="https://github.com/OWASP/Top10.git",
        weight=1.5,
        include_patterns=["**/*.md"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="OWASP Top 10 security vulnerabilities",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-cheatsheets": SourceConfig(
        name="OWASP Cheat Sheets",
        path="owasp-cheatsheets",
        repo_url="https://github.com/OWASP/CheatSheetSeries.git",
        weight=2.0,
        include_patterns=["**/*.md"],  # All markdown files
        exclude_patterns=["CONTRIBUTING.md"],
        description="Security cheat sheets and best practices",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-wstg": SourceConfig(
        name="OWASP Web Security Testing Guide",
        path="owasp-wstg",
        repo_url="https://github.com/OWASP/wstg.git",
        weight=1.5,
        include_patterns=["**/*.md"],  # All markdown
        exclude_patterns=["CONTRIBUTING.md"],
        description="Web application security testing methodology",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-asvs": SourceConfig(
        name="OWASP Application Security Verification Standard",
        path="owasp-asvs",
        repo_url="https://github.com/OWASP/ASVS.git",
        weight=1.5,
        include_patterns=["**/*.md", "**/*.txt"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="Application security verification standard",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-samm": SourceConfig(
        name="OWASP Software Assurance Maturity Model",
        path="owasp-samm",
        repo_url="https://github.com/OWASP/samm.git",
        weight=1.2,
        include_patterns=["**/*.md", "**/*.yaml"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="Software assurance maturity model",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-mstg": SourceConfig(
        name="OWASP Mobile Security Testing Guide",
        path="owasp-mstg",
        repo_url="https://github.com/OWASP/owasp-mstg.git",
        weight=1.5,
        include_patterns=["**/*.md"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="Mobile application security testing guide",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-mastg": SourceConfig(
        name="OWASP Mobile Application Security",
        path="owasp-mastg",
        repo_url="https://github.com/OWASP/owasp-mastg.git",
        weight=1.5,
        include_patterns=["**/*.md"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="Mobile application security testing guide (new)",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-wrongsecrets": SourceConfig(
        name="OWASP WrongSecrets",
        path="owasp-wrongsecrets",
        repo_url="https://github.com/OWASP/wrongsecrets.git",
        weight=1.0,
        include_patterns=["**/*.md", "**/*.adoc"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="Vulnerable app for secrets management education",
        license_hint="MIT"
    ),
    "owasp-juice-shop": SourceConfig(
        name="OWASP Juice Shop",
        path="owasp-juice-shop",
        repo_url="https://github.com/juice-shop/juice-shop.git",
        weight=1.0,
        include_patterns=["**/*.md"],
        exclude_patterns=["CONTRIBUTING.md", "**/node_modules/**"],
        description="Vulnerable web application for security training",
        license_hint="MIT"
    ),
    "owasp-api-security": SourceConfig(
        name="OWASP API Security",
        path="owasp-api-security",
        repo_url="https://github.com/OWASP/API-Security.git",
        weight=1.5,
        include_patterns=["**/*.md"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="API security top 10 and best practices",
        license_hint="CC-BY-SA-4.0"
    ),
    "owasp-cornucopia": SourceConfig(
        name="OWASP Cornucopia",
        path="owasp-cornucopia",
        repo_url="https://github.com/OWASP/cornucopia.git",
        weight=1.0,
        include_patterns=["**/*.md", "**/*.xml"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="Card game for threat modeling",
        license_hint="CC-BY-SA-3.0"
    ),
    "owasp-secure-coding": SourceConfig(
        name="OWASP Secure Coding Practices",
        path="owasp-secure-coding",
        repo_url="https://github.com/OWASP/secure-coding-practices-quick-reference-guide.git",
        weight=1.2,
        include_patterns=["**/*.md", "**/*.rst"],
        exclude_patterns=["CONTRIBUTING.md"],
        description="Secure coding practices quick reference",
        license_hint="CC-BY-SA-3.0"
    ),
    "mitre-cti": SourceConfig(
        name="MITRE ATT&CK",
        path="mitre-cti",
        repo_url="https://github.com/mitre/cti.git",
        weight=1.0,
        include_patterns=["**/*.json"],
        description="MITRE ATT&CK threat intelligence",
        license_hint="Apache-2.0"
    ),
    
    # RFCs and Protocols - Use direct .txt downloads instead of rfcxml repo
    "rfc-txt": SourceConfig(
        name="RFC Text Documents",
        path="rfc-txt",
        repo_url="__RFC_DOWNLOAD__",  # Special marker for RFC download
        weight=1.5,
        include_patterns=["**/*.txt"],
        description="IETF RFC specifications (text format)",
        license_hint="Public Domain / IETF Trust"
    ),
    
    # Unix/Bash
    "man-pages": SourceConfig(
        name="Linux Man Pages",
        path="man-pages",
        repo_url="https://github.com/mkerrisk/man-pages.git",
        weight=1.2,
        include_patterns=[
            "man*/*.1", "man*/*.2", "man*/*.3", "man*/*.4",
            "man*/*.5", "man*/*.6", "man*/*.7", "man*/*.8",
            "man*/*.1.gz", "man*/*.2.gz", "man*/*.3.gz", "man*/*.4.gz",
            "man*/*.5.gz", "man*/*.6.gz", "man*/*.7.gz", "man*/*.8.gz",
        ],
        description="Linux manual pages",
        license_hint="GPL/BSD/MIT"
    ),
    "bash-docs": SourceConfig(
        name="GNU Bash Documentation",
        path="bash-docs",
        repo_url="https://github.com/bminor/bash.git",
        weight=1.0,
        include_patterns=["doc/*.texi", "doc/*.txt", "*.md"],
        description="Bash shell documentation",
        license_hint="GPL-3.0"
    ),
    
    # Python - ALL documentation
    "python-docs": SourceConfig(
        name="Python Documentation",
        path="python-docs",
        repo_url="https://github.com/python/cpython.git",
        weight=1.5,
        include_patterns=["Doc/**/*.rst", "Doc/**/*.txt"],  # All docs
        exclude_patterns=[],  # Include everything
        description="Official Python documentation (full)",
        license_hint="PSF-2.0"
    ),
    "python-peps": SourceConfig(
        name="Python Enhancement Proposals",
        path="python-peps",
        repo_url="https://github.com/python/peps.git",
        weight=1.0,
        include_patterns=["peps/*.rst", "peps/*.txt"],
        description="Python PEPs",
        license_hint="PSF-2.0"
    ),
    
    # Node.js
    "nodejs-docs": SourceConfig(
        name="Node.js Documentation",
        path="nodejs-docs",
        repo_url="https://github.com/nodejs/node.git",
        weight=1.2,
        include_patterns=["doc/api/*.md", "doc/guides/*.md"],
        description="Node.js API documentation",
        license_hint="MIT"
    ),
    
    # MDN - ALL English content (massive documentation source)
    "mdn-content": SourceConfig(
        name="MDN Web Docs",
        path="mdn-content",
        repo_url="https://github.com/mdn/content.git",
        weight=2.0,
        include_patterns=[
            "files/en-us/**/*.md",  # ALL English MDN content
        ],
        exclude_patterns=["**/games/**"],  # Only exclude games
        description="Mozilla Developer Network documentation (full)",
        license_hint="CC-BY-SA-2.5"
    ),
    
    # Java - ALL java.base classes
    "openjdk": SourceConfig(
        name="OpenJDK Documentation",
        path="openjdk",
        repo_url="https://github.com/openjdk/jdk.git",
        weight=1.0,
        include_patterns=[
            "src/java.base/share/classes/**/*.java",  # All java.base
        ],
        exclude_patterns=["**/test/**", "**/*Test*.java", "**/internal/**"],
        description="OpenJDK source documentation (Javadoc)",
        license_hint="GPL-2.0-classpath"
    ),
    
    # Linux Kernel Documentation - Massive high-quality source
    "linux-docs": SourceConfig(
        name="Linux Kernel Documentation",
        path="linux-docs",
        repo_url="https://github.com/torvalds/linux.git",
        weight=1.5,
        include_patterns=[
            "Documentation/**/*.rst",
            "Documentation/**/*.txt",
        ],
        exclude_patterns=["**/translations/**"],
        description="Linux kernel documentation",
        license_hint="GPL-2.0"
    ),
    
    # Rust Book & Documentation - Modern systems programming
    "rust-book": SourceConfig(
        name="The Rust Programming Language",
        path="rust-book",
        repo_url="https://github.com/rust-lang/book.git",
        weight=1.2,
        include_patterns=["**/*.md"],
        exclude_patterns=["**/redirects/**"],
        description="The Rust Programming Language book",
        license_hint="MIT/Apache-2.0"
    ),
    
    # Go Documentation
    "go-docs": SourceConfig(
        name="Go Documentation",
        path="go-docs",
        repo_url="https://github.com/golang/go.git",
        weight=1.0,
        include_patterns=["doc/**/*.html", "src/**/*.go"],
        exclude_patterns=["**/test/**", "**/testdata/**"],
        description="Go language documentation and source",
        license_hint="BSD-3-Clause"
    ),
    
    # TypeScript Documentation
    "typescript-docs": SourceConfig(
        name="TypeScript Documentation",
        path="typescript-docs",
        repo_url="https://github.com/microsoft/TypeScript.git",
        weight=1.0,
        include_patterns=["doc/**/*.md", "src/**/*.ts"],
        exclude_patterns=["**/test/**", "**/tests/**"],
        description="TypeScript documentation and source",
        license_hint="Apache-2.0"
    ),
}


# ---------------------------------------------------------------------------
# Git Repository Fetching
# ---------------------------------------------------------------------------

def clone_repository(
    name: str, 
    url: str, 
    target_dir: Path, 
    logger: logging.Logger,
    depth: int = 1
) -> Tuple[bool, float, str]:
    """
    Clone a git repository.
    
    Returns:
        (success, elapsed_time, message)
    """
    start_time = time.time()
    target_path = target_dir / name
    
    if target_path.exists():
        elapsed = time.time() - start_time
        logger.info(f"Repository '{name}' already exists, skipping")
        return True, elapsed, "already_exists"
    
    logger.info(f"Cloning {name} from {url}")
    
    try:
        cmd = ["git", "clone", "--depth", str(depth), "--quiet", url, str(target_path)]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600,
            encoding='utf-8',
            errors='replace'
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Calculate directory size
            total_size = sum(
                f.stat().st_size 
                for f in target_path.rglob('*') 
                if f.is_file()
            )
            logger.info(f"Cloned {name} ({format_size(total_size)}) in {elapsed:.1f}s")
            return True, elapsed, "cloned"
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            logger.error(f"Failed to clone {name}: {error_msg}")
            return False, elapsed, f"error: {error_msg}"
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"Clone timeout for {name} after {elapsed:.1f}s")
        return False, elapsed, "timeout"
    except FileNotFoundError:
        elapsed = time.time() - start_time
        logger.error(f"Git not found - please install git")
        return False, elapsed, "git_not_found"
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Clone error for {name}: {e}")
        return False, elapsed, f"exception: {e}"


def download_rfc_documents(
    target_dir: Path,
    logger: logging.Logger,
) -> Tuple[bool, float, str]:
    """
    Download RFC documents from rfc-editor.org.
    
    Returns:
        (success, elapsed_time, message)
    """
    start_time = time.time()
    rfc_dir = target_dir / "rfc-txt"
    
    if rfc_dir.exists() and len(list(rfc_dir.glob("rfc*.txt"))) > 10:
        # Already have RFCs
        existing = len(list(rfc_dir.glob("rfc*.txt")))
        elapsed = time.time() - start_time
        logger.info(f"RFC directory already has {existing} files")
        return True, elapsed, f"exists_{existing}_files"
    
    logger.info("Downloading RFC documents from rfc-editor.org...")
    
    try:
        downloader = RFCDownloader(str(rfc_dir), delay=0.3)
        
        def progress(current, total, rfc_num, status):
            if current % 10 == 0 or current == total:
                logger.info(f"  RFC download progress: {current}/{total}")
        
        results = downloader.download_key_rfcs(progress_callback=progress)
        
        successful = sum(1 for _, path, _ in results if path is not None)
        stats = downloader.get_stats()
        
        elapsed = time.time() - start_time
        logger.info(f"Downloaded {successful}/{len(results)} RFCs ({stats['total_size_mb']:.1f} MB)")
        
        return True, elapsed, f"downloaded_{successful}"
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"RFC download error: {e}")
        return False, elapsed, f"error: {e}"


def fetch_all_sources(
    sources: Dict[str, SourceConfig],
    target_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Tuple[bool, str]]:
    """
    Fetch all source repositories.
    
    Returns:
        Dict mapping source name to (success, status_message)
    """
    results = {}
    target_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(sources)
    for i, (name, config) in enumerate(sources.items(), 1):
        if not config.repo_url:
            logger.warning(f"[{i}/{total}] No repo URL for {name}, skipping")
            results[name] = (False, "no_url")
            continue
        
        logger.info(f"[{i}/{total}] Fetching {config.name}...")
        
        # Special case: RFC download
        if config.repo_url == "__RFC_DOWNLOAD__":
            success, elapsed, msg = download_rfc_documents(target_dir, logger)
            results[name] = (success, msg)
            continue
        
        success, elapsed, msg = clone_repository(
            name=config.path,
            url=config.repo_url,
            target_dir=target_dir,
            logger=logger,
        )
        results[name] = (success, msg)
    
    return results


# ---------------------------------------------------------------------------
# Simhash implementation for near-duplicate detection
# ---------------------------------------------------------------------------

def tokenize_for_simhash(text: str, n: int = 5) -> List[str]:
    """Tokenize text into n-grams for simhash."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    
    if len(words) < n:
        return [' '.join(words)]
    
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def compute_simhash(text: str, bits: int = 64) -> int:
    """Compute simhash fingerprint for text."""
    ngrams = tokenize_for_simhash(text)
    
    if not ngrams:
        return 0
    
    v = [0] * bits
    
    for ngram in ngrams:
        h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        for i in range(bits):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Compute Hamming distance between two integers."""
    return bin(a ^ b).count('1')


class SimhashIndex:
    """Index for finding near-duplicates using simhash."""
    
    def __init__(self, threshold: int = 4):
        self.threshold = threshold
        self.fingerprints: List[Tuple[int, str]] = []
    
    def add(self, fingerprint: int, doc_id: str) -> bool:
        """Add a fingerprint. Returns False if near-duplicate exists."""
        for existing_fp, existing_id in self.fingerprints:
            if hamming_distance(fingerprint, existing_fp) <= self.threshold:
                return False
        
        self.fingerprints.append((fingerprint, doc_id))
        return True
    
    def is_duplicate(self, fingerprint: int) -> Optional[str]:
        """Check if fingerprint is a near-duplicate."""
        for existing_fp, existing_id in self.fingerprints:
            if hamming_distance(fingerprint, existing_fp) <= self.threshold:
                return existing_id
        return None


# ---------------------------------------------------------------------------
# Text cleaning and normalization
# ---------------------------------------------------------------------------

class TextCleaner:
    """Clean and normalize text documents."""
    
    BOILERPLATE_PATTERNS = [
        r'Edit this page on GitHub.*',
        r'Edit on GitHub.*',
        r'View on GitHub.*',
        r'Previous\s*[|/]\s*Next',
        r'← Previous.*',
        r'Next →.*',
        r'Table of Contents',
        r'Skip to (main )?content.*',
        r'Skip to navigation.*',
        r'© \d{4}.*',
        r'Copyright \d{4}.*',
        r'All rights reserved.*',
        r'Terms of Service.*',
        r'Privacy Policy.*',
        r'Cookie.*preferences.*',
        r'Subscribe to.*newsletter.*',
        r'Follow us on.*',
        r'Share this.*',
        r'Was this page helpful.*',
        r'Report a bug.*',
        r'Feedback.*',
        r'Last updated:.*',
        r'Last modified:.*',
    ]
    
    def __init__(self, min_chars: int = 400):
        self.min_chars = min_chars
        self.boilerplate_re = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) 
            for p in self.BOILERPLATE_PATTERNS
        ]
    
    def clean(self, text: str) -> Optional[str]:
        """Clean and normalize text. Returns None if too short."""
        # Remove boilerplate
        for pattern in self.boilerplate_re:
            text = pattern.sub('', text)
        
        # Normalize Unicode
        text = self._normalize_unicode(text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Normalize whitespace
        lines = []
        for line in text.split('\n'):
            if line.startswith('  ') or line.startswith('\t'):
                parts = line.split()
                if parts:
                    indent = len(line) - len(line.lstrip())
                    lines.append(' ' * min(indent, 8) + ' '.join(parts))
                else:
                    lines.append('')
            else:
                lines.append(' '.join(line.split()))
        text = '\n'.join(lines)
        
        # Collapse blank lines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = text.strip()
        
        if len(text) < self.min_chars:
            return None
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        replacements = {
            '\u00a0': ' ', '\u2018': "'", '\u2019': "'",
            '\u201c': '"', '\u201d': '"', '\u2013': '-',
            '\u2014': '--', '\u2026': '...', '\u00ad': '',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text


# ---------------------------------------------------------------------------
# File scanner and converter
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """Process documents from various source formats."""
    
    EXTENSION_MAP = {
        '.md': 'markdown', '.markdown': 'markdown', '.rst': 'rst',
        '.1': 'man', '.2': 'man', '.3': 'man', '.4': 'man',
        '.5': 'man', '.6': 'man', '.7': 'man', '.8': 'man',
        '.1p': 'man', '.3p': 'man', '.gz': 'man_gz',
        '.html': 'html', '.htm': 'html',
        '.xml': 'xml', '.txt': 'text', '.text': 'text',
        '.json': 'json',
        '.java': 'java',  # Javadoc extraction
        '.texi': 'texinfo', '.texinfo': 'texinfo', '.txi': 'texinfo',  # GNU Texinfo
        '.go': 'go',  # Go source with doc comments
        '.ts': 'typescript', '.tsx': 'typescript',  # TypeScript with JSDoc
        '.adoc': 'asciidoc',  # AsciiDoc format
        '.yaml': 'yaml', '.yml': 'yaml',  # YAML (extract readable content)
    }
    
    def __init__(self):
        self.man_converter = ManPageConverter()
        self.rfc_converter = RFCXMLConverter()
        self.html_converter = HTMLConverter()
        self.md_converter = MarkdownRSTConverter()
        self.java_converter = JavaDocConverter()
        self.texinfo_converter = TexinfoConverter()
        self.code_converter = CodeDocExtractor()
    
    def get_file_type(self, path: str) -> Optional[str]:
        """Determine file type from path."""
        p = Path(path)
        ext = p.suffix.lower()
        
        if ext == '.gz':
            inner_ext = Path(p.stem).suffix.lower()
            if inner_ext in ('.1', '.2', '.3', '.5', '.7', '.8'):
                return 'man'
        
        return self.EXTENSION_MAP.get(ext)
    
    def convert(self, path: str) -> Tuple[Optional[str], str]:
        """Convert a file to plaintext."""
        file_type = self.get_file_type(path)
        
        if file_type is None:
            return None, "unsupported_type"
        
        try:
            if file_type == 'man':
                return self.man_converter.convert(path)
            elif file_type == 'xml':
                if self.rfc_converter.is_rfc_xml(path):
                    return self.rfc_converter.convert(path)
                else:
                    return None, "non_rfc_xml"
            elif file_type == 'html':
                return self.html_converter.convert(path)
            elif file_type in ('markdown', 'rst'):
                return self.md_converter.convert(path)
            elif file_type == 'text':
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read(), "plaintext"
            elif file_type == 'json':
                return self._extract_mitre_json(path)
            elif file_type == 'java':
                return self.java_converter.convert(path)
            elif file_type == 'texinfo':
                return self.texinfo_converter.convert(path)
            elif file_type in ('go', 'typescript'):
                return self.code_converter.convert(path)
            elif file_type == 'asciidoc':
                # AsciiDoc - treat similar to markdown
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read(), "asciidoc"
            elif file_type == 'yaml':
                # YAML - extract text content
                return self._extract_yaml_text(path)
            else:
                return None, f"unhandled_type:{file_type}"
        except Exception as e:
            return None, f"error:{e}"
    
    def _extract_mitre_json(self, path: str) -> Tuple[Optional[str], str]:
        """Extract text from MITRE ATT&CK JSON files."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            texts = []
            if 'objects' in data:
                for obj in data['objects']:
                    obj_type = obj.get('type', '')
                    if obj_type in ('attack-pattern', 'malware', 'tool', 
                                   'course-of-action', 'intrusion-set'):
                        name = obj.get('name', '')
                        desc = obj.get('description', '')
                        if name and desc:
                            texts.append(f"# {name}\n\n{desc}")
            
            if texts:
                return '\n\n---\n\n'.join(texts), "mitre_json"
            return None, "empty_json"
        except json.JSONDecodeError:
            return None, "invalid_json"
    
    def _extract_yaml_text(self, path: str) -> Tuple[Optional[str], str]:
        """Extract readable text content from YAML files."""
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # For YAML, we extract string values that look like documentation
            texts = []
            for line in content.split('\n'):
                # Look for description/text fields
                if ':' in line:
                    key, _, value = line.partition(':')
                    value = value.strip().strip('"').strip("'")
                    # Only extract meaningful text fields
                    key_lower = key.strip().lower()
                    if key_lower in ('description', 'text', 'content', 'summary', 
                                    'title', 'name', 'explanation', 'note', 'details'):
                        if len(value) > 20:
                            texts.append(value)
            
            if texts:
                return '\n\n'.join(texts), "yaml_text"
            return None, "empty_yaml"
        except Exception:
            return None, "yaml_error"


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


# ---------------------------------------------------------------------------
# Main corpus builder
# ---------------------------------------------------------------------------

class CorpusBuilder:
    """Build domain corpus from source repositories."""
    
    def __init__(
        self,
        sources_dir: str,
        work_dir: str,
        out_dir: str,
        logger: logging.Logger,
        min_chars: int = 400,
        shard_mb: int = 25,
        dedupe_methods: List[str] = None,
        simhash_threshold: int = 4,
        include_ext: Set[str] = None,
        max_docs_per_repo: int = 0,
        seed: int = 42,
    ):
        self.sources_dir = Path(sources_dir)
        self.work_dir = Path(work_dir)
        self.out_dir = Path(out_dir)
        self.logger = logger
        self.min_chars = min_chars
        self.shard_mb = shard_mb
        self.dedupe_methods = dedupe_methods or ['exact']
        self.simhash_threshold = simhash_threshold
        self.include_ext = include_ext or {'.md', '.rst', '.txt', '.html', '.xml'}
        self.max_docs_per_repo = max_docs_per_repo
        self.seed = seed
        
        # Set random seed for determinism
        random.seed(seed)
        
        # Initialize components
        self.processor = DocumentProcessor()
        self.cleaner = TextCleaner(min_chars=min_chars)
        
        # Deduplication state
        self.seen_hashes: Set[str] = set()
        self.simhash_index = SimhashIndex(threshold=simhash_threshold)
        
        # Statistics
        self.stats = defaultdict(int)
        self.source_stats: Dict[str, Dict[str, int]] = {}
        
        # Create directories
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'corpus_shards').mkdir(exist_ok=True)
    
    def scan_source(self, source_config: SourceConfig) -> Iterator[Path]:
        """Scan a source directory for matching files."""
        source_path = self.sources_dir / source_config.path
        
        if not source_path.exists():
            self.logger.warning(f"Source not found: {source_path}")
            return
        
        include_patterns = source_config.include_patterns or ['**/*']
        exclude_patterns = source_config.exclude_patterns or []
        
        self.logger.info(f"Scanning {source_path} with patterns: {include_patterns}")
        
        # Man page extensions (sections 1-8)
        man_extensions = {'.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.1p', '.3p'}
        
        files_found = 0
        files_filtered_ext = 0
        files_excluded = 0
        files_yielded = 0
        
        for pattern in include_patterns:
            pattern_matches = list(source_path.glob(pattern))
            self.logger.debug(f"  Pattern '{pattern}' matched {len(pattern_matches)} items")
            
            for path in pattern_matches:
                if not path.is_file():
                    continue
                
                files_found += 1
                
                ext = path.suffix.lower()
                original_ext = ext
                if ext == '.gz':
                    ext = Path(path.stem).suffix.lower()
                
                if ext not in self.include_ext and ext not in man_extensions:
                    files_filtered_ext += 1
                    if files_filtered_ext <= 3:
                        self.logger.debug(f"    Filtered by ext: {path.name} (ext={original_ext}, inner={ext})")
                    continue
                
                rel_path = str(path.relative_to(source_path))
                excluded = any(Path(rel_path).match(exc) for exc in exclude_patterns)
                
                if excluded:
                    files_excluded += 1
                    continue
                
                files_yielded += 1
                yield path
        
        self.logger.info(f"  Scan summary: found={files_found}, filtered_ext={files_filtered_ext}, excluded={files_excluded}, yielded={files_yielded}")
    
    def process_document(self, path: Path, source_name: str) -> Optional[Dict]:
        """Process a single document."""
        content, status = self.processor.convert(str(path))
        
        if content is None:
            self.stats['convert_failed'] += 1
            return None
        
        cleaned = self.cleaner.clean(content)
        
        if cleaned is None:
            self.stats['too_short'] += 1
            return None
        
        content_hash = hashlib.sha256(cleaned.encode()).hexdigest()
        
        # Exact deduplication
        if 'exact' in self.dedupe_methods:
            if content_hash in self.seen_hashes:
                self.stats['exact_dupe'] += 1
                return None
            self.seen_hashes.add(content_hash)
        
        # Simhash deduplication
        if 'simhash' in self.dedupe_methods:
            fingerprint = compute_simhash(cleaned)
            if not self.simhash_index.add(fingerprint, content_hash):
                self.stats['near_dupe'] += 1
                return None
        
        self.stats['accepted'] += 1
        
        return {
            'content': cleaned,
            'source': source_name,
            'path': str(path.relative_to(self.sources_dir)),
            'sha256': content_hash,
            'bytes': len(cleaned.encode('utf-8')),
            'chars': len(cleaned),
            'status': status,
        }
    
    def build_corpus(self, source_configs: Dict[str, SourceConfig]) -> Tuple[List[Dict], List[Dict]]:
        """Build the corpus from all configured sources."""
        self.logger.info("=" * 60)
        self.logger.info("Building Phase 2 Domain Corpus")
        self.logger.info("=" * 60)
        self.logger.info(f"Sources directory: {self.sources_dir}")
        self.logger.info(f"Output directory:  {self.out_dir}")
        self.logger.info(f"Deduplication:     {', '.join(self.dedupe_methods)}")
        self.logger.info(f"Min chars:         {self.min_chars}")
        self.logger.info(f"Shard size:        {self.shard_mb} MB")
        self.logger.info(f"Random seed:       {self.seed}")
        self.logger.info(f"Include ext:       {self.include_ext}")
        
        start_time = time.time()
        
        all_docs: List[Dict] = []
        manifest_entries: List[Dict] = []
        
        for source_key, source_config in source_configs.items():
            self.logger.info("-" * 60)
            self.logger.info(f"Processing: {source_config.name}")
            
            source_start = time.time()
            source_docs = 0
            source_bytes = 0
            files_scanned = 0
            
            for path in self.scan_source(source_config):
                files_scanned += 1
                
                if self.max_docs_per_repo > 0 and source_docs >= self.max_docs_per_repo:
                    break
                
                doc = self.process_document(path, source_key)
                
                if doc:
                    all_docs.append(doc)
                    manifest_entries.append({
                        'source': doc['source'],
                        'path': doc['path'],
                        'sha256': doc['sha256'],
                        'bytes': doc['bytes'],
                        'license_hint': source_config.license_hint or 'see_source',
                    })
                    source_docs += 1
                    source_bytes += doc['bytes']
                
                if files_scanned % 500 == 0:
                    self.logger.debug(f"  Scanned {files_scanned} files, accepted {source_docs}")
            
            source_time = time.time() - source_start
            self.source_stats[source_key] = {
                'files_scanned': files_scanned,
                'docs_accepted': source_docs,
                'bytes': source_bytes,
                'time': source_time,
            }
            
            self.logger.info(
                f"  {source_config.name}: {source_docs:,} docs, "
                f"{format_size(source_bytes)}, {format_duration(source_time)}"
            )
        
        # Write shards
        self.logger.info("-" * 60)
        self.logger.info("Writing corpus shards")
        self._write_shards(all_docs)
        
        # Write manifest
        manifest_path = self.out_dir / 'manifest.jsonl'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + '\n')
        self.logger.info(f"Manifest written: {manifest_path}")
        
        # Write build metadata
        self._write_build_metadata(source_configs, all_docs)
        
        # Write dedupe report
        self._write_dedupe_report()
        
        total_time = time.time() - start_time
        self._log_summary(total_time, all_docs)
        
        return all_docs, manifest_entries
    
    def _write_shards(self, docs: List[Dict]) -> None:
        """Write documents to sharded text files."""
        shard_bytes = self.shard_mb * 1024 * 1024
        shard_dir = self.out_dir / 'corpus_shards'
        current_shard = []
        current_size = 0
        shard_num = 0
        
        for doc in docs:
            formatted = (
                "==== DOC START ====\n"
                f"SOURCE: {doc['source']} | PATH: {doc['path']}\n"
                "====\n"
                f"{doc['content']}\n"
                "==== DOC END ====\n\n"
            )
            
            doc_bytes = len(formatted.encode('utf-8'))
            
            if current_size + doc_bytes > shard_bytes and current_shard:
                self._write_shard_file(shard_dir, shard_num, current_shard)
                shard_num += 1
                current_shard = []
                current_size = 0
            
            current_shard.append(formatted)
            current_size += doc_bytes
        
        if current_shard:
            self._write_shard_file(shard_dir, shard_num, current_shard)
            shard_num += 1
        
        self.stats['total_shards'] = shard_num
        self.logger.info(f"Written {shard_num} shards to {shard_dir}")
    
    def _write_shard_file(self, shard_dir: Path, shard_num: int, docs: List[str]) -> None:
        """Write a single shard file."""
        shard_path = shard_dir / f"shard_{shard_num:04d}.txt"
        content = ''.join(docs)
        
        with open(shard_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        size = len(content.encode('utf-8'))
        self.logger.debug(f"  Shard {shard_num:04d}: {len(docs):,} docs, {format_size(size)}")
    
    def _write_build_metadata(self, source_configs: Dict[str, SourceConfig], docs: List[Dict]) -> None:
        """Write build metadata for reproducibility."""
        metadata_path = self.out_dir / 'build_metadata.json'
        
        metadata = {
            'build_timestamp': datetime.now().isoformat(),
            'seed': self.seed,
            'min_chars': self.min_chars,
            'shard_mb': self.shard_mb,
            'dedupe_methods': self.dedupe_methods,
            'simhash_threshold': self.simhash_threshold,
            'include_extensions': sorted(list(self.include_ext)),
            'max_docs_per_repo': self.max_docs_per_repo,
            'sources': {
                name: {
                    'path': cfg.path,
                    'weight': cfg.weight,
                    'include_patterns': cfg.include_patterns,
                    'exclude_patterns': cfg.exclude_patterns,
                    'license_hint': cfg.license_hint,
                }
                for name, cfg in source_configs.items()
            },
            'statistics': {
                'total_documents': len(docs),
                'total_bytes': sum(d['bytes'] for d in docs),
                'total_shards': self.stats.get('total_shards', 0),
                'accepted': self.stats.get('accepted', 0),
                'too_short': self.stats.get('too_short', 0),
                'convert_failed': self.stats.get('convert_failed', 0),
                'exact_dupes': self.stats.get('exact_dupe', 0),
                'near_dupes': self.stats.get('near_dupe', 0),
            },
            'per_source': self.source_stats,
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Build metadata: {metadata_path}")
    
    def _write_dedupe_report(self) -> None:
        """Write deduplication report."""
        report_path = self.work_dir / 'dedupe_report.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'methods': self.dedupe_methods,
            'simhash_threshold': self.simhash_threshold,
            'stats': {
                'exact_hashes_seen': len(self.seen_hashes),
                'simhash_fingerprints': len(self.simhash_index.fingerprints),
                'exact_dupes_found': self.stats.get('exact_dupe', 0),
                'near_dupes_found': self.stats.get('near_dupe', 0),
            }
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Dedupe report: {report_path}")
    
    def _log_summary(self, total_time: float, docs: List[Dict]) -> None:
        """Log build summary."""
        total_bytes = sum(d['bytes'] for d in docs)
        
        self.logger.info("=" * 60)
        self.logger.info("BUILD SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total documents:     {len(docs):,}")
        self.logger.info(f"Total size:          {format_size(total_bytes)}")
        self.logger.info(f"Total shards:        {self.stats.get('total_shards', 0)}")
        self.logger.info(f"Build time:          {format_duration(total_time)}")
        self.logger.info("")
        self.logger.info("Processing stats:")
        self.logger.info(f"  Accepted:          {self.stats.get('accepted', 0):,}")
        self.logger.info(f"  Too short:         {self.stats.get('too_short', 0):,}")
        self.logger.info(f"  Convert failed:    {self.stats.get('convert_failed', 0):,}")
        self.logger.info(f"  Exact duplicates:  {self.stats.get('exact_dupe', 0):,}")
        self.logger.info(f"  Near duplicates:   {self.stats.get('near_dupe', 0):,}")


def run_prepare_weighted_dataset(
    corpus_dir: Path,
    out_dir: Path,
    total_tokens: int,
    logger: logging.Logger,
) -> bool:
    """Run prepare_weighted_dataset.py on the built corpus."""
    logger.info("=" * 60)
    logger.info("Running Tokenizer Pipeline")
    logger.info("=" * 60)
    
    shard_dir = corpus_dir / 'corpus_shards'
    shards = sorted(shard_dir.glob('shard_*.txt'))
    
    if not shards:
        logger.error("No shard files found!")
        return False
    
    logger.info(f"Found {len(shards)} shard files")
    
    shard_paths = ','.join(str(s) for s in shards)
    
    script_path = Path(__file__).parent.parent / 'scripts' / 'prepare_weighted_dataset.py'
    
    if not script_path.exists():
        logger.error(f"Cannot find prepare_weighted_dataset.py at {script_path}")
        return False
    
    cmd = [
        sys.executable,
        str(script_path),
        f'--source=domain_corpus:{shard_paths}',
        '--weight=domain_corpus:1.0',
        f'--total_tokens={total_tokens}',
        f'--out_dir={out_dir}',
        '--tokenization=gpt2',
        '--tokens_per_shard=10000000',
        '--val_fraction=0.05',
        '--no_normalize',
        '--no_filter',
    ]
    
    logger.info(f"Running: {script_path.name}")
    
    try:
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            logger.info("Tokenization complete!")
            return True
        else:
            logger.error(f"Tokenization failed with code {result.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Tokenization failed: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Phase 2 Domain Corpus for MyPT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument("--sources_dir", type=str, default="./sources",
                        help="Directory containing cloned source repositories")
    parser.add_argument("--work_dir", type=str, default="./work",
                        help="Working directory for intermediate files")
    parser.add_argument("--out_dir", type=str, default="./out",
                        help="Output directory for corpus")
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch/clone source repositories before building")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Optional JSON config file for source definitions")
    parser.add_argument("--min_chars", type=int, default=400,
                        help="Minimum characters per document (default: 400)")
    parser.add_argument("--shard_mb", type=int, default=25,
                        help="Target shard size in MB (default: 25)")
    parser.add_argument("--dedupe", type=str, default="exact",
                        help="Deduplication methods: exact,simhash (default: exact)")
    parser.add_argument("--simhash_threshold", type=int, default=4,
                        help="Simhash Hamming distance threshold (default: 4)")
    parser.add_argument("--include_ext", type=str, default=".md,.rst,.txt,.html,.xml,.json,.java,.texi,.go,.ts,.adoc,.yaml,.yml",
                        help="File extensions to process")
    parser.add_argument("--max_docs_per_repo", type=int, default=0,
                        help="Max documents per repo, 0 = unlimited (default: 0)")
    parser.add_argument("--run_tokenizer", action="store_true",
                        help="Run prepare_weighted_dataset.py after building corpus")
    parser.add_argument("--total_tokens", type=int, default=100_000_000,
                        help="Total tokens for tokenizer (default: 100000000)")
    parser.add_argument("--tokenized_out_dir", type=str, default=None,
                        help="Output directory for tokenized dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="Which sources to process (default: all available)")
    
    args = parser.parse_args()
    
    # Print banner
    print_banner("MyPT Phase 2", "Domain Corpus Builder")
    
    print(f"\n  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Random seed: {args.seed}\n")
    
    # Setup logging
    work_dir = Path(args.work_dir)
    logger = setup_logging(work_dir / 'logs')
    logger.info(f"Build started at {datetime.now().isoformat()}")
    logger.info(f"Random seed: {args.seed}")
    
    # Audit: Build started
    if AUDIT_AVAILABLE:
        audit.training(
            "corpus_build_start",
            corpus_type="phase2_domain",
            sources_dir=args.sources_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            details=f"Phase 2 domain corpus build started"
        )
    
    # Parse extensions
    include_ext = set(ext.strip() for ext in args.include_ext.split(','))
    
    # Parse dedupe methods
    dedupe_methods = [m.strip() for m in args.dedupe.split(',')]
    
    # Load or use default source configs
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config_data = json.load(f)
        source_configs = {
            k: SourceConfig(**v) for k, v in config_data.get('sources', {}).items()
        }
    else:
        source_configs = DEFAULT_REPOS
    
    # Filter sources if specified
    if args.sources:
        source_configs = {k: v for k, v in source_configs.items() if k in args.sources}
        if not source_configs:
            logger.error(f"No matching sources found!")
            logger.error(f"Available: {list(DEFAULT_REPOS.keys())}")
            sys.exit(1)
    
    # Fetch sources if requested
    sources_dir = Path(args.sources_dir)
    if args.fetch:
        print("\n" + "=" * 60)
        print("  STEP 1: Fetching Source Repositories")
        print("=" * 60)
        
        fetch_results = fetch_all_sources(source_configs, sources_dir, logger)
        
        success_count = sum(1 for s, _ in fetch_results.values() if s)
        logger.info(f"Fetch complete: {success_count}/{len(fetch_results)} successful")
        
        if AUDIT_AVAILABLE:
            audit.training(
                "corpus_fetch_complete",
                corpus_type="phase2_domain",
                success_count=success_count,
                total_count=len(fetch_results),
            )
    
    # Filter to only sources that exist
    available_sources = {}
    for key, config in source_configs.items():
        source_path = sources_dir / config.path
        if source_path.exists():
            available_sources[key] = config
        else:
            logger.warning(f"Source not found: {key} ({source_path})")
    
    if not available_sources:
        logger.error("No source directories found!")
        logger.error(f"Run with --fetch to clone repositories, or check --sources_dir")
        
        if AUDIT_AVAILABLE:
            audit.training(
                "corpus_build_error",
                level=audit.AuditLevel.ERROR,
                corpus_type="phase2_domain",
                error="no_sources",
                details="No source directories found"
            )
        sys.exit(1)
    
    print(f"\n  Sources to process: {len(available_sources)}")
    for key in available_sources:
        print(f"    - {key}")
    
    # Build corpus
    builder = CorpusBuilder(
        sources_dir=args.sources_dir,
        work_dir=args.work_dir,
        out_dir=args.out_dir,
        logger=logger,
        min_chars=args.min_chars,
        shard_mb=args.shard_mb,
        dedupe_methods=dedupe_methods,
        simhash_threshold=args.simhash_threshold,
        include_ext=include_ext,
        max_docs_per_repo=args.max_docs_per_repo,
        seed=args.seed,
    )
    
    docs, manifest = builder.build_corpus(available_sources)
    
    # Audit: Corpus built
    if AUDIT_AVAILABLE:
        audit.training(
            "corpus_build_complete",
            corpus_type="phase2_domain",
            total_docs=len(docs),
            total_bytes=sum(d['bytes'] for d in docs),
            total_shards=builder.stats.get('total_shards', 0),
        )
    
    # Optionally run tokenizer
    if args.run_tokenizer:
        tokenized_dir = args.tokenized_out_dir or (args.out_dir + '_tokenized')
        
        success = run_prepare_weighted_dataset(
            corpus_dir=Path(args.out_dir),
            out_dir=Path(tokenized_dir),
            total_tokens=args.total_tokens,
            logger=logger,
        )
        
        if success:
            print("\n" + "=" * 60)
            print("  SUCCESS! Dataset ready for training.")
            print("=" * 60)
            print(f"\n  Corpus shards:    {args.out_dir}/corpus_shards/")
            print(f"  Tokenized data:   {tokenized_dir}/")
            print(f"\n  Next steps:")
            print(f"    python train.py --dataset_dir {tokenized_dir} --model_name phase2_domain")
            
            if AUDIT_AVAILABLE:
                audit.training(
                    "corpus_tokenize_complete",
                    corpus_type="phase2_domain",
                    tokenized_dir=tokenized_dir,
                )
        else:
            if AUDIT_AVAILABLE:
                audit.training(
                    "corpus_tokenize_error",
                    level=audit.AuditLevel.ERROR,
                    corpus_type="phase2_domain",
                    error="tokenization_failed",
                )
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("  Corpus build complete!")
        print("=" * 60)
        print(f"\n  To tokenize and prepare for training:")
        print(f"    python scripts/prepare_weighted_dataset.py \\")
        print(f"      --source domain:{args.out_dir}/corpus_shards/*.txt \\")
        print(f"      --total_tokens {args.total_tokens} \\")
        print(f"      --out_dir {args.out_dir}_tokenized")
    
    logger.info(f"Build completed at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
