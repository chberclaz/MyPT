#!/usr/bin/env python
"""
Download and Format Retrieval QA Data for Pre-Training

Downloads reading comprehension datasets from HuggingFace and formats
them as pre-training text with the pattern:

    [passage text]

    Question: [question]
    Answer: [answer grounded in passage]

    -- or for unanswerable questions (SQuAD v2): --

    [passage text]

    Question: [question]
    Answer: The passage does not contain enough information to answer this question.

This creates a strong retrieval head training signal: every example forces
the model to learn that answers come FROM the preceding passage, AND that
sometimes the answer is NOT in the passage (critical for RAG).

Sources:
1. TriviaQA (mandarjoshi/trivia_qa) -- PRIMARY
   - 138K trivia questions + 650K evidence documents
   - Multiple evidence passages per question for context diversity
   - License: Apache 2.0

2. SQuAD v2 (rajpurkar/squad_v2) -- SUPPLEMENT
   - 130K extractive QA examples from Wikipedia paragraphs
   - ~20% are unanswerable: teaches "answer is NOT in context"
   - Precise answer spans with character offsets
   - License: CC BY-SA 4.0

3. Natural Questions (google-research-datasets/natural_questions) -- DISABLED
   - Disabled due to 42GB download size (full HTML Wikipedia pages)
   - TriviaQA + SQuAD v2 provide equivalent retrieval signal

Target: ~420M tokens (7% of 6B mix)

Usage:
    # Download TriviaQA + SQuAD v2 (default)
    python scripts/unified_build/download_nq_triviaqa.py --output_dir data/unified_clean

    # Download only TriviaQA
    python scripts/unified_build/download_nq_triviaqa.py --output_dir data/unified_clean --sources triviaqa

    # Download only SQuAD v2
    python scripts/unified_build/download_nq_triviaqa.py --output_dir data/unified_clean --sources squad_v2
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4.0
DOC_DELIMITER = "\n\n"
SHARD_MB = 25.0

# Minimum lengths for quality filtering
MIN_PASSAGE_CHARS = 200
MIN_ANSWER_CHARS = 10
MAX_PASSAGE_TOKENS = 2048  # Truncate very long passages


# ---------------------------------------------------------------------------
# Shard Writer (reused pattern from download_unified_sources.py)
# ---------------------------------------------------------------------------

class ShardWriter:
    """Write documents into fixed-size text shards."""

    def __init__(self, output_dir: str, shard_mb: float = SHARD_MB):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = int(shard_mb * 1024 * 1024)

        self.shard_idx = 0
        self.current_bytes = 0
        self.current_file = None
        self.total_docs = 0
        self.total_bytes = 0
        self._open_new_shard()

    def _open_new_shard(self):
        if self.current_file:
            self.current_file.close()
        path = self.output_dir / f"shard_{self.shard_idx:04d}.txt"
        self.current_file = open(path, "w", encoding="utf-8")
        self.current_bytes = 0

    def write_document(self, content: str):
        doc = content + DOC_DELIMITER
        doc_bytes = len(doc.encode("utf-8"))

        if self.current_bytes + doc_bytes > self.shard_bytes and self.current_bytes > 0:
            self.shard_idx += 1
            self._open_new_shard()

        self.current_file.write(doc)
        self.current_bytes += doc_bytes
        self.total_bytes += doc_bytes
        self.total_docs += 1

    def close(self):
        if self.current_file:
            self.current_file.close()
            self.current_file = None

    def stats(self) -> dict:
        return {
            "total_docs": self.total_docs,
            "total_shards": self.shard_idx + 1,
            "total_bytes": self.total_bytes,
            "total_mb": self.total_bytes / (1024 * 1024),
        }


class ExactDeduplicator:
    """Deduplicate by SHA-256 hash of passage content."""

    def __init__(self):
        self.seen_hashes = set()
        self.duplicates_found = 0

    def is_duplicate(self, content: str) -> bool:
        h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        if h in self.seen_hashes:
            self.duplicates_found += 1
            return True
        self.seen_hashes.add(h)
        return False


# ---------------------------------------------------------------------------
# Text Formatting
# ---------------------------------------------------------------------------

def clean_passage(text: str) -> Optional[str]:
    """Clean a passage / evidence document."""
    if not text or not text.strip():
        return None

    text = text.strip()

    # Remove HTML tags if any remain
    text = re.sub(r"<[^>]+>", " ", text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if len(text) < MIN_PASSAGE_CHARS:
        return None

    # Truncate very long passages to keep within context window
    max_chars = int(MAX_PASSAGE_TOKENS * CHARS_PER_TOKEN)
    if len(text) > max_chars:
        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.7:
            truncated = truncated[: last_period + 1]
        text = truncated

    return text


def clean_answer(text: str) -> Optional[str]:
    """Clean an answer string."""
    if not text or not text.strip():
        return None

    text = text.strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < MIN_ANSWER_CHARS:
        return None

    return text


def format_reading_comprehension(passage: str, question: str, answer: str) -> str:
    """
    Format a single reading comprehension example for pre-training.

    The format teaches the model:
    1. Read the passage (builds context representation)
    2. See a question about it (activates retrieval)
    3. Generate an answer grounded in the passage (reinforces retrieval heads)
    """
    return f"{passage}\n\nQuestion: {question}\nAnswer: {answer}"


# ---------------------------------------------------------------------------
# Natural Questions
# ---------------------------------------------------------------------------

def download_natural_questions(
    output_dir: Path,
    target_tokens: int,
    report_every: int = 10000,
) -> dict:
    """
    Download and format Natural Questions.

    NQ has long_answer (paragraph from Wikipedia) and short_answer (exact span).
    We use the long_answer context as the passage and format with the question.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"\n  Loading Natural Questions from HuggingFace...")
    print(f"  Target: {target_tokens:,} tokens (~{target_tokens * CHARS_PER_TOKEN / 1e9:.1f} GB text)")

    ds = load_dataset(
        "google-research-datasets/natural_questions",
        split="train",
        streaming=True,
    )

    nq_dir = output_dir / "nq_triviaqa"
    writer = ShardWriter(str(nq_dir), shard_mb=SHARD_MB)
    deduper = ExactDeduplicator()

    stats = {
        "seen": 0,
        "kept": 0,
        "no_long_answer": 0,
        "no_short_answer": 0,
        "too_short": 0,
        "duplicate": 0,
        "est_tokens": 0,
    }
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    total_chars = 0
    start_time = time.time()

    for record in ds:
        stats["seen"] += 1

        # Extract question
        question = record.get("question", {})
        if isinstance(question, dict):
            question_text = question.get("text", "")
        else:
            question_text = str(question)

        question_text = question_text.strip()
        if not question_text:
            continue

        # Capitalize first letter if needed, add question mark
        if question_text[0].islower():
            question_text = question_text[0].upper() + question_text[1:]
        if not question_text.endswith("?"):
            question_text += "?"

        # Extract document tokens (the Wikipedia page)
        doc_tokens = record.get("document", {}).get("tokens", {})
        if not doc_tokens:
            stats["no_long_answer"] += 1
            continue

        tokens = doc_tokens.get("token", [])
        is_html = doc_tokens.get("is_html", [])

        if not tokens:
            stats["no_long_answer"] += 1
            continue

        # Reconstruct plain text from tokens (skip HTML tokens)
        plain_tokens = []
        for tok, html_flag in zip(tokens, is_html):
            if not html_flag:
                plain_tokens.append(tok)

        # Extract the long answer span (the relevant paragraph)
        annotations = record.get("annotations", {})
        long_answers = annotations.get("long_answer", [])

        passage_text = None
        if long_answers:
            # Take the first annotated long answer
            la = long_answers[0]
            start_tok_raw = la.get("start_token", -1)
            end_tok_raw = la.get("end_token", -1)
            # Handle HF NQ format where values may be lists
            start_tok = start_tok_raw[0] if isinstance(start_tok_raw, list) and start_tok_raw else (start_tok_raw if isinstance(start_tok_raw, int) else -1)
            end_tok = end_tok_raw[0] if isinstance(end_tok_raw, list) and end_tok_raw else (end_tok_raw if isinstance(end_tok_raw, int) else -1)

            if start_tok >= 0 and end_tok > start_tok:
                # Get tokens in the long answer span, skip HTML
                la_tokens = []
                for i in range(start_tok, min(end_tok, len(tokens))):
                    if i < len(is_html) and not is_html[i]:
                        la_tokens.append(tokens[i])
                if la_tokens:
                    passage_text = " ".join(la_tokens)

        if not passage_text:
            # Fallback: use first N tokens of the plain text
            if len(plain_tokens) > 100:
                passage_text = " ".join(plain_tokens[:500])
            else:
                stats["no_long_answer"] += 1
                continue

        # Extract short answer (the actual answer span)
        short_answers = annotations.get("short_answers", [])
        answer_text = None

        if short_answers:
            sa = short_answers[0]
            sa_start_raw = sa.get("start_token", -1)
            sa_end_raw = sa.get("end_token", -1)

            # HF NQ stores short answer spans as lists (one annotator may have multiple spans)
            sa_start = sa_start_raw[0] if isinstance(sa_start_raw, list) and sa_start_raw else (sa_start_raw if isinstance(sa_start_raw, int) else -1)
            sa_end = sa_end_raw[0] if isinstance(sa_end_raw, list) and sa_end_raw else (sa_end_raw if isinstance(sa_end_raw, int) else -1)

            if sa_start >= 0 and sa_end > sa_start:
                sa_tokens = []
                for i in range(sa_start, min(sa_end, len(tokens))):
                    if i < len(is_html) and not is_html[i]:
                        sa_tokens.append(tokens[i])
                if sa_tokens:
                    answer_text = " ".join(sa_tokens)

        # Also check yes_no_answer
        if not answer_text:
            yn_answers = annotations.get("yes_no_answer", [])
            if yn_answers:
                yn = yn_answers[0]
                if yn == 1:
                    answer_text = "Yes"
                elif yn == 0:
                    answer_text = "No"

        if not answer_text:
            stats["no_short_answer"] += 1
            continue

        # Clean
        passage_text = clean_passage(passage_text)
        answer_text = clean_answer(answer_text)

        if not passage_text or not answer_text:
            stats["too_short"] += 1
            continue

        # Dedup on passage
        if deduper.is_duplicate(passage_text):
            stats["duplicate"] += 1
            continue

        # Format and write
        formatted = format_reading_comprehension(passage_text, question_text, answer_text)
        writer.write_document(formatted)
        stats["kept"] += 1

        total_chars += len(formatted)
        stats["est_tokens"] = int(total_chars / CHARS_PER_TOKEN)

        # Progress
        if stats["seen"] % report_every == 0:
            elapsed = time.time() - start_time
            rate = stats["seen"] / elapsed if elapsed > 0 else 0
            print(
                f"    NQ: {stats['seen']:,} seen | {stats['kept']:,} kept | "
                f"~{stats['est_tokens'] / 1e6:.0f}M tokens | "
                f"{rate:.0f} docs/s"
            )

        # Stop when we have enough (NQ portion = roughly 60% of target)
        if total_chars >= target_chars * 0.6:
            print(f"    NQ: reached 60% target ({stats['est_tokens']:,} tokens), stopping")
            break

    writer.close()
    elapsed = time.time() - start_time

    print(f"\n  Natural Questions complete:")
    print(f"    Seen: {stats['seen']:,}")
    print(f"    Kept: {stats['kept']:,}")
    print(f"    No long answer: {stats['no_long_answer']:,}")
    print(f"    No short answer: {stats['no_short_answer']:,}")
    print(f"    Too short: {stats['too_short']:,}")
    print(f"    Duplicates: {stats['duplicate']:,}")
    print(f"    Est. tokens: {stats['est_tokens']:,}")
    print(f"    Shards: {writer.stats()['total_shards']}")
    print(f"    Time: {elapsed:.0f}s")

    return {**stats, **writer.stats(), "source": "natural_questions", "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# TriviaQA
# ---------------------------------------------------------------------------

def download_triviaqa(
    output_dir: Path,
    target_tokens: int,
    report_every: int = 5000,
) -> dict:
    """
    Download and format TriviaQA.

    Uses the rc (reading comprehension) subset which includes Wikipedia
    evidence documents paired with questions and answers.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"\n  Loading TriviaQA (rc subset) from HuggingFace...")
    print(f"  Target: {target_tokens:,} tokens (~{target_tokens * CHARS_PER_TOKEN / 1e9:.1f} GB text)")

    ds = load_dataset(
        "mandarjoshi/trivia_qa",
        name="rc",
        split="train",
        streaming=True,
    )

    # Write to the SAME directory as NQ (they get combined)
    nq_dir = output_dir / "nq_triviaqa"
    # Continue shard numbering from where NQ left off
    existing_shards = sorted(nq_dir.glob("shard_*.txt")) if nq_dir.exists() else []
    start_shard = len(existing_shards)

    writer = ShardWriter(str(nq_dir), shard_mb=SHARD_MB)
    writer.shard_idx = start_shard  # continue numbering
    if start_shard > 0:
        # Reopen at correct index
        writer.current_file.close()
        writer._open_new_shard()

    deduper = ExactDeduplicator()

    stats = {
        "seen": 0,
        "kept": 0,
        "no_evidence": 0,
        "no_answer": 0,
        "too_short": 0,
        "duplicate": 0,
        "est_tokens": 0,
    }
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    total_chars = 0
    start_time = time.time()

    for record in ds:
        stats["seen"] += 1

        # Extract question
        question_text = record.get("question", "").strip()
        if not question_text:
            continue

        if question_text[0].islower():
            question_text = question_text[0].upper() + question_text[1:]
        if not question_text.endswith("?"):
            question_text += "?"

        # Extract answer
        answer_obj = record.get("answer", {})
        answer_text = None

        if isinstance(answer_obj, dict):
            # Primary answer value
            answer_text = answer_obj.get("value", "")
            if not answer_text:
                aliases = answer_obj.get("aliases", [])
                if aliases:
                    answer_text = aliases[0]
        elif isinstance(answer_obj, str):
            answer_text = answer_obj

        answer_text = clean_answer(answer_text) if answer_text else None
        if not answer_text:
            stats["no_answer"] += 1
            continue

        # Extract evidence documents (Wikipedia contexts)
        entity_pages = record.get("entity_pages", {})
        search_results = record.get("search_results", {})

        # Collect evidence texts
        evidence_texts = []

        # Wikipedia entity pages
        if isinstance(entity_pages, dict):
            wiki_contexts = entity_pages.get("wiki_context", [])
            if isinstance(wiki_contexts, list):
                evidence_texts.extend(wiki_contexts)

        # Search result contexts (web evidence)
        if isinstance(search_results, dict):
            search_contexts = search_results.get("search_context", [])
            if isinstance(search_contexts, list):
                evidence_texts.extend(search_contexts)

        if not evidence_texts:
            stats["no_evidence"] += 1
            continue

        # Use each grounded evidence document as a separate training example.
        # Multiple passages per question is beneficial: the model sees the same
        # answer extracted from different contexts, strengthening retrieval circuits.
        for evidence in evidence_texts:
            if not evidence or not isinstance(evidence, str):
                continue

            passage = clean_passage(evidence)
            if not passage:
                continue

            # Skip if passage doesn't seem relevant (answer not in passage)
            # This ensures the retrieval signal is real
            if answer_text.lower() not in passage.lower():
                continue

            if deduper.is_duplicate(passage):
                stats["duplicate"] += 1
                continue

            formatted = format_reading_comprehension(passage, question_text, answer_text)
            writer.write_document(formatted)
            stats["kept"] += 1

            total_chars += len(formatted)
            stats["est_tokens"] = int(total_chars / CHARS_PER_TOKEN)

            # Check token target inside the evidence loop
            if total_chars >= target_chars:
                break

        # Early exit if we've reached target
        if total_chars >= target_chars:
            print(f"    TriviaQA: reached target ({stats['est_tokens']:,} tokens), stopping")
            break

        # Progress
        if stats["seen"] % report_every == 0:
            elapsed = time.time() - start_time
            rate = stats["seen"] / elapsed if elapsed > 0 else 0
            print(
                f"    TriviaQA: {stats['seen']:,} seen | {stats['kept']:,} kept | "
                f"~{stats['est_tokens'] / 1e6:.0f}M tokens | "
                f"{rate:.0f} docs/s"
            )

    writer.close()
    elapsed = time.time() - start_time

    print(f"\n  TriviaQA complete:")
    print(f"    Seen: {stats['seen']:,}")
    print(f"    Kept: {stats['kept']:,}")
    print(f"    No evidence: {stats['no_evidence']:,}")
    print(f"    No answer: {stats['no_answer']:,}")
    print(f"    Too short: {stats['too_short']:,}")
    print(f"    Duplicates: {stats['duplicate']:,}")
    print(f"    Est. tokens: {stats['est_tokens']:,}")
    print(f"    Shards written: {writer.stats()['total_shards']}")
    print(f"    Time: {elapsed:.0f}s")

    return {**stats, **writer.stats(), "source": "triviaqa_rc", "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# SQuAD v2 (unanswerable questions + precise extractive QA)
# ---------------------------------------------------------------------------

UNANSWERABLE_RESPONSES = [
    "The passage does not contain enough information to answer this question.",
    "This question cannot be answered based on the given passage.",
    "The provided context does not address this question.",
    "No answer can be found in the passage above.",
    "The passage does not mention information relevant to this question.",
]

def download_squad_v2(
    output_dir: Path,
    target_tokens: int,
    report_every: int = 10000,
) -> dict:
    """
    Download and format SQuAD v2.

    SQuAD v2 adds ~50K unanswerable questions (20% of dataset) to SQuAD 1.1's
    extractive QA. This teaches the model that sometimes the answer is NOT in
    the passage -- a critical signal for RAG, where retrieved passages may be
    irrelevant.

    Answerable examples: passage + question + extracted answer span
    Unanswerable examples: passage + question + "cannot be answered" response
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: 'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"\n  Loading SQuAD v2 from HuggingFace...")
    print(f"  Target: {target_tokens:,} tokens (~{target_tokens * CHARS_PER_TOKEN / 1e9:.1f} GB text)")

    ds = load_dataset(
        "rajpurkar/squad_v2",
        split="train",
        streaming=True,
    )

    # Write to the SAME directory as TriviaQA (they get combined)
    nq_dir = output_dir / "nq_triviaqa"
    # Continue shard numbering from where previous sources left off
    existing_shards = sorted(nq_dir.glob("shard_*.txt")) if nq_dir.exists() else []
    start_shard = len(existing_shards)

    writer = ShardWriter(str(nq_dir), shard_mb=SHARD_MB)
    writer.shard_idx = start_shard
    if start_shard > 0:
        writer.current_file.close()
        writer._open_new_shard()

    deduper = ExactDeduplicator()
    import random
    rng = random.Random(42)  # Deterministic unanswerable response selection

    stats = {
        "seen": 0,
        "kept": 0,
        "kept_answerable": 0,
        "kept_unanswerable": 0,
        "too_short": 0,
        "duplicate": 0,
        "est_tokens": 0,
    }
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    total_chars = 0
    start_time = time.time()

    for record in ds:
        stats["seen"] += 1

        # Extract fields
        context = record.get("context", "").strip()
        question = record.get("question", "").strip()
        answers = record.get("answers", {})

        if not context or not question:
            continue

        # Clean the passage
        passage = clean_passage(context)
        if not passage:
            stats["too_short"] += 1
            continue

        # Format question
        if question[0].islower():
            question = question[0].upper() + question[1:]
        if not question.endswith("?"):
            question += "?"

        # Determine if answerable
        answer_texts = answers.get("text", [])

        if answer_texts and answer_texts[0].strip():
            # Answerable: use the first answer
            answer_text = answer_texts[0].strip()

            # Verify answer is in passage (should always be true for SQuAD)
            if answer_text.lower() not in passage.lower():
                continue

            if deduper.is_duplicate(passage + question):
                stats["duplicate"] += 1
                continue

            formatted = format_reading_comprehension(passage, question, answer_text)
            stats["kept_answerable"] += 1
        else:
            # Unanswerable: use varied "no answer" responses
            if deduper.is_duplicate(passage + question):
                stats["duplicate"] += 1
                continue

            unanswerable_response = rng.choice(UNANSWERABLE_RESPONSES)
            formatted = format_reading_comprehension(passage, question, unanswerable_response)
            stats["kept_unanswerable"] += 1

        writer.write_document(formatted)
        stats["kept"] += 1
        total_chars += len(formatted)
        stats["est_tokens"] = int(total_chars / CHARS_PER_TOKEN)

        # Progress
        if stats["seen"] % report_every == 0:
            elapsed = time.time() - start_time
            rate = stats["seen"] / elapsed if elapsed > 0 else 0
            pct_unans = stats["kept_unanswerable"] / max(1, stats["kept"]) * 100
            print(
                f"    SQuAD v2: {stats['seen']:,} seen | {stats['kept']:,} kept "
                f"({pct_unans:.0f}% unans) | "
                f"~{stats['est_tokens'] / 1e6:.0f}M tokens | "
                f"{rate:.0f} docs/s"
            )

        # Stop when we have enough
        if total_chars >= target_chars:
            print(f"    SQuAD v2: reached target ({stats['est_tokens']:,} tokens), stopping")
            break

    writer.close()
    elapsed = time.time() - start_time

    pct_unans = stats["kept_unanswerable"] / max(1, stats["kept"]) * 100
    print(f"\n  SQuAD v2 complete:")
    print(f"    Seen: {stats['seen']:,}")
    print(f"    Kept: {stats['kept']:,} ({stats['kept_answerable']:,} answerable, "
          f"{stats['kept_unanswerable']:,} unanswerable [{pct_unans:.0f}%])")
    print(f"    Too short: {stats['too_short']:,}")
    print(f"    Duplicates: {stats['duplicate']:,}")
    print(f"    Est. tokens: {stats['est_tokens']:,}")
    print(f"    Shards written: {writer.stats()['total_shards']}")
    print(f"    Time: {elapsed:.0f}s")

    return {**stats, **writer.stats(), "source": "squad_v2", "elapsed_s": elapsed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download TriviaQA + SQuAD v2 for retrieval head training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/unified_clean",
        help="Output base directory",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["triviaqa", "squad_v2"],
        choices=["nq", "triviaqa", "squad_v2"],
        help="Which sources to download (default: triviaqa + squad_v2)",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=420_000_000,
        help="Total target tokens (default: 420M)",
    )
    parser.add_argument(
        "--squad_tokens",
        type=int,
        default=70_000_000,
        help="Target tokens for SQuAD v2 specifically (default: 70M). "
             "Remainder goes to TriviaQA.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Compute per-source targets
    squad_target = args.squad_tokens if "squad_v2" in args.sources else 0
    triviaqa_target = args.target_tokens - squad_target if "triviaqa" in args.sources else 0

    print("=" * 70)
    print("  Retrieval QA Downloader")
    print("  TriviaQA + SQuAD v2 for Retrieval Head Training")
    print("=" * 70)
    print(f"\n  Output: {output_dir / 'nq_triviaqa'}")
    print(f"  Sources: {', '.join(args.sources)}")
    print(f"  Target: {args.target_tokens:,} tokens total")
    if "triviaqa" in args.sources:
        print(f"    TriviaQA: ~{triviaqa_target:,} tokens (extractive retrieval)")
    if "squad_v2" in args.sources:
        print(f"    SQuAD v2: ~{squad_target:,} tokens (extractive + unanswerable)")
    print(f"  Format: passage + question + grounded answer")

    results = {}

    if "nq" in args.sources:
        results["nq"] = download_natural_questions(
            output_dir, args.target_tokens
        )

    if "triviaqa" in args.sources:
        results["triviaqa"] = download_triviaqa(
            output_dir, triviaqa_target
        )

    if "squad_v2" in args.sources:
        results["squad_v2"] = download_squad_v2(
            output_dir, squad_target
        )

    # Write metadata
    nq_dir = output_dir / "nq_triviaqa"
    nq_dir.mkdir(parents=True, exist_ok=True)

    total_shards = sorted(nq_dir.glob("shard_*.txt"))
    total_bytes = sum(f.stat().st_size for f in total_shards)
    est_tokens = int(total_bytes / CHARS_PER_TOKEN)

    meta = {
        "downloaded_at": datetime.now().isoformat(),
        "description": "TriviaQA + SQuAD v2 formatted as reading comprehension "
                       "passages for retrieval head pre-training. SQuAD v2 includes "
                       "unanswerable questions (~20%) teaching the model to recognize "
                       "when context does not contain the answer.",
        "format": "passage\\n\\nQuestion: question\\nAnswer: grounded answer (or unanswerable response)",
        "licenses": {
            "triviaqa": "Apache 2.0",
            "squad_v2": "CC BY-SA 4.0",
        },
        "total_shards": len(total_shards),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "est_tokens": est_tokens,
        "sources": results,
    }

    meta_path = nq_dir / "download_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Combined results:")
    print(f"    Shards: {len(total_shards)}")
    print(f"    Size: {total_bytes / (1024 * 1024):.0f} MB")
    print(f"    Est. tokens: {est_tokens:,}")
    print(f"    Metadata: {meta_path}")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
