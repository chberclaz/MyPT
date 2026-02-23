#!/usr/bin/env python
"""
Build Context Extension Dataset from HuggingFace Sources
=========================================================
Downloads and adapts 6 proven QA datasets into a single plain-text file
for continued pre-training at 4096 context (Position Interpolation phase).

Sources (all episodes used, no downsampling -- proportions emerge from dataset sizes):
    - HotpotQA distractor (~108M tokens): Long multi-passage, position-varied gold paragraphs
    - MS MARCO v2.1 (~100M tokens):       10-passage search results, position-varied, real Bing queries
    - TriviaQA evidence (~78M tokens):    Medium-long trivia grounded in Wikipedia
    - SQuAD v2 (~32M tokens):             Short extractive EN QA, incl. unanswerable
    - MuSiQue (~20M tokens):              Hard multi-hop EN (2-4 hops)
    - GermanQuAD (~1.5M tokens):          Short extractive DE QA (reduced)

This produces the QA portion (~320M tokens, 40% of final dataset).
The general text portion (60%) is added by prepare_context_extension.py
from existing pre-training shards.

Output is plain text (no <myPT_*> tags) with <|endoftext|> as document delimiter.

Prerequisites:
    pip install datasets

Usage:
    python scripts/data_prep/build_context_extension_dataset.py
    python scripts/data_prep/build_context_extension_dataset.py --output_file data/context_extension_raw/context_ext.txt
    python scripts/data_prep/build_context_extension_dataset.py --germanquad_max 6000 --seed 42
"""

import argparse
import io
import json
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed. Run: pip install datasets")
    sys.exit(1)


EOS = "<|endoftext|>"


# ---------------------------------------------------------------------------
# Adapter: SQuAD v2 (short extractive QA, ~150-300 tokens each)
# ---------------------------------------------------------------------------

def adapt_squad_v2(max_episodes=None, seed=42):
    """Load SQuAD v2 and convert to plain-text passage/question/answer episodes."""
    print("\n[1/6] Loading SQuAD v2 from HuggingFace...")
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    print(f"  Loaded {len(ds)} examples")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if max_episodes is not None:
        indices = indices[:max_episodes]

    episodes = []
    n_answerable = 0
    n_unanswerable = 0

    for idx in indices:
        ex = ds[idx]
        context = ex["context"].strip()
        question = ex["question"].strip()
        answers = ex["answers"]["text"]

        if answers:
            answer = answers[0].strip()
            n_answerable += 1
        else:
            answer = "The passage does not contain enough information to answer this question."
            n_unanswerable += 1

        episode = f"Passage: {context}\n\nQuestion: {question}\nAnswer: {answer}"
        episodes.append(episode)

    print(f"  Adapted {len(episodes)} episodes ({n_answerable} answerable, {n_unanswerable} unanswerable)")
    return episodes


# ---------------------------------------------------------------------------
# Adapter: GermanQuAD (short extractive DE QA)
# ---------------------------------------------------------------------------

def adapt_germanquad(max_episodes=None, seed=42):
    """Load GermanQuAD and convert to plain-text passage/question/answer episodes.
    
    Downloads directly from S3 since deepset/germanquad uses a legacy HF loading
    script that newer datasets versions refuse to load.
    """
    print("\n[2/6] Loading GermanQuAD (direct download)...")
    url = "https://germanquad.s3.amazonaws.com/GermanQuAD.zip"

    print(f"  Downloading from {url}...")
    response = urllib.request.urlopen(url)
    zip_data = io.BytesIO(response.read())

    with zipfile.ZipFile(zip_data) as zf:
        with zf.open("GermanQuAD/GermanQuAD_train.json") as f:
            data = json.loads(f.read().decode("utf-8"))

    # Parse SQuAD-format JSON
    examples = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answers = [a["text"] for a in qa["answers"]]
                examples.append({"context": context, "question": question, "answers": answers})

    print(f"  Loaded {len(examples)} examples")

    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_episodes is not None:
        examples = examples[:max_episodes]

    episodes = []
    for ex in examples:
        context = ex["context"].strip()
        question = ex["question"].strip()
        answer = ex["answers"][0].strip() if ex["answers"] else ""

        if not answer:
            continue

        episode = f"Passage: {context}\n\nFrage: {question}\nAntwort: {answer}"
        episodes.append(episode)

    print(f"  Adapted {len(episodes)} episodes (all answerable, German)")
    return episodes


# ---------------------------------------------------------------------------
# Adapter: HotpotQA distractor (multi-passage, position-controlled)
# ---------------------------------------------------------------------------

def _flatten_hotpot_paragraph(title, sentences):
    """Convert [title, [sent1, sent2, ...]] to plain text block."""
    text = " ".join(s.strip() for s in sentences if s.strip())
    return f"[{title}]\n{text}"


def adapt_hotpotqa(max_episodes=None, seed=42):
    """Load HotpotQA distractor setting and build position-varied multi-passage episodes."""
    print("\n[3/6] Loading HotpotQA (distractor) from HuggingFace...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    print(f"  Loaded {len(ds)} examples")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if max_episodes is not None:
        indices = indices[:max_episodes]

    episodes = []
    position_bucket_counts = [0, 0, 0, 0]

    for i, idx in enumerate(indices):
        ex = ds[idx]
        question = ex["question"].strip()
        answer = ex["answer"].strip()
        context_titles = ex["context"]["title"]
        context_sentences = ex["context"]["sentences"]
        supporting_titles = set(ex["supporting_facts"]["title"])

        gold_paragraphs = []
        distractor_paragraphs = []

        for title, sentences in zip(context_titles, context_sentences):
            block = _flatten_hotpot_paragraph(title, sentences)
            if title in supporting_titles:
                gold_paragraphs.append(block)
            else:
                distractor_paragraphs.append(block)

        if not gold_paragraphs:
            continue

        rng.shuffle(distractor_paragraphs)

        bucket = i % 4
        position_bucket_counts[bucket] += 1
        total_paras = len(gold_paragraphs) + len(distractor_paragraphs)

        if total_paras <= 1:
            ordered = gold_paragraphs + distractor_paragraphs
        else:
            n_distractors = len(distractor_paragraphs)
            if bucket == 0:
                insert_pos = 0
            elif bucket == 1:
                insert_pos = max(1, n_distractors // 3)
            elif bucket == 2:
                insert_pos = max(1, (2 * n_distractors) // 3)
            else:
                insert_pos = n_distractors

            ordered = list(distractor_paragraphs)
            for g in gold_paragraphs:
                ordered.insert(min(insert_pos, len(ordered)), g)
                insert_pos += 1

        passages_text = "\n\n".join(ordered)
        episode = f"{passages_text}\n\nQuestion: {question}\nAnswer: {answer}"
        episodes.append(episode)

    print(f"  Adapted {len(episodes)} episodes (multi-passage, position-varied)")
    print(f"  Position buckets: beginning={position_bucket_counts[0]}, "
          f"early-mid={position_bucket_counts[1]}, "
          f"late-mid={position_bucket_counts[2]}, "
          f"end={position_bucket_counts[3]}")
    return episodes


# ---------------------------------------------------------------------------
# Adapter: MuSiQue (hard multi-hop, 2-4 hops)
# ---------------------------------------------------------------------------

def adapt_musique(max_episodes=None, seed=42):
    """Load MuSiQue and build multi-hop episodes with position-varied supporting paragraphs."""
    print("\n[4/6] Loading MuSiQue from HuggingFace...")
    ds = load_dataset("dgslibisey/MuSiQue", split="train")
    print(f"  Loaded {len(ds)} examples")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if max_episodes is not None:
        indices = indices[:max_episodes]

    episodes = []

    for i, idx in enumerate(indices):
        ex = ds[idx]

        question = ex.get("question", "").strip()
        answer = ex.get("answer", "").strip()

        if not question or not answer:
            continue

        paragraphs = ex.get("paragraphs", [])
        if not paragraphs:
            continue

        gold_blocks = []
        distractor_blocks = []

        for para in paragraphs:
            title = para.get("title", "Document")
            text = para.get("paragraph_text", "").strip()
            is_supporting = para.get("is_supporting", False)

            if not text:
                continue

            block = f"[{title}]\n{text}"
            if is_supporting:
                gold_blocks.append(block)
            else:
                distractor_blocks.append(block)

        if not gold_blocks:
            continue

        rng.shuffle(distractor_blocks)

        bucket = i % 4
        n_dist = len(distractor_blocks)
        if bucket == 0:
            insert_pos = 0
        elif bucket == 1:
            insert_pos = max(1, n_dist // 3)
        elif bucket == 2:
            insert_pos = max(1, (2 * n_dist) // 3)
        else:
            insert_pos = n_dist

        ordered = list(distractor_blocks)
        for g in gold_blocks:
            ordered.insert(min(insert_pos, len(ordered)), g)
            insert_pos += 1

        passages_text = "\n\n".join(ordered)
        episode = f"{passages_text}\n\nQuestion: {question}\nAnswer: {answer}"
        episodes.append(episode)

    print(f"  Adapted {len(episodes)} episodes (multi-hop, 2-4 hops)")
    return episodes


# ---------------------------------------------------------------------------
# Adapter: MS MARCO v2.1 (10-passage search results, position-controlled)
# ---------------------------------------------------------------------------

def adapt_msmarco(max_episodes=100000, seed=42):
    """Load MS MARCO v2.1 and build position-varied multi-passage search episodes.

    Each example has 10 Bing search result passages with relevance labels.
    Selected (relevant) passages are placed in 4 uniform position buckets
    among unselected passages, same strategy as HotpotQA.
    """
    print("\n[5/6] Loading MS MARCO v2.1 from HuggingFace...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train")
    print(f"  Loaded {len(ds)} examples")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if max_episodes is not None:
        indices = indices[:max_episodes]

    episodes = []
    n_answerable = 0
    n_unanswerable = 0
    n_skipped = 0
    position_bucket_counts = [0, 0, 0, 0]

    for i, idx in enumerate(indices):
        ex = ds[idx]

        query = ex.get("query", "").strip()
        answers = ex.get("answers", [])

        if not query:
            n_skipped += 1
            continue

        is_unanswerable = (not answers or
                           (len(answers) == 1 and answers[0].strip() == "No Answer Present."))

        # Keep ~20% of unanswerable for negative retrieval signal
        if is_unanswerable:
            if rng.random() > 0.20:
                n_skipped += 1
                continue
            answer = "No answer found in the provided passages."
            n_unanswerable += 1
        else:
            answer = answers[0].strip()
            n_answerable += 1

        passages_data = ex.get("passages", {})
        passage_texts = passages_data.get("passage_text", [])
        is_selected = passages_data.get("is_selected", [])

        if not passage_texts:
            n_skipped += 1
            continue

        selected_passages = []
        unselected_passages = []

        for j, (text, sel) in enumerate(zip(passage_texts, is_selected)):
            text = text.strip()
            if not text:
                continue
            block = f"[Source {j + 1}]\n{text}"
            if sel:
                selected_passages.append(block)
            else:
                unselected_passages.append(block)

        if not selected_passages and not is_unanswerable:
            n_skipped += 1
            continue

        rng.shuffle(unselected_passages)

        # Position control: place selected passages in 4 uniform buckets
        bucket = i % 4
        position_bucket_counts[bucket] += 1
        n_unsel = len(unselected_passages)

        if not selected_passages:
            ordered = unselected_passages
        elif n_unsel == 0:
            ordered = selected_passages
        else:
            if bucket == 0:
                insert_pos = 0
            elif bucket == 1:
                insert_pos = max(1, n_unsel // 3)
            elif bucket == 2:
                insert_pos = max(1, (2 * n_unsel) // 3)
            else:
                insert_pos = n_unsel

            ordered = list(unselected_passages)
            for sp in selected_passages:
                ordered.insert(min(insert_pos, len(ordered)), sp)
                insert_pos += 1

        passages_text = "\n\n".join(ordered)
        episode = f"{passages_text}\n\nQuery: {query}\nAnswer: {answer}"
        episodes.append(episode)

    print(f"  Adapted {len(episodes)} episodes ({n_answerable} answerable, "
          f"{n_unanswerable} unanswerable, {n_skipped} skipped)")
    print(f"  Position buckets: beginning={position_bucket_counts[0]}, "
          f"early-mid={position_bucket_counts[1]}, "
          f"late-mid={position_bucket_counts[2]}, "
          f"end={position_bucket_counts[3]}")
    return episodes


# ---------------------------------------------------------------------------
# Adapter: TriviaQA (evidence-based trivia with Wikipedia passages)
# ---------------------------------------------------------------------------

def adapt_triviaqa(max_episodes=None, seed=42):
    """Load TriviaQA with Wikipedia evidence and convert to grounded QA episodes."""
    print("\n[6/6] Loading TriviaQA (rc.wikipedia) from HuggingFace...")
    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train")
    print(f"  Loaded {len(ds)} examples")

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if max_episodes is not None:
        indices = indices[:max_episodes]

    episodes = []
    n_skipped = 0

    for idx in indices:
        ex = ds[idx]
        question = ex.get("question", "").strip()
        answer_data = ex.get("answer", {})
        answer = answer_data.get("value", "").strip() if isinstance(answer_data, dict) else str(answer_data).strip()

        if not question or not answer:
            n_skipped += 1
            continue

        # Get Wikipedia evidence passages
        entity_pages = ex.get("entity_pages", {})
        wiki_contexts = entity_pages.get("wiki_context", [])
        
        if not wiki_contexts:
            n_skipped += 1
            continue

        # Use first Wikipedia passage, truncated to reasonable length
        context = wiki_contexts[0].strip()
        if len(context) > 4000:
            context = context[:4000]

        # Grounding filter: answer must appear in context
        if answer.lower() not in context.lower():
            n_skipped += 1
            continue

        episode = f"Passage: {context}\n\nQuestion: {question}\nAnswer: {answer}"
        episodes.append(episode)

    print(f"  Adapted {len(episodes)} episodes (skipped {n_skipped} ungrounded/missing)")
    return episodes


# ---------------------------------------------------------------------------
# Main: combine, shuffle, write
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build context extension QA dataset from 6 HuggingFace sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_file", type=str,
                        default="data/context_extension_raw/context_ext.txt",
                        help="Output plain-text file")
    parser.add_argument("--germanquad_max", type=int, default=6000,
                        help="Max GermanQuAD episodes (default: 6000, reduced for balance)")
    parser.add_argument("--msmarco_max", type=int, default=100000,
                        help="Max MS MARCO episodes (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    print("=" * 70)
    print("Context Extension QA Dataset Builder (6 sources)")
    print("=" * 70)
    print(f"Output: {args.output_file}")
    print(f"Seed: {args.seed}")
    print(f"GermanQuAD cap: {args.germanquad_max}")
    print(f"MS MARCO cap:   {args.msmarco_max}")
    print("All other sources: use all available episodes (no downsampling)")

    # Load and adapt all sources (use all episodes except GermanQuAD capped, MS MARCO capped)
    squad_eps = adapt_squad_v2(seed=args.seed)
    german_eps = adapt_germanquad(max_episodes=args.germanquad_max, seed=args.seed)
    hotpot_eps = adapt_hotpotqa(seed=args.seed)
    musique_eps = adapt_musique(seed=args.seed)
    msmarco_eps = adapt_msmarco(max_episodes=args.msmarco_max, seed=args.seed)
    trivia_eps = adapt_triviaqa(seed=args.seed)

    # Combine and shuffle
    all_episodes = squad_eps + german_eps + hotpot_eps + musique_eps + msmarco_eps + trivia_eps
    rng_final = random.Random(args.seed + 2)
    rng_final.shuffle(all_episodes)

    total = len(all_episodes)
    print(f"\n{'=' * 70}")
    print(f"Final QA dataset: {total} episodes")
    print(f"  HotpotQA:    {len(hotpot_eps):>6} ({len(hotpot_eps)/total:.1%})")
    print(f"  MS MARCO:    {len(msmarco_eps):>6} ({len(msmarco_eps)/total:.1%})")
    print(f"  TriviaQA:    {len(trivia_eps):>6} ({len(trivia_eps)/total:.1%})")
    print(f"  SQuAD v2:    {len(squad_eps):>6} ({len(squad_eps)/total:.1%})")
    print(f"  MuSiQue:     {len(musique_eps):>6} ({len(musique_eps)/total:.1%})")
    print(f"  GermanQuAD:  {len(german_eps):>6} ({len(german_eps)/total:.1%})")

    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, episode in enumerate(all_episodes):
            f.write(episode)
            if i < len(all_episodes) - 1:
                f.write(f"\n{EOS}\n")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nWritten to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"\nThis is the QA portion (~40% of final dataset).")
    print(f"Next: tokenize with prepare_context_extension.py --general_shards_dir data/unified_6B")
    print("=" * 70)


if __name__ == "__main__":
    main()
