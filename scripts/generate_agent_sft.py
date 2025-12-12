#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic agentic RAG training data from workspace documents.

This script creates tool-calling conversation examples for SFT training:
1. Loads documents from a workspace directory
2. Builds or loads a RAG index
3. Generates synthetic conversations with realistic tool usage patterns
4. Outputs JSONL ready for prepare_tool_sft.py

Generated conversation patterns:
- workspace.search → find relevant docs → answer based on results
- workspace.list_docs → see available docs → describe workspace
- workspace.get_doc → retrieve full document → summarize or quote
- workspace.summarize → summarize long content → report findings
- Multi-step: search → get_doc → answer

Usage:
    # Generate from workspace
    python scripts/generate_agent_sft.py --workspace_dir workspace/ --output data/agent_sft.jsonl
    
    # Generate more examples
    python scripts/generate_agent_sft.py --workspace_dir workspace/ --output data/agent_sft.jsonl \
        --num_examples 1000 --seed 42

Then prepare the SFT dataset:
    python scripts/prepare_tool_sft.py --input data/agent_sft.jsonl --output_dir data/agent_sft
"""

import argparse
import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.document import DocumentLoader, TextChunker
import hashlib


def get_doc_id(doc) -> str:
    """Generate a unique ID for a document."""
    return hashlib.md5(doc.source.encode()).hexdigest()[:12]


def get_doc_title(doc) -> str:
    """Get a human-readable title for a document."""
    name = os.path.splitext(doc.filename)[0]
    return name.replace("_", " ").replace("-", " ").title()


# ============================================================
# Question Templates - These simulate realistic user questions
# ============================================================

SEARCH_QUESTIONS = [
    "What information do we have about {topic}?",
    "Find documents related to {topic}.",
    "Search for anything about {topic}.",
    "Can you look up {topic} in the workspace?",
    "What do our docs say about {topic}?",
    "I need information on {topic}.",
    "Tell me about {topic} from the documents.",
    "Look for {topic} in the knowledge base.",
]

LIST_QUESTIONS = [
    "What documents do we have?",
    "List all available documents.",
    "Show me what's in the workspace.",
    "What files are in the knowledge base?",
    "What documents can I access?",
    "Give me an overview of the workspace.",
]

GET_DOC_QUESTIONS = [
    "Show me the contents of {filename}.",
    "What's in {filename}?",
    "Read {filename} for me.",
    "Display the document {filename}.",
    "I need to see {filename}.",
    "Open {filename} and tell me what's there.",
]

SUMMARIZE_QUESTIONS = [
    "Summarize {filename}.",
    "Give me a summary of {filename}.",
    "What's the main point of {filename}?",
    "TL;DR for {filename}?",
    "Can you condense {filename}?",
]

MULTI_STEP_QUESTIONS = [
    "Find documents about {topic} and summarize the key points.",
    "What's in our {topic} documentation? Give me the highlights.",
    "Search for {topic} and explain what you find.",
    "Look up {topic} and provide a detailed answer.",
]


# ============================================================
# Answer Templates - These simulate model responses
# ============================================================

SEARCH_ANSWERS = [
    "Based on the search results, I found information about {topic}:\n\n{content}",
    "Here's what I found about {topic}:\n\n{content}",
    "The workspace contains the following relevant information about {topic}:\n\n{content}",
    "From the documents, here's what relates to {topic}:\n\n{content}",
]

LIST_ANSWERS = [
    "The workspace contains {count} documents:\n\n{content}",
    "Here are the available documents ({count} total):\n\n{content}",
    "I found {count} documents in the workspace:\n\n{content}",
]

GET_DOC_ANSWERS = [
    "Here's the content of {filename}:\n\n{content}",
    "The document {filename} contains:\n\n{content}",
    "{filename} says:\n\n{content}",
]

SUMMARIZE_ANSWERS = [
    "Here's a summary of {filename}:\n\n{summary}",
    "The main points from {filename}:\n\n{summary}",
    "Summary of {filename}:\n\n{summary}",
]

NO_RESULTS_ANSWERS = [
    "I couldn't find any documents related to {topic}. The workspace might not have information on this topic.",
    "No relevant documents found for {topic}. You may need to add relevant documentation first.",
    "Sorry, the search for {topic} didn't return any results.",
]


# ============================================================
# System Messages
# ============================================================

SYSTEM_MESSAGES = [
    "You are MyPT, an AI assistant with access to a document workspace. You can search, list, read, and summarize documents to answer questions.",
    "You are a helpful workspace assistant. Use the available tools (workspace.search, workspace.list_docs, workspace.get_doc, workspace.summarize) to find and present information.",
    "You are MyPT assistant with access to a knowledge base. Search documents and provide accurate answers based on the available content.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic agentic RAG training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--workspace_dir", type=str, default="workspace/",
                        help="Workspace directory with documents (default: workspace/)")
    parser.add_argument("--docs_dir", type=str, default=None,
                        help="Documents directory (default: workspace_dir/docs/)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    
    parser.add_argument("--num_examples", type=int, default=500,
                        help="Number of synthetic examples to generate (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--max_doc_chars", type=int, default=2000,
                        help="Max chars to include from documents in answers (default: 2000)")
    parser.add_argument("--include_errors", action="store_true",
                        help="Include examples with search errors (no results)")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def extract_topics_from_documents(documents: List) -> List[str]:
    """Extract meaningful topics/keywords from documents."""
    topics = set()
    
    for doc in documents:
        # Use filename (without extension) as a topic
        name = os.path.splitext(doc.filename)[0]
        topics.add(name.replace("_", " ").replace("-", " "))
        
        # Extract words from title/first line
        first_line = doc.text.split("\n")[0].strip()
        # Remove markdown headers
        first_line = first_line.lstrip("#").strip()
        if first_line and len(first_line) < 50:
            topics.add(first_line)
        
        # Extract capitalized phrases (likely important concepts)
        words = doc.text.split()
        for i, word in enumerate(words[:200]):  # Check first 200 words
            clean = word.strip(".,;:()[]{}\"'")
            if clean and clean[0].isupper() and len(clean) > 3:
                topics.add(clean)
    
    # Filter and clean topics
    clean_topics = []
    for t in topics:
        t = t.strip()
        if len(t) >= 3 and len(t) <= 50:
            clean_topics.append(t)
    
    return list(clean_topics)


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to max_chars, trying to break at sentence boundaries."""
    if len(text) <= max_chars:
        return text
    
    # Find a good break point
    truncated = text[:max_chars]
    
    # Try to break at sentence end
    for punct in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
        last_punct = truncated.rfind(punct)
        if last_punct > max_chars * 0.7:  # At least 70% of content
            return truncated[:last_punct + 1] + "..."
    
    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        return truncated[:last_space] + "..."
    
    return truncated + "..."


def generate_search_example(
    documents: List,
    chunks: List,
    topics: List[str],
    max_chars: int,
    rng: random.Random
) -> Optional[Dict]:
    """Generate a workspace.search conversation example."""
    if not topics or not chunks:
        return None
    
    topic = rng.choice(topics)
    
    # Find relevant chunks (simulate search)
    relevant_chunks = []
    topic_lower = topic.lower()
    for chunk in chunks:
        if topic_lower in chunk.text.lower():
            relevant_chunks.append(chunk)
    
    if not relevant_chunks:
        return None
    
    # Limit to top 3 chunks
    relevant_chunks = relevant_chunks[:3]
    
    # Build tool result
    tool_result = {
        "documents": [
            {
                "chunk_id": c.chunk_id,
                "text": truncate_text(c.text, 500),
                "source": c.source,
            }
            for c in relevant_chunks
        ],
        "total": len(relevant_chunks)
    }
    
    # Build answer content
    content_parts = []
    for i, c in enumerate(relevant_chunks, 1):
        src = c.source.get("filename", "unknown")
        content_parts.append(f"[{i}] From {src}:\n{truncate_text(c.text, 300)}")
    
    content = "\n\n".join(content_parts)
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(SEARCH_QUESTIONS).format(topic=topic)},
            {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": topic, "top_k": 3}},
            {"role": "toolresult", "name": "workspace.search", "content": tool_result},
            {"role": "assistant", "content": rng.choice(SEARCH_ANSWERS).format(topic=topic, content=content)},
        ]
    }


def generate_list_docs_example(
    documents: List,
    rng: random.Random
) -> Optional[Dict]:
    """Generate a workspace.list_docs conversation example."""
    if not documents:
        return None
    
    # Build tool result
    tool_result = {
        "documents": [
            {"doc_id": get_doc_id(doc), "title": get_doc_title(doc), "filename": doc.filename}
            for doc in documents
        ],
        "total": len(documents)
    }
    
    # Build answer content
    content_parts = []
    for doc in documents:
        content_parts.append(f"- **{get_doc_title(doc)}** ({doc.filename})")
    
    content = "\n".join(content_parts)
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(LIST_QUESTIONS)},
            {"role": "assistant_toolcall", "name": "workspace.list_docs", "arguments": {}},
            {"role": "toolresult", "name": "workspace.list_docs", "content": tool_result},
            {"role": "assistant", "content": rng.choice(LIST_ANSWERS).format(count=len(documents), content=content)},
        ]
    }


def generate_get_doc_example(
    documents: List,
    max_chars: int,
    rng: random.Random
) -> Optional[Dict]:
    """Generate a workspace.get_doc conversation example."""
    if not documents:
        return None
    
    doc = rng.choice(documents)
    doc_id = get_doc_id(doc)
    
    # Build tool result
    tool_result = {
        "doc_id": doc_id,
        "title": get_doc_title(doc),
        "text": truncate_text(doc.text, max_chars),
        "length": len(doc.text),
    }
    
    content = truncate_text(doc.text, max_chars)
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(GET_DOC_QUESTIONS).format(filename=doc.filename)},
            {"role": "assistant_toolcall", "name": "workspace.get_doc", "arguments": {"doc_id": doc_id}},
            {"role": "toolresult", "name": "workspace.get_doc", "content": tool_result},
            {"role": "assistant", "content": rng.choice(GET_DOC_ANSWERS).format(filename=doc.filename, content=content)},
        ]
    }


def generate_summarize_example(
    documents: List,
    max_chars: int,
    rng: random.Random
) -> Optional[Dict]:
    """Generate a workspace.summarize conversation example."""
    if not documents:
        return None
    
    doc = rng.choice(documents)
    doc_id = get_doc_id(doc)
    
    # Create a simple extractive summary (first paragraph + key sentences)
    lines = doc.text.split("\n")
    summary_lines = []
    char_count = 0
    max_summary = 500
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if char_count + len(line) > max_summary:
            break
        summary_lines.append(line)
        char_count += len(line)
    
    summary = " ".join(summary_lines)
    if len(summary) < 50:
        summary = doc.text[:500] + "..."
    
    # Build tool result
    tool_result = {"summary": summary}
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(SUMMARIZE_QUESTIONS).format(filename=doc.filename)},
            {"role": "assistant_toolcall", "name": "workspace.summarize", "arguments": {"doc_id": doc_id}},
            {"role": "toolresult", "name": "workspace.summarize", "content": tool_result},
            {"role": "assistant", "content": rng.choice(SUMMARIZE_ANSWERS).format(filename=doc.filename, summary=summary)},
        ]
    }


def generate_multi_step_example(
    documents: List,
    chunks: List,
    topics: List[str],
    max_chars: int,
    rng: random.Random
) -> Optional[Dict]:
    """Generate a multi-tool conversation example (search → get_doc → answer)."""
    if not topics or not chunks or not documents:
        return None
    
    topic = rng.choice(topics)
    
    # Find a relevant document
    topic_lower = topic.lower()
    relevant_docs = [d for d in documents if topic_lower in d.text.lower()]
    
    if not relevant_docs:
        return None
    
    doc = rng.choice(relevant_docs)
    doc_id = get_doc_id(doc)
    
    # First: search
    search_result = {
        "documents": [{"doc_id": doc_id, "text": truncate_text(doc.text, 200), "source": {"filename": doc.filename}}],
        "total": 1
    }
    
    # Second: get_doc for more detail
    get_doc_result = {
        "doc_id": doc_id,
        "title": get_doc_title(doc),
        "text": truncate_text(doc.text, max_chars),
        "length": len(doc.text),
    }
    
    # Build comprehensive answer
    content = f"Based on searching for '{topic}' and reading {doc.filename}:\n\n{truncate_text(doc.text, max_chars)}"
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(MULTI_STEP_QUESTIONS).format(topic=topic)},
            {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": topic, "top_k": 3}},
            {"role": "toolresult", "name": "workspace.search", "content": search_result},
            {"role": "assistant_toolcall", "name": "workspace.get_doc", "arguments": {"doc_id": doc_id}},
            {"role": "toolresult", "name": "workspace.get_doc", "content": get_doc_result},
            {"role": "assistant", "content": content},
        ]
    }


def generate_no_results_example(
    rng: random.Random
) -> Dict:
    """Generate a search-with-no-results example."""
    fake_topics = ["quantum computing", "blockchain", "mars exploration", 
                   "ancient history", "underwater photography", "jazz music"]
    topic = rng.choice(fake_topics)
    
    tool_result = {"documents": [], "total": 0}
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(SEARCH_QUESTIONS).format(topic=topic)},
            {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": topic, "top_k": 3}},
            {"role": "toolresult", "name": "workspace.search", "content": tool_result},
            {"role": "assistant", "content": rng.choice(NO_RESULTS_ANSWERS).format(topic=topic)},
        ]
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Agentic RAG SFT Data Generator")
    print("=" * 60)
    
    # Determine docs directory
    docs_dir = args.docs_dir or os.path.join(args.workspace_dir, "docs")
    
    if not os.path.exists(docs_dir):
        print(f"Error: Documents directory not found: {docs_dir}")
        print(f"Create {docs_dir}/ and add .txt or .md files.")
        sys.exit(1)
    
    # Load documents
    print(f"\nLoading documents from: {docs_dir}")
    loader = DocumentLoader()
    documents = loader.load_directory(docs_dir)
    
    if not documents:
        print("Error: No documents found. Add .txt or .md files to the docs directory.")
        sys.exit(1)
    
    print(f"  Loaded {len(documents)} documents")
    for doc in documents[:5]:
        print(f"    - {doc.filename}: {doc.num_chars} chars")
    if len(documents) > 5:
        print(f"    ... and {len(documents) - 5} more")
    
    # Create chunks
    print("\nChunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")
    
    # Extract topics
    print("\nExtracting topics...")
    topics = extract_topics_from_documents(documents)
    print(f"  Found {len(topics)} topics")
    if args.verbose:
        for t in topics[:10]:
            print(f"    - {t}")
    
    # Initialize random generator
    rng = random.Random(args.seed)
    
    # Generate examples
    print(f"\nGenerating {args.num_examples} synthetic examples...")
    
    examples = []
    pattern_weights = [
        ("search", 0.35, generate_search_example),
        ("list_docs", 0.15, generate_list_docs_example),
        ("get_doc", 0.20, generate_get_doc_example),
        ("summarize", 0.15, generate_summarize_example),
        ("multi_step", 0.15, generate_multi_step_example),
    ]
    
    pattern_counts = {p[0]: 0 for p in pattern_weights}
    attempts = 0
    max_attempts = args.num_examples * 10
    
    while len(examples) < args.num_examples and attempts < max_attempts:
        attempts += 1
        
        # Select pattern based on weights
        r = rng.random()
        cumulative = 0
        selected_pattern = pattern_weights[0]
        for pattern in pattern_weights:
            cumulative += pattern[1]
            if r <= cumulative:
                selected_pattern = pattern
                break
        
        pattern_name, _, generator = selected_pattern
        
        # Generate example
        if pattern_name == "search":
            example = generator(documents, chunks, topics, args.max_doc_chars, rng)
        elif pattern_name == "list_docs":
            example = generator(documents, rng)
        elif pattern_name == "get_doc":
            example = generator(documents, args.max_doc_chars, rng)
        elif pattern_name == "summarize":
            example = generator(documents, args.max_doc_chars, rng)
        elif pattern_name == "multi_step":
            example = generator(documents, chunks, topics, args.max_doc_chars, rng)
        
        if example:
            examples.append(example)
            pattern_counts[pattern_name] += 1
        
        if args.verbose and len(examples) % 100 == 0:
            print(f"  Generated {len(examples)} examples...")
    
    # Add some no-results examples if requested
    if args.include_errors:
        num_errors = min(50, args.num_examples // 10)
        print(f"\nAdding {num_errors} no-results examples...")
        for _ in range(num_errors):
            examples.append(generate_no_results_example(rng))
        pattern_counts["no_results"] = num_errors
    
    # Shuffle examples
    rng.shuffle(examples)
    
    # Save to JSONL
    print(f"\nSaving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"  Saved {len(examples)} examples")
    
    # Summary
    print("\n" + "=" * 60)
    print("[SUCCESS] Synthetic dataset generated!")
    print("=" * 60)
    
    print(f"\nPattern distribution:")
    for pattern, count in sorted(pattern_counts.items()):
        pct = count / len(examples) * 100 if examples else 0
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput: {args.output}")
    print(f"Total examples: {len(examples)}")
    
    print(f"\nNext step - prepare SFT dataset:")
    print(f"  python scripts/prepare_tool_sft.py \\")
    print(f"      --input {args.output} \\")
    print(f"      --output_dir data/agent_sft")
    
    print(f"\nThen train:")
    print(f"  python train.py --model_name my_agent \\")
    print(f"      --dataset_dir data/agent_sft \\")
    print(f"      --config_file configs/sft2/toolchat.json \\")
    print(f"      --init_from_model base_model")


if __name__ == "__main__":
    main()

