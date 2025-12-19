#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test RAG components without a trained model.

This script validates:
1. Document loading
2. Text chunking
3. Embedding generation
4. Index building & saving
5. Retrieval (search)

Usage:
    python tests/test_rag_components.py
    
    # With custom test documents
    python tests/test_rag_components.py --docs_dir my_docs/
"""

import argparse
import os
import sys
import tempfile
import shutil

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_documents(test_dir: str):
    """Create sample documents for testing."""
    os.makedirs(test_dir, exist_ok=True)
    
    # Document 1: About Python
    with open(os.path.join(test_dir, "python.md"), "w") as f:
        f.write("""# Python Programming

Python is a high-level, interpreted programming language known for its 
simplicity and readability. Created by Guido van Rossum in 1991, Python 
has become one of the most popular programming languages in the world.

## Key Features

- Easy to learn and read
- Extensive standard library
- Dynamic typing
- Cross-platform compatibility
- Large ecosystem of packages

## Common Use Cases

Python is widely used in:
- Web development (Django, Flask)
- Data science and machine learning (NumPy, Pandas, TensorFlow)
- Automation and scripting
- Scientific computing
- Artificial intelligence
""")
    
    # Document 2: About Machine Learning
    with open(os.path.join(test_dir, "machine_learning.txt"), "w") as f:
        f.write("""Machine Learning: An Introduction

Machine learning is a subset of artificial intelligence that enables 
computers to learn from data without being explicitly programmed.

Types of Machine Learning:

1. Supervised Learning
   - Uses labeled data
   - Examples: classification, regression
   - Algorithms: linear regression, decision trees, neural networks

2. Unsupervised Learning
   - Uses unlabeled data
   - Examples: clustering, dimensionality reduction
   - Algorithms: k-means, PCA, autoencoders

3. Reinforcement Learning
   - Learns through trial and error
   - Uses rewards and penalties
   - Examples: game playing, robotics

Neural Networks are a key technology in modern machine learning. They
consist of layers of interconnected nodes that process information
similar to the human brain.
""")
    
    # Document 3: About Transformers
    with open(os.path.join(test_dir, "transformers.md"), "w") as f:
        f.write("""# Transformer Architecture

The Transformer is a neural network architecture introduced in the paper
"Attention Is All You Need" by Vaswani et al. in 2017.

## Key Innovation: Self-Attention

Unlike RNNs, Transformers process all positions in parallel using 
self-attention mechanisms. This allows them to:

- Capture long-range dependencies
- Train more efficiently on GPUs
- Scale to larger models

## Components

1. **Encoder**: Processes input sequence
2. **Decoder**: Generates output sequence
3. **Multi-Head Attention**: Attends to different representation subspaces
4. **Feed-Forward Networks**: Process attention outputs
5. **Positional Encoding**: Adds sequence order information

## Applications

- GPT (Generative Pre-trained Transformer)
- BERT (Bidirectional Encoder Representations)
- T5, LLaMA, and many others

Transformers have revolutionized natural language processing and are now
being applied to computer vision, audio, and other domains.
""")
    
    print(f"Created 3 test documents in {test_dir}")
    return test_dir


def test_document_loader(docs_dir: str):
    """Test document loading."""
    print("\n" + "=" * 50)
    print("TEST 1: Document Loading")
    print("=" * 50)
    
    from core.document import DocumentLoader
    
    loader = DocumentLoader()
    docs = loader.load_directory(docs_dir)
    
    print(f"✅ Loaded {len(docs)} documents:")
    for doc in docs:
        print(f"   - {doc.filename}: {doc.num_chars} chars, {doc.num_lines} lines")
    
    assert len(docs) > 0, "No documents loaded!"
    return docs


def test_chunker(docs):
    """Test text chunking."""
    print("\n" + "=" * 50)
    print("TEST 2: Text Chunking")
    print("=" * 50)
    
    from core.document import TextChunker
    
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(docs)
    
    print(f"✅ Created {len(chunks)} chunks:")
    for chunk in chunks[:5]:  # Show first 5
        src = chunk.source
        print(f"   [{chunk.chunk_id}] {src['filename']} (lines {src['start_line']}-{src['end_line']}): {chunk.num_chars} chars")
    if len(chunks) > 5:
        print(f"   ... and {len(chunks) - 5} more")
    
    # Verify provenance
    for chunk in chunks:
        assert "file" in chunk.source, "Missing source file!"
        assert "filename" in chunk.source, "Missing filename!"
        assert "start_line" in chunk.source, "Missing start_line!"
    
    print("✅ All chunks have proper provenance!")
    return chunks


def test_embedder(chunks):
    """Test embedding generation."""
    print("\n" + "=" * 50)
    print("TEST 3: Embedding Generation")
    print("=" * 50)
    
    from core.embeddings import LocalEmbedder
    import numpy as np
    import time
    
    embedder = LocalEmbedder(dim=256)
    
    # Test single encoding
    vec = embedder.encode("Hello world")
    print(f"✅ Single encoding: shape={vec.shape}, dtype={vec.dtype}")
    
    # Verify normalization
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 0.001, f"Vector not normalized: {norm}"
    print(f"✅ Vector is L2-normalized (norm={norm:.6f})")
    
    # Test batch encoding
    texts = [chunk.text for chunk in chunks]
    start = time.time()
    embeddings = embedder.encode_batch(texts)
    elapsed = time.time() - start
    
    print(f"✅ Batch encoding: {len(texts)} texts in {elapsed:.3f}s ({len(texts)/elapsed:.0f}/sec)")
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Test similarity
    q1 = embedder.encode("Python programming language")
    q2 = embedder.encode("Machine learning algorithms")
    q3 = embedder.encode("Python is a programming language")
    
    sim_1_3 = embedder.similarity(q1, q3)  # Should be high
    sim_1_2 = embedder.similarity(q1, q2)  # Should be lower
    
    print(f"✅ Similarity tests:")
    print(f"   'Python programming' vs 'Python is a programming language': {sim_1_3:.4f}")
    print(f"   'Python programming' vs 'Machine learning algorithms': {sim_1_2:.4f}")
    
    assert sim_1_3 > sim_1_2, "Similarity ordering unexpected!"
    print("✅ Semantically similar texts have higher similarity!")
    
    return embeddings


def test_index_building(docs_dir: str, index_dir: str):
    """Test full index building pipeline."""
    print("\n" + "=" * 50)
    print("TEST 4: Index Building")
    print("=" * 50)
    
    import subprocess
    
    result = subprocess.run([
        sys.executable, "scripts/build_rag_index.py",
        "--docs_dir", docs_dir,
        "--out_dir", index_dir,
        "--chunk_size", "500",
        "--chunk_overlap", "100",
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Index building failed:")
        print(result.stderr)
        raise RuntimeError("Index building failed")
    
    print(result.stdout)
    
    # Verify files exist
    import os
    assert os.path.exists(os.path.join(index_dir, "embeddings.npy")), "Missing embeddings.npy!"
    assert os.path.exists(os.path.join(index_dir, "meta.jsonl")), "Missing meta.jsonl!"
    assert os.path.exists(os.path.join(index_dir, "config.json")), "Missing config.json!"
    
    print("✅ All index files created!")


def test_retriever(index_dir: str):
    """Test retrieval functionality."""
    print("\n" + "=" * 50)
    print("TEST 5: Retrieval")
    print("=" * 50)
    
    from core.rag import Retriever
    
    retriever = Retriever(index_dir)
    print(f"✅ Loaded index with {retriever.num_chunks} chunks")
    
    # Test queries
    queries = [
        "What is Python used for?",
        "Explain neural networks",
        "What is self-attention in transformers?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"   [{i}] {r['source']['filename']} (score: {r['score']:.3f})")
            print(f"       {r['text'][:80]}...")
    
    # Verify provenance in results
    for r in results:
        assert "source" in r, "Missing source in result!"
        assert "filename" in r["source"], "Missing filename in source!"
    
    print("\n✅ Retrieval working with full provenance!")
    return retriever


def test_tags():
    """Test tag helpers."""
    print("\n" + "=" * 50)
    print("TEST 6: Tag Helpers")
    print("=" * 50)
    
    from core.rag.tags import (
        wrap_system, wrap_context, wrap_user, wrap_assistant,
        build_rag_prompt, SYSTEM_OPEN, CONTEXT_OPEN
    )
    from core.special_tokens import SPECIAL_TOKEN_STRINGS
    
    # Verify tags come from special_tokens.py
    assert SYSTEM_OPEN == SPECIAL_TOKEN_STRINGS["myPT_system_open"]
    assert CONTEXT_OPEN == SPECIAL_TOKEN_STRINGS["myPT_user_context_open"]
    print("✅ Tags correctly imported from special_tokens.py")
    
    # Test wrappers
    system = wrap_system("You are helpful.")
    user = wrap_user("What is Python?")
    assistant = wrap_assistant("Python is a programming language.")
    
    print(f"✅ wrap_system: {system}")
    print(f"✅ wrap_user: {user}")
    print(f"✅ wrap_assistant: {assistant}")
    
    # Test prompt builder
    chunks = [
        {"text": "Python is great.", "source": {"filename": "doc.md"}},
        {"text": "Machine learning is cool.", "source": {"filename": "ml.txt"}},
    ]
    prompt = build_rag_prompt("Tell me about Python", chunks, system="Be helpful")
    
    print(f"\n✅ Full RAG prompt:\n{prompt[:300]}...")
    
    assert "<myPT_system>" in prompt
    assert "<myPT_user_context>" in prompt
    assert "(doc.md)" in prompt  # Source attribution
    print("\n✅ Prompt includes source attribution!")


def test_pipeline_mock(retriever):
    """Test pipeline with a mock model."""
    print("\n" + "=" * 50)
    print("TEST 7: Pipeline (with mock model)")
    print("=" * 50)
    
    from core.rag import RAGPipeline
    
    # Create a mock model that just returns the prompt
    class MockModel:
        def generate(self, prompt, max_new_tokens=100):
            return prompt + "This is a mock response."
    
    mock_model = MockModel()
    pipeline = RAGPipeline(mock_model, retriever, default_system="You are helpful.")
    
    print("✅ Pipeline created with mock model")
    
    # Test prompt building
    question = "What is Python?"
    result = pipeline.answer_with_sources(question, top_k=2, max_new_tokens=50)
    
    print(f"\n🔍 Question: {question}")
    print(f"📝 Answer: {result['answer'][:100]}...")
    print(f"📚 Sources ({result['num_sources']}):")
    for src in result["sources"]:
        print(f"   - {src['filename']} (score: {src['score']:.3f})")
    
    print("\n✅ Pipeline works with mock model!")
    print("   (Replace MockModel with real model for actual generation)")


def main():
    parser = argparse.ArgumentParser(description="Test RAG components")
    parser.add_argument("--docs_dir", type=str, default=None,
                        help="Use existing docs directory instead of creating test docs")
    parser.add_argument("--keep_temp", action="store_true",
                        help="Keep temporary test files after running")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 RAG Component Test Suite")
    print("=" * 60)
    
    # Setup temp directories
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    test_docs_dir = args.docs_dir or os.path.join(temp_dir, "docs")
    index_dir = os.path.join(temp_dir, "index")
    
    try:
        # Create test documents if not provided
        if not args.docs_dir:
            create_test_documents(test_docs_dir)
        
        # Run tests
        docs = test_document_loader(test_docs_dir)
        chunks = test_chunker(docs)
        embeddings = test_embedder(chunks)
        test_index_building(test_docs_dir, index_dir)
        retriever = test_retriever(index_dir)
        test_tags()
        test_pipeline_mock(retriever)
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
        print("\n📋 Summary:")
        print(f"   - Documents loaded: {len(docs)}")
        print(f"   - Chunks created: {len(chunks)}")
        print(f"   - Embeddings generated: {embeddings.shape}")
        print(f"   - Index built: {index_dir}")
        print(f"   - Retrieval working: ✅")
        print(f"   - Pipeline ready: ✅ (needs real model for generation)")
        
        print("\n🚀 Next steps:")
        print("   1. Train a model: python train.py --model_name my_model ...")
        print("   2. Use workspace chat: python scripts/workspace_chat.py --model_name my_model --workspace_dir workspace/")
        
        if args.keep_temp:
            print(f"\n📁 Test files kept at: {temp_dir}")
        
    finally:
        if not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()




