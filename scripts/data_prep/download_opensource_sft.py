#!/usr/bin/env python3
"""
Download and convert open-source SFT datasets to MyPT JSONL format.

These are FREE, high-quality, curated datasets:
- Dolly-15K: Databricks instruction-following (English)
- OASST1: OpenAssistant conversations (multilingual)
- Security StackExchange: IT security Q&A

Usage:
    python scripts/download_opensource_sft.py --dataset dolly --output data/sft_opensource/dolly.jsonl
    python scripts/download_opensource_sft.py --dataset oasst --output data/sft_opensource/oasst.jsonl
    python scripts/download_opensource_sft.py --dataset all --output data/sft_opensource/
"""

import argparse
import json
import os
from pathlib import Path

def download_dolly(output_path: str, max_samples: int = None):
    """
    Download Dolly-15K dataset from Hugging Face.
    High-quality instruction-following examples from Databricks employees.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return
    
    print("Downloading Dolly-15K from Hugging Face...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    conversations = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # Dolly format: instruction, context (optional), response
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        response = item.get("response", "")
        category = item.get("category", "general")
        
        # Combine instruction with context if present
        if context:
            user_content = f"{instruction}\n\nContext: {context}"
        else:
            user_content = instruction
        
        conv = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response}
            ],
            "context": {
                "episode_id": f"dolly_{i:05d}",
                "language": "en",
                "source": "dolly-15k",
                "category": category
            }
        }
        conversations.append(conv)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved {len(conversations)} Dolly conversations to {output_path}")
    return conversations


def download_oasst(output_path: str, max_samples: int = None, languages: list = None):
    """
    Download OpenAssistant dataset.
    High-quality human-written conversations in multiple languages.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return
    
    if languages is None:
        languages = ["en", "de"]  # English and German for your use case
    
    print(f"Downloading OpenAssistant (languages: {languages})...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    # OASST is tree-structured, we need to extract linear conversations
    # Group by conversation tree
    from collections import defaultdict
    
    messages_by_parent = defaultdict(list)
    messages_by_id = {}
    roots = []
    
    for item in dataset:
        msg_id = item["message_id"]
        parent_id = item["parent_id"]
        messages_by_id[msg_id] = item
        
        if parent_id is None:
            roots.append(msg_id)
        else:
            messages_by_parent[parent_id].append(msg_id)
    
    def extract_conversation(root_id, max_depth=4):
        """Extract a linear conversation from a tree."""
        conv = []
        current_id = root_id
        depth = 0
        
        while current_id and depth < max_depth:
            msg = messages_by_id.get(current_id)
            if not msg:
                break
            
            role = "user" if msg["role"] == "prompter" else "assistant"
            conv.append({"role": role, "content": msg["text"], "lang": msg.get("lang", "en")})
            
            # Get best child (highest rank or first)
            children = messages_by_parent.get(current_id, [])
            if not children:
                break
            
            # Pick first child (simplified - could pick by rank)
            current_id = children[0]
            depth += 1
        
        return conv
    
    conversations = []
    for i, root_id in enumerate(roots):
        if max_samples and i >= max_samples:
            break
        
        conv_messages = extract_conversation(root_id)
        
        # Filter by language (check first message)
        if conv_messages and conv_messages[0].get("lang") not in languages:
            continue
        
        # Need at least user + assistant
        if len(conv_messages) < 2:
            continue
        
        # Remove lang field from messages
        clean_messages = [{"role": m["role"], "content": m["content"]} for m in conv_messages]
        
        conv = {
            "messages": clean_messages,
            "context": {
                "episode_id": f"oasst_{i:05d}",
                "language": conv_messages[0].get("lang", "en"),
                "source": "openassistant"
            }
        }
        conversations.append(conv)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved {len(conversations)} OpenAssistant conversations to {output_path}")
    return conversations


def download_security_stackexchange(output_path: str, max_samples: int = 2000):
    """
    Download Security StackExchange Q&A.
    Real IT security questions with community-vetted answers.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install datasets: pip install datasets")
        return
    
    print("Downloading Security StackExchange...")
    
    # Try to load from HuggingFace (there are several SE datasets)
    try:
        dataset = load_dataset("flax-sentence-embeddings/stackexchange_xml", 
                              "security", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Could not load Security SE dataset: {e}")
        print("Falling back to general StackExchange...")
        try:
            dataset = load_dataset("HuggingFaceH4/stack-exchange-preferences", 
                                  split="train")
        except Exception as e2:
            print(f"Could not load fallback dataset: {e2}")
            return []
    
    conversations = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        
        # Format depends on dataset structure
        question = item.get("question", item.get("title", ""))
        answer = item.get("answer", item.get("body", ""))
        
        if not question or not answer:
            continue
        
        conv = {
            "messages": [
                {"role": "user", "content": question[:2000]},  # Truncate long questions
                {"role": "assistant", "content": answer[:3000]}  # Truncate long answers
            ],
            "context": {
                "episode_id": f"secse_{i:05d}",
                "language": "en",
                "source": "security-stackexchange"
            }
        }
        conversations.append(conv)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved {len(conversations)} Security SE conversations to {output_path}")
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Download open-source SFT datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["dolly", "oasst", "security", "all"],
                       help="Dataset to download")
    parser.add_argument("--output", type=str, required=True,
                       help="Output path (file for single dataset, directory for 'all')")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per dataset")
    parser.add_argument("--languages", type=str, nargs="+", default=["en", "de"],
                       help="Languages to include (for OASST)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("OPEN-SOURCE SFT DATASET DOWNLOADER")
    print("=" * 60)
    
    if args.dataset == "all":
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        download_dolly(str(output_dir / "dolly.jsonl"), args.max_samples)
        download_oasst(str(output_dir / "oasst.jsonl"), args.max_samples, args.languages)
        download_security_stackexchange(str(output_dir / "security_se.jsonl"), args.max_samples)
        
        print(f"\n✅ All datasets saved to {output_dir}/")
        
    elif args.dataset == "dolly":
        download_dolly(args.output, args.max_samples)
        
    elif args.dataset == "oasst":
        download_oasst(args.output, args.max_samples, args.languages)
        
    elif args.dataset == "security":
        download_security_stackexchange(args.output, args.max_samples)
    
    print("\nNext steps:")
    print("  1. Review the downloaded data")
    print("  2. Combine with your gold set if desired")
    print("  3. Run: python scripts/prepare_chat_sft.py --input <jsonl> --output data/sft_combined")


if __name__ == "__main__":
    main()
