"""
Merge English and German JSONL files into a single bilingual dataset.

Usage:
    python scripts/merge_bilingual_episodes.py \
        --english data/sft_conversation_goldset/mypt_phase3a_gold_en.jsonl \
        --german data/sft_conversation_goldset/mypt_phase3a_gold_de.jsonl \
        --output data/sft_conversation_goldset/mypt_phase3a_gold_bilingual.jsonl \
        --shuffle
"""

import json
import sys
import argparse
import random
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def merge_bilingual(en_file: str, de_file: str, output_file: str, shuffle: bool = True):
    """Merge English and German episodes into one file."""
    
    print(f"Reading English episodes from {en_file}...")
    en_episodes = []
    with open(en_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ep = json.loads(line)
                ep['language'] = 'en'  # Ensure language tag
                en_episodes.append(ep)
    
    print(f"Reading German episodes from {de_file}...")
    de_episodes = []
    with open(de_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ep = json.loads(line)
                ep['language'] = 'de'  # Ensure language tag
                de_episodes.append(ep)
    
    print(f"\nDataset composition:")
    print(f"  English: {len(en_episodes)} episodes")
    print(f"  German:  {len(de_episodes)} episodes")
    print(f"  Total:   {len(en_episodes) + len(de_episodes)} episodes")
    
    # Combine
    all_episodes = en_episodes + de_episodes
    
    # Shuffle if requested
    if shuffle:
        print(f"\nShuffling episodes for better language mixing during training...")
        random.shuffle(all_episodes)
    
    # Write output
    print(f"\nWriting bilingual dataset to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ep in all_episodes:
            json_line = json.dumps(ep, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("\n" + "="*60)
    print("  Merge Complete!")
    print("="*60)
    print(f"\nOutput: {output_file}")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"  English: {len(en_episodes)} ({len(en_episodes)/len(all_episodes)*100:.1f}%)")
    print(f"  German:  {len(de_episodes)} ({len(de_episodes)/len(all_episodes)*100:.1f}%)")
    print(f"Shuffled: {'Yes' if shuffle else 'No'}")
    print("\nNext step: Use this file for augmentation or training preparation")
    print()


def main():
    from core.banner import print_banner
    print_banner("MyPT Bilingual Merger", "Combine English & German Episodes")
    
    parser = argparse.ArgumentParser(description="Merge English and German episodes")
    parser.add_argument('--english', required=True, help='English JSONL file')
    parser.add_argument('--german', required=True, help='German JSONL file')
    parser.add_argument('--output', required=True, help='Output bilingual JSONL file')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle episodes for language mixing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling (default: 42)')
    
    args = parser.parse_args()
    
    if args.shuffle:
        random.seed(args.seed)
    
    merge_bilingual(args.english, args.german, args.output, args.shuffle)


if __name__ == "__main__":
    main()

