"""
Deduplicate JSONL episodes based on message content.

Removes duplicate episodes, keeping only the first occurrence.
Duplicates are identified by comparing user + assistant message content.

Usage:
    python scripts/deduplicate_episodes.py \
        --input data/sft_conversation_goldset/mypt_phase3a_gold_en.jsonl \
        --output data/sft_conversation_goldset/mypt_phase3a_gold_en_dedup.jsonl \
        --verbose
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def deduplicate_episodes(input_file: str, output_file: str, verbose: bool = False):
    """Remove duplicate episodes from JSONL file."""
    
    print(f"Reading episodes from {input_file}...")
    
    episodes = []
    seen_content = set()
    duplicates = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                episode = json.loads(line)
                
                # Create content signature (user + assistant content)
                messages = episode.get('messages', [])
                content_parts = []
                
                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role in ['user', 'assistant']:
                        content_parts.append(f"{role}:{content}")
                
                content_signature = '||'.join(content_parts)
                
                # Check if we've seen this content before
                if content_signature in seen_content:
                    duplicates.append({
                        'line': line_num,
                        'id': episode.get('id', episode.get('context', 'unknown')),
                        'user_preview': messages[0].get('content', '')[:60] if messages else 'N/A'
                    })
                    if verbose:
                        print(f"  Duplicate at line {line_num}: {duplicates[-1]['user_preview']}...")
                else:
                    seen_content.add(content_signature)
                    episodes.append(episode)
            
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"  Deduplication Results")
    print(f"{'='*60}")
    print(f"Total input episodes:   {len(episodes) + len(duplicates)}")
    print(f"Unique episodes:        {len(episodes)}")
    print(f"Duplicates removed:     {len(duplicates)}")
    print(f"Deduplication rate:     {len(duplicates)/(len(episodes)+len(duplicates))*100:.1f}%")
    
    if verbose and duplicates:
        print(f"\nDuplicate episodes (first 10):")
        for dup in duplicates[:10]:
            print(f"  Line {dup['line']:3d} (id: {dup['id']}): {dup['user_preview']}...")
    
    # Write deduplicated output
    print(f"\nWriting {len(episodes)} unique episodes to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for episode in episodes:
            json_line = json.dumps(episode, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n{'='*60}")
    print(f"  Deduplication Complete!")
    print(f"{'='*60}")
    print(f"\nOutput: {output_file}")
    print(f"Episodes: {len(episodes)} unique")
    print()
    
    return len(episodes), len(duplicates)


def main():
    from core.banner import print_banner
    print_banner("MyPT Episode Deduplicator", "Remove Duplicate Episodes (Full Content)")
    
    parser = argparse.ArgumentParser(description="Deduplicate JSONL episodes")
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output deduplicated JSONL file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed duplicate information')
    
    args = parser.parse_args()
    
    deduplicate_episodes(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()

