"""
Deduplicate JSONL episodes based on USER MESSAGE ONLY.

This removes episodes where the user message is identical,
keeping only the first occurrence (regardless of assistant response).

Usage:
    python scripts/deduplicate_by_user_message.py \
        --input data/sft_conversation_goldset/mypt_phase3a_gold_en_dedup.jsonl \
        --output data/sft_conversation_goldset/mypt_phase3a_gold_en_unique.jsonl \
        --verbose
"""

import json
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def deduplicate_by_user_message(input_file: str, output_file: str, verbose: bool = False):
    """Remove duplicate episodes based on user message content only."""
    
    print(f"Reading episodes from {input_file}...")
    
    episodes = []
    seen_user_messages = {}
    duplicates = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                episode = json.loads(line)
                
                # Extract user message
                messages = episode.get('messages', [])
                user_message = None
                
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_message = msg.get('content', '')
                        break
                
                if user_message is None:
                    print(f"Warning: No user message found at line {line_num}")
                    continue
                
                # Check if we've seen this user message before
                if user_message in seen_user_messages:
                    first_line = seen_user_messages[user_message]
                    duplicates.append({
                        'line': line_num,
                        'first_seen': first_line,
                        'id': episode.get('id', episode.get('context', 'unknown')),
                        'user_preview': user_message[:60]
                    })
                    if verbose:
                        print(f"  Duplicate at line {line_num} (first seen at line {first_line}): {user_message[:60]}...")
                else:
                    seen_user_messages[user_message] = line_num
                    episodes.append(episode)
            
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"  Deduplication Results (User Message Only)")
    print(f"{'='*60}")
    print(f"Total input episodes:   {len(episodes) + len(duplicates)}")
    print(f"Unique user messages:   {len(episodes)}")
    print(f"Duplicates removed:     {len(duplicates)}")
    print(f"Deduplication rate:     {len(duplicates)/(len(episodes)+len(duplicates))*100:.1f}%")
    
    if verbose and duplicates:
        print(f"\nDuplicate episodes (showing first 20):")
        for dup in duplicates[:20]:
            print(f"  Line {dup['line']:3d} (duplicate of line {dup['first_seen']:3d}): {dup['user_preview']}...")
    
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
    print(f"Episodes: {len(episodes)} unique user messages")
    print()
    
    return len(episodes), len(duplicates)


def main():
    from core.banner import print_banner
    print_banner("MyPT Episode Deduplicator", "Remove Duplicate User Messages")
    
    parser = argparse.ArgumentParser(description="Deduplicate JSONL episodes by user message")
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output deduplicated JSONL file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed duplicate information')
    
    args = parser.parse_args()
    
    deduplicate_by_user_message(args.input, args.output, args.verbose)


if __name__ == "__main__":
    main()

