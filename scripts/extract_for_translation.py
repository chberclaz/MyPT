"""
Extract user and assistant content from JSONL episodes for translation.

This script creates two files:
- One with all user messages (one per line)
- One with all assistant messages (one per line)

These can be fed to a translator, then recombined using the companion script.

Usage:
    python scripts/extract_for_translation.py \
        --input data/sft_conversation_goldset/mypt_phase3a_gold_en.jsonl \
        --output_user data/temp/user_messages_for_translation.txt \
        --output_assistant data/temp/assistant_messages_for_translation.txt
"""

import json
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_messages(input_file: str, output_user: str, output_assistant: str):
    """Extract user and assistant messages to separate files."""
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create output directories
    Path(output_user).parent.mkdir(parents=True, exist_ok=True)
    Path(output_assistant).parent.mkdir(parents=True, exist_ok=True)
    
    user_messages = []
    assistant_messages = []
    episode_ids = []
    
    print(f"Reading episodes from {input_file}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                episode = json.loads(line)
                episode_id = episode.get('id') or episode.get('context', f'line_{line_num}')
                
                messages = episode.get('messages', [])
                
                # Extract user message (should be first)
                user_msg = None
                assistant_msg = None
                
                for msg in messages:
                    if msg.get('role') == 'user' and user_msg is None:
                        user_msg = msg.get('content', '')
                    elif msg.get('role') == 'assistant' and assistant_msg is None:
                        assistant_msg = msg.get('content', '')
                
                if user_msg is not None and assistant_msg is not None:
                    user_messages.append(user_msg)
                    assistant_messages.append(assistant_msg)
                    episode_ids.append(episode_id)
                else:
                    print(f"Warning: Line {line_num} missing user or assistant message")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"\nExtracted {len(user_messages)} episodes")
    
    # Write user messages
    print(f"Writing user messages to {output_user}...")
    with open(output_user, 'w', encoding='utf-8') as f:
        for msg in user_messages:
            # Write one episode per line (no newlines within the message)
            msg_oneline = msg.replace('\n', '\\n').replace('\r', '')
            f.write(msg_oneline + '\n')
    
    # Write assistant messages
    print(f"Writing assistant messages to {output_assistant}...")
    with open(output_assistant, 'w', encoding='utf-8') as f:
        for msg in assistant_messages:
            # Write one episode per line (no newlines within the message)
            msg_oneline = msg.replace('\n', '\\n').replace('\r', '')
            f.write(msg_oneline + '\n')
    
    # Write episode IDs for reference
    ids_file = Path(output_user).parent / 'episode_ids.txt'
    print(f"Writing episode IDs to {ids_file}...")
    with open(ids_file, 'w', encoding='utf-8') as f:
        for ep_id in episode_ids:
            f.write(str(ep_id) + '\n')
    
    print("\n" + "="*60)
    print("  Extraction Complete!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  User messages:      {output_user}")
    print(f"  Assistant messages: {output_assistant}")
    print(f"  Episode IDs:        {ids_file}")
    print(f"\nTotal episodes: {len(user_messages)}")
    print("\nNext steps:")
    print("1. Translate the content in these files")
    print("2. Use recombine_translations.py to merge back into JSONL")
    print()


def main():
    from core.banner import print_banner
    print_banner("MyPT Translation Extractor", "Episode Message Extraction for Translation")
    
    parser = argparse.ArgumentParser(description="Extract messages for translation")
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output_user', required=True, help='Output file for user messages')
    parser.add_argument('--output_assistant', required=True, help='Output file for assistant messages')
    
    args = parser.parse_args()
    
    extract_messages(args.input, args.output_user, args.output_assistant)


if __name__ == "__main__":
    main()

