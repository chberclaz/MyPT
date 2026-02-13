"""
Extract user and assistant content from JSONL episodes for translation.

This script creates two files with episode markers that work with the full
translation pipeline (translate_deepl.py → recombine_translations.py).

OUTPUT FORMAT (matches translate_deepl.py and recombine_translations.py):
    === episode_id ===
    Full message content with
    multiple lines preserved.
    
    === multi_turn_episode ===
    First message content...
    <<<MSG_SEP>>>
    Second message content...

Multi-turn episodes use <<<MSG_SEP>>> to separate individual messages.

Usage:
    python scripts/extract_for_translation.py \\
        --input data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1.jsonl \\
        --output_dir data/temp

    This creates:
        data/temp/user_messages_en.txt
        data/temp/assistant_messages_en.txt
"""

import json
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Message separator for multi-turn episodes (must match recombine_translations.py)
MSG_SEP = '<<<MSG_SEP>>>'


def extract_episode_id(episode: dict, line_num: int) -> str:
    """Extract episode ID from episode data."""
    # Try 'id' field first
    if 'id' in episode:
        return str(episode['id'])
    
    # Try to extract from context field
    context = episode.get('context', '')
    if 'episode_id:' in context:
        # Format: "episode_id: 0001, ..." or "episode_id: some_id, ..."
        ep_id = context.split('episode_id:')[-1].strip().split(',')[0].strip()
        return ep_id
    
    # Fallback to line number
    return f"ep_{line_num:04d}"


def extract_messages(input_file: str, output_dir: str):
    """Extract user and assistant messages to separate files with episode markers."""
    
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Output files (names must match translate_deepl.py expectations)
    user_file = output_path / "user_messages_en.txt"
    assistant_file = output_path / "assistant_messages_en.txt"
    
    user_episodes = []  # List of (episode_id, [messages])
    assistant_episodes = []
    
    print(f"Reading episodes from {input_file}...")
    
    total_episodes = 0
    multi_turn_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                episode = json.loads(line)
                episode_id = extract_episode_id(episode, line_num)
                
                messages = episode.get('messages', [])
                
                # Extract ALL user and assistant messages (for multi-turn support)
                user_msgs = []
                assistant_msgs = []
                
                for msg in messages:
                    role = msg.get('role')
                    content = msg.get('content', '')
                    
                    if role == 'user':
                        user_msgs.append(content)
                    elif role == 'assistant':
                        assistant_msgs.append(content)
                
                if user_msgs and assistant_msgs:
                    user_episodes.append((episode_id, user_msgs))
                    assistant_episodes.append((episode_id, assistant_msgs))
                    total_episodes += 1
                    
                    if len(user_msgs) > 1 or len(assistant_msgs) > 1:
                        multi_turn_count += 1
                else:
                    print(f"Warning: Line {line_num} ({episode_id}) missing user or assistant message")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"\nExtracted {total_episodes} episodes ({multi_turn_count} multi-turn)")
    
    # Write user messages with episode markers
    print(f"Writing user messages to {user_file}...")
    with open(user_file, 'w', encoding='utf-8') as f:
        for episode_id, msgs in user_episodes:
            f.write(f"=== {episode_id} ===\n")
            if len(msgs) == 1:
                # Single message - write directly
                f.write(msgs[0] + "\n\n")
            else:
                # Multi-turn - separate with MSG_SEP
                f.write(f"\n{MSG_SEP}\n".join(msgs) + "\n\n")
    
    # Write assistant messages with episode markers
    print(f"Writing assistant messages to {assistant_file}...")
    with open(assistant_file, 'w', encoding='utf-8') as f:
        for episode_id, msgs in assistant_episodes:
            f.write(f"=== {episode_id} ===\n")
            if len(msgs) == 1:
                # Single message - write directly
                f.write(msgs[0] + "\n\n")
            else:
                # Multi-turn - separate with MSG_SEP
                f.write(f"\n{MSG_SEP}\n".join(msgs) + "\n\n")
    
    # Calculate character counts
    user_chars = sum(len(m) for _, msgs in user_episodes for m in msgs)
    assistant_chars = sum(len(m) for _, msgs in assistant_episodes for m in msgs)
    total_chars = user_chars + assistant_chars
    
    print("\n" + "="*60)
    print("  Extraction Complete!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  User messages:      {user_file}")
    print(f"  Assistant messages: {assistant_file}")
    print(f"\nStatistics:")
    print(f"  Total episodes:     {total_episodes}")
    print(f"  Multi-turn:         {multi_turn_count}")
    print(f"  User characters:    {user_chars:,}")
    print(f"  Assistant chars:    {assistant_chars:,}")
    print(f"  Total characters:   {total_chars:,} (for DeepL quota)")
    print("\nNext step:")
    print("  python scripts/translate_deepl.py")
    print()


def main():
    from core.banner import print_banner
    print_banner("MyPT Translation Extractor", "Episode Message Extraction for Translation")
    
    parser = argparse.ArgumentParser(
        description="Extract messages for translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/extract_for_translation.py \\
        --input data/sft_run2_minimal_qa/mypt_run2_minimal_qa_v1.jsonl \\
        --output_dir data/temp
        
Output files will be:
    data/temp/user_messages_en.txt
    data/temp/assistant_messages_en.txt
"""
    )
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output_dir', default='data/temp', 
                        help='Output directory (default: data/temp)')
    
    args = parser.parse_args()
    
    extract_messages(args.input, args.output_dir)


if __name__ == "__main__":
    main()
