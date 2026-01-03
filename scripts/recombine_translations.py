"""
Recombine translated messages back into JSONL format.

After translating the user and assistant messages, this script merges them
back into the original JSONL structure.

Usage:
    python scripts/recombine_translations.py \
        --original data/sft_conversation_goldset/mypt_phase3a_gold_en.jsonl \
        --user_translated data/temp/user_messages_de.txt \
        --assistant_translated data/temp/assistant_messages_de.txt \
        --output data/sft_conversation_goldset/mypt_phase3a_gold_de.jsonl \
        --language de
"""

import json
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def recombine_translations(
    original_file: str,
    user_translated_file: str,
    assistant_translated_file: str,
    output_file: str,
    target_language: str = 'de'
):
    """Recombine translated messages with original structure."""
    
    # Read original episodes
    print(f"Reading original episodes from {original_file}...")
    original_episodes = []
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                original_episodes.append(json.loads(line))
    
    # Read translated user messages
    print(f"Reading translated user messages from {user_translated_file}...")
    with open(user_translated_file, 'r', encoding='utf-8') as f:
        user_translated = [line.strip().replace('\\n', '\n') for line in f]
    
    # Read translated assistant messages
    print(f"Reading translated assistant messages from {assistant_translated_file}...")
    with open(assistant_translated_file, 'r', encoding='utf-8') as f:
        assistant_translated = [line.strip().replace('\\n', '\n') for line in f]
    
    # Verify counts match
    if len(original_episodes) != len(user_translated) or len(original_episodes) != len(assistant_translated):
        print(f"ERROR: Episode counts don't match!")
        print(f"  Original: {len(original_episodes)}")
        print(f"  User translated: {len(user_translated)}")
        print(f"  Assistant translated: {len(assistant_translated)}")
        return
    
    print(f"\nRecombining {len(original_episodes)} episodes...")
    
    # Create new episodes with translated content
    translated_episodes = []
    for i, orig_ep in enumerate(original_episodes):
        new_ep = orig_ep.copy()
        
        # Update messages with translations
        new_messages = []
        for msg in orig_ep.get('messages', []):
            if msg.get('role') == 'user':
                new_messages.append({
                    'role': 'user',
                    'content': user_translated[i]
                })
            elif msg.get('role') == 'assistant':
                new_messages.append({
                    'role': 'assistant',
                    'content': assistant_translated[i]
                })
            else:
                # Keep other message types as-is (e.g., system)
                new_messages.append(msg.copy())
        
        new_ep['messages'] = new_messages
        
        # Update metadata
        if 'language' in new_ep:
            new_ep['language'] = target_language
        else:
            new_ep['language'] = target_language
        
        # Update episode ID to indicate translation
        orig_id = new_ep.get('id', new_ep.get('context', f'ep_{i:04d}'))
        if isinstance(orig_id, str) and 'episode_id:' in orig_id:
            # Extract ID from context
            ep_num = orig_id.split('episode_id:')[-1].strip()
            new_ep['id'] = f"ep_{ep_num}_{target_language}"
        elif 'id' in new_ep:
            new_ep['id'] = f"{new_ep['id']}_{target_language}"
        
        # Update context
        if 'context' in new_ep:
            context_parts = new_ep['context'].split(',')
            new_context = [p.strip() for p in context_parts]
            new_context.append(f"language={target_language}")
            new_ep['context'] = ', '.join(new_context)
        
        translated_episodes.append(new_ep)
    
    # Write output
    print(f"Writing translated episodes to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ep in translated_episodes:
            json_line = json.dumps(ep, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("\n" + "="*60)
    print("  Recombination Complete!")
    print("="*60)
    print(f"\nOutput file: {output_file}")
    print(f"Episodes: {len(translated_episodes)}")
    print(f"Language: {target_language}")
    print("\nNext steps:")
    print("1. Review a few episodes to verify translation quality")
    print("2. Merge with English episodes if creating bilingual dataset")
    print()


def main():
    from core.banner import print_banner
    print_banner("MyPT Translation Recombiner", "Merge Translated Messages into Episodes")
    
    parser = argparse.ArgumentParser(description="Recombine translated messages into JSONL")
    parser.add_argument('--original', required=True, help='Original JSONL file')
    parser.add_argument('--user_translated', required=True, help='File with translated user messages')
    parser.add_argument('--assistant_translated', required=True, help='File with translated assistant messages')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--language', default='de', help='Target language code (default: de)')
    
    args = parser.parse_args()
    
    recombine_translations(
        args.original,
        args.user_translated,
        args.assistant_translated,
        args.output,
        args.language
    )


if __name__ == "__main__":
    main()

