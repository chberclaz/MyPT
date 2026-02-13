"""
Recombine translated messages back into JSONL format.

After translating the user and assistant messages, this script merges them
back into the original JSONL structure.

FILE FORMATS (UNIFIED):
-----------------------
Both user and assistant messages use the SAME format:
    - Multi-line content between episode markers
    - <<<MSG_SEP>>> separates individual messages in multi-turn episodes

User messages (user_messages_de.txt):
    === episode_id ===
    Full user message with
    code blocks, multiple lines, etc.
    
    === multi_turn_id ===
    First user message...
    ```python
    code_here()
    ```
    <<<MSG_SEP>>>
    Second user message...
    also with multiple lines...

Assistant messages (assistant_messages_de.txt):
    === episode_id ===
    Full assistant response with
    multiple lines, code blocks, etc.
    
    === multi_turn_id ===
    First assistant response...
    <<<MSG_SEP>>>
    Second assistant response...

MULTI-TURN EPISODES REQUIRING <<<MSG_SEP>>>:
    0096, 0097, 0098, multi_001 through multi_008

NEW EPISODES (German-only):
    Episodes that exist in translation files but not in the original
    will be created as new episodes with auto-generated metadata.

Usage:
    python scripts/recombine_translations.py \\
        --original data/sft_conversation_goldset/mypt_phase3a_gold_en_v2.jsonl \\
        --user_translated data/temp/user_messages_de.txt \\
        --assistant_translated data/temp/assistant_messages_de.txt \\
        --output data/sft_conversation_goldset/mypt_phase3a_gold_de_v2.jsonl \\
        --language de
"""

import json
import sys
import re
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Message separator for multi-turn episodes
MSG_SEP = '<<<MSG_SEP>>>'


def parse_messages_file(file_path: str) -> dict:
    """
    Parse a messages file (user or assistant) with unified format.
    
    File format:
        === episode_id ===
        Multi-line message content
        with code blocks, etc.
        
        === multi_turn_id ===
        First message...
        <<<MSG_SEP>>>
        Second message...
        
    Returns:
        dict mapping episode_id to list of message strings
    """
    episodes = {}
    current_episode_id = None
    content_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n\r')
            
            # Check for episode marker
            match = re.match(r'^===\s*(.+?)\s*===$', line)
            if match:
                # Save previous episode if exists
                if current_episode_id is not None:
                    episodes[current_episode_id] = _split_content(content_lines)
                
                # Start new episode
                current_episode_id = match.group(1).strip()
                content_lines = []
            elif current_episode_id is not None:
                content_lines.append(line)
        
        # Save last episode
        if current_episode_id is not None:
            episodes[current_episode_id] = _split_content(content_lines)
    
    return episodes


def _split_content(lines: list) -> list:
    """
    Split collected lines into individual messages.
    
    Uses <<<MSG_SEP>>> as delimiter for multi-turn episodes.
    If no separator found, all content is one message.
    """
    # Join all lines
    full_content = '\n'.join(lines)
    
    # Split by separator marker
    if MSG_SEP in full_content:
        messages = full_content.split(MSG_SEP)
    else:
        messages = [full_content]
    
    # Clean up each message (strip leading/trailing whitespace)
    cleaned = []
    for msg in messages:
        msg = msg.strip()
        if msg:
            cleaned.append(msg)
    
    return cleaned


def create_new_episode(ep_id: str, user_msgs: list, assistant_msgs: list, 
                       target_language: str) -> dict:
    """
    Create a new episode structure for translation-only episodes.
    
    These are episodes that exist in the translation files but not
    in the original English file.
    """
    # Default system prompt (same as all other episodes)
    DEFAULT_SYSTEM_PROMPT = "You are MyPT, an offline, privacy-first assistant. Be accurate, practical, and concise. Ask clarifying questions only when essential. If unsure, state assumptions."
    
    # Build messages array alternating user/assistant
    messages = []
    max_turns = max(len(user_msgs), len(assistant_msgs))
    
    for i in range(max_turns):
        if i < len(user_msgs):
            messages.append({
                'role': 'user',
                'content': user_msgs[i]
            })
        if i < len(assistant_msgs):
            messages.append({
                'role': 'assistant',
                'content': assistant_msgs[i]
            })
    
    return {
        'system': DEFAULT_SYSTEM_PROMPT,
        'context': f"episode_id: {ep_id}, language={target_language}",
        'messages': messages,
        'language': target_language
    }


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
    original_ids = set()
    
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ep = json.loads(line)
                original_episodes.append(ep)
                
                # Extract episode ID
                context = ep.get('context', '')
                if 'episode_id:' in context:
                    ep_id = context.split('episode_id:')[-1].strip().split(',')[0].strip()
                    original_ids.add(ep_id)
    
    print(f"  Found {len(original_episodes)} episodes")
    
    # Parse translated messages (unified format with <<<MSG_SEP>>>)
    print(f"Parsing translated user messages from {user_translated_file}...")
    print(f"  (Format: multi-line content, {MSG_SEP} separates turns)")
    user_translations = parse_messages_file(user_translated_file)
    print(f"  Found {len(user_translations)} episode sections")
    
    print(f"Parsing translated assistant messages from {assistant_translated_file}...")
    print(f"  (Format: multi-line content, {MSG_SEP} separates turns)")
    assistant_translations = parse_messages_file(assistant_translated_file)
    print(f"  Found {len(assistant_translations)} episode sections")
    
    # Find new episodes (in translations but not in original)
    translation_ids = set(user_translations.keys()) | set(assistant_translations.keys())
    new_episode_ids = translation_ids - original_ids
    
    if new_episode_ids:
        print(f"\n  Found {len(new_episode_ids)} new episode(s) only in translations:")
        for ep_id in sorted(new_episode_ids):
            print(f"    + {ep_id}")
    
    # Create translated episodes from originals
    print(f"\nRecombining {len(original_episodes)} original episodes...")
    
    translated_episodes = []
    warnings = []
    
    for i, orig_ep in enumerate(original_episodes):
        new_ep = orig_ep.copy()
        
        # Extract episode ID from context field
        context = orig_ep.get('context', '')
        if 'episode_id:' in context:
            ep_id = context.split('episode_id:')[-1].strip().split(',')[0].strip()
        else:
            ep_id = orig_ep.get('id', f'ep_{i:04d}')
        
        # Get translated messages for this episode
        user_msgs = user_translations.get(ep_id, [])
        assistant_msgs = assistant_translations.get(ep_id, [])
        
        # Count original user/assistant messages
        orig_user_count = sum(1 for m in orig_ep.get('messages', []) if m.get('role') == 'user')
        orig_assistant_count = sum(1 for m in orig_ep.get('messages', []) if m.get('role') == 'assistant')
        
        # Verify counts match
        if len(user_msgs) != orig_user_count:
            warnings.append(f"  [{ep_id}] User message count mismatch: original={orig_user_count}, translated={len(user_msgs)}")
        if len(assistant_msgs) != orig_assistant_count:
            warnings.append(f"  [{ep_id}] Assistant message count mismatch: original={orig_assistant_count}, translated={len(assistant_msgs)}")
        
        # Rebuild messages array with translations
        new_messages = []
        user_idx = 0
        assistant_idx = 0
        
        for msg in orig_ep.get('messages', []):
            role = msg.get('role')
            
            if role == 'user':
                if user_idx < len(user_msgs):
                    new_messages.append({
                        'role': 'user',
                        'content': user_msgs[user_idx]
                    })
                    user_idx += 1
                else:
                    # Fallback to original if translation missing
                    new_messages.append(msg.copy())
                    
            elif role == 'assistant':
                if assistant_idx < len(assistant_msgs):
                    new_messages.append({
                        'role': 'assistant',
                        'content': assistant_msgs[assistant_idx]
                    })
                    assistant_idx += 1
                else:
                    # Fallback to original if translation missing
                    new_messages.append(msg.copy())
            else:
                # Keep other message types as-is (e.g., system)
                new_messages.append(msg.copy())
        
        new_ep['messages'] = new_messages
        
        # Update metadata
        new_ep['language'] = target_language
        
        # Update episode ID to indicate translation
        if 'id' in new_ep:
            if not new_ep['id'].endswith(f'_{target_language}'):
                new_ep['id'] = f"{new_ep['id']}_{target_language}"
        
        # Update context
        if 'context' in new_ep:
            if f'language={target_language}' not in new_ep['context']:
                new_ep['context'] = f"{new_ep['context']}, language={target_language}"
        
        translated_episodes.append(new_ep)
    
    # Add new episodes (translation-only)
    new_episodes_added = 0
    if new_episode_ids:
        print(f"\nCreating {len(new_episode_ids)} new episode(s) from translations...")
        
        for ep_id in sorted(new_episode_ids):
            user_msgs = user_translations.get(ep_id, [])
            assistant_msgs = assistant_translations.get(ep_id, [])
            
            if user_msgs or assistant_msgs:
                new_ep = create_new_episode(ep_id, user_msgs, assistant_msgs, target_language)
                translated_episodes.append(new_ep)
                new_episodes_added += 1
                print(f"    Created: {ep_id} ({len(user_msgs)} user, {len(assistant_msgs)} assistant)")
    
    # Print warnings
    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for w in warnings[:20]:  # Show first 20
            print(w)
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")
    
    # Write output
    print(f"\nWriting translated episodes to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ep in translated_episodes:
            json_line = json.dumps(ep, ensure_ascii=False)
            f.write(json_line + '\n')
    
    # Summary
    multi_turn_count = sum(1 for ep in translated_episodes 
                          if sum(1 for m in ep.get('messages', []) if m.get('role') == 'user') > 1)
    
    print("\n" + "="*60)
    print("  Recombination Complete!")
    print("="*60)
    print(f"\n  Output file: {output_file}")
    print(f"  Episodes from original: {len(original_episodes)}")
    print(f"  New episodes added: {new_episodes_added}")
    print(f"  Total episodes: {len(translated_episodes)}")
    print(f"  Multi-turn episodes: {multi_turn_count}")
    print(f"  Language: {target_language}")
    print(f"  Warnings: {len(warnings)}")
    print("\nNext steps:")
    print("  1. Review a few episodes to verify translation quality")
    print("  2. Merge with English episodes: python scripts/merge_bilingual_episodes.py")
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
