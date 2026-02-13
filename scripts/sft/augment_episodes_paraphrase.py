"""
Episode Paraphrasing for Data Augmentation

This script takes gold conversation episodes and generates paraphrased variations
to expand your training dataset while maintaining quality.

Strategy:
1. Load gold episodes from JSON
2. Use your trained model to generate paraphrases
3. Apply quality filters
4. Output augmented dataset for human review

Usage:
    python scripts/augment_episodes_paraphrase.py \
        --input data/sft_conversation_goldset/mypt_phase3a_gold_en_verified.jsonl \
        --output data/sft_conversation_goldset_augmented.jsonl \
        --model checkpoints/750M_gold_2.22 \
        --target_count 1200 \
        --lang_filter both \
        --review_mode
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import load_model

# Paraphrasing templates for system/user messages
PARAPHRASE_TEMPLATES = {
    'user_question': [
        "{original}",  # Keep some originals
        "Could you help me with this: {content}",
        "I have a question: {content}",
        "Can you explain {content}",
        "What about {content}",
        "Tell me about {content}",
    ],
    'user_request': [
        "{original}",
        "Please {content}",
        "I need you to {content}",
        "Would you mind {content}",
        "Can you help me {content}",
    ],
    'user_greeting': [
        "Hello",
        "Hi",
        "Hey",
        "Hi there",
        "Greetings",
        "Good day",
    ],
}


def load_episodes(input_path: str) -> List[Dict[str, Any]]:
    """
    Load episodes from input path.
    Supports both:
    - JSONL file (one episode per line)
    - Directory with individual JSON files
    """
    episodes = []
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    # Case 1: JSONL file
    if path.is_file() and path.suffix in ['.jsonl', '.json']:
        print(f"Loading JSONL file: {input_path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        episode = json.loads(line)
                        
                        # Convert format if needed
                        if 'messages' in episode:
                            # Already in correct format, just add metadata
                            if 'id' not in episode:
                                # Extract from context if available
                                context = episode.get('context', '')
                                if 'episode_id:' in context:
                                    ep_id = context.split('episode_id:')[-1].strip()
                                    episode['id'] = f"ep_{ep_id}"
                                else:
                                    episode['id'] = f"ep_{line_num:05d}"
                            
                            # Keep system field separate (don't duplicate into messages array)
                            # The format keeps system at top level, messages only has user/assistant
                            
                            # Detect language (simple heuristic)
                            if 'language' not in episode:
                                # Check if messages contain German-specific words
                                all_text = ' '.join(m.get('content', '') for m in episode['messages'])
                                # Common German indicators
                                german_indicators = ['der ', 'die ', 'das ', 'ist ', 'und ', 'für ', 'mit ', 'von ']
                                german_count = sum(1 for word in german_indicators if word in all_text.lower())
                                episode['language'] = 'de' if german_count >= 3 else 'en'
                            
                            episodes.append(episode)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
            raise
    
    # Case 2: Directory with JSON files
    elif path.is_dir():
        print(f"Loading JSON files from directory: {input_path}")
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    episode = json.load(f)
                    episodes.append(episode)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    else:
        raise ValueError(f"Input must be a JSONL file or directory, got: {input_path}")
    
    print(f"Loaded {len(episodes)} episodes from {input_path}")
    return episodes


def analyze_dataset(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze dataset composition."""
    stats = {
        'total': len(episodes),
        'by_language': defaultdict(int),
        'by_category': defaultdict(int),
        'by_length': {'short': 0, 'medium': 0, 'long': 0},
    }
    
    for ep in episodes:
        # Language
        lang = ep.get('language', 'unknown')
        stats['by_language'][lang] += 1
        
        # Category
        category = ep.get('category', 'uncategorized')
        stats['by_category'][category] += 1
        
        # Length (based on message count)
        msg_count = len(ep.get('messages', []))
        if msg_count <= 2:
            stats['by_length']['short'] += 1
        elif msg_count <= 5:
            stats['by_length']['medium'] += 1
        else:
            stats['by_length']['long'] += 1
    
    return stats


def print_dataset_stats(stats: Dict[str, Any], label: str = "Dataset"):
    """Print formatted dataset statistics."""
    print(f"\n{'='*60}")
    print(f"  {label} Statistics")
    print(f"{'='*60}")
    print(f"Total episodes: {stats['total']}")
    
    print(f"\nBy language:")
    for lang, count in sorted(stats['by_language'].items()):
        pct = (count / stats['total']) * 100
        print(f"  {lang:10s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nBy category:")
    for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total']) * 100
        print(f"  {cat:20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nBy length:")
    for length, count in stats['by_length'].items():
        pct = (count / stats['total']) * 100
        print(f"  {length:10s}: {count:4d} ({pct:5.1f}%)")
    print(f"{'='*60}\n")


def paraphrase_message_simple(message: Dict[str, str], role: str) -> Dict[str, str]:
    """
    Simple rule-based paraphrasing for user messages.
    
    For assistant messages, we'll use the model to generate variations.
    """
    if role != 'user':
        return message  # Don't paraphrase system/assistant with rules
    
    content = message.get('content', '')
    
    # Detect message type
    is_greeting = any(g in content.lower() for g in ['hello', 'hi', 'hey', 'greetings'])
    is_question = '?' in content
    
    # Apply template
    if is_greeting and len(content.split()) <= 3:
        # Short greeting - use greeting templates
        new_content = random.choice(PARAPHRASE_TEMPLATES['user_greeting'])
    elif is_question:
        # Keep questions mostly unchanged (high risk with paraphrasing)
        new_content = content
    else:
        # Other user messages - minimal changes
        new_content = content
    
    return {'role': 'user', 'content': new_content}


def paraphrase_episode_model_assisted(
    episode: Dict[str, Any],
    model,
    temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Use the model to paraphrase assistant responses while keeping
    user messages and system prompts mostly unchanged.
    
    This maintains conversation flow while adding variation.
    """
    # Create copy
    new_episode = episode.copy()
    new_episode['messages'] = []
    new_episode['augmented'] = True
    new_episode['augmentation_method'] = 'model_assisted_paraphrase'
    new_episode['source_id'] = episode.get('id', 'unknown')
    
    # Keep system field at top level (don't modify)
    # It should NOT be in the messages array
    
    messages = episode.get('messages', [])
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        
        if role == 'system':
            # Skip - system should be at top level, not in messages
            # This shouldn't happen if load is correct, but handle it
            continue
        
        elif role == 'user':
            # Light paraphrasing for user messages
            paraphrased = paraphrase_message_simple(msg, role)
            new_episode['messages'].append(paraphrased)
        
        elif role == 'assistant':
            # Use model to generate a paraphrase of assistant response
            # Build context up to this point
            context = ""
            
            # Add system prompt first (from top level field)
            if 'system' in new_episode and new_episode['system']:
                context += f"<myPT_system>{new_episode['system']}</myPT_system>\n"
            
            # Add conversation history
            for prev_msg in new_episode['messages']:
                r = prev_msg['role']
                c = prev_msg['content']
                if r == 'user':
                    context += f"<myPT_user>{c}</myPT_user>\n"
                elif r == 'assistant':
                    context += f"<myPT_assistant>{c}</myPT_assistant>\n"
            
            # Add paraphrasing instruction
            paraphrase_instruction = (
                f"<myPT_user>Please rephrase your previous response while "
                f"keeping the same meaning and information.</myPT_user>\n"
                f"<myPT_assistant>"
            )
            
            full_prompt = context + paraphrase_instruction
            
            try:
                # Generate paraphrase
                output = model.generate(
                    full_prompt,
                    max_new_tokens=200,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.9
                )
                
                # Extract generated part
                if full_prompt in output:
                    generated = output[len(full_prompt):]
                    # Remove closing tag if present
                    if '</myPT_assistant>' in generated:
                        generated = generated.split('</myPT_assistant>')[0]
                    
                    paraphrased_content = generated.strip()
                    
                    # Quality check: not too short, not too long
                    original_len = len(msg['content'].split())
                    new_len = len(paraphrased_content.split())
                    
                    # Accept if within 50%-200% of original length
                    if 0.5 <= (new_len / max(original_len, 1)) <= 2.0:
                        new_episode['messages'].append({
                            'role': 'assistant',
                            'content': paraphrased_content
                        })
                    else:
                        # Use original if paraphrase is bad quality
                        new_episode['messages'].append(msg.copy())
                        new_episode['paraphrase_failed'] = True
                else:
                    # Fallback to original
                    new_episode['messages'].append(msg.copy())
                    new_episode['paraphrase_failed'] = True
            
            except Exception as e:
                print(f"Warning: Paraphrase generation failed: {e}")
                new_episode['messages'].append(msg.copy())
                new_episode['paraphrase_failed'] = True
    
    return new_episode


def paraphrase_episode_simple(episode: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple rule-based paraphrasing without model.
    Less quality but faster.
    """
    new_episode = episode.copy()
    new_episode['messages'] = []
    new_episode['augmented'] = True
    new_episode['augmentation_method'] = 'rule_based_paraphrase'
    new_episode['source_id'] = episode.get('id', 'unknown')
    
    # Keep system field at top level unchanged
    # Don't add it to messages array
    
    for msg in episode.get('messages', []):
        role = msg.get('role', '')
        
        if role == 'system':
            # Skip - system should stay at top level only
            continue
        else:
            # Light paraphrasing
            paraphrased = paraphrase_message_simple(msg, role)
            new_episode['messages'].append(paraphrased)
    
    return new_episode


def augment_dataset(
    episodes: List[Dict[str, Any]],
    target_count: int,
    model=None,
    use_model: bool = True,
    lang_filter: str = 'both'
) -> List[Dict[str, Any]]:
    """
    Augment dataset to reach target count.
    
    Args:
        episodes: Original episodes
        target_count: Desired total count
        model: Model for paraphrasing (optional)
        use_model: Whether to use model-assisted paraphrasing
        lang_filter: 'en', 'de', or 'both'
    """
    # Filter by language if needed
    if lang_filter != 'both':
        filtered = [ep for ep in episodes if ep.get('language') == lang_filter]
        print(f"Filtered to {len(filtered)} {lang_filter} episodes")
        episodes = filtered
    
    original_count = len(episodes)
    needed = target_count - original_count
    
    if needed <= 0:
        print(f"Already have {original_count} episodes (target: {target_count})")
        return episodes
    
    print(f"\nAugmenting dataset:")
    print(f"  Original: {original_count} episodes")
    print(f"  Target:   {target_count} episodes")
    print(f"  Need:     {needed} new episodes")
    print(f"  Method:   {'Model-assisted' if use_model and model else 'Rule-based'}")
    
    # Start with originals
    augmented = episodes.copy()
    
    # Generate needed paraphrases
    for i in range(needed):
        # Select a random episode to paraphrase
        source_episode = random.choice(episodes)
        
        # Generate paraphrase
        if use_model and model is not None:
            new_episode = paraphrase_episode_model_assisted(source_episode, model)
        else:
            new_episode = paraphrase_episode_simple(source_episode)
        
        # Assign new ID
        new_episode['id'] = f"aug_{i:05d}"
        
        augmented.append(new_episode)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{needed} paraphrases...")
    
    print(f"✓ Augmentation complete: {len(augmented)} total episodes")
    return augmented


def save_augmented_dataset(
    episodes: List[Dict[str, Any]],
    output_path: str,
    review_mode: bool = False
):
    """
    Save augmented episodes.
    Default: JSONL file output (recommended for training pipelines)
    Alternative: Directory with individual JSON files (if path has no extension)
    """
    path = Path(output_path)
    
    # Add review marker if in review mode
    if review_mode:
        for episode in episodes:
            if episode.get('augmented'):
                episode['needs_review'] = True
    
    # Default to JSONL if no extension specified
    if not path.suffix:
        output_path = output_path + '.jsonl'
        path = Path(output_path)
        print(f"Note: No extension provided, defaulting to JSONL: {output_path}")
    
    # Case 1: Save as JSONL file (default and recommended)
    if output_path.endswith('.jsonl'):
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving as JSONL file: {output_path}")
        with open(path, 'w', encoding='utf-8') as f:
            for episode in episodes:
                json_line = json.dumps(episode, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"✓ Saved {len(episodes)} episodes to {output_path}")
    
    # Case 2: Save as individual JSON files in directory
    else:
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving as individual JSON files to: {output_path}")
        for i, episode in enumerate(episodes):
            episode_id = episode.get('id', f'unknown_{random.randint(1000, 9999)}')
            # Sanitize filename: remove/replace invalid characters
            safe_id = str(episode_id).replace('\n', '_').replace('\r', '_')
            safe_id = safe_id.replace('/', '_').replace('\\', '_')
            safe_id = safe_id.replace(':', '-').replace('*', '_')
            safe_id = safe_id.replace('?', '_').replace('"', '_')
            safe_id = safe_id.replace('<', '_').replace('>', '_')
            safe_id = safe_id.replace('|', '_')
            # Truncate if too long (Windows path limit)
            if len(safe_id) > 100:
                safe_id = safe_id[:100]
            output_file = path / f"{safe_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(episode, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(episodes)} episodes to {output_path}")
    
    # Save metadata
    metadata = {
        'total_episodes': len(episodes),
        'original_episodes': sum(1 for ep in episodes if not ep.get('augmented')),
        'augmented_episodes': sum(1 for ep in episodes if ep.get('augmented')),
        'needs_review': sum(1 for ep in episodes if ep.get('needs_review')),
    }
    
    if output_path.endswith('.jsonl'):
        metadata_file = path.parent / f"{path.stem}_metadata.json"
    else:
        metadata_file = path / 'augmentation_metadata.json'
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to {metadata_file}")


def main():
    from core.banner import print_banner
    print_banner("MyPT Episode Augmentation", "Paraphrase-Based Dataset Expansion")
    
    parser = argparse.ArgumentParser(description="Paraphrase augmentation for conversation episodes")
    parser.add_argument('--input', required=True, help='Input JSONL file or directory with gold episodes')
    parser.add_argument('--output', required=True, help='Output JSONL file (recommended) or directory')
    parser.add_argument('--model', help='Model checkpoint to use for paraphrasing (optional)')
    parser.add_argument('--target_count', type=int, default=1200, help='Target total episode count (default: 1200)')
    parser.add_argument('--lang_filter', choices=['en', 'de', 'both'], default='both', help='Language filter')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model generation')
    parser.add_argument('--review_mode', action='store_true', help='Mark augmented episodes for human review')
    parser.add_argument('--no_model', action='store_true', help='Use rule-based paraphrasing only (faster but lower quality)')
    
    args = parser.parse_args()
    
    # Load original episodes
    print(f"\nLoading episodes from {args.input}...")
    episodes = load_episodes(args.input)
    
    if not episodes:
        print("Error: No episodes found!")
        return
    
    # Analyze original dataset
    original_stats = analyze_dataset(episodes)
    print_dataset_stats(original_stats, "Original Dataset")
    
    # Load model if requested
    model = None
    use_model = not args.no_model
    
    if use_model and args.model:
        print(f"\nLoading model from {args.model}...")
        try:
            model = load_model(args.model)
            model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load model: {e}")
            print("Falling back to rule-based paraphrasing")
            use_model = False
    elif use_model:
        print("\nWarning: --model not specified, using rule-based paraphrasing")
        use_model = False
    
    # Augment dataset
    augmented_episodes = augment_dataset(
        episodes,
        target_count=args.target_count,
        model=model,
        use_model=use_model,
        lang_filter=args.lang_filter
    )
    
    # Analyze augmented dataset
    augmented_stats = analyze_dataset(augmented_episodes)
    print_dataset_stats(augmented_stats, "Augmented Dataset")
    
    # Save results
    save_augmented_dataset(augmented_episodes, args.output, args.review_mode)
    
    print("\n" + "="*60)
    print("  Augmentation Complete!")
    print("="*60)
    print(f"\nNext steps:")
    if args.review_mode:
        print(f"1. Review augmented episodes in {args.output}")
        print(f"   - Look for episodes with 'needs_review': true")
        print(f"   - Check quality, fix any issues")
        print(f"2. Run prepare_chat_sft.py on reviewed dataset")
    else:
        print(f"1. Run prepare_chat_sft.py on {args.output}")
    print(f"2. Train with augmented dataset")
    print()


if __name__ == "__main__":
    main()

