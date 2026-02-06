#!/usr/bin/env python3
"""
Generate operator learning dataset with UNIQUE payloads and template splits.

This dataset tests whether the model can learn ABSTRACT operators, not memorize pairs.

Key properties:
1. Every payload appears exactly ONCE in train (no memorization possible)
2. Val/test use DIFFERENT templates than train (tests generalization)
3. Exact-match metric: transformation either matches or doesn't
4. Mechanical operators only: COPY, WRAP, EXTRACT
5. Multi-word payloads (1-4 words) with configurable distribution

Operators:
- COPY: "Repeat exactly: {X}" → "{X}"
- WRAP: "Wrap in brackets: {X}" → "[{X}]"
- EXTRACT: 'Return text between quotes: "{X}"' → "{X}"

Usage:
    # Default: multi-word payloads (35% 1-word, 30% 2-word, 20% 3-word, 15% 4-word)
    python scripts/generate_operator_dataset.py --output_dir data/sft_operator
    
    # Single-word only (legacy mode)
    python scripts/generate_operator_dataset.py --output_dir data/sft_operator --max_words 1
    
    # Custom distribution
    python scripts/generate_operator_dataset.py --output_dir data/sft_operator --word_dist "0.25,0.25,0.25,0.25"
"""

import argparse
import json
import random
import string
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# =============================================================================
# TEMPLATE DEFINITIONS - STRICTLY SPLIT BETWEEN TRAIN AND VAL
# =============================================================================

# COPY operator templates (mix of with/without colons to avoid bias)
COPY_TEMPLATES_TRAIN = [
    # With colon
    "Repeat exactly: {X}",
    "Say this back: {X}",
    "Copy: {X}",
    "Echo: {X}",
    # Without colon
    "Repeat exactly {X}",
    "Say back {X}",
    "Copy {X}",
    "Echo {X}",
    "Output {X}",
    "Just say {X}",
]

COPY_TEMPLATES_VAL = [
    # With colon
    "Parrot this: {X}",
    "Return verbatim: {X}",
    # Without colon
    "Parrot {X}",
    "Mirror {X}",
    "Reproduce {X}",
    "Output exactly {X}",
]

# WRAP operator templates (wrap in square brackets, mix with/without colons)
WRAP_TEMPLATES_TRAIN = [
    # With colon
    "Wrap in brackets: {X}",
    "Put square brackets around: {X}",
    "Surround with []: {X}",
    # Without colon
    "Wrap in brackets {X}",
    "Add brackets around {X}",
    "Bracket {X}",
    "Enclose in [] {X}",
]

WRAP_TEMPLATES_VAL = [
    # With colon
    "Put in square brackets: {X}",
    # Without colon
    "Wrap with [] {X}",
    "Add [] around {X}",
    "Put brackets around {X}",
]

# EXTRACT operator templates (extract from quotes, mix with/without colons)
EXTRACT_TEMPLATES_TRAIN = [
    # With colon
    'Return only the text between quotes: "{X}"',
    'Extract the quoted part: "{X}"',
    'Get what\'s in quotes: "{X}"',
    # Without colon (question style)
    'What is inside "{X}"?',
    'Extract "{X}"',
    'Return the content of "{X}"',
    'Pull out "{X}"',
]

EXTRACT_TEMPLATES_VAL = [
    # With colon
    'Output the quoted content: "{X}"',
    # Without colon
    'What\'s inside "{X}"?',
    'Get the text from "{X}"',
    'Return what\'s in "{X}"',
]

# German variants (all go to train, German val uses same templates - tests language, not template)
COPY_TEMPLATES_TRAIN_DE = [
    "Wiederhole genau: {X}",
    "Sage das zurueck: {X}",
    "Kopiere: {X}",
    "Gib aus: {X}",
]

WRAP_TEMPLATES_TRAIN_DE = [
    "Setze in Klammern: {X}",
    "Umgib mit []: {X}",
    "Klammere: {X}",
]

EXTRACT_TEMPLATES_TRAIN_DE = [
    'Gib nur den Text in Anfuehrungszeichen aus: "{X}"',
    'Extrahiere das Zitat: "{X}"',
]


# =============================================================================
# PAYLOAD GENERATION - ALL UNIQUE
# =============================================================================

# Word lists for payload generation
COMMON_WORDS = [
    "apple", "banana", "cherry", "dog", "elephant", "forest", "garden", "house",
    "island", "jungle", "kitchen", "lemon", "mountain", "night", "ocean", "paper",
    "queen", "river", "sunset", "tower", "umbrella", "village", "window", "yellow",
    "zebra", "anchor", "bridge", "castle", "desert", "engine", "falcon", "glacier",
    "harbor", "ivory", "jasmine", "knight", "lantern", "marble", "nebula", "olive",
    "phoenix", "quartz", "rainbow", "silver", "thunder", "uranium", "velvet", "whisper",
    "crystal", "dragon", "emerald", "flame", "golden", "horizon", "infinite", "journey",
    "kingdom", "liberty", "mystery", "northern", "optical", "paradise", "quantum", "radiant",
    "sapphire", "twilight", "universe", "victory", "wonder", "xenon", "yearning", "zenith",
]

GERMAN_WORDS = [
    "Apfel", "Birne", "Kirsche", "Hund", "Elefant", "Wald", "Garten", "Haus",
    "Insel", "Dschungel", "Kueche", "Zitrone", "Berg", "Nacht", "Ozean", "Papier",
    "Koenigin", "Fluss", "Sonnenuntergang", "Turm", "Regenschirm", "Dorf", "Fenster",
    "Schmetterling", "Donner", "Blitz", "Wolke", "Stern", "Mond", "Sonne",
]

# Syllables for generating novel words (BPE-friendly)
SYLLABLES = [
    "ba", "be", "bi", "bo", "bu", "da", "de", "di", "do", "du",
    "fa", "fe", "fi", "fo", "fu", "ga", "ge", "gi", "go", "gu",
    "ka", "ke", "ki", "ko", "ku", "la", "le", "li", "lo", "lu",
    "ma", "me", "mi", "mo", "mu", "na", "ne", "ni", "no", "nu",
    "pa", "pe", "pi", "po", "pu", "ra", "re", "ri", "ro", "ru",
    "sa", "se", "si", "so", "su", "ta", "te", "ti", "to", "tu",
    "va", "ve", "vi", "vo", "vu", "wa", "we", "wi", "wo", "za",
    "zel", "zon", "zar", "tex", "nex", "pix", "lux", "max", "dex",
    "tron", "plex", "flux", "grid", "node", "core", "link", "sync",
]


def generate_unique_payloads(
    count: int, 
    seed: int, 
    exclude: Set[str] = None,
    tokenizer = None,
    max_tokens: int = 12,
    max_words: int = 4,
    word_distribution: Tuple[float, ...] = (0.35, 0.30, 0.20, 0.15),
) -> List[str]:
    """
    Generate unique payloads that never repeat, optionally filtered by BPE token count.
    
    Args:
        count: Number of payloads to generate
        seed: Random seed for reproducibility
        exclude: Set of payloads to exclude (for train/val separation)
        tokenizer: Tokenizer for BPE filtering (optional)
        max_tokens: Maximum BPE tokens per payload (default: 12 for multi-word)
        max_words: Maximum words per payload (1-4, default: 4)
        word_distribution: Probability for (1-word, 2-word, 3-word, 4-word) payloads
                          Default: (0.35, 0.30, 0.20, 0.15)
    
    Returns:
        List of unique payload strings
    """
    random.seed(seed)
    exclude = exclude or set()
    payloads = []
    seen = set(exclude)
    
    # Normalize word distribution to max_words
    word_probs = list(word_distribution[:max_words])
    total = sum(word_probs)
    word_probs = [p / total for p in word_probs]
    
    # Strategy generators by word count
    def gen_1_word():
        """Single word strategies"""
        strats = [
            lambda: random.choice(COMMON_WORDS),
            lambda: random.choice(GERMAN_WORDS),
            lambda: ''.join(random.choices(SYLLABLES, k=random.randint(2, 3))),  # Novel word
            lambda: f"{random.choice(COMMON_WORDS)}{random.randint(10, 99)}",  # Word + number
            lambda: f"{random.choice(COMMON_WORDS).upper()}",  # Uppercase
            lambda: f"{random.choice(COMMON_WORDS).capitalize()}",  # Capitalized
            lambda: f"{random.randint(100, 9999)}",  # Just a number
        ]
        return random.choice(strats)()
    
    def gen_2_words():
        """Two word strategies"""
        strats = [
            lambda: f"{random.choice(COMMON_WORDS)} {random.choice(COMMON_WORDS)}",
            lambda: f"{random.choice(COMMON_WORDS)} {random.randint(1, 100)}",
            lambda: f"{random.choice(GERMAN_WORDS)} {random.choice(GERMAN_WORDS)}",
            lambda: f"{random.choice(COMMON_WORDS).upper()} {random.choice(COMMON_WORDS)}",
            lambda: f"{random.choice(SYLLABLES)}{random.choice(SYLLABLES)} {random.choice(COMMON_WORDS)}",
            lambda: f"the {random.choice(COMMON_WORDS)}",
            lambda: f"{random.choice(['big', 'small', 'red', 'blue', 'old', 'new'])} {random.choice(COMMON_WORDS)}",
        ]
        return random.choice(strats)()
    
    def gen_3_words():
        """Three word strategies"""
        strats = [
            lambda: ' '.join(random.choices(COMMON_WORDS, k=3)),
            lambda: f"the {random.choice(COMMON_WORDS)} {random.choice(COMMON_WORDS)}",
            lambda: f"{random.choice(COMMON_WORDS)} and {random.choice(COMMON_WORDS)}",
            lambda: f"{random.choice(['big', 'small', 'red', 'blue'])} {random.choice(COMMON_WORDS)} {random.randint(1, 99)}",
            lambda: f"{random.choice(GERMAN_WORDS)} {random.choice(GERMAN_WORDS)} {random.choice(GERMAN_WORDS)}",
            lambda: f"a {random.choice(COMMON_WORDS)} {random.choice(['house', 'tree', 'river', 'mountain'])}",
        ]
        return random.choice(strats)()
    
    def gen_4_words():
        """Four word strategies"""
        strats = [
            lambda: ' '.join(random.choices(COMMON_WORDS, k=4)),
            lambda: f"the {random.choice(COMMON_WORDS)} {random.choice(['and', 'or', 'with'])} {random.choice(COMMON_WORDS)}",
            lambda: f"{random.choice(COMMON_WORDS)} {random.choice(COMMON_WORDS)} {random.choice(COMMON_WORDS)} {random.randint(1, 99)}",
            lambda: f"a {random.choice(['big', 'small', 'red', 'blue'])} {random.choice(COMMON_WORDS)} {random.choice(COMMON_WORDS)}",
            lambda: f"{random.choice(GERMAN_WORDS)} {random.choice(GERMAN_WORDS)} und {random.choice(GERMAN_WORDS)}",
        ]
        return random.choice(strats)()
    
    word_generators = [gen_1_word, gen_2_words, gen_3_words, gen_4_words][:max_words]
    
    attempts = 0
    max_attempts = count * 30  # More attempts since we filter
    rejected_bpe = 0
    rejected_duplicate = 0
    
    # Track word count distribution for reporting
    word_count_stats = [0] * max_words
    
    while len(payloads) < count and attempts < max_attempts:
        attempts += 1
        
        # Select word count based on distribution
        word_count_idx = random.choices(range(len(word_probs)), weights=word_probs, k=1)[0]
        generator = word_generators[word_count_idx]
        payload = generator()
        
        # Skip duplicates
        if payload in seen:
            rejected_duplicate += 1
            continue
        
        # BPE token count filter (if tokenizer provided)
        if tokenizer is not None:
            tokens = tokenizer.encode(payload)
            if len(tokens) > max_tokens:
                rejected_bpe += 1
                continue
        
        seen.add(payload)
        payloads.append(payload)
        word_count_stats[word_count_idx] += 1
    
    if len(payloads) < count:
        # Fill remaining with guaranteed-unique short strings
        print(f"  Warning: Needed fallback strings. BPE rejected: {rejected_bpe}, duplicates: {rejected_duplicate}")
        fallback_count = 0
        while len(payloads) < count:
            # Generate unique fallback: word + unique number
            base = random.choice(COMMON_WORDS)
            payload = f"{base}_{fallback_count:04d}"
            fallback_count += 1
            if payload not in seen:
                if tokenizer is None or len(tokenizer.encode(payload)) <= max_tokens:
                    seen.add(payload)
                    payloads.append(payload)
    
    # Report distribution
    total_generated = sum(word_count_stats)
    if total_generated > 0:
        dist_str = ", ".join([f"{i+1}w:{c}({100*c/total_generated:.0f}%)" for i, c in enumerate(word_count_stats)])
        print(f"  Payload distribution: {dist_str}")
    
    random.shuffle(payloads)
    return payloads


# =============================================================================
# OPERATOR TRANSFORMATIONS
# =============================================================================

def apply_operator(operator: str, payload: str) -> str:
    """Apply operator transformation to payload."""
    if operator == "COPY":
        return payload
    elif operator == "WRAP":
        return f"[{payload}]"
    elif operator == "EXTRACT":
        return payload  # Input has quotes, output doesn't
    else:
        raise ValueError(f"Unknown operator: {operator}")


def format_question(template: str, payload: str) -> str:
    """Format question with payload."""
    return template.replace("{X}", payload)


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_operator_pairs(
    payloads: List[str],
    operator: str,
    templates: List[str],
    language: str = "en"
) -> List[Dict]:
    """Generate (question, answer, metadata) for an operator."""
    pairs = []
    
    for payload in payloads:
        template = random.choice(templates)
        question = format_question(template, payload)
        answer = apply_operator(operator, payload)
        
        pairs.append({
            "question": question,
            "answer": answer,
            "operator": operator,
            "template": template,
            "payload": payload,
            "language": language,
        })
    
    return pairs


def create_episode(question: str, answer: str, language: str = "en") -> Dict:
    """Create a single episode in MyPT format."""
    if language == "de":
        system_prompt = "Du bist MyPT. Fuehre die Anweisung genau aus."
    else:
        system_prompt = "You are MyPT. Execute the instruction exactly."
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }


def generate_dataset(
    n_train: int = 5000,
    n_val: int = 500,
    seed_train: int = 42,
    seed_val: int = 12345,
    include_german: bool = True,
    tokenizer = None,
    max_tokens: int = 12,
    max_words: int = 4,
    word_distribution: Tuple[float, ...] = (0.35, 0.30, 0.20, 0.15),
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Generate train and val datasets with strict separation.
    
    Args:
        n_train: Number of training examples
        n_val: Number of validation examples
        seed_train: Random seed for training data
        seed_val: Random seed for validation data (must differ)
        include_german: Include German language variants
        tokenizer: Tokenizer for BPE filtering
        max_tokens: Maximum BPE tokens per payload
        max_words: Maximum words per payload (1-4)
        word_distribution: Probability for (1w, 2w, 3w, 4w) payloads
    
    Returns:
        train_episodes: List of training episodes
        val_episodes: List of validation episodes (DIFFERENT templates)
        metadata: Dataset statistics
    """
    
    # Calculate per-operator counts
    n_operators = 3
    n_train_per_op = n_train // n_operators
    n_val_per_op = n_val // n_operators
    
    # Generate UNIQUE payloads for train (BPE-filtered if tokenizer provided)
    print(f"  Generating {n_train} unique train payloads (max {max_words} words, max {max_tokens} BPE tokens)...")
    train_payloads = generate_unique_payloads(
        n_train, seed_train, 
        tokenizer=tokenizer, max_tokens=max_tokens,
        max_words=max_words, word_distribution=word_distribution
    )
    
    # Generate UNIQUE payloads for val (different seed, no overlap)
    print(f"  Generating {n_val} unique val payloads...")
    val_payloads = generate_unique_payloads(
        n_val, seed_val, exclude=set(train_payloads), 
        tokenizer=tokenizer, max_tokens=max_tokens,
        max_words=max_words, word_distribution=word_distribution
    )
    
    # Verify no overlap
    train_set = set(train_payloads)
    val_set = set(val_payloads)
    overlap = train_set & val_set
    assert len(overlap) == 0, f"Payload overlap detected: {overlap}"
    
    # Split payloads by operator
    random.seed(seed_train)
    random.shuffle(train_payloads)
    random.seed(seed_val)
    random.shuffle(val_payloads)
    
    train_pairs = []
    val_pairs = []
    
    # COPY operator
    copy_train = generate_operator_pairs(
        train_payloads[:n_train_per_op],
        "COPY",
        COPY_TEMPLATES_TRAIN,
        "en"
    )
    copy_val = generate_operator_pairs(
        val_payloads[:n_val_per_op],
        "COPY",
        COPY_TEMPLATES_VAL,  # DIFFERENT templates
        "en"
    )
    train_pairs.extend(copy_train)
    val_pairs.extend(copy_val)
    
    # WRAP operator
    wrap_train = generate_operator_pairs(
        train_payloads[n_train_per_op:2*n_train_per_op],
        "WRAP",
        WRAP_TEMPLATES_TRAIN,
        "en"
    )
    wrap_val = generate_operator_pairs(
        val_payloads[n_val_per_op:2*n_val_per_op],
        "WRAP",
        WRAP_TEMPLATES_VAL,  # DIFFERENT templates
        "en"
    )
    train_pairs.extend(wrap_train)
    val_pairs.extend(wrap_val)
    
    # EXTRACT operator
    extract_train = generate_operator_pairs(
        train_payloads[2*n_train_per_op:3*n_train_per_op],
        "EXTRACT",
        EXTRACT_TEMPLATES_TRAIN,
        "en"
    )
    extract_val = generate_operator_pairs(
        val_payloads[2*n_val_per_op:3*n_val_per_op],
        "EXTRACT",
        EXTRACT_TEMPLATES_VAL,  # DIFFERENT templates
        "en"
    )
    train_pairs.extend(extract_train)
    val_pairs.extend(extract_val)
    
    # Add German variants to train (optional)
    if include_german:
        n_german = n_train // 10  # 10% German
        german_payloads = generate_unique_payloads(
            n_german * 3, seed_train + 1000, exclude=train_set | val_set,
            tokenizer=tokenizer, max_tokens=max_tokens,
            max_words=max_words, word_distribution=word_distribution
        )
        
        n_per_op = n_german
        german_copy = generate_operator_pairs(
            german_payloads[:n_per_op],
            "COPY",
            COPY_TEMPLATES_TRAIN_DE,
            "de"
        )
        german_wrap = generate_operator_pairs(
            german_payloads[n_per_op:2*n_per_op],
            "WRAP",
            WRAP_TEMPLATES_TRAIN_DE,
            "de"
        )
        german_extract = generate_operator_pairs(
            german_payloads[2*n_per_op:3*n_per_op],
            "EXTRACT",
            EXTRACT_TEMPLATES_TRAIN_DE,
            "de"
        )
        train_pairs.extend(german_copy)
        train_pairs.extend(german_wrap)
        train_pairs.extend(german_extract)
    
    # Shuffle
    random.seed(seed_train)
    random.shuffle(train_pairs)
    random.seed(seed_val)
    random.shuffle(val_pairs)
    
    # Convert to episodes
    train_episodes = []
    for pair in train_pairs:
        episode = create_episode(pair["question"], pair["answer"], pair["language"])
        episode["_meta"] = {
            "operator": pair["operator"],
            "payload": pair["payload"],
            "expected": pair["answer"],
        }
        train_episodes.append(episode)
    
    val_episodes = []
    for pair in val_pairs:
        episode = create_episode(pair["question"], pair["answer"], pair["language"])
        episode["_meta"] = {
            "operator": pair["operator"],
            "payload": pair["payload"],
            "expected": pair["answer"],
        }
        val_episodes.append(episode)
    
    # Metadata
    metadata = {
        "n_train": len(train_episodes),
        "n_val": len(val_episodes),
        "operators": ["COPY", "WRAP", "EXTRACT"],
        "train_templates": {
            "COPY": COPY_TEMPLATES_TRAIN,
            "WRAP": WRAP_TEMPLATES_TRAIN,
            "EXTRACT": EXTRACT_TEMPLATES_TRAIN,
        },
        "val_templates": {
            "COPY": COPY_TEMPLATES_VAL,
            "WRAP": WRAP_TEMPLATES_VAL,
            "EXTRACT": EXTRACT_TEMPLATES_VAL,
        },
        "template_overlap": False,
        "payload_overlap": False,
        "seed_train": seed_train,
        "seed_val": seed_val,
        "max_tokens": max_tokens,
        "max_words": max_words,
        "word_distribution": word_distribution,
    }
    
    return train_episodes, val_episodes, metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate operator learning dataset")
    parser.add_argument("--output_dir", type=str, default="data/sft_operator",
                        help="Output directory")
    parser.add_argument("--n_train", type=int, default=5000,
                        help="Number of training examples")
    parser.add_argument("--n_val", type=int, default=500,
                        help="Number of validation examples")
    parser.add_argument("--seed_train", type=int, default=42,
                        help="Random seed for training data")
    parser.add_argument("--seed_val", type=int, default=12345,
                        help="Random seed for validation data (MUST differ from train)")
    parser.add_argument("--no_german", action="store_true",
                        help="Exclude German examples")
    parser.add_argument("--max_tokens", type=int, default=12,
                        help="Max BPE tokens per payload (default: 12, for multi-word support)")
    parser.add_argument("--max_words", type=int, default=4, choices=[1, 2, 3, 4],
                        help="Max words per payload (1-4, default: 4)")
    parser.add_argument("--word_dist", type=str, default="0.35,0.30,0.20,0.15",
                        help="Word count distribution (comma-separated, default: 0.35,0.30,0.20,0.15 for 1w,2w,3w,4w)")
    parser.add_argument("--no_bpe_filter", action="store_true",
                        help="Disable BPE token filtering (not recommended)")
    args = parser.parse_args()
    
    # Parse word distribution
    word_distribution = tuple(float(x) for x in args.word_dist.split(","))
    if len(word_distribution) < args.max_words:
        # Pad with zeros
        word_distribution = word_distribution + (0.0,) * (args.max_words - len(word_distribution))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  OPERATOR DATASET GENERATOR")
    print("  Testing abstract operator learning with unique payloads")
    print("=" * 60)
    print()
    
    # Load tokenizer for BPE filtering
    tokenizer = None
    if not args.no_bpe_filter:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.tokenizer import Tokenizer
        from core.model import GPTConfig
        config = GPTConfig(vocab_size=50304)
        tokenizer = Tokenizer(config, "gpt2")
        print(f"  BPE filtering enabled (max {args.max_tokens} tokens per payload)")
    else:
        print("  WARNING: BPE filtering disabled")
    
    print(f"\nConfiguration:")
    print(f"  Train examples: {args.n_train}")
    print(f"  Val examples: {args.n_val}")
    print(f"  Train seed: {args.seed_train}")
    print(f"  Val seed: {args.seed_val}")
    print(f"  Include German: {not args.no_german}")
    print(f"  Max BPE tokens: {args.max_tokens}")
    print(f"  Max words: {args.max_words}")
    print(f"  Word distribution: {word_distribution[:args.max_words]}")
    print()
    
    # Generate dataset
    train_episodes, val_episodes, metadata = generate_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        seed_train=args.seed_train,
        seed_val=args.seed_val,
        include_german=not args.no_german,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        max_words=args.max_words,
        word_distribution=word_distribution,
    )
    
    # Write train file
    train_file = output_dir / "operator_train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for episode in train_episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    # Write val file
    val_file = output_dir / "operator_val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for episode in val_episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    # Write metadata
    meta_file = output_dir / "dataset_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    # Statistics
    print("Dataset generated:")
    print(f"  Train: {len(train_episodes)} episodes → {train_file}")
    print(f"  Val: {len(val_episodes)} episodes → {val_file}")
    print(f"  Metadata: {meta_file}")
    print()
    
    # Operator breakdown
    train_ops = defaultdict(int)
    val_ops = defaultdict(int)
    for ep in train_episodes:
        train_ops[ep["_meta"]["operator"]] += 1
    for ep in val_episodes:
        val_ops[ep["_meta"]["operator"]] += 1
    
    print("Operator breakdown (train):")
    for op, count in sorted(train_ops.items()):
        print(f"  {op}: {count}")
    
    print("\nOperator breakdown (val):")
    for op, count in sorted(val_ops.items()):
        print(f"  {op}: {count}")
    
    # Sample examples
    print("\n" + "=" * 60)
    print("Sample TRAIN examples:")
    print("=" * 60)
    for ep in train_episodes[:6]:
        meta = ep["_meta"]
        q = ep["messages"][1]["content"]
        a = ep["messages"][2]["content"]
        print(f"\n[{meta['operator']}] payload='{meta['payload']}'")
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print(f"  Expected: {meta['expected']}")
    
    print("\n" + "=" * 60)
    print("Sample VAL examples (DIFFERENT TEMPLATES!):")
    print("=" * 60)
    for ep in val_episodes[:6]:
        meta = ep["_meta"]
        q = ep["messages"][1]["content"]
        a = ep["messages"][2]["content"]
        print(f"\n[{meta['operator']}] payload='{meta['payload']}'")
        print(f"  Q: {q}")
        print(f"  A: {a}")
        print(f"  Expected: {meta['expected']}")
    
    # Training recommendations
    print("\n" + "=" * 60)
    print("TRAINING RECOMMENDATIONS")
    print("=" * 60)
    batch_size = 16
    batches_per_epoch = len(train_episodes) // batch_size
    recommended_iters = int(batches_per_epoch * 3.0)  # 3x coverage
    
    print(f"""
Dataset size: {len(train_episodes)} episodes
Batch size: {batch_size}
Batches per epoch: {batches_per_epoch}

Recommended max_iters (3x coverage): {recommended_iters}
Absolute max (3.5x coverage): {int(batches_per_epoch * 3.5)}

DO NOT EXCEED 3.5x coverage - overtraining causes brittleness!

Next steps:
  1. Prepare: python scripts/prepare_chat_sft.py --input {train_file} --output_dir {output_dir}/prepared --val_file {val_file}
  2. Train:   python train.py --model_name phase3a_operator --init_from_model phase3a1_alpha_v2 --dataset_dir {output_dir}/prepared --max_iters {recommended_iters}
  3. Eval:    python scripts/eval_operator.py --model phase3a_operator
""")


if __name__ == "__main__":
    main()
