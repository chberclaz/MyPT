#!/usr/bin/env python3
"""
Generate operator learning dataset with CROSS-OPERATOR contrastive design.

This dataset tests whether the model can learn ABSTRACT operators, not memorize pairs.

Key design: Every payload appears in ALL 3 operators (COPY, WRAP, EXTRACT),
each with multiple template phrasings. This creates two contrastive signals:

  Signal 1: Same payload + same operator + different template = same output
    → "sunset" with COPY template A → "sunset"
    → "sunset" with COPY template B → "sunset"
    → Forces model to ignore template surface form

  Signal 2: Same payload + different operator = different output
    → "sunset" with COPY → "sunset"
    → "sunset" with WRAP → "[sunset]"
    → "sunset" with EXTRACT → "sunset"
    → Forces model to attend to operator instruction

Properties:
1. Every payload appears in ALL 3 operators (cross-operator)
2. Each operator gets N different template phrasings per payload
3. Val uses DIFFERENT templates AND DIFFERENT payloads (clean eval)
4. Exact-match metric: transformation either matches or doesn't
5. Multi-word payloads (1-4 words) with configurable distribution

Operators:
- COPY: "Repeat exactly: {X}" → "{X}"
- WRAP: "Wrap in brackets: {X}" → "[{X}]"
- EXTRACT: 'Return text between quotes: "{X}"' → "{X}"

Usage:
    # Default: 16500 payloads × 3 ops × 2 templates = ~99k episodes
    python scripts/generate_operator_dataset.py --output_dir data/sft_operator_v3

    # Fewer payloads, more template diversity
    python scripts/generate_operator_dataset.py --output_dir data/sft_operator_v3 \\
        --n_train 11000 --templates_per_op 3

    # Legacy mode: each payload in only 1 operator, 1 template
    python scripts/generate_operator_dataset.py --output_dir data/sft_operator \\
        --templates_per_op 1 --no_cross_operator
"""

import argparse
import json
import random
import string
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from core.dataset_lineage import iso_now, write_lineage_sidecar

# =============================================================================
# TEMPLATE DEFINITIONS - STRICTLY SPLIT BETWEEN TRAIN AND VAL
# =============================================================================

# COPY operator templates - USE VARIED DELIMITERS to teach abstract "delimited content"
# Mix of: backticks `, single quotes ', pipes |, curly braces {}, asterisks *
COPY_TEMPLATES_TRAIN = [
    # Backticks
    "Repeat exactly: `{X}`",
    "Copy: `{X}`",
    "Echo `{X}`",
    # Single quotes
    "Say this back: '{X}'",
    "Output: '{X}'",
    "Just say '{X}'",
    # Pipes
    "Repeat exactly |{X}|",
    "Copy |{X}|",
    # Curly braces (different from template placeholder - these are literal)
    "Echo: {{{X}}}",
    "Say back {{{X}}}",
]

COPY_TEMPLATES_VAL = [
    # Mix delimiters in val too
    "Parrot this: `{X}`",
    "Return verbatim: '{X}'",
    "Parrot |{X}|",
    "Mirror `{X}`",
    "Reproduce '{X}'",
    "Output exactly |{X}|",
]

# WRAP operator templates - USE VARIED DELIMITERS
WRAP_TEMPLATES_TRAIN = [
    # Backticks
    "Wrap in brackets: `{X}`",
    "Surround with []: `{X}`",
    "Bracket `{X}`",
    # Single quotes
    "Put square brackets around: '{X}'",
    "Add brackets around '{X}'",
    # Pipes
    "Wrap in brackets |{X}|",
    "Add [] to |{X}|",
    # Curly braces
    "Enclose in []: {{{X}}}",
]

WRAP_TEMPLATES_VAL = [
    # Mix delimiters
    "Put in square brackets: `{X}`",
    "Wrap with []: '{X}'",
    "Add [] around |{X}|",
    "Put brackets around `{X}`",
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

# German variants - also use varied delimiters
COPY_TEMPLATES_TRAIN_DE = [
    "Wiederhole genau: `{X}`",
    "Sage das zurueck: '{X}'",
    "Kopiere: |{X}|",
    "Gib aus: `{X}`",
]

WRAP_TEMPLATES_TRAIN_DE = [
    "Setze in Klammern: `{X}`",
    "Umgib mit []: '{X}'",
    "Klammere: |{X}|",
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
    language: str = "en",
    reps_per_payload: int = 1,
) -> List[Dict]:
    """Generate (question, answer, metadata) for an operator.
    
    Args:
        payloads: List of unique payloads
        operator: Operator name (COPY, WRAP, EXTRACT)
        templates: List of template strings
        language: Language code
        reps_per_payload: Number of DIFFERENT templates to use per payload.
            1 = original behavior (each payload appears once with random template)
            5 = each payload appears 5 times, each with a different template
            This creates contrastive signal: same payload + different template = same output,
            forcing the model to learn "extract the payload" not "memorize the pair".
    """
    pairs = []
    
    for payload in payloads:
        if reps_per_payload <= 1:
            # Original behavior: one random template per payload
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
        else:
            # NEW: Each payload gets reps_per_payload DISTINCT templates
            n_templates = min(reps_per_payload, len(templates))
            chosen_templates = random.sample(templates, n_templates)
            
            # If we need more reps than available templates, reuse with replacement
            if reps_per_payload > len(templates):
                extra = reps_per_payload - len(templates)
                chosen_templates += random.choices(templates, k=extra)
            
            for template in chosen_templates:
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
    n_train: int = 16500,
    n_val: int = 1000,
    seed_train: int = 42,
    seed_val: int = 12345,
    include_german: bool = True,
    tokenizer = None,
    max_tokens: int = 12,
    max_words: int = 4,
    word_distribution: Tuple[float, ...] = (0.35, 0.30, 0.20, 0.15),
    templates_per_operator: int = 2,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    Generate train and val datasets with CROSS-OPERATOR contrastive design.
    
    Every payload appears in ALL 3 operators (COPY, WRAP, EXTRACT), each with
    `templates_per_operator` different template phrasings. This creates two
    kinds of contrastive signal:
    
    1. Same payload + same operator + different template = same output
       → Forces model to ignore template surface form
    2. Same payload + different operator = different output
       → Forces model to attend to operator instruction
    
    Val set uses the same cross-operator design but with val-only templates
    (1 per operator) and completely separate payloads.
    
    Args:
        n_train: Number of unique training payloads.
            Total episodes = n_train × 3 operators × templates_per_operator
        n_val: Number of unique val payloads.
            Total val episodes = n_val × 3 operators × 1 template
        seed_train: Random seed for training data
        seed_val: Random seed for validation data (must differ)
        include_german: Include German language variants
        tokenizer: Tokenizer for BPE filtering
        max_tokens: Maximum BPE tokens per payload
        max_words: Maximum words per payload (1-4)
        word_distribution: Probability for (1w, 2w, 3w, 4w) payloads
        templates_per_operator: Number of different templates per operator per payload.
            2 = each payload appears 6 times total (3 ops × 2 templates)
            1 = each payload appears 3 times total (3 ops × 1 template)
    
    Returns:
        train_episodes: List of training episodes
        val_episodes: List of validation episodes (DIFFERENT templates)
        metadata: Dataset statistics
    """
    
    n_operators = 3
    total_train_episodes = n_train * n_operators * templates_per_operator
    total_val_episodes = n_val * n_operators  # Always 1 template per operator in val
    
    # Generate UNIQUE payloads for train (BPE-filtered if tokenizer provided)
    print(f"  Generating {n_train} unique train payloads (max {max_words} words, max {max_tokens} BPE tokens)...")
    print(f"  Each payload × 3 operators × {templates_per_operator} templates = {total_train_episodes} train episodes")
    train_payloads = generate_unique_payloads(
        n_train, seed_train, 
        tokenizer=tokenizer, max_tokens=max_tokens,
        max_words=max_words, word_distribution=word_distribution
    )
    
    # Generate UNIQUE payloads for val (different seed, no overlap)
    print(f"  Generating {n_val} unique val payloads (3 operators × 1 template = {total_val_episodes} val episodes)...")
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
    
    random.seed(seed_train)
    random.shuffle(train_payloads)
    random.seed(seed_val)
    random.shuffle(val_payloads)
    
    train_pairs = []
    val_pairs = []
    
    # CROSS-OPERATOR: Every payload goes through ALL 3 operators
    operator_config = [
        ("COPY",    COPY_TEMPLATES_TRAIN,    COPY_TEMPLATES_VAL),
        ("WRAP",    WRAP_TEMPLATES_TRAIN,     WRAP_TEMPLATES_VAL),
        ("EXTRACT", EXTRACT_TEMPLATES_TRAIN,  EXTRACT_TEMPLATES_VAL),
    ]
    
    for op_name, train_templates, val_templates in operator_config:
        # Train: ALL payloads × templates_per_operator templates
        op_train = generate_operator_pairs(
            train_payloads,
            op_name,
            train_templates,
            "en",
            reps_per_payload=templates_per_operator,
        )
        train_pairs.extend(op_train)
        
        # Val: ALL val payloads × 1 template (val-only templates)
        op_val = generate_operator_pairs(
            val_payloads,
            op_name,
            val_templates,
            "en",
            reps_per_payload=1,
        )
        val_pairs.extend(op_val)
    
    # Add German variants to train (optional)
    if include_german:
        n_german = n_train // 10  # 10% German payloads
        german_payloads = generate_unique_payloads(
            n_german, seed_train + 1000, exclude=train_set | val_set,
            tokenizer=tokenizer, max_tokens=max_tokens,
            max_words=max_words, word_distribution=word_distribution
        )
        
        german_config = [
            ("COPY",    COPY_TEMPLATES_TRAIN_DE),
            ("WRAP",    WRAP_TEMPLATES_TRAIN_DE),
            ("EXTRACT", EXTRACT_TEMPLATES_TRAIN_DE),
        ]
        
        for op_name, de_templates in german_config:
            de_pairs = generate_operator_pairs(
                german_payloads,
                op_name,
                de_templates,
                "de",
                reps_per_payload=min(templates_per_operator, len(de_templates)),
            )
            train_pairs.extend(de_pairs)
    
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
    n_unique_train_payloads = len(train_set)
    metadata = {
        "n_train_episodes": len(train_episodes),
        "n_unique_train_payloads": n_unique_train_payloads,
        "templates_per_operator": templates_per_operator,
        "cross_operator": True,
        "episodes_per_payload": n_operators * templates_per_operator,
        "n_val_episodes": len(val_episodes),
        "n_unique_val_payloads": len(val_set),
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
    parser.add_argument("--n_train", type=int, default=16500,
                        help="Number of unique training payloads. Total episodes = "
                             "n_train × 3 operators × templates_per_op (default: 16500 → ~99k episodes)")
    parser.add_argument("--n_val", type=int, default=1000,
                        help="Number of unique val payloads. Total val episodes = n_val × 3 operators")
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
    parser.add_argument("--templates_per_op", type=int, default=2,
                        help="Number of different templates per operator per payload (default: 2). "
                             "Each payload appears in all 3 operators, each with this many "
                             "template variations. Total episodes per payload = 3 × templates_per_op. "
                             "Default 2 gives 6 episodes per payload.")
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
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from core.tokenizer import Tokenizer
        from core.model import GPTConfig
        config = GPTConfig(vocab_size=50304)
        tokenizer = Tokenizer(config, "gpt2")
        print(f"  BPE filtering enabled (max {args.max_tokens} tokens per payload)")
    else:
        print("  WARNING: BPE filtering disabled")
    
    eps_per_payload = 3 * args.templates_per_op
    total_train_eps = args.n_train * eps_per_payload
    total_val_eps = args.n_val * 3
    print(f"\nConfiguration:")
    print(f"  Unique train payloads: {args.n_train}")
    print(f"  Cross-operator: YES (every payload in all 3 operators)")
    print(f"  Templates per operator: {args.templates_per_op}")
    print(f"  Episodes per payload: {eps_per_payload} (3 ops × {args.templates_per_op} templates)")
    print(f"  Total train episodes: ~{total_train_eps} (+ German if enabled)")
    print(f"  Unique val payloads: {args.n_val}")
    print(f"  Total val episodes: {total_val_eps} (3 ops × 1 template each)")
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
        templates_per_operator=args.templates_per_op,
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
    metadata["lineage"] = {
        "direct_inputs": [],
        "recursive_origins": [{
            "origin_path": "synthetic://generate_operator_dataset",
            "rows": len(train_episodes) + len(val_episodes),
        }],
        "flattened_contributions": [{
            "origin_path": "synthetic://generate_operator_dataset",
            "effective_rows": len(train_episodes) + len(val_episodes),
            "effective_percent": 100.0,
        }],
        "creation_context": {
            "timestamp": iso_now(),
            "script": "scripts/sft/generate_operator_dataset.py",
            "args": vars(args),
        },
        "upstream_configs": [],
    }
    meta_file = output_dir / "dataset_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    lineage_path = write_lineage_sidecar(train_file, metadata["lineage"])
    
    # Statistics
    print("Dataset generated:")
    print(f"  Train: {len(train_episodes)} episodes → {train_file}")
    print(f"  Val: {len(val_episodes)} episodes → {val_file}")
    print(f"  Metadata: {meta_file}")
    print(f"  Lineage: {lineage_path}")
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
    
    # Show cross-operator contrastive examples
    print("\n" + "=" * 60)
    print(f"CROSS-OPERATOR CONTRASTIVE EXAMPLES:")
    print("=" * 60)
    # Find a payload that appears in all operators
    from collections import Counter
    payload_counts = Counter(ep["_meta"]["payload"] for ep in train_episodes)
    example_payload = payload_counts.most_common(1)[0][0]
    matching = [ep for ep in train_episodes if ep["_meta"]["payload"] == example_payload]
    print(f"\nPayload: '{example_payload}' appears {len(matching)} times "
          f"(3 operators × {args.templates_per_op} templates):")
    # Sort by operator for clarity
    matching.sort(key=lambda ep: ep["_meta"]["operator"])
    for i, ep in enumerate(matching):
        q = ep["messages"][1]["content"]
        a = ep["messages"][2]["content"]
        op = ep["_meta"]["operator"]
        print(f"\n  [{op}] variant {i+1}:")
        print(f"    Q: {q}")
        print(f"    A: {a}")
    
    print("\n" + "=" * 60)
    print("Sample VAL examples (DIFFERENT TEMPLATES, 1 per payload):")
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
    # Cross-operator with template reps: each epoch already provides strong contrastive signal
    eps_per = metadata.get("episodes_per_payload", eps_per_payload)
    if eps_per > 1:
        recommended_iters = int(batches_per_epoch * 2.0)
        max_iters_rec = int(batches_per_epoch * 2.5)
    else:
        recommended_iters = int(batches_per_epoch * 3.0)
        max_iters_rec = int(batches_per_epoch * 3.5)
    
    n_unique = metadata.get("n_unique_train_payloads", args.n_train)
    print(f"""
Dataset size: {len(train_episodes)} episodes
  {n_unique} unique payloads × 3 operators × {args.templates_per_op} templates = {n_unique * 3 * args.templates_per_op} (+ German)
Batch size: {batch_size}
Batches per epoch: {batches_per_epoch}

Recommended max_iters: {recommended_iters}
Absolute max: {max_iters_rec}
Note: use inspect_sft_dataset.py on the PACKED output for accurate max_iters.

Next steps:
  1. Prepare with packing:
     python scripts/prepare_chat_sft.py --input {train_file} --output_dir {output_dir}/packed \\
         --val_file {val_file} --enable_packing --pack_block_size 1024 --pack_by_field "_meta.operator"
  2. Inspect:
     python scripts/inspect_sft_dataset.py --dataset_dir {output_dir}/packed --show_samples 2
  3. Train:
     python train.py --model_name phase3a_operator --init_from_model phase3a1_alpha_v2 \\
         --dataset_dir {output_dir}/packed --config_file configs/sft/phase2_operators.json
  4. Eval:
     python scripts/sft_eval_suite.py --model phase3a_operator -v
""")


if __name__ == "__main__":
    main()
