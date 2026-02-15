#!/usr/bin/env python3
"""
Generate diverse echo/repeat instruction dataset with BPE-safe gibberish and contrast pairs.

Features:
1. Multiple instruction patterns (Say:, Repeat:, Output:, etc.)
2. Diverse random content (words, phrases, numbers, sentences)
3. BPE-safe gibberish (filtered by token count to ensure learnability)
4. Anti-echo templates (prevents blind copying)
5. Contrast pairs (same string in echo AND non-echo contexts)

This teaches the model abstract instruction following, not just "see text → copy text".
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use canonical system prompt from core
from core.system_prompts import CONVERSATION_SYSTEM_PROMPT
from core.special_tokens import BASE_VOCAB_SIZE
SYSTEM_PROMPT = CONVERSATION_SYSTEM_PROMPT

# =============================================================================
# INSTRUCTION PREFIXES
# =============================================================================

# Echo instruction prefixes - the model should learn these all mean "repeat what follows"
ECHO_PREFIXES_EN = [
    "Say:", "Say the word:", "Say the phrase:", "Say exactly:",
    "Please say:", "Just say:",
    "Repeat:", "Repeat after me:", "Repeat this:", "Repeat exactly:",
    "Please repeat:", "Please repeat exactly:",
    "Echo:", "Echo this:",
    "Output:", "Output exactly:", "Output the following:",
    "Write:", "Write exactly:", "Write out:",
    "Type:", "Type out:",
    "Respond with:", "Respond with only:",
    "Reply with:", "Reply with just:",
    "Give me:", "Return:", "Print:",
]

# Quote-style echo prefixes (for contrast pairs)
ECHO_QUOTE_PREFIXES_EN = [
    'Say "{X}"', 'Repeat "{X}"', 'Echo "{X}"', 'Output "{X}"',
    'Write "{X}"', 'Type "{X}"', 'Reply with "{X}"',
]

# German echo prefixes
ECHO_PREFIXES_DE = [
    "Sage:", "Sag:", "Sag das Wort:", "Sag genau:",
    "Bitte sage:", "Bitte sag:",
    "Wiederhole:", "Wiederhole dies:", "Wiederhole genau:",
    "Bitte wiederhole:", "Bitte wiederhole genau:",
    "Gib aus:", "Gib genau aus:",
    "Schreibe:", "Schreibe genau:",
    "Antworte mit:", "Antworte nur mit:",
]

# =============================================================================
# ANTI-ECHO TEMPLATES (model should NOT copy, but answer appropriately)
# =============================================================================

ANTI_ECHO_TEMPLATES_EN = [
    ('What is "{X}"?', "Unknown."),
    ('What does "{X}" mean?', "Unknown."),
    ('Is "{X}" a real word?', "No."),
    ('Is "{X}" a valid English word?', "No."),
    ('Does "{X}" have a meaning?', "No."),
    ('Translate "{X}" to German.', "Unknown."),
    ('Define "{X}".', "Unknown."),
]

ANTI_ECHO_TEMPLATES_DE = [
    ('Was bedeutet "{X}"?', "Unbekannt."),
    ('Ist "{X}" ein echtes Wort?', "Nein."),
    ('Hat "{X}" eine Bedeutung?', "Nein."),
    ('Übersetze "{X}" ins Englische.', "Unbekannt."),
]

# =============================================================================
# CONTENT POOLS
# =============================================================================

SINGLE_WORDS_EN = [
    "Yes", "No", "Hello", "Goodbye", "Thanks", "OK", "Done", "Ready",
    "Start", "Stop", "Go", "Wait", "Here", "There", "Now", "Later",
    "True", "False", "Pass", "Fail", "Success", "Error", "Valid", "Invalid",
    "Accept", "Reject", "Confirm", "Cancel", "Submit", "Reset", "Clear", "Save",
    "Open", "Close", "Send", "Receive", "Upload", "Download", "Connect", "Disconnect",
    "Enable", "Disable", "Active", "Inactive", "Online", "Offline", "Busy", "Free",
    "Apple", "Banana", "Orange", "Lemon", "Cherry", "Grape", "Melon", "Peach",
    "Red", "Blue", "Green", "Yellow", "Purple", "Black", "White",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "Cat", "Dog", "Bird", "Fish", "Horse", "Lion", "Tiger", "Bear",
    "Sun", "Moon", "Star", "Cloud", "Rain", "Snow", "Wind", "Fire",
    "Book", "Pen", "Paper", "Desk", "Chair", "Door", "Window", "Floor",
]

SINGLE_WORDS_DE = [
    "Ja", "Nein", "Hallo", "Danke", "OK", "Fertig", "Bereit",
    "Start", "Stopp", "Los", "Warte", "Hier", "Dort", "Jetzt",
    "Wahr", "Falsch", "Fehler", "Erfolg",
    "Apfel", "Banane", "Orange", "Zitrone", "Kirsche",
    "Rot", "Blau", "Grün", "Gelb", "Schwarz", "Weiss",
    "Eins", "Zwei", "Drei", "Vier", "Fünf",
    "Katze", "Hund", "Vogel", "Fisch", "Pferd",
    "Sonne", "Mond", "Stern", "Wolke", "Regen",
    "Buch", "Stift", "Papier", "Tisch", "Stuhl",
]

PHRASES_EN = [
    "Hello world", "Thank you", "Good morning", "Good night", "See you later",
    "Nice to meet you", "How are you", "I understand", "No problem", "Of course",
    "Well done", "Great job", "Good work", "Keep going", "Try again",
    "The answer is 42", "Python is great", "MyPT assistant", "Machine learning",
    "Hello there", "Testing testing", "One two three", "A B C", "X Y Z",
    "Red and blue", "Yes and no", "Start to finish", "Left or right",
    "Input output", "Request response", "Query result", "Send receive",
    "Coffee and tea", "Bread and butter", "Salt and pepper",
    "Day and night", "Black and white", "Hot and cold", "Up and down",
]

PHRASES_DE = [
    "Hallo Welt", "Danke schön", "Guten Morgen", "Gute Nacht", "Bis später",
    "Freut mich", "Wie geht es", "Ich verstehe", "Kein Problem", "Natürlich",
    "Gut gemacht", "Tolle Arbeit", "Weiter so",
    "Die Antwort ist 42", "Python ist toll", "MyPT Assistent",
    "Hallo zusammen", "Test Test", "Eins zwei drei",
    "Rot und blau", "Ja und nein", "Links oder rechts",
    "Tag und Nacht", "Schwarz und weiss", "Heiss und kalt",
]

NUMBERS_AND_CODES = [
    "42", "123", "007", "404", "500", "200", "301", "100",
    "3.14", "2.71", "1.41", "9.81", "0.5", "0.25",
    "A1", "B2", "C3", "X1", "Y2", "Z3",
    "ABC123", "XYZ789", "TEST001", "CODE42",
    "12345", "67890", "11111", "00000", "99999",
    "2024", "2025", "2026", "1984", "2001",
    "ID-001", "ID-002", "ID-003", "ID-004", "ID-005",
]

SENTENCES_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a test sentence.",
    "MyPT is ready to help.",
    "The answer to your question is yes.",
    "Please try again later.",
    "Operation completed successfully.",
    "Connection established.",
    "Data saved to file.",
    "Processing your request.",
    "Task finished.",
    "All systems operational.",
    "Ready for next command.",
]

SENTENCES_DE = [
    "Dies ist ein Testsatz.",
    "MyPT ist bereit zu helfen.",
    "Die Antwort auf Ihre Frage ist ja.",
    "Bitte versuchen Sie es später erneut.",
    "Vorgang erfolgreich abgeschlossen.",
    "Verbindung hergestellt.",
    "Daten in Datei gespeichert.",
    "Aufgabe beendet.",
    "Alle Systeme betriebsbereit.",
]

# =============================================================================
# BPE-SAFE GIBBERISH GENERATION
# =============================================================================

# Syllables that typically encode to single or few BPE tokens
BPE_SYLLABLES = [
    # Common short syllables
    "bla", "ble", "bli", "blo", "blu",
    "cra", "cre", "cri", "cro", "cru",
    "dra", "dre", "dri", "dro", "dru",
    "fla", "fle", "fli", "flo", "flu",
    "gla", "gle", "gli", "glo", "glu",
    "pla", "ple", "pli", "plo", "plu",
    "sla", "sle", "sli", "slo", "slu",
    "tra", "tre", "tri", "tro", "tru",
    # Common endings
    "tion", "ing", "ness", "ment", "able",
    "ful", "less", "ous", "ive", "ical",
    # Short common patterns
    "an", "en", "in", "on", "un",
    "ar", "er", "ir", "or", "ur",
    "al", "el", "il", "ol", "ul",
    "at", "et", "it", "ot", "ut",
    # Consonant clusters
    "ch", "sh", "th", "wh", "ph",
    "st", "sp", "sk", "sc", "sm", "sn",
    "pr", "br", "tr", "dr", "cr", "gr", "fr",
    # Single letters for variety
    "x", "z", "q", "k", "v", "w",
    # Numbers (single digit)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]

# Pre-defined BPE-safe gibberish (known to be short in tokens)
BPE_SAFE_GIBBERISH_WORDS = [
    # 1-2 syllable combinations
    "Blax", "Crex", "Drip", "Flox", "Grix", "Plun", "Trax", "Vex",
    "Blon", "Crin", "Drun", "Flen", "Glon", "Plix", "Trin", "Vrex",
    "Blip", "Crax", "Drix", "Flon", "Grex", "Plon", "Trix", "Vron",
    # With numbers
    "Blax7", "Crex3", "Drip9", "Flox2", "Grix5", "Plun8", "Trax1", "Vex4",
    "X7", "Z3", "Q9", "K2", "V5", "W8",
    # Short combos
    "AB", "CD", "EF", "GH", "XY", "ZZ",
    "A1", "B2", "C3", "D4", "E5",
    # Mixed case short
    "aB", "cD", "eF", "gH", "xY",
]

# German-ish BPE-safe
BPE_SAFE_GIBBERISH_DE = [
    "Blöx", "Crüx", "Drüp", "Flöx", "Grüx", "Plün", "Träx", "Vöx",
    "Blön", "Crün", "Drün", "Flön", "Grön", "Plöx", "Trön", "Vröx",
]


def generate_bpe_safe_gibberish(
    tokenizer,
    count: int,
    max_tokens: int = 4,
    seed: int = 42,
    include_phrases: bool = True
) -> List[str]:
    """Generate BPE-safe gibberish by combining syllables and filtering by token count.
    
    Args:
        tokenizer: Tokenizer to check token counts
        count: Number of gibberish items to generate
        max_tokens: Maximum tokens allowed per item
        seed: Random seed
        include_phrases: Whether to include multi-word phrases
    
    Returns:
        List of BPE-safe gibberish strings
    """
    rng = random.Random(seed)
    results = []
    attempts = 0
    max_attempts = count * 20  # Prevent infinite loops
    
    # Start with pre-defined safe words
    for word in BPE_SAFE_GIBBERISH_WORDS:
        tokens = tokenizer.encode(word)
        if len(tokens) <= max_tokens and all(t < BASE_VOCAB_SIZE for t in tokens):
            results.append(word)
            if len(results) >= count:
                return results[:count]
    
    # Generate more by combining syllables
    while len(results) < count and attempts < max_attempts:
        attempts += 1
        
        # Decide structure
        structure = rng.choice([
            "syllable",           # Single syllable combo
            "two_syllable",       # Two syllables
            "syllable_number",    # Syllable + number
            "caps_short",         # Short uppercase
        ] + (["two_word"] if include_phrases else []))
        
        if structure == "syllable":
            # 1-2 syllables, capitalized
            n_syl = rng.randint(1, 2)
            word = "".join(rng.choice(BPE_SYLLABLES) for _ in range(n_syl))
            word = word.capitalize()
        
        elif structure == "two_syllable":
            # Two syllables
            word = rng.choice(BPE_SYLLABLES) + rng.choice(BPE_SYLLABLES)
            word = word.capitalize()
        
        elif structure == "syllable_number":
            # Syllable + single digit
            word = rng.choice(BPE_SYLLABLES).capitalize() + str(rng.randint(0, 9))
        
        elif structure == "caps_short":
            # 2-3 uppercase letters
            word = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(2, 3)))
        
        elif structure == "two_word":
            # Two short words
            w1 = rng.choice(BPE_SYLLABLES).capitalize()
            w2 = rng.choice(BPE_SYLLABLES).capitalize()
            word = f"{w1} {w2}"
        
        else:
            continue
        
        # Check token count
        try:
            tokens = tokenizer.encode(word)
            if len(tokens) <= max_tokens and all(t < BASE_VOCAB_SIZE for t in tokens):
                if word not in results:
                    results.append(word)
        except Exception:
            continue
    
    return results[:count]


# =============================================================================
# CONTRAST PAIR GENERATION
# =============================================================================

def generate_contrast_group(
    payload: str,
    is_gibberish: bool,
    language: str = "en"
) -> List[Tuple[str, str, str]]:
    """Generate a contrast group for a single payload.
    
    For the same payload X, generates:
    1. echo_quote: Say "X" → X
    2. echo_prefix: Repeat: X → X  
    3. anti_echo_meta: Is "X" a real word? → No. (for gibberish)
    
    Returns list of (question, answer, category) tuples.
    """
    pairs = []
    
    if language == "en":
        # Echo with quotes
        quote_template = random.choice(ECHO_QUOTE_PREFIXES_EN)
        q = quote_template.replace("{X}", payload)
        pairs.append((q, payload, "contrast_echo_quote"))
        
        # Echo with prefix
        prefix = random.choice(ECHO_PREFIXES_EN)
        pairs.append((f"{prefix} {payload}", payload, "contrast_echo_prefix"))
        
        # Anti-echo (only for gibberish or if random chance)
        if is_gibberish:
            anti_template, anti_answer = random.choice(ANTI_ECHO_TEMPLATES_EN)
            q = anti_template.replace("{X}", payload)
            pairs.append((q, anti_answer, "contrast_anti_echo"))
    
    else:  # German
        # Echo with prefix
        prefix = random.choice(ECHO_PREFIXES_DE)
        pairs.append((f"{prefix} {payload}", payload, "contrast_echo_prefix_de"))
        
        # Anti-echo for gibberish
        if is_gibberish:
            anti_template, anti_answer = random.choice(ANTI_ECHO_TEMPLATES_DE)
            q = anti_template.replace("{X}", payload)
            pairs.append((q, anti_answer, "contrast_anti_echo_de"))
    
    return pairs


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_echo_pairs(
    max_examples: int = None,
    seed: int = 42,
    gibberish_mode: str = "exclude",
    bpe_safe: bool = True,
    max_target_tokens: int = 4,
    anti_echo_ratio: float = 0.2,
    contrast_ratio: float = 0.2,
    tokenizer = None,
) -> List[Tuple[str, str, str]]:
    """Generate echo instruction pairs with optional BPE-safe gibberish and contrast pairs.
    
    Args:
        max_examples: Optional cap on total examples
        seed: Random seed
        gibberish_mode: "include", "exclude", or "only"
        bpe_safe: If True, filter gibberish by token count (requires tokenizer)
        max_target_tokens: Max tokens for BPE-safe filtering
        anti_echo_ratio: Fraction of examples that are anti-echo
        contrast_ratio: Fraction of examples that are contrast groups
        tokenizer: Required if bpe_safe=True
    
    Returns list of (question, answer, category) tuples.
    """
    random.seed(seed)
    pairs = []
    
    # Build content pools based on gibberish mode
    en_content = []
    de_content = []
    gibberish_content = []
    
    if gibberish_mode in ["include", "exclude"]:
        # Regular content
        en_content.extend([(w, "word_en", False) for w in SINGLE_WORDS_EN])
        en_content.extend([(p, "phrase_en", False) for p in PHRASES_EN])
        en_content.extend([(n, "number", False) for n in NUMBERS_AND_CODES])
        en_content.extend([(s, "sentence_en", False) for s in SENTENCES_EN])
        
        de_content.extend([(w, "word_de", False) for w in SINGLE_WORDS_DE])
        de_content.extend([(p, "phrase_de", False) for p in PHRASES_DE])
        de_content.extend([(s, "sentence_de", False) for s in SENTENCES_DE])
    
    if gibberish_mode in ["include", "only"]:
        # Generate BPE-safe gibberish if requested
        if bpe_safe and tokenizer is not None:
            safe_gibberish_en = generate_bpe_safe_gibberish(
                tokenizer, count=200, max_tokens=max_target_tokens, seed=seed
            )
            gibberish_content.extend([(w, "gibberish_bpe_safe", True) for w in safe_gibberish_en])
            
            # Add pre-defined German gibberish (already short)
            for w in BPE_SAFE_GIBBERISH_DE:
                tokens = tokenizer.encode(w)
                if len(tokens) <= max_target_tokens:
                    gibberish_content.append((w, "gibberish_bpe_safe_de", True))
        else:
            # Use pre-defined lists without filtering
            gibberish_content.extend([(w, "gibberish_word", True) for w in BPE_SAFE_GIBBERISH_WORDS])
            gibberish_content.extend([(w, "gibberish_word_de", True) for w in BPE_SAFE_GIBBERISH_DE])
    
    # Combine all content for generation
    all_en = en_content + [g for g in gibberish_content if "_de" not in g[1]]
    all_de = de_content + [g for g in gibberish_content if "_de" in g[1]]
    
    # 1) Generate standard echo pairs
    for prefix in ECHO_PREFIXES_EN:
        for content, cat, is_gib in all_en:
            q = f"{prefix} {content}"
            pairs.append((q, content, f"echo_{cat}"))
    
    for prefix in ECHO_PREFIXES_DE:
        for content, cat, is_gib in all_de:
            q = f"{prefix} {content}"
            pairs.append((q, content, f"echo_{cat}"))
    
    # 2) Generate anti-echo pairs (for gibberish only)
    n_anti_echo = int(len(pairs) * anti_echo_ratio)
    anti_echo_pairs = []
    
    for content, cat, is_gib in gibberish_content:
        if is_gib:
            if "_de" in cat:
                template, answer = random.choice(ANTI_ECHO_TEMPLATES_DE)
            else:
                template, answer = random.choice(ANTI_ECHO_TEMPLATES_EN)
            q = template.replace("{X}", content)
            anti_echo_pairs.append((q, answer, f"anti_echo_{cat}"))
    
    random.shuffle(anti_echo_pairs)
    pairs.extend(anti_echo_pairs[:n_anti_echo])
    
    # 3) Generate contrast groups (same payload in echo + anti-echo context)
    n_contrast = int(len(pairs) * contrast_ratio)
    contrast_pairs = []
    
    # Sample payloads for contrast groups
    contrast_payloads = []
    for content, cat, is_gib in all_en[:50]:  # Sample from English content
        contrast_payloads.append((content, is_gib, "en"))
    for content, cat, is_gib in gibberish_content[:30]:  # Extra gibberish
        if "_de" not in cat:
            contrast_payloads.append((content, True, "en"))
    
    for payload, is_gib, lang in contrast_payloads:
        group = generate_contrast_group(payload, is_gib, lang)
        contrast_pairs.extend(group)
    
    random.shuffle(contrast_pairs)
    pairs.extend(contrast_pairs[:n_contrast])
    
    # Shuffle all pairs
    random.shuffle(pairs)
    
    # Optional cap
    if max_examples and len(pairs) > max_examples:
        pairs = pairs[:max_examples]
    
    return pairs


def create_episode(question: str, answer: str, episode_id: int, category: str) -> dict:
    """Create a single episode in the expected format."""
    is_german = "_de" in category
    return {
        "system": SYSTEM_PROMPT,
        "context": f"episode_id: echo_{episode_id:04d}, category: {category}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "language": "de" if is_german else "en"
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate diverse echo instruction dataset with BPE-safe gibberish"
    )
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Optional cap on examples (default: no cap)")
    parser.add_argument("--output_dir", type=str, default="data/sft_echo",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--gibberish", type=str, default="exclude",
                        choices=["include", "exclude", "only"],
                        help="Gibberish mode: include, exclude, only. Default: exclude")
    parser.add_argument("--bpe_safe", action="store_true", default=True,
                        help="Filter gibberish by BPE token count (default: True)")
    parser.add_argument("--no_bpe_safe", action="store_false", dest="bpe_safe",
                        help="Disable BPE-safe filtering")
    parser.add_argument("--max_target_tokens", type=int, default=4,
                        help="Max tokens for BPE-safe gibberish (default: 4)")
    parser.add_argument("--anti_echo_ratio", type=float, default=0.2,
                        help="Fraction of anti-echo examples (default: 0.2)")
    parser.add_argument("--contrast_ratio", type=float, default=0.2,
                        help="Fraction of contrast pair examples (default: 0.2)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mypt_echo_diverse.jsonl"
    
    # Load tokenizer if BPE-safe mode
    tokenizer = None
    if args.bpe_safe and args.gibberish in ["include", "only"]:
        print("Loading tokenizer for BPE-safe filtering...")
        try:
            from core import Tokenizer
            tokenizer = Tokenizer()
            print(f"  Tokenizer loaded, vocab size: {tokenizer.vocab_size}")
        except Exception as e:
            print(f"  Warning: Could not load tokenizer ({e})")
            print("  Falling back to pre-defined BPE-safe gibberish only")
            tokenizer = None
    
    # Generate pairs
    cap_msg = f"max: {args.max_examples}" if args.max_examples else "no cap"
    print(f"\nGenerating echo dataset ({cap_msg}, gibberish: {args.gibberish}, bpe_safe: {args.bpe_safe})...")
    
    pairs = generate_echo_pairs(
        max_examples=args.max_examples,
        seed=args.seed,
        gibberish_mode=args.gibberish,
        bpe_safe=args.bpe_safe,
        max_target_tokens=args.max_target_tokens,
        anti_echo_ratio=args.anti_echo_ratio,
        contrast_ratio=args.contrast_ratio,
        tokenizer=tokenizer,
    )
    
    # Create episodes
    episodes = []
    for i, (q, a, cat) in enumerate(pairs):
        episode = create_episode(q, a, i, cat)
        episodes.append(episode)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    
    # Stats
    print(f"\n✅ Generated {len(episodes)} echo episodes")
    print(f"   Output: {output_file}")
    
    # Category breakdown
    categories = {}
    for _, _, cat in pairs:
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    
    # Count special categories
    n_anti = sum(c for cat, c in categories.items() if "anti_echo" in cat)
    n_contrast = sum(c for cat, c in categories.items() if "contrast" in cat)
    n_gibberish = sum(c for cat, c in categories.items() if "gibberish" in cat)
    
    print(f"\nSpecial categories:")
    print(f"  Anti-echo examples: {n_anti}")
    print(f"  Contrast examples: {n_contrast}")
    print(f"  Gibberish examples: {n_gibberish}")
    
    # Sample examples
    print("\nSample examples:")
    samples = pairs[:5] + [p for p in pairs if "anti_echo" in p[2]][:3] + [p for p in pairs if "contrast" in p[2]][:3]
    for q, a, cat in samples[:12]:
        print(f"  [{cat}]")
        print(f"    Q: {q[:60]}{'...' if len(q) > 60 else ''}")
        print(f"    A: {a[:40]}{'...' if len(a) > 40 else ''}")


if __name__ == "__main__":
    main()
