#!/usr/bin/env python3
"""
Generate diverse echo/repeat instruction dataset.

Combines:
1. Multiple instruction patterns (Say:, Repeat:, Output:, etc.)
2. Diverse random content (words, phrases, numbers, sentences)

This teaches the model "where to look" - the pattern of extracting
content after an instruction prefix.
"""

import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Use canonical system prompt from core
from core.system_prompts import CONVERSATION_SYSTEM_PROMPT
SYSTEM_PROMPT = CONVERSATION_SYSTEM_PROMPT

# Instruction prefixes - the model should learn these all mean "repeat what follows"
INSTRUCTION_PREFIXES = [
    "Say:",
    "Say the word:",
    "Say the phrase:",
    "Say exactly:",
    "Please say:",
    "Just say:",
    "Repeat:",
    "Repeat after me:",
    "Repeat this:",
    "Repeat exactly:",
    "Please repeat:",
    "Please repeat exactly:",
    "Please repeat this:",
    "Please repeat this exactly:",
    "Echo:",
    "Echo this:",
    "Output:",
    "Output exactly:",
    "Output the following:",
    "Write:",
    "Write exactly:",
    "Write out:",
    "would you please say:",
    "would you please write:",
    "would you please repeat:",
    "would you please output:",
    "would you please type:",
    "would you please respond:",
    "would you please reply:",
    "would you please give me:",
    "would you please return:",
    "would you please print:",
    "Type:",
    "Type out:",
    "Respond with:",
    "Respond with only:",
    "Reply with:",
    "Reply with just:",
    "Your answer is:",
    "Your response should be:",
    "Give me:",
    "Return:",
    "Print:",
]

# German instruction prefixes
GERMAN_PREFIXES = [
    "Sage:",
    "Sag:",
    "Sag das Wort:",
    "Sag genau:",
    "Bitte sage:",
    "Bitte wiederhole genau:",
    "Bitte wiederhole dies:",
    "Bitte wiederhole dies genau:",
    "Wiederhole:",
    "Wiederhole dies:",
    "Wiederhole genau:",
    "Bitte wiederhole:",
    "Bitte sag:",
    "Bitte schreibe:",
    "Bitte antworte mit:",
    "Bitte antworte nur mit:",
    "Bitte gib aus:",
    "Bitte gib genau aus:",
    "Bitte wiederhole:",
    "Gib aus:",
    "Gib genau aus:",
    "Schreibe:",
    "Schreibe genau:",
    "Antworte mit:",
    "Antworte nur mit:",
    "Deine Antwort ist:",
    "Deine Antwort soll sein:",
    "Würdest du bitte folgendes sagen:",
    "Würdest du bitte folgendes schreiben:",
    "Würdest du bitte folgendes antworten:",
    "Würdest du bitte folgendes ausgeben:",
    "Würdest du bitte folgendes wiederholen:",
    "Würdest du bitte folgendes antworten:",
    "Würdest du bitte folgendes ausgeben:",
]

# Diverse content to echo - organized by category
SINGLE_WORDS_EN = [
    "Yes", "No", "Hello", "Goodbye", "Thanks", "OK", "Done", "Ready",
    "Start", "Stop", "Go", "Wait", "Here", "There", "Now", "Later",
    "True", "False", "Pass", "Fail", "Success", "Error", "Valid", "Invalid",
    "Accept", "Reject", "Confirm", "Cancel", "Submit", "Reset", "Clear", "Save",
    "Open", "Close", "Send", "Receive", "Upload", "Download", "Connect", "Disconnect",
    "Enable", "Disable", "Active", "Inactive", "Online", "Offline", "Busy", "Free",
    "Apple", "Banana", "Orange", "Lemon", "Cherry", "Grape", "Melon", "Peach",
    "Red", "Blue", "Green", "Yellow", "Purple", "Orange", "Black", "White",
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "Cat", "Dog", "Bird", "Fish", "Horse", "Lion", "Tiger", "Bear",
    "Sun", "Moon", "Star", "Cloud", "Rain", "Snow", "Wind", "Fire",
    "Book", "Pen", "Paper", "Desk", "Chair", "Door", "Window", "Floor",
]

# Gibberish words - forces true copy/echo learning, not word memorization
GIBBERISH_WORDS = [
    "Blurpix", "Zanthor", "Quexling", "Flimzap", "Grobnak", "Wibblyx", "Snorfle", "Plonkus",
    "Xyrpod", "Krazbit", "Muffnix", "Drazzle", "Skronf", "Blibber", "Zwomp", "Grindax",
    "Fluxorp", "Qwibble", "Splonk", "Vrazzle", "Nimbux", "Glarfle", "Triznak", "Wompus",
    "Blixtar", "Zorfnik", "Krumple", "Snazzix", "Plorfin", "Gruxoid", "Flibbix", "Quondar",
    "asdfgh", "qwerty", "zxcvbn", "poiuyt", "lkjhgf", "mnbvcx", "rewqas", "ytrewq",
    "Blub123", "Xyz789", "Qrs456", "Abc-xy", "Mno_pq", "Rst.uv", "Wxy;ab", "Cde:fg",
    "fl0rpz", "gr1nkx", "bl4rpq", "zw0nky", "kr1spx", "sn0rfl", "pl0nkz", "tr1zzy",
    "BLURP", "ZONK", "GRAX", "FLIB", "QUEX", "SNORF", "PLONK", "WIBZ",
    "bLuRpIx", "ZaNtHoR", "QuExLiNg", "fLiMzAp", "gRoBnAk", "WiBbLyX", "sNoRfLe", "pLoNkUs",
    "blurp-zant", "quex_flim", "grob.nak", "wibb;lyx", "snor:fle", "plon-kus", "xyr_pod", "kraz.bit",
    "123blurp", "456zant", "789quex", "012flim", "345grob", "678wibb", "901snor", "234plon",
    "blurp123zant", "quex456flim", "grob789nak", "wibb012lyx", "snor345fle", "plon678kus",
]

SINGLE_WORDS_DE = [
    "Ja", "Nein", "Hallo", "Tschüss", "Danke", "OK", "Fertig", "Bereit",
    "Start", "Stopp", "Los", "Warte", "Hier", "Dort", "Jetzt", "Später",
    "Wahr", "Falsch", "Bestanden", "Fehler", "Erfolg", "Ungültig", "Gültig",
    "Akzeptieren", "Ablehnen", "Bestätigen", "Abbrechen", "Senden", "Zurücksetzen",
    "Öffnen", "Schliessen", "Hochladen", "Herunterladen", "Verbinden", "Trennen",
    "Aktivieren", "Deaktivieren", "Aktiv", "Inaktiv", "Online", "Offline",
    "Apfel", "Banane", "Orange", "Zitrone", "Kirsche", "Traube", "Melone",
    "Rot", "Blau", "Grün", "Gelb", "Lila", "Schwarz", "Weiss",
    "Eins", "Zwei", "Drei", "Vier", "Fünf", "Sechs", "Sieben", "Acht",
    "Katze", "Hund", "Vogel", "Fisch", "Pferd", "Löwe", "Tiger", "Bär",
    "Sonne", "Mond", "Stern", "Wolke", "Regen", "Schnee", "Wind", "Feuer",
    "Buch", "Stift", "Papier", "Tisch", "Stuhl", "Tür", "Fenster", "Boden", "Blub",
    # Gibberish with German-ish flavor
    "Blörpix", "Zänthör", "Quöxling", "Flümzap", "Gröbnak", "Wübblyx", "Schnörfle", "Plönkus",
    "Xürpöd", "Kräzbit", "Müffnix", "Dräzzle", "Skrönf", "Blübber", "Zwömp", "Gründax",
    "äöüss", "Öäüss", "äÖüÜ", "ssäöü", "üöäss", "Äöüss", "öÄüÜ", "ÜöÄss",
]

PHRASES_EN = [
    "Hello world", "Thank you", "Good morning", "Good night", "See you later",
    "Nice to meet you", "How are you", "I understand", "No problem", "Of course",
    "Well done", "Great job", "Good work", "Keep going", "Try again",
    "The answer is 42", "Python is great", "MyPT assistant", "Machine learning",
    "Artificial intelligence", "Neural network", "Deep learning", "Data science",
    "Hello there", "Testing testing", "One two three", "A B C", "X Y Z",
    "Red and blue", "Yes and no", "Start to finish", "Left or right",
    "North south east west", "First second third", "Alpha beta gamma",
    "Input output", "Request response", "Query result", "Send receive",
    "Open sesame", "Abracadabra", "Hocus pocus", "Lorem ipsum",
    "Coffee and tea", "Bread and butter", "Salt and pepper",
    "Day and night", "Black and white", "Hot and cold", "Up and down",
]

PHRASES_DE = [
    "Hallo Welt", "Danke schön", "Guten Morgen", "Gute Nacht", "Bis später",
    "Freut mich", "Wie geht es", "Ich verstehe", "Kein Problem", "Natürlich",
    "Gut gemacht", "Tolle Arbeit", "Weiter so", "Versuch es nochmal",
    "Die Antwort ist 42", "Python ist toll", "MyPT Assistent",
    "Künstliche Intelligenz", "Neuronales Netzwerk", "Maschinelles Lernen",
    "Hallo zusammen", "Test Test", "Eins zwei drei", "A B C", "X Y Z",
    "Rot und blau", "Ja und nein", "Anfang bis Ende", "Links oder rechts",
    "Nord Süd Ost West", "Erste zweite dritte", "Alpha Beta Gamma",
    "Eingabe Ausgabe", "Anfrage Antwort", "Senden Empfangen",
    "Kaffee und Tee", "Brot und Butter", "Salz und Pfeffer",
    "Tag und Nacht", "Schwarz und weiss", "Heiss und kalt", "Auf und ab",
]

NUMBERS_AND_CODES = [
    "42", "123", "007", "404", "500", "200", "301", "100",
    "3.14", "2.71", "1.41", "9.81", "0.5", "0.25", "0.1",
    "A1", "B2", "C3", "X1", "Y2", "Z3",
    "ABC123", "XYZ789", "TEST001", "CODE42",
    "UID_042", "UID_123", "UID_404", "UID_500", "UID_200", "UID_301", "UID_100",
    "CH-2026-01", "CH-2026-02", "CH-2026-03", "CH-2026-04",
    "12345", "67890", "11111", "00000", "99999",
    "2024", "2025", "2026", "1984", "2001",
    "ID-001", "ID-002", "ID-003", "ID-004", "ID-005",

    "1 + 1", "2 + 2", "3 x 3", "10 - 5", "100 / 10",
]

SENTENCES_EN = [
    "The quick brown fox jumps over the lazy dog.",
    "This is a test sentence.",
    "MyPT is ready to help.",
    "The answer to your question is yes.",
    "Please try again later.",
    "Operation completed successfully.",
    "An error has occurred.",
    "Connection established.",
    "Data saved to file.",
    "Processing your request.",
    "Waiting for input.",
    "Task finished.",
    "All systems operational.",
    "Ready for next command.",
    "Initialization complete.",
]

# Gibberish sentences - forces true copy/echo, not semantic understanding
GIBBERISH_SENTENCES = [
    "The blurpix zanthor quexled over the flimzap.",
    "Grobnak wibblyx snorfled the plonkus xyrpod.",
    "When krazbit muffnixed, drazzle skronfed blibber.",
    "Zwomp grindax fluxorped qwibble splonk vrazzle.",
    "Nimbux glarfled triznak wompus blixtar zorfnik.",
    "If krumple snazzixed, plorfin gruxoid flibbixed.",
    "Quondar blurpix zanthor quexling flimzap grobnak.",
    "asdfgh qwerty zxcvbn poiuyt lkjhgf mnbvcx.",
    "The fl0rpz gr1nkx bl4rpq zw0nky kr1spx.",
    "Blub123 xyz789 qrs456 abc-xy mno_pq rst.uv.",
    "BLURP ZONK GRAX FLIB QUEX SNORF PLONK WIBZ.",
    "bLuRpIx ZaNtHoR QuExLiNg fLiMzAp gRoBnAk WiBbLyX.",
    "blurp-zant quex_flim grob.nak wibb;lyx snor:fle.",
    "123blurp 456zant 789quex 012flim 345grob 678wibb.",
    "Xyrpod krazbit muffnix drazzle skronf blibber zwomp.",
    "Grindax fluxorp qwibble splonk vrazzle nimbux glarfle.",
    "Triznak wompus blixtar zorfnik krumple snazzix plorfin.",
    "qwerty123 asdf456 zxcv789 poiu012 lkjh345 mnbv678.",
    "The gr0bnak fl1mzapped wh1le zant-hor qu3xled.",
    "Snorfle_plonkus xyr.pod kraz;bit muff:nix drazzle!",
]

SENTENCES_DE = [
    "Dies ist ein Testsatz.",
    "MyPT ist bereit zu helfen.",
    "Die Antwort auf Ihre Frage ist ja.",
    "Bitte versuchen Sie es später erneut.",
    "Vorgang erfolgreich abgeschlossen.",
    "Ein Fehler ist aufgetreten.",
    "Verbindung hergestellt.",
    "Daten in Datei gespeichert.",
    "Ihre Anfrage wird bearbeitet.",
    "Warte auf Eingabe.",
    "Aufgabe beendet.",
    "Alle Systeme betriebsbereit.",
    "Bereit für nächsten Befehl.",
    "Initialisierung abgeschlossen.",
    "Der frühe Vogel fängt den Wurm",
    "Den Wald vor lauter Bäumen nicht mehr sehen",
    "Wer zusammenhängende Sätze schreibt, der wird auch zusammenhängende Gedanken haben.",
    "Wer zuspäht kommt, den Bestraft das Leben",
    "Nur im Wörterbuch steht Erfolg vor Fleiss",
    "Den letzten beissen die Hunde",
    "Gelegenheit macht Diebe",
]

# German gibberish sentences
GIBBERISH_SENTENCES_DE = [
    "Der Blörpix zänthörte über den Quöxling.",
    "Gröbnak wübblyx schnörflte den Plönkus xürpöd.",
    "Wenn Kräzbit müffnixte, dräzzle skrönfte blübber.",
    "Zwömp gründax flüxörpte qwübble splönk vräzzle.",
    "äöüss qwerty zxcvbn poiuyt lkjhgf mnbvcx.",
    "Der fl0rpz gr1nkx bl4rpq zw0nky kr1spx.",
    "BLÖRP ZÖNK GRÄX FLÜB QUÖX SCHNÖRF PLÖNK WÜBZ.",
    "bLöRpIx ZäNtHöR QuÖxLiNg fLüMzAp gRöBnAk WüBbLyX.",
    "blörp-zänt quöx_flüm gröb.näk wübb;lyx schnör:fle.",
    "123blörp 456zänt 789quöx 012flüm 345gröb 678wübb.",
    "Xürpöd kräzbit müffnix dräzzle skrönf blübber zwömp.",
    "Gründax flüxörp qwübble splönk vräzzle nümbux glärfle.",
]


def generate_echo_pairs(max_examples: int = None, seed: int = 42) -> List[Tuple[str, str, str]]:
    """Generate ALL unique echo instruction pairs (prefix × content combinations).
    
    Args:
        max_examples: Optional cap on total examples. If None, generates all combinations.
        seed: Random seed for shuffling.
    
    Returns list of (question, answer, category) tuples.
    """
    random.seed(seed)
    pairs = []
    
    # Combine all content pools
    en_content = (
        [(w, "word_en") for w in SINGLE_WORDS_EN] +
        [(w, "gibberish_word") for w in GIBBERISH_WORDS] +
        [(p, "phrase_en") for p in PHRASES_EN] +
        [(n, "number") for n in NUMBERS_AND_CODES] +
        [(s, "sentence_en") for s in SENTENCES_EN] +
        [(s, "gibberish_sentence") for s in GIBBERISH_SENTENCES]
    )
    
    de_content = (
        [(w, "word_de") for w in SINGLE_WORDS_DE] +
        [(p, "phrase_de") for p in PHRASES_DE] +
        [(s, "sentence_de") for s in SENTENCES_DE] +
        [(s, "gibberish_sentence_de") for s in GIBBERISH_SENTENCES_DE]
    )
    
    # Generate ALL English combinations (prefix × content)
    for prefix in INSTRUCTION_PREFIXES:
        for content, cat in en_content:
            question = f"{prefix} {content}"
            answer = content
            pairs.append((question, answer, f"echo_{cat}"))
    
    # Generate ALL German combinations (prefix × content)
    for prefix in GERMAN_PREFIXES:
        for content, cat in de_content:
            question = f"{prefix} {content}"
            answer = content
            pairs.append((question, answer, f"echo_{cat}"))
    
    # Shuffle
    random.shuffle(pairs)
    
    # Optional cap
    if max_examples and len(pairs) > max_examples:
        pairs = pairs[:max_examples]
    
    return pairs


def create_episode(question: str, answer: str, episode_id: int, category: str) -> dict:
    """Create a single episode in the expected format."""
    return {
        "system": SYSTEM_PROMPT,
        "context": f"episode_id: echo_{episode_id:04d}, category: {category}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ],
        "language": "en" if "_en" in category or "number" in category else "de"
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate diverse echo instruction dataset")
    parser.add_argument("--max_examples", type=int, default=4300, help="Cap on examples (default: 4300)")
    parser.add_argument("--output_dir", type=str, default="data/sft_echo", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mypt_echo_diverse.jsonl"
    
    # Generate pairs (capped to max_examples)
    print(f"Generating echo instruction combinations (max: {args.max_examples})...")
    pairs = generate_echo_pairs(max_examples=args.max_examples, seed=args.seed)
    
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
    
    # Sample examples
    print("\nSample examples:")
    for i in range(min(10, len(pairs))):
        q, a, cat = pairs[i]
        print(f"  [{cat}] Q: {q[:50]}{'...' if len(q) > 50 else ''}")
        print(f"          A: {a[:50]}{'...' if len(a) > 50 else ''}")


if __name__ == "__main__":
    main()
