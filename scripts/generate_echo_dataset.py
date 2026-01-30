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
from pathlib import Path
from typing import List, Tuple

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

SINGLE_WORDS_DE = [
    "Ja", "Nein", "Hallo", "Tschüss", "Danke", "OK", "Fertig", "Bereit",
    "Start", "Stopp", "Los", "Warte", "Hier", "Dort", "Jetzt", "Später",
    "Wahr", "Falsch", "Bestanden", "Fehler", "Erfolg", "Ungültig", "Gültig",
    "Akzeptieren", "Ablehnen", "Bestätigen", "Abbrechen", "Senden", "Zurücksetzen",
    "Öffnen", "Schließen", "Hochladen", "Herunterladen", "Verbinden", "Trennen",
    "Aktivieren", "Deaktivieren", "Aktiv", "Inaktiv", "Online", "Offline",
    "Apfel", "Banane", "Orange", "Zitrone", "Kirsche", "Traube", "Melone",
    "Rot", "Blau", "Grün", "Gelb", "Lila", "Schwarz", "Weiß",
    "Eins", "Zwei", "Drei", "Vier", "Fünf", "Sechs", "Sieben", "Acht",
    "Katze", "Hund", "Vogel", "Fisch", "Pferd", "Löwe", "Tiger", "Bär",
    "Sonne", "Mond", "Stern", "Wolke", "Regen", "Schnee", "Wind", "Feuer",
    "Buch", "Stift", "Papier", "Tisch", "Stuhl", "Tür", "Fenster", "Boden", "Blub",
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
    "Tag und Nacht", "Schwarz und weiß", "Heiß und kalt", "Auf und ab",
]

NUMBERS_AND_CODES = [
    "42", "123", "007", "404", "500", "200", "301", "100",
    "3.14", "2.71", "1.41", "9.81", "0.5", "0.25", "0.1",
    "A1", "B2", "C3", "X1", "Y2", "Z3",
    "ABC123", "XYZ789", "TEST001", "CODE42",
    "12345", "67890", "11111", "00000", "99999",
    "2024", "2025", "2026", "1984", "2001",
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


def generate_echo_pairs(n_examples: int = 500, seed: int = 42) -> List[Tuple[str, str, str]]:
    """Generate diverse echo instruction pairs.
    
    Returns list of (question, answer, category) tuples.
    """
    random.seed(seed)
    pairs = []
    
    # Combine all content pools
    en_content = (
        [(w, "word_en") for w in SINGLE_WORDS_EN] +
        [(p, "phrase_en") for p in PHRASES_EN] +
        [(n, "number") for n in NUMBERS_AND_CODES] +
        [(s, "sentence_en") for s in SENTENCES_EN]
    )
    
    de_content = (
        [(w, "word_de") for w in SINGLE_WORDS_DE] +
        [(p, "phrase_de") for p in PHRASES_DE] +
        [(s, "sentence_de") for s in SENTENCES_DE]
    )
    
    # Generate English examples
    en_count = int(n_examples * 0.6)  # 60% English
    for _ in range(en_count):
        prefix = random.choice(INSTRUCTION_PREFIXES)
        content, cat = random.choice(en_content)
        question = f"{prefix} {content}"
        answer = content
        pairs.append((question, answer, f"echo_{cat}"))
    
    # Generate German examples
    de_count = int(n_examples * 0.4)  # 40% German
    for _ in range(de_count):
        prefix = random.choice(GERMAN_PREFIXES)
        content, cat = random.choice(de_content)
        question = f"{prefix} {content}"
        answer = content
        pairs.append((question, answer, f"echo_{cat}"))
    
    # Shuffle
    random.shuffle(pairs)
    
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
    parser.add_argument("--n_examples", type=int, default=500, help="Number of examples to generate")
    parser.add_argument("--output_dir", type=str, default="data/sft_echo_diverse", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "echo_diverse.jsonl"
    
    # Generate pairs
    print(f"Generating {args.n_examples} echo instruction examples...")
    pairs = generate_echo_pairs(n_examples=args.n_examples, seed=args.seed)
    
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
