#!/usr/bin/env python3
"""
Generate Multi-turn Conversation SFT episodes for Phase 4.

Creates multi-turn conversations (2-4 turns) that teach the model:
1. Clean turn boundaries (</myPT_assistant><myPT_eot> after each turn)
2. Context carryover across turns
3. Language consistency within a conversation
4. Handling topic switches and clarifications

Patterns:
    FOLLOWUP (40%):         2-3 turns, user asks follow-up on same topic
    CLARIFICATION (25%):    2 turns, user asks for clarification of the answer
    TOPIC_SWITCH (15%):     2-3 turns, topic changes mid-conversation
    CONTEXT_MULTITURN (20%):2-3 turns with assistant_context injected between turns

Input: Workspace docs from workspace/docs/ (markdown files)
Output: JSONL compatible with prepare_chat_sft.py

Usage:
    python scripts/sft/generate_multiturn_sft.py \\
        --docs_dir workspace/docs \\
        --output data/sft_phase4_intermediate/multiturn.jsonl

    python scripts/sft/generate_multiturn_sft.py \\
        --docs_dir workspace/docs \\
        --output data/sft_phase4_intermediate/multiturn.jsonl \\
        --num_examples 3000 --language mixed --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.system_prompts import CONVERSATION_SYSTEM_PROMPT

SYSTEM_PROMPT = CONVERSATION_SYSTEM_PROMPT


# =============================================================================
# Topic Knowledge Base (for generating conversations without docs)
# =============================================================================

TOPICS_EN = [
    {
        "topic": "Python data types",
        "initial_q": "What are the main data types in Python?",
        "initial_a": "Python has several built-in data types: integers, floats, strings, booleans, lists, tuples, dictionaries, and sets. Each serves a different purpose for storing and manipulating data.",
        "followup_q": "What's the difference between a list and a tuple?",
        "followup_a": "Lists are mutable (can be changed after creation) while tuples are immutable (cannot be changed). Lists use square brackets [] and tuples use parentheses (). Tuples are slightly faster and can be used as dictionary keys.",
        "clarify_q": "Can you give me an example of when to use a tuple instead of a list?",
        "clarify_a": "Use a tuple when you have a fixed collection that shouldn't change, like coordinates (x, y), RGB colors (255, 128, 0), or database records. Since tuples are immutable, they're hashable and can be dictionary keys.",
    },
    {
        "topic": "version control with Git",
        "initial_q": "How does Git work?",
        "initial_a": "Git is a distributed version control system that tracks changes to files. It uses commits to save snapshots of your project, branches to work on features in isolation, and merges to combine changes. Every developer has a full copy of the repository.",
        "followup_q": "What is a Git branch?",
        "followup_a": "A Git branch is an independent line of development. It lets you work on features or fixes without affecting the main codebase. The default branch is usually called 'main' or 'master'. You can create, switch between, and merge branches.",
        "clarify_q": "How do I create and switch to a new branch?",
        "clarify_a": "Use 'git checkout -b branch-name' to create and switch to a new branch in one command. Alternatively, use 'git branch branch-name' to create it and 'git checkout branch-name' to switch. With newer Git versions, 'git switch -c branch-name' also works.",
    },
    {
        "topic": "neural networks",
        "initial_q": "What is a neural network?",
        "initial_a": "A neural network is a machine learning model inspired by the human brain. It consists of layers of interconnected nodes (neurons) that process input data through weighted connections. The network learns by adjusting these weights during training.",
        "followup_q": "What are the different types of layers?",
        "followup_a": "The main layer types are: input layers (receive raw data), hidden layers (process information), and output layers (produce predictions). Common specialized layers include convolutional layers (for images), recurrent layers (for sequences), and attention layers (for transformers).",
        "clarify_q": "What does an attention layer do exactly?",
        "clarify_a": "An attention layer allows the model to focus on different parts of the input when producing each output. It computes relevance scores between all pairs of positions, then uses these scores to create weighted combinations. This is the core mechanism behind transformer models like GPT.",
    },
    {
        "topic": "API design",
        "initial_q": "What makes a good REST API?",
        "initial_a": "A good REST API uses clear resource-based URLs, appropriate HTTP methods (GET, POST, PUT, DELETE), consistent response formats (usually JSON), proper status codes, and meaningful error messages. It should also be versioned and well-documented.",
        "followup_q": "What HTTP status codes should I use?",
        "followup_a": "Common status codes: 200 (OK), 201 (Created), 204 (No Content), 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 404 (Not Found), 409 (Conflict), and 500 (Internal Server Error). Always use the most specific code that fits.",
        "clarify_q": "What is the difference between 401 and 403?",
        "clarify_a": "401 Unauthorized means the request lacks valid authentication credentials -- the user hasn't logged in or the token is invalid. 403 Forbidden means the user is authenticated but doesn't have permission to access the resource. Use 401 when identity is unknown, 403 when identity is known but access is denied.",
    },
    {
        "topic": "database indexing",
        "initial_q": "What is database indexing?",
        "initial_a": "Database indexing creates data structures that speed up data retrieval. An index is like a book's index -- instead of scanning every row, the database can jump directly to relevant records. Common types include B-tree indexes, hash indexes, and full-text indexes.",
        "followup_q": "When should I create an index?",
        "followup_a": "Create indexes on columns frequently used in WHERE clauses, JOIN conditions, and ORDER BY statements. Also index columns with high cardinality (many unique values). Avoid over-indexing -- each index slows down INSERT, UPDATE, and DELETE operations.",
        "clarify_q": "What do you mean by high cardinality?",
        "clarify_a": "Cardinality refers to the number of unique values in a column. A column with high cardinality has many distinct values (like user IDs or email addresses), making indexes very effective. Low cardinality columns (like boolean flags or status with 3 values) benefit less from indexing.",
    },
    {
        "topic": "encryption",
        "initial_q": "What is the difference between symmetric and asymmetric encryption?",
        "initial_a": "Symmetric encryption uses one shared key for both encrypting and decrypting (like AES). Asymmetric encryption uses a key pair: a public key for encryption and a private key for decryption (like RSA). Symmetric is faster, asymmetric is more secure for key exchange.",
        "followup_q": "How is HTTPS related to this?",
        "followup_a": "HTTPS uses both. The TLS handshake starts with asymmetric encryption to securely exchange a session key. Then the actual data transfer uses symmetric encryption with that session key. This combines the security of asymmetric key exchange with the speed of symmetric data encryption.",
        "clarify_q": "What happens during the TLS handshake?",
        "clarify_a": "During a TLS handshake: the client sends supported cipher suites, the server responds with its certificate and chosen cipher, the client verifies the certificate, both sides generate a shared secret using Diffie-Hellman or RSA key exchange, and finally both derive symmetric session keys from that secret.",
    },
    {
        "topic": "containerization",
        "initial_q": "What is Docker?",
        "initial_a": "Docker is a platform for building, shipping, and running applications in containers. Containers package an application with all its dependencies into a standardized unit, ensuring it runs consistently across different environments.",
        "followup_q": "How is a container different from a virtual machine?",
        "followup_a": "Containers share the host OS kernel and isolate only the application layer, making them lightweight (MBs, seconds to start). Virtual machines include a full guest OS, making them heavier (GBs, minutes to start). Containers are more resource-efficient but provide less isolation than VMs.",
        "clarify_q": "Can I run containers inside a VM?",
        "clarify_a": "Yes, this is actually a common setup. Cloud providers run containers inside VMs for added security. Docker Desktop on Windows and macOS uses a lightweight Linux VM behind the scenes. Running containers in VMs combines VM-level isolation with container-level efficiency.",
    },
    {
        "topic": "tokenization in NLP",
        "initial_q": "How does BPE tokenization work?",
        "initial_a": "BPE (Byte Pair Encoding) starts with individual characters and iteratively merges the most frequent adjacent pairs. For example, 't' and 'h' might merge into 'th', then 'th' and 'e' into 'the'. This creates a vocabulary of subword units that balances vocabulary size with coverage.",
        "followup_q": "Why not just use word-level tokenization?",
        "followup_a": "Word-level tokenization requires a huge vocabulary to cover all words, and can't handle unseen words (out-of-vocabulary problem). BPE uses subwords, so rare words are split into known pieces. For example, 'unhappiness' becomes 'un' + 'happiness' or 'un' + 'happy' + 'ness'.",
        "clarify_q": "What vocabulary size do modern models typically use?",
        "clarify_a": "GPT-2 uses 50,257 tokens, GPT-4 uses around 100,000, and LLaMA uses 32,000. Larger vocabularies encode text more efficiently (fewer tokens per sentence) but increase the embedding matrix size. The choice is a trade-off between compression efficiency and model parameters.",
    },
]

TOPICS_DE = [
    {
        "topic": "Python Datentypen",
        "initial_q": "Was sind die wichtigsten Datentypen in Python?",
        "initial_a": "Python hat mehrere eingebaute Datentypen: Integers, Floats, Strings, Booleans, Listen, Tupel, Dictionaries und Sets. Jeder dient einem anderen Zweck zur Speicherung und Verarbeitung von Daten.",
        "followup_q": "Was ist der Unterschied zwischen einer Liste und einem Tupel?",
        "followup_a": "Listen sind veraenderbar (koennen nach der Erstellung geaendert werden), waehrend Tupel unveraenderlich sind. Listen verwenden eckige Klammern [] und Tupel verwenden runde Klammern (). Tupel sind etwas schneller und koennen als Dictionary-Schluessel verwendet werden.",
        "clarify_q": "Kannst du ein Beispiel geben, wann man ein Tupel statt einer Liste verwenden sollte?",
        "clarify_a": "Verwende ein Tupel, wenn du eine feste Sammlung hast, die sich nicht aendern sollte, wie Koordinaten (x, y), RGB-Farben (255, 128, 0) oder Datenbankeintraege. Da Tupel unveraenderlich sind, sind sie hashbar und koennen als Dictionary-Schluessel dienen.",
    },
    {
        "topic": "Versionskontrolle mit Git",
        "initial_q": "Wie funktioniert Git?",
        "initial_a": "Git ist ein verteiltes Versionskontrollsystem, das Aenderungen an Dateien verfolgt. Es verwendet Commits um Schnappschuesse des Projekts zu speichern, Branches um isoliert an Features zu arbeiten, und Merges um Aenderungen zusammenzufuehren.",
        "followup_q": "Was ist ein Git Branch?",
        "followup_a": "Ein Git Branch ist eine unabhaengige Entwicklungslinie. Er ermoeglicht es, an Features oder Fixes zu arbeiten, ohne den Hauptcode zu beeinflussen. Der Standard-Branch heisst meist 'main' oder 'master'. Man kann Branches erstellen, zwischen ihnen wechseln und sie zusammenfuehren.",
        "clarify_q": "Wie erstelle und wechsle ich zu einem neuen Branch?",
        "clarify_a": "Verwende 'git checkout -b branch-name' um einen neuen Branch zu erstellen und gleichzeitig zu wechseln. Alternativ 'git branch branch-name' zum Erstellen und 'git checkout branch-name' zum Wechseln. Mit neueren Git-Versionen funktioniert auch 'git switch -c branch-name'.",
    },
    {
        "topic": "Neuronale Netze",
        "initial_q": "Was ist ein neuronales Netz?",
        "initial_a": "Ein neuronales Netz ist ein maschinelles Lernmodell, das vom menschlichen Gehirn inspiriert ist. Es besteht aus Schichten verbundener Knoten (Neuronen), die Eingabedaten durch gewichtete Verbindungen verarbeiten. Das Netzwerk lernt, indem es diese Gewichte waehrend des Trainings anpasst.",
        "followup_q": "Welche verschiedenen Schichttypen gibt es?",
        "followup_a": "Die wichtigsten Schichttypen sind: Eingabeschichten (empfangen Rohdaten), versteckte Schichten (verarbeiten Informationen) und Ausgabeschichten (erzeugen Vorhersagen). Spezialisierte Schichten umfassen Faltungsschichten (fuer Bilder), rekurrente Schichten (fuer Sequenzen) und Aufmerksamkeitsschichten (fuer Transformer).",
        "clarify_q": "Was genau macht eine Aufmerksamkeitsschicht?",
        "clarify_a": "Eine Aufmerksamkeitsschicht ermoeglicht es dem Modell, sich auf verschiedene Teile der Eingabe zu konzentrieren. Sie berechnet Relevanzwerte zwischen allen Positionspaaren und verwendet diese Werte, um gewichtete Kombinationen zu erstellen. Dies ist der Kernmechanismus hinter Transformer-Modellen wie GPT.",
    },
    {
        "topic": "API-Design",
        "initial_q": "Was macht eine gute REST-API aus?",
        "initial_a": "Eine gute REST-API verwendet klare ressourcenbasierte URLs, passende HTTP-Methoden (GET, POST, PUT, DELETE), konsistente Antwortformate (meist JSON), korrekte Statuscodes und aussagekraeftige Fehlermeldungen. Sie sollte auch versioniert und gut dokumentiert sein.",
        "followup_q": "Welche HTTP-Statuscodes sollte ich verwenden?",
        "followup_a": "Haeufige Statuscodes: 200 (OK), 201 (Erstellt), 204 (Kein Inhalt), 400 (Fehlerhafte Anfrage), 401 (Nicht autorisiert), 403 (Verboten), 404 (Nicht gefunden), 409 (Konflikt) und 500 (Interner Serverfehler). Verwende immer den spezifischsten Code.",
        "clarify_q": "Was ist der Unterschied zwischen 401 und 403?",
        "clarify_a": "401 bedeutet, dass gueltige Authentifizierungsdaten fehlen -- der Benutzer ist nicht eingeloggt oder das Token ist ungueltig. 403 bedeutet, dass der Benutzer authentifiziert ist, aber keine Berechtigung fuer die Ressource hat. Verwende 401 wenn die Identitaet unbekannt ist, 403 wenn sie bekannt ist aber der Zugriff verweigert wird.",
    },
    {
        "topic": "Datenbank-Indizierung",
        "initial_q": "Was ist Datenbank-Indizierung?",
        "initial_a": "Datenbank-Indizierung erstellt Datenstrukturen, die den Datenabruf beschleunigen. Ein Index ist wie ein Buchindex -- statt jede Zeile zu durchsuchen, kann die Datenbank direkt zu relevanten Eintraegen springen. Gaengige Typen sind B-Tree-Indizes, Hash-Indizes und Volltextindizes.",
        "followup_q": "Wann sollte ich einen Index erstellen?",
        "followup_a": "Erstelle Indizes auf Spalten, die haeufig in WHERE-Klauseln, JOIN-Bedingungen und ORDER BY-Anweisungen verwendet werden. Auch Spalten mit hoher Kardinalitaet (viele einzigartige Werte) sollten indiziert werden. Vermeide Ueberindizierung, da jeder Index INSERT, UPDATE und DELETE verlangsamt.",
        "clarify_q": "Was meinst du mit hoher Kardinalitaet?",
        "clarify_a": "Kardinalitaet bezieht sich auf die Anzahl einzigartiger Werte in einer Spalte. Eine Spalte mit hoher Kardinalitaet hat viele verschiedene Werte (wie Benutzer-IDs oder E-Mail-Adressen), wodurch Indizes sehr effektiv sind. Spalten mit niedriger Kardinalitaet (wie boolesche Flags) profitieren weniger von der Indizierung.",
    },
    {
        "topic": "Verschluesselung",
        "initial_q": "Was ist der Unterschied zwischen symmetrischer und asymmetrischer Verschluesselung?",
        "initial_a": "Symmetrische Verschluesselung verwendet einen gemeinsamen Schluessel zum Ver- und Entschluesseln (wie AES). Asymmetrische Verschluesselung verwendet ein Schluesselpaar: einen oeffentlichen Schluessel zum Verschluesseln und einen privaten zum Entschluesseln (wie RSA). Symmetrisch ist schneller, asymmetrisch ist sicherer fuer den Schluesselaustausch.",
        "followup_q": "Wie haengt HTTPS damit zusammen?",
        "followup_a": "HTTPS verwendet beides. Der TLS-Handshake beginnt mit asymmetrischer Verschluesselung zum sicheren Austausch eines Sitzungsschluessels. Dann verwendet die eigentliche Datenuebertragung symmetrische Verschluesselung mit diesem Sitzungsschluessel.",
        "clarify_q": "Was passiert waehrend des TLS-Handshakes?",
        "clarify_a": "Waehrend eines TLS-Handshakes: Der Client sendet unterstuetzte Cipher-Suites, der Server antwortet mit seinem Zertifikat, der Client verifiziert das Zertifikat, beide Seiten generieren ein gemeinsames Geheimnis und leiten daraus symmetrische Sitzungsschluessel ab.",
    },
]

# Switch topics for TOPIC_SWITCH pattern
SWITCH_TOPICS_EN = [
    ("Actually, I have a different question. What is JSON?",
     "JSON (JavaScript Object Notation) is a lightweight data interchange format. It uses key-value pairs and arrays, is human-readable, and widely used for APIs and configuration files. Example: {\"name\": \"Alice\", \"age\": 30}."),
    ("Switching topics -- how do I read a file in Python?",
     "Use the open() function: with open('file.txt', 'r') as f: content = f.read(). The 'with' statement ensures the file is properly closed. Use 'r' for reading, 'w' for writing, and 'a' for appending."),
    ("New question: what is a hash function?",
     "A hash function takes input of any size and produces a fixed-size output (hash). Good hash functions are fast, deterministic, and resistant to collisions. Common uses include password storage, data integrity verification, and hash tables."),
    ("Can we talk about something else? What is recursion?",
     "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems. Every recursive function needs a base case (stopping condition) to prevent infinite recursion. Classic example: factorial(n) = n * factorial(n-1)."),
    ("Different topic: what is a linked list?",
     "A linked list is a data structure where each element (node) contains data and a pointer to the next node. Unlike arrays, linked lists don't require contiguous memory. Insertion and deletion are O(1) at known positions, but random access is O(n)."),
]

SWITCH_TOPICS_DE = [
    ("Eigentlich habe ich eine andere Frage. Was ist JSON?",
     "JSON (JavaScript Object Notation) ist ein leichtgewichtiges Datenaustauschformat. Es verwendet Schluessel-Wert-Paare und Arrays, ist menschenlesbar und wird haeufig fuer APIs und Konfigurationsdateien verwendet."),
    ("Themenwechsel -- wie lese ich eine Datei in Python?",
     "Verwende die open()-Funktion: with open('datei.txt', 'r') as f: inhalt = f.read(). Die 'with'-Anweisung stellt sicher, dass die Datei ordnungsgemaess geschlossen wird. Verwende 'r' zum Lesen, 'w' zum Schreiben und 'a' zum Anhaengen."),
    ("Neue Frage: Was ist eine Hashfunktion?",
     "Eine Hashfunktion nimmt Eingaben beliebiger Groesse und erzeugt eine Ausgabe fester Groesse. Gute Hashfunktionen sind schnell, deterministisch und kollisionsresistent. Haeufige Anwendungen sind Passwortspeicherung und Datenintegritaetspruefung."),
    ("Koennen wir ueber etwas anderes reden? Was ist Rekursion?",
     "Rekursion ist, wenn eine Funktion sich selbst aufruft, um ein Problem zu loesen, indem es in kleinere Teilprobleme zerlegt wird. Jede rekursive Funktion braucht einen Basisfall (Abbruchbedingung), um unendliche Rekursion zu verhindern."),
    ("Anderes Thema: Was ist eine verkettete Liste?",
     "Eine verkettete Liste ist eine Datenstruktur, bei der jedes Element (Knoten) Daten und einen Zeiger auf den naechsten Knoten enthaelt. Anders als Arrays benoetigen verkettete Listen keinen zusammenhaengenden Speicher. Einfuegen und Loeschen sind O(1) an bekannten Positionen."),
]

# Context injections for CONTEXT_MULTITURN
CONTEXT_INJECTIONS_EN = [
    "The user is working on a Python web application using Flask.",
    "The user's project uses PostgreSQL as the database backend.",
    "The user is a beginner learning programming for the first time.",
    "The user is preparing for a technical interview.",
    "The user is building a machine learning pipeline for text classification.",
    "The user is migrating an application from Python 2 to Python 3.",
]

CONTEXT_INJECTIONS_DE = [
    "Der Benutzer arbeitet an einer Python-Webanwendung mit Flask.",
    "Das Projekt des Benutzers verwendet PostgreSQL als Datenbank-Backend.",
    "Der Benutzer ist ein Anfaenger, der zum ersten Mal Programmieren lernt.",
    "Der Benutzer bereitet sich auf ein technisches Vorstellungsgespraech vor.",
    "Der Benutzer baut eine Machine-Learning-Pipeline fuer Textklassifikation.",
    "Der Benutzer migriert eine Anwendung von Python 2 zu Python 3.",
]


# =============================================================================
# Episode Generation
# =============================================================================

def generate_followup(topic_data: Dict, rng: random.Random, num_turns: int = 2) -> Dict:
    """Generate a FOLLOWUP episode: initial question + follow-up(s)."""
    messages = [
        {"role": "user", "content": topic_data["initial_q"]},
        {"role": "assistant", "content": topic_data["initial_a"]},
        {"role": "user", "content": topic_data["followup_q"]},
        {"role": "assistant", "content": topic_data["followup_a"]},
    ]

    if num_turns >= 3 and "clarify_q" in topic_data:
        messages.extend([
            {"role": "user", "content": topic_data["clarify_q"]},
            {"role": "assistant", "content": topic_data["clarify_a"]},
        ])

    return {"system": SYSTEM_PROMPT, "messages": messages}


def generate_clarification(topic_data: Dict, rng: random.Random) -> Dict:
    """Generate a CLARIFICATION episode: question + answer + clarification request."""
    messages = [
        {"role": "user", "content": topic_data["initial_q"]},
        {"role": "assistant", "content": topic_data["initial_a"]},
        {"role": "user", "content": topic_data["clarify_q"]},
        {"role": "assistant", "content": topic_data["clarify_a"]},
    ]
    return {"system": SYSTEM_PROMPT, "messages": messages}


def generate_topic_switch(
    topic_data: Dict, switch_topics: List, rng: random.Random
) -> Dict:
    """Generate a TOPIC_SWITCH episode: start one topic, switch to another."""
    switch_q, switch_a = rng.choice(switch_topics)

    messages = [
        {"role": "user", "content": topic_data["initial_q"]},
        {"role": "assistant", "content": topic_data["initial_a"]},
        {"role": "user", "content": switch_q},
        {"role": "assistant", "content": switch_a},
    ]
    return {"system": SYSTEM_PROMPT, "messages": messages}


def generate_context_multiturn(
    topic_data: Dict, context_injections: List, rng: random.Random
) -> Dict:
    """Generate a CONTEXT_MULTITURN episode with assistant_context between turns."""
    context = rng.choice(context_injections)

    messages = [
        {"role": "user", "content": topic_data["initial_q"]},
        {"role": "assistant", "content": topic_data["initial_a"]},
        {"role": "assistant_context", "content": context},
        {"role": "user", "content": topic_data["followup_q"]},
        {"role": "assistant", "content": topic_data["followup_a"]},
    ]
    return {"system": SYSTEM_PROMPT, "messages": messages}


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Multi-turn Conversation SFT episodes (Phase 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs_dir", type=str, default="workspace/docs",
                        help="Directory with workspace docs (for future enrichment)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--num_examples", type=int, default=2000,
                        help="Number of episodes to generate (default: 2000)")
    parser.add_argument("--language", type=str, default="mixed",
                        choices=["en", "de", "mixed"],
                        help="Language: en, de, or mixed (default: mixed)")
    parser.add_argument("--de_ratio", type=float, default=0.35,
                        help="Fraction of DE episodes when language=mixed (default: 0.35)")
    parser.add_argument("--followup_ratio", type=float, default=0.40,
                        help="Fraction FOLLOWUP (default: 0.40)")
    parser.add_argument("--clarification_ratio", type=float, default=0.25,
                        help="Fraction CLARIFICATION (default: 0.25)")
    parser.add_argument("--topic_switch_ratio", type=float, default=0.15,
                        help="Fraction TOPIC_SWITCH (default: 0.15)")
    parser.add_argument("--context_multiturn_ratio", type=float, default=0.20,
                        help="Fraction CONTEXT_MULTITURN (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 60)
    print("  MyPT Phase 4: Multi-turn Conversation SFT Generator")
    print("=" * 60)

    print(f"\n  Language: {args.language} (DE ratio: {args.de_ratio})")
    print(f"  Target: {args.num_examples} episodes")
    print(f"  EN topics: {len(TOPICS_EN)}, DE topics: {len(TOPICS_DE)}")

    episodes = []
    pattern_counts = {
        "followup": 0, "clarification": 0,
        "topic_switch": 0, "context_multiturn": 0,
    }

    for i in range(args.num_examples):
        if args.language == "mixed":
            lang = "de" if rng.random() < args.de_ratio else "en"
        else:
            lang = args.language

        topics = TOPICS_DE if lang == "de" else TOPICS_EN
        switch_topics = SWITCH_TOPICS_DE if lang == "de" else SWITCH_TOPICS_EN
        context_inj = CONTEXT_INJECTIONS_DE if lang == "de" else CONTEXT_INJECTIONS_EN

        topic_data = rng.choice(topics)

        r = rng.random()
        if r < args.followup_ratio:
            pattern = "followup"
            num_turns = rng.choice([2, 2, 3])
            ep = generate_followup(topic_data, rng, num_turns)
        elif r < args.followup_ratio + args.clarification_ratio:
            pattern = "clarification"
            ep = generate_clarification(topic_data, rng)
        elif r < args.followup_ratio + args.clarification_ratio + args.topic_switch_ratio:
            pattern = "topic_switch"
            ep = generate_topic_switch(topic_data, switch_topics, rng)
        else:
            pattern = "context_multiturn"
            ep = generate_context_multiturn(topic_data, context_inj, rng)

        ep["language"] = lang
        episodes.append(ep)
        pattern_counts[pattern] += 1

    rng.shuffle(episodes)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\n  Generated {len(episodes)} episodes -> {args.output}")
    print(f"\n  Pattern distribution:")
    for pattern, count in sorted(pattern_counts.items()):
        pct = count / len(episodes) * 100 if episodes else 0
        print(f"    {pattern}: {count} ({pct:.1f}%)")

    lang_counts = {"en": 0, "de": 0}
    for ep in episodes:
        lang_counts[ep.get("language", "en")] += 1
    print(f"\n  Language distribution:")
    for lang, count in sorted(lang_counts.items()):
        pct = count / len(episodes) * 100 if episodes else 0
        print(f"    {lang}: {count} ({pct:.1f}%)")

    turn_counts = {}
    for ep in episodes:
        n = sum(1 for m in ep["messages"] if m["role"] == "assistant")
        turn_counts[n] = turn_counts.get(n, 0) + 1
    print(f"\n  Turn distribution:")
    for turns, count in sorted(turn_counts.items()):
        pct = count / len(episodes) * 100 if episodes else 0
        print(f"    {turns} assistant turns: {count} ({pct:.1f}%)")

    print(f"\n  Next step:")
    print(f"    python scripts/sft/prepare_chat_sft.py \\")
    print(f"        --input {args.output} \\")
    print(f"        --output_dir data/sft_phase4_multiturn \\")
    print(f"        --enable_packing --pack_block_size 1024")


if __name__ == "__main__":
    main()
