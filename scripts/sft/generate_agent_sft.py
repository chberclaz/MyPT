#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic agentic RAG training data from workspace documents (Phase 5).

Creates single-step tool-calling conversation examples for SFT training:
1. Loads documents from a workspace directory
2. Generates synthetic conversations with realistic tool usage patterns
3. Adds <myPT_think> reasoning before tool calls (optional)
4. Adds <myPT_cite> source attribution on final answers
5. Includes NO_TOOL pattern (answer directly without tools)
6. Supports bilingual generation (EN + DE)
7. Outputs JSONL ready for prepare_tool_sft.py

Generated conversation patterns:
- workspace.search -> find relevant docs -> answer based on results
- workspace.list_docs -> see available docs -> describe workspace
- workspace.get_doc -> retrieve full document -> summarize or quote
- workspace.summarize -> summarize long content -> report findings
- Multi-step: search -> get_doc -> answer
- NO_TOOL: answer directly from general knowledge (no tool call)

Usage:
    python scripts/sft/generate_agent_sft.py \\
        --workspace_dir workspace/ --output data/sft_phase5_intermediate/tool_episodes.jsonl

    python scripts/sft/generate_agent_sft.py \\
        --workspace_dir workspace/ --output data/sft_phase5_intermediate/tool_episodes.jsonl \\
        --num_examples 5000 --language mixed --include_think --seed 42

Then prepare the SFT dataset:
    python scripts/sft/prepare_tool_sft.py \\
        --input data/sft_phase5_intermediate/tool_episodes.jsonl \\
        --output_dir data/sft_phase5_toolcall
"""

import argparse
import os
import sys
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.document import DocumentLoader, TextChunker


def get_doc_id(doc) -> str:
    """Generate a unique ID for a document."""
    return hashlib.md5(doc.source.encode()).hexdigest()[:12]


def get_doc_title(doc) -> str:
    """Get a human-readable title for a document."""
    name = os.path.splitext(doc.filename)[0]
    return name.replace("_", " ").replace("-", " ").title()


# ============================================================
# Question Templates - These simulate realistic user questions
# ============================================================

SEARCH_QUESTIONS_EN = [
    "What information do we have about {topic}?",
    "Find documents related to {topic}.",
    "Search for anything about {topic}.",
    "Can you look up {topic} in the workspace?",
    "What do our docs say about {topic}?",
    "I need information on {topic}.",
    "Tell me about {topic} from the documents.",
    "Look for {topic} in the knowledge base.",
    "Do we have anything about {topic}?",
    "Check if there's documentation on {topic}.",
    "Search the workspace for {topic}.",
    "Any docs mentioning {topic}?",
    "Pull up everything we have on {topic}.",
    "Is there a document about {topic}?",
]

SEARCH_QUESTIONS_DE = [
    "Welche Informationen haben wir ueber {topic}?",
    "Finde Dokumente zu {topic}.",
    "Suche nach {topic} im Workspace.",
    "Kannst du {topic} in der Wissensdatenbank nachschlagen?",
    "Was sagen unsere Dokumente ueber {topic}?",
    "Ich brauche Informationen zu {topic}.",
    "Erzaehl mir ueber {topic} aus den Dokumenten.",
    "Suche nach {topic} in der Wissensdatenbank.",
    "Haben wir irgendwas zu {topic}?",
    "Pruefe ob es Dokumentation zu {topic} gibt.",
    "Durchsuche den Workspace nach {topic}.",
    "Gibt es Dokumente, die {topic} erwaehnen?",
    "Zeig mir alles was wir zu {topic} haben.",
    "Gibt es ein Dokument ueber {topic}?",
]

LIST_QUESTIONS_EN = [
    "What documents do we have?",
    "List all available documents.",
    "Show me what's in the workspace.",
    "What files are in the knowledge base?",
    "What documents can I access?",
    "Give me an overview of the workspace.",
    "How many documents are available?",
    "Show me all files.",
    "What's in the document library?",
    "List the workspace contents.",
]

LIST_QUESTIONS_DE = [
    "Welche Dokumente haben wir?",
    "Liste alle verfuegbaren Dokumente auf.",
    "Zeig mir was im Workspace ist.",
    "Welche Dateien sind in der Wissensdatenbank?",
    "Auf welche Dokumente kann ich zugreifen?",
    "Gib mir einen Ueberblick ueber den Workspace.",
    "Wie viele Dokumente sind verfuegbar?",
    "Zeig mir alle Dateien.",
    "Was ist in der Dokumentenbibliothek?",
    "Liste den Workspace-Inhalt auf.",
]

GET_DOC_QUESTIONS_EN = [
    "Show me the contents of {filename}.",
    "What's in {filename}?",
    "Read {filename} for me.",
    "Display the document {filename}.",
    "I need to see {filename}.",
    "Open {filename} and tell me what's there.",
    "Get me the full text of {filename}.",
    "What does {filename} contain?",
    "Pull up {filename} please.",
    "Retrieve {filename} from the workspace.",
]

GET_DOC_QUESTIONS_DE = [
    "Zeig mir den Inhalt von {filename}.",
    "Was steht in {filename}?",
    "Lies {filename} fuer mich.",
    "Zeige das Dokument {filename} an.",
    "Ich muss {filename} sehen.",
    "Oeffne {filename} und sag mir was drin steht.",
    "Gib mir den vollen Text von {filename}.",
    "Was enthaelt {filename}?",
    "Zeig mir {filename} bitte.",
    "Rufe {filename} aus dem Workspace ab.",
]

SUMMARIZE_QUESTIONS_EN = [
    "Summarize {filename}.",
    "Give me a summary of {filename}.",
    "What's the main point of {filename}?",
    "TL;DR for {filename}?",
    "Can you condense {filename}?",
    "What are the key takeaways from {filename}?",
    "Break down {filename} for me briefly.",
    "Give me the highlights of {filename}.",
]

SUMMARIZE_QUESTIONS_DE = [
    "Fasse {filename} zusammen.",
    "Gib mir eine Zusammenfassung von {filename}.",
    "Was ist der Hauptpunkt von {filename}?",
    "Kurzfassung von {filename}?",
    "Kannst du {filename} zusammenfassen?",
    "Was sind die wichtigsten Erkenntnisse aus {filename}?",
    "Erklaere mir {filename} kurz.",
    "Gib mir die Highlights von {filename}.",
]

MULTI_STEP_QUESTIONS_EN = [
    "Find documents about {topic} and summarize the key points.",
    "What's in our {topic} documentation? Give me the highlights.",
    "Search for {topic} and explain what you find.",
    "Look up {topic} and provide a detailed answer.",
    "Research {topic} in our docs and give me a thorough answer.",
    "Find and read the most relevant document about {topic}.",
    "I need a deep dive on {topic}. Search and summarize what we have.",
    "Check our knowledge base for {topic} and report back.",
]

MULTI_STEP_QUESTIONS_DE = [
    "Finde Dokumente ueber {topic} und fasse die wichtigsten Punkte zusammen.",
    "Was steht in unserer {topic}-Dokumentation? Gib mir die Highlights.",
    "Suche nach {topic} und erklaere was du findest.",
    "Schlag {topic} nach und gib eine detaillierte Antwort.",
    "Recherchiere {topic} in unseren Docs und gib mir eine gruendliche Antwort.",
    "Finde und lies das relevanteste Dokument zu {topic}.",
    "Ich brauche eine tiefe Analyse zu {topic}. Suche und fasse zusammen.",
    "Pruefe unsere Wissensdatenbank zu {topic} und berichte.",
]


# ============================================================
# Answer Templates - These simulate model responses
# ============================================================

SEARCH_ANSWERS_EN = [
    "Based on the search results, I found information about {topic}:\n\n{content}",
    "Here's what I found about {topic}:\n\n{content}",
    "The workspace contains the following relevant information about {topic}:\n\n{content}",
    "From the documents, here's what relates to {topic}:\n\n{content}",
    "I found relevant information about {topic}. {content}",
    "The workspace contains the following about {topic}: {content}",
    "Here are the results about {topic}: {content}",
    "The documents mention the following regarding {topic}: {content}",
]

SEARCH_ANSWERS_DE = [
    "Basierend auf den Suchergebnissen habe ich Informationen ueber {topic} gefunden:\n\n{content}",
    "Hier ist was ich ueber {topic} gefunden habe:\n\n{content}",
    "Der Workspace enthaelt folgende relevante Informationen zu {topic}:\n\n{content}",
    "Aus den Dokumenten, hier ist was sich auf {topic} bezieht:\n\n{content}",
    "Ich habe relevante Informationen zu {topic} gefunden. {content}",
    "Der Workspace enthaelt Folgendes zu {topic}: {content}",
    "Hier sind die Ergebnisse zu {topic}: {content}",
    "Die Dokumente erwaehnen Folgendes bezueglich {topic}: {content}",
]

LIST_ANSWERS_EN = [
    "The workspace contains {count} documents:\n\n{content}",
    "Here are the available documents ({count} total):\n\n{content}",
    "I found {count} documents in the workspace:\n\n{content}",
    "I found the following documents in your workspace:\n\n{content}",
    "Your workspace has these files:\n\n{content}",
    "The available documents are:\n\n{content}",
]

LIST_ANSWERS_DE = [
    "Der Workspace enthaelt {count} Dokumente:\n\n{content}",
    "Hier sind die verfuegbaren Dokumente ({count} insgesamt):\n\n{content}",
    "Ich habe {count} Dokumente im Workspace gefunden:\n\n{content}",
    "Ich habe folgende Dokumente in Ihrem Workspace gefunden:\n\n{content}",
    "Ihr Workspace enthaelt diese Dateien:\n\n{content}",
    "Die verfuegbaren Dokumente sind:\n\n{content}",
]

GET_DOC_ANSWERS_EN = [
    "Here's the content of {filename}:\n\n{content}",
    "The document {filename} contains:\n\n{content}",
    "{filename} says:\n\n{content}",
    "Here is what's in {filename}:\n\n{content}",
    "The full text of {filename}:\n\n{content}",
    "I've retrieved {filename} for you:\n\n{content}",
]

GET_DOC_ANSWERS_DE = [
    "Hier ist der Inhalt von {filename}:\n\n{content}",
    "Das Dokument {filename} enthaelt:\n\n{content}",
    "In {filename} steht:\n\n{content}",
    "Folgendes steht in {filename}:\n\n{content}",
    "Der vollstaendige Text von {filename}:\n\n{content}",
    "Ich habe {filename} fuer Sie abgerufen:\n\n{content}",
]

SUMMARIZE_ANSWERS_EN = [
    "Here's a summary of {filename}:\n\n{summary}",
    "The main points from {filename}:\n\n{summary}",
    "Summary of {filename}:\n\n{summary}",
    "The key takeaways from {filename} are:\n\n{summary}",
    "In short, {filename} covers:\n\n{summary}",
    "Here are the highlights from {filename}:\n\n{summary}",
]

SUMMARIZE_ANSWERS_DE = [
    "Hier ist eine Zusammenfassung von {filename}:\n\n{summary}",
    "Die wichtigsten Punkte aus {filename}:\n\n{summary}",
    "Zusammenfassung von {filename}:\n\n{summary}",
    "Die wichtigsten Erkenntnisse aus {filename} sind:\n\n{summary}",
    "Kurz gesagt behandelt {filename}:\n\n{summary}",
    "Hier sind die Highlights aus {filename}:\n\n{summary}",
]

NO_RESULTS_ANSWERS_EN = [
    "I couldn't find any documents related to {topic}. The workspace might not have information on this topic.",
    "No relevant documents found for {topic}. You may need to add relevant documentation first.",
    "Sorry, the search for {topic} didn't return any results.",
    "The workspace doesn't seem to contain documents about {topic}. Try adding some relevant files first.",
    "I searched for {topic} but found nothing in the current workspace.",
    "There are no documents matching {topic} at the moment.",
]

NO_RESULTS_ANSWERS_DE = [
    "Ich konnte keine Dokumente zu {topic} finden. Der Workspace hat moeglicherweise keine Informationen zu diesem Thema.",
    "Keine relevanten Dokumente fuer {topic} gefunden. Moeglicherweise muessen erst entsprechende Dokumente hinzugefuegt werden.",
    "Entschuldigung, die Suche nach {topic} hat keine Ergebnisse geliefert.",
    "Der Workspace scheint keine Dokumente zu {topic} zu enthalten. Versuchen Sie, relevante Dateien hinzuzufuegen.",
    "Ich habe nach {topic} gesucht, aber nichts im aktuellen Workspace gefunden.",
    "Derzeit gibt es keine Dokumente, die zu {topic} passen.",
]


# ============================================================
# Think Templates (for <myPT_think> reasoning before tool calls)
# ============================================================

THINK_SEARCH_EN = [
    "The user wants information about {topic}. I should search the workspace for relevant documents.",
    "I need to find documents about {topic}. Let me search the knowledge base.",
    "To answer this question about {topic}, I'll search the workspace first.",
    "Let me look up information about {topic} in the workspace documents.",
    "The user needs information on {topic}. I'll search for relevant content.",
    "To answer this, I should check what documents mention {topic}.",
    "I'll query the workspace to find documents related to {topic}.",
]

THINK_SEARCH_DE = [
    "Der Benutzer moechte Informationen ueber {topic}. Ich sollte den Workspace nach relevanten Dokumenten durchsuchen.",
    "Ich muss Dokumente zu {topic} finden. Lass mich die Wissensdatenbank durchsuchen.",
    "Um diese Frage zu {topic} zu beantworten, durchsuche ich zuerst den Workspace.",
    "Lass mich Informationen zu {topic} in den Workspace-Dokumenten nachschlagen.",
    "Der Benutzer braucht Informationen zu {topic}. Ich suche nach relevanten Inhalten.",
    "Um das zu beantworten, sollte ich pruefen, welche Dokumente {topic} erwaehnen.",
    "Ich durchsuche den Workspace nach Dokumenten zu {topic}.",
]

THINK_LIST_EN = [
    "The user wants to see what documents are available. I'll list all documents.",
    "Let me check what's in the workspace by listing all documents.",
    "I should show the user what files are available in their workspace.",
    "The user is curious about their documents. Let me list everything.",
    "I'll retrieve a list of all documents for the user.",
]

THINK_LIST_DE = [
    "Der Benutzer moechte sehen welche Dokumente verfuegbar sind. Ich liste alle Dokumente auf.",
    "Lass mich pruefen was im Workspace ist, indem ich alle Dokumente aufliste.",
    "Ich sollte dem Benutzer zeigen, welche Dateien in seinem Workspace verfuegbar sind.",
    "Der Benutzer ist neugierig auf seine Dokumente. Lass mich alles auflisten.",
    "Ich rufe eine Liste aller Dokumente fuer den Benutzer ab.",
]

THINK_GETDOC_EN = [
    "The user wants to see {filename}. I'll retrieve the document content.",
    "Let me fetch the contents of {filename} for the user.",
    "I'll retrieve {filename} so the user can see its contents.",
    "The user wants to read {filename}. Let me get it.",
    "I need to fetch the content of {filename} for the user.",
]

THINK_GETDOC_DE = [
    "Der Benutzer moechte {filename} sehen. Ich rufe den Dokumentinhalt ab.",
    "Lass mich den Inhalt von {filename} fuer den Benutzer abrufen.",
    "Ich rufe {filename} ab, damit der Benutzer den Inhalt sehen kann.",
    "Der Benutzer moechte {filename} lesen. Lass mich es holen.",
    "Ich muss den Inhalt von {filename} fuer den Benutzer abrufen.",
]

THINK_SUMMARIZE_EN = [
    "The user wants a summary of {filename}. I'll use the summarize tool.",
    "Let me summarize {filename} for a concise overview.",
    "I'll generate a concise summary of {filename} for the user.",
    "The user wants the key points from {filename}. Let me summarize it.",
    "A summary of {filename} would help the user. I'll use the summarize tool.",
]

THINK_SUMMARIZE_DE = [
    "Der Benutzer moechte eine Zusammenfassung von {filename}. Ich verwende das Zusammenfassungs-Tool.",
    "Lass mich {filename} fuer einen kompakten Ueberblick zusammenfassen.",
    "Ich erstelle eine praegnante Zusammenfassung von {filename} fuer den Benutzer.",
    "Der Benutzer moechte die wichtigsten Punkte aus {filename}. Lass mich zusammenfassen.",
    "Eine Zusammenfassung von {filename} waere hilfreich. Ich verwende das Zusammenfassungs-Tool.",
]

THINK_MULTI_STEP_EN = [
    "The user wants detailed information about {topic}. I'll search first, then get the full document.",
    "To give a thorough answer about {topic}, I should search and then read the relevant document.",
    "I need to search first and then read the most relevant document about {topic}.",
    "For a thorough answer about {topic}, I'll do a search followed by retrieving the best match.",
    "Let me find documents about {topic} first, then get the full content of the most relevant one.",
    "This requires multiple steps: search for {topic}, then retrieve the document for details.",
]

THINK_MULTI_STEP_DE = [
    "Der Benutzer moechte detaillierte Informationen zu {topic}. Ich suche zuerst und hole dann das vollstaendige Dokument.",
    "Um eine gruendliche Antwort zu {topic} zu geben, sollte ich suchen und dann das relevante Dokument lesen.",
    "Ich muss zuerst suchen und dann das relevanteste Dokument zu {topic} lesen.",
    "Fuer eine gruendliche Antwort zu {topic} mache ich zuerst eine Suche und rufe dann das beste Ergebnis ab.",
    "Lass mich zuerst Dokumente zu {topic} finden und dann den vollstaendigen Inhalt des relevantesten abrufen.",
    "Das erfordert mehrere Schritte: nach {topic} suchen, dann das Dokument fuer Details abrufen.",
]


# ============================================================
# NO_TOOL Pattern Templates
# ============================================================

NO_TOOL_QUESTIONS_EN = [
    "What is Python?",
    "How does version control work?",
    "Explain the concept of an API.",
    "What is machine learning?",
    "Tell me about neural networks.",
    "What is a database index?",
    "How does encryption work?",
    "What is a REST API?",
    "Explain what a hash function does.",
    "What is the difference between HTTP and HTTPS?",
    "How do you write a for loop in Python?",
    "What are environment variables?",
    "Explain JSON format.",
    "What is a Docker container?",
    "How does a transformer model work?",
]

NO_TOOL_QUESTIONS_DE = [
    "Was ist Python?",
    "Wie funktioniert Versionskontrolle?",
    "Erklaere das Konzept einer API.",
    "Was ist maschinelles Lernen?",
    "Erzaehl mir ueber neuronale Netze.",
    "Was ist ein Datenbankindex?",
    "Wie funktioniert Verschluesselung?",
    "Was ist eine REST-API?",
    "Erklaere was eine Hashfunktion macht.",
    "Was ist der Unterschied zwischen HTTP und HTTPS?",
    "Wie schreibt man eine for-Schleife in Python?",
    "Was sind Umgebungsvariablen?",
    "Erklaere das JSON-Format.",
    "Was ist ein Docker-Container?",
    "Wie funktioniert ein Transformer-Modell?",
]

NO_TOOL_ANSWERS_EN = [
    ("What is Python?",
     "Python is a high-level, interpreted programming language known for its clear syntax and readability. It supports multiple paradigms including object-oriented and functional programming."),
    ("How does version control work?",
     "Version control tracks changes to files over time. Git, the most popular system, uses commits (snapshots), branches (parallel development lines), and merges to manage code history and collaboration."),
    ("Explain the concept of an API.",
     "An API (Application Programming Interface) defines how software components communicate. It specifies the methods, data formats, and protocols for requesting and exchanging information between systems."),
    ("What is machine learning?",
     "Machine learning is a subset of AI where systems learn patterns from data without explicit programming. It includes supervised learning (labeled data), unsupervised learning (finding structure), and reinforcement learning (learning from rewards)."),
    ("Tell me about neural networks.",
     "Neural networks are computational models with layers of interconnected nodes. Data flows through input, hidden, and output layers. The network learns by adjusting connection weights during training to minimize prediction errors."),
    ("What is a database index?",
     "A database index is a data structure that speeds up data retrieval, similar to a book index. It lets the database jump directly to relevant rows instead of scanning the entire table. Common types are B-tree and hash indexes."),
    ("How does encryption work?",
     "Encryption converts readable data into an encoded format using a key. Symmetric encryption uses one shared key, while asymmetric uses a public/private key pair. The encrypted data can only be decoded with the correct key."),
    ("What is a REST API?",
     "A REST API follows representational state transfer principles: it uses HTTP methods (GET, POST, PUT, DELETE) with resource-based URLs, returns JSON responses, and is stateless -- each request contains all needed information."),
    ("Explain what a hash function does.",
     "A hash function takes input of any size and produces a fixed-size output. Good hash functions are deterministic, fast, and collision-resistant. They're used for password storage, data integrity checks, and hash tables."),
    ("What is the difference between HTTP and HTTPS?",
     "HTTPS is HTTP with TLS encryption. HTTP sends data in plain text, while HTTPS encrypts all communication between client and server. HTTPS uses certificates to verify server identity and protect against eavesdropping."),
    ("How do you write a for loop in Python?",
     "In Python, a for loop iterates over a sequence: for item in [1, 2, 3]: print(item). You can also use range(): for i in range(10): print(i). Python's for loops work with any iterable object."),
    ("What are environment variables?",
     "Environment variables are key-value pairs set in the operating system that applications can read. They store configuration like API keys, database URLs, and paths. In Python, access them via os.environ['VAR_NAME']."),
    ("Explain JSON format.",
     "JSON (JavaScript Object Notation) is a lightweight data format using key-value pairs and arrays. It's human-readable, language-independent, and widely used for APIs and configuration. Example: {\"name\": \"Alice\", \"age\": 30}."),
    ("What is a Docker container?",
     "A Docker container packages an application with all its dependencies into a standardized unit. Containers are lightweight (share the host OS kernel), portable, and start in seconds. They ensure consistent behavior across environments."),
    ("How does a transformer model work?",
     "Transformers process sequences using self-attention, which computes relevance scores between all positions in parallel. This replaced sequential processing in older models. Key components are multi-head attention, feed-forward layers, and positional encodings."),
]

NO_TOOL_ANSWERS_DE = [
    ("Was ist Python?",
     "Python ist eine hoeherwertige, interpretierte Programmiersprache, bekannt fuer ihre klare Syntax und Lesbarkeit. Sie unterstuetzt mehrere Paradigmen einschliesslich objektorientierter und funktionaler Programmierung."),
    ("Wie funktioniert Versionskontrolle?",
     "Versionskontrolle verfolgt Aenderungen an Dateien ueber die Zeit. Git verwendet Commits (Schnappschuesse), Branches (parallele Entwicklungslinien) und Merges zur Verwaltung der Code-Historie und Zusammenarbeit."),
    ("Erklaere das Konzept einer API.",
     "Eine API (Application Programming Interface) definiert wie Softwarekomponenten kommunizieren. Sie spezifiziert die Methoden, Datenformate und Protokolle fuer den Informationsaustausch zwischen Systemen."),
    ("Was ist maschinelles Lernen?",
     "Maschinelles Lernen ist ein Teilgebiet der KI, bei dem Systeme Muster aus Daten lernen. Es umfasst ueberwachtes Lernen (markierte Daten), unueberwachtes Lernen (Strukturfindung) und Reinforcement Learning (Lernen durch Belohnung)."),
    ("Erzaehl mir ueber neuronale Netze.",
     "Neuronale Netze sind Rechenmodelle mit Schichten verbundener Knoten. Daten fliessen durch Eingabe-, versteckte und Ausgabeschichten. Das Netzwerk lernt durch Anpassung der Verbindungsgewichte waehrend des Trainings."),
    ("Was ist ein Datenbankindex?",
     "Ein Datenbankindex ist eine Datenstruktur, die den Datenabruf beschleunigt. Er laesst die Datenbank direkt zu relevanten Zeilen springen, statt die gesamte Tabelle zu durchsuchen. Gaengige Typen sind B-Tree- und Hash-Indizes."),
    ("Wie funktioniert Verschluesselung?",
     "Verschluesselung wandelt lesbare Daten mit einem Schluessel in ein codiertes Format um. Symmetrische Verschluesselung verwendet einen gemeinsamen Schluessel, asymmetrische ein oeffentliches/privates Schluesselpaar."),
    ("Was ist eine REST-API?",
     "Eine REST-API folgt den Prinzipien des Representational State Transfer: Sie verwendet HTTP-Methoden (GET, POST, PUT, DELETE) mit ressourcenbasierten URLs, gibt JSON-Antworten zurueck und ist zustandslos."),
    ("Erklaere was eine Hashfunktion macht.",
     "Eine Hashfunktion nimmt Eingaben beliebiger Groesse und erzeugt eine Ausgabe fester Groesse. Gute Hashfunktionen sind deterministisch, schnell und kollisionsresistent. Sie werden fuer Passwortspeicherung und Datenintegritaet verwendet."),
    ("Was ist der Unterschied zwischen HTTP und HTTPS?",
     "HTTPS ist HTTP mit TLS-Verschluesselung. HTTP sendet Daten im Klartext, waehrend HTTPS die gesamte Kommunikation verschluesselt. HTTPS verwendet Zertifikate zur Verifizierung der Serveridentitaet."),
    ("Wie schreibt man eine for-Schleife in Python?",
     "In Python iteriert eine for-Schleife ueber eine Sequenz: for item in [1, 2, 3]: print(item). Man kann auch range() verwenden: for i in range(10): print(i). Pythons for-Schleifen funktionieren mit jedem iterierbaren Objekt."),
    ("Was sind Umgebungsvariablen?",
     "Umgebungsvariablen sind Schluessel-Wert-Paare im Betriebssystem, die Anwendungen lesen koennen. Sie speichern Konfigurationen wie API-Schluessel, Datenbank-URLs und Pfade. In Python: os.environ['VAR_NAME']."),
    ("Erklaere das JSON-Format.",
     "JSON (JavaScript Object Notation) ist ein leichtgewichtiges Datenformat mit Schluessel-Wert-Paaren und Arrays. Es ist menschenlesbar, sprachunabhaengig und wird haeufig fuer APIs und Konfiguration verwendet."),
    ("Was ist ein Docker-Container?",
     "Ein Docker-Container verpackt eine Anwendung mit allen Abhaengigkeiten in eine standardisierte Einheit. Container sind leichtgewichtig, portabel und starten in Sekunden. Sie sorgen fuer konsistentes Verhalten ueber verschiedene Umgebungen."),
    ("Wie funktioniert ein Transformer-Modell?",
     "Transformer verarbeiten Sequenzen mit Self-Attention, die Relevanzwerte zwischen allen Positionen parallel berechnet. Kernkomponenten sind Multi-Head Attention, Feed-Forward-Schichten und Positionscodierungen."),
]

THINK_NO_TOOL_EN = [
    "This is a general knowledge question. I can answer directly without searching the workspace.",
    "The user is asking about a general concept. No workspace search needed.",
    "I know about this topic. I'll answer directly.",
    "I already know about this topic. No need to search the workspace.",
    "This question is about general knowledge that I can answer directly.",
    "No workspace search required for this kind of question.",
    "I have enough knowledge to answer this without using any tools.",
]

THINK_NO_TOOL_DE = [
    "Das ist eine allgemeine Wissensfrage. Ich kann direkt antworten ohne den Workspace zu durchsuchen.",
    "Der Benutzer fragt nach einem allgemeinen Konzept. Keine Workspace-Suche noetig.",
    "Ich weiss ueber dieses Thema Bescheid. Ich antworte direkt.",
    "Ich weiss bereits ueber dieses Thema Bescheid. Keine Workspace-Suche noetig.",
    "Diese Frage betrifft allgemeines Wissen, das ich direkt beantworten kann.",
    "Fuer diese Art von Frage ist keine Workspace-Suche erforderlich.",
    "Ich habe genug Wissen, um das ohne Tools zu beantworten.",
]


# ============================================================
# System Messages
# ============================================================

SYSTEM_MESSAGES = [
    "You are MyPT, an AI assistant with access to a document workspace. You can search, list, read, and summarize documents to answer questions.",
    "You are a helpful workspace assistant. Use the available tools (workspace.search, workspace.list_docs, workspace.get_doc, workspace.summarize) to find and present information.",
    "You are MyPT assistant with access to a knowledge base. Search documents and provide accurate answers based on the available content.",
    "You are MyPT, an AI workspace assistant. Use the available tools to find information in the user's documents.",
    "You are MyPT. You help users by searching and reading documents in their workspace. Always cite your sources.",
    "You are MyPT, a helpful assistant for document management. You can search, list, retrieve, and summarize documents.",
    "You are MyPT. When the user asks about their documents, use workspace tools to find accurate answers. Cite sources when possible.",
    "You are MyPT, an intelligent assistant. You have access to workspace tools for document search and retrieval.",
]


# ============================================================
# Language-aware template selection helper
# ============================================================

def pick_templates(lang: str):
    """Return the correct template sets for the given language."""
    if lang == "de":
        return {
            "search_q": SEARCH_QUESTIONS_DE,
            "list_q": LIST_QUESTIONS_DE,
            "get_doc_q": GET_DOC_QUESTIONS_DE,
            "summarize_q": SUMMARIZE_QUESTIONS_DE,
            "multi_step_q": MULTI_STEP_QUESTIONS_DE,
            "search_a": SEARCH_ANSWERS_DE,
            "list_a": LIST_ANSWERS_DE,
            "get_doc_a": GET_DOC_ANSWERS_DE,
            "summarize_a": SUMMARIZE_ANSWERS_DE,
            "no_results_a": NO_RESULTS_ANSWERS_DE,
            "think_search": THINK_SEARCH_DE,
            "think_list": THINK_LIST_DE,
            "think_getdoc": THINK_GETDOC_DE,
            "think_summarize": THINK_SUMMARIZE_DE,
            "think_multi_step": THINK_MULTI_STEP_DE,
            "think_no_tool": THINK_NO_TOOL_DE,
            "no_tool_q": NO_TOOL_QUESTIONS_DE,
            "no_tool_qa": NO_TOOL_ANSWERS_DE,
        }
    else:
        return {
            "search_q": SEARCH_QUESTIONS_EN,
            "list_q": LIST_QUESTIONS_EN,
            "get_doc_q": GET_DOC_QUESTIONS_EN,
            "summarize_q": SUMMARIZE_QUESTIONS_EN,
            "multi_step_q": MULTI_STEP_QUESTIONS_EN,
            "search_a": SEARCH_ANSWERS_EN,
            "list_a": LIST_ANSWERS_EN,
            "get_doc_a": GET_DOC_ANSWERS_EN,
            "summarize_a": SUMMARIZE_ANSWERS_EN,
            "no_results_a": NO_RESULTS_ANSWERS_EN,
            "think_search": THINK_SEARCH_EN,
            "think_list": THINK_LIST_EN,
            "think_getdoc": THINK_GETDOC_EN,
            "think_summarize": THINK_SUMMARIZE_EN,
            "think_multi_step": THINK_MULTI_STEP_EN,
            "think_no_tool": THINK_NO_TOOL_EN,
            "no_tool_q": NO_TOOL_QUESTIONS_EN,
            "no_tool_qa": NO_TOOL_ANSWERS_EN,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic agentic RAG training data (Phase 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--workspace_dir", type=str, default="workspace/",
                        help="Workspace directory with documents (default: workspace/)")
    parser.add_argument("--docs_dir", type=str, default=None,
                        help="Documents directory (default: workspace_dir/docs/)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    
    parser.add_argument("--num_examples", type=int, default=5000,
                        help="Number of synthetic examples to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--max_doc_chars", type=int, default=2000,
                        help="Max chars to include from documents in answers (default: 2000)")
    parser.add_argument("--include_errors", action="store_true", default=True,
                        help="Include examples with search errors/no results (default: True)")
    parser.add_argument("--no_errors", action="store_false", dest="include_errors",
                        help="Exclude error/no-results examples")
    
    parser.add_argument("--language", type=str, default="mixed",
                        choices=["en", "de", "mixed"],
                        help="Language: en, de, or mixed (default: mixed)")
    parser.add_argument("--de_ratio", type=float, default=0.40,
                        help="Fraction of DE episodes when language=mixed (default: 0.40)")
    parser.add_argument("--include_think", action="store_true", default=True,
                        help="Add <myPT_think> reasoning to tool calls (default: True)")
    parser.add_argument("--no_think", action="store_false", dest="include_think",
                        help="Disable <myPT_think> generation")
    parser.add_argument("--think_omit_ratio", type=float, default=0.15,
                        help="When include_think is on, randomly omit think block from this "
                             "fraction of episodes to teach the model that thinking is optional "
                             "(default: 0.15, i.e. 15%% of think-eligible episodes have no think)")
    parser.add_argument("--no_tool_ratio", type=float, default=0.20,
                        help="Fraction of NO_TOOL episodes (default: 0.20)")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def extract_topics_from_documents(documents: List) -> List[str]:
    """Extract meaningful topics/keywords from documents."""
    topics = set()
    
    for doc in documents:
        # Use filename (without extension) as a topic
        name = os.path.splitext(doc.filename)[0]
        topics.add(name.replace("_", " ").replace("-", " "))
        
        # Extract words from title/first line
        first_line = doc.text.split("\n")[0].strip()
        # Remove markdown headers
        first_line = first_line.lstrip("#").strip()
        if first_line and len(first_line) < 50:
            topics.add(first_line)
        
        # Extract capitalized phrases (likely important concepts)
        words = doc.text.split()
        for i, word in enumerate(words[:200]):  # Check first 200 words
            clean = word.strip(".,;:()[]{}\"'")
            if clean and clean[0].isupper() and len(clean) > 3:
                topics.add(clean)
    
    # Filter and clean topics
    clean_topics = []
    for t in topics:
        t = t.strip()
        if len(t) >= 3 and len(t) <= 50:
            clean_topics.append(t)
    
    return list(clean_topics)


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to max_chars, trying to break at sentence boundaries."""
    if len(text) <= max_chars:
        return text
    
    # Find a good break point
    truncated = text[:max_chars]
    
    # Try to break at sentence end
    for punct in [". ", ".\n", "! ", "!\n", "? ", "?\n"]:
        last_punct = truncated.rfind(punct)
        if last_punct > max_chars * 0.7:  # At least 70% of content
            return truncated[:last_punct + 1] + "..."
    
    # Fall back to word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        return truncated[:last_space] + "..."
    
    return truncated + "..."


def generate_search_example(
    documents: List,
    chunks: List,
    topics: List[str],
    max_chars: int,
    rng: random.Random,
    lang: str = "en",
    use_think: bool = False,
) -> Optional[Dict]:
    """Generate a workspace.search conversation example."""
    if not topics or not chunks:
        return None
    
    tpl = pick_templates(lang)
    topic = rng.choice(topics)
    
    # Find relevant chunks (simulate search)
    relevant_chunks = []
    topic_lower = topic.lower()
    for chunk in chunks:
        if topic_lower in chunk.text.lower():
            relevant_chunks.append(chunk)
    
    if not relevant_chunks:
        return None
    
    relevant_chunks = relevant_chunks[:3]
    
    tool_result = {
        "documents": [
            {
                "chunk_id": c.chunk_id,
                "text": truncate_text(c.text, 500),
                "source": c.source,
            }
            for c in relevant_chunks
        ],
        "total": len(relevant_chunks)
    }
    
    content_parts = []
    cite_source = None
    for i, c in enumerate(relevant_chunks, 1):
        src = c.source.get("filename", "unknown")
        if cite_source is None:
            cite_source = src
        content_parts.append(f"[{i}] From {src}:\n{truncate_text(c.text, 300)}")
    
    content = "\n\n".join(content_parts)
    
    toolcall_msg = {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": topic, "top_k": 3}}
    if use_think:
        toolcall_msg["think"] = rng.choice(tpl["think_search"]).format(topic=topic)
    
    answer_msg = {"role": "assistant", "content": rng.choice(tpl["search_a"]).format(topic=topic, content=content)}
    if cite_source:
        answer_msg["cite"] = cite_source
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(tpl["search_q"]).format(topic=topic)},
            toolcall_msg,
            {"role": "toolresult", "name": "workspace.search", "content": tool_result},
            answer_msg,
        ],
        "language": lang,
    }


def generate_list_docs_example(
    documents: List,
    rng: random.Random,
    lang: str = "en",
    use_think: bool = False,
) -> Optional[Dict]:
    """Generate a workspace.list_docs conversation example."""
    if not documents:
        return None
    
    tpl = pick_templates(lang)
    
    tool_result = {
        "documents": [
            {"doc_id": get_doc_id(doc), "title": get_doc_title(doc), "filename": doc.filename}
            for doc in documents
        ],
        "total": len(documents)
    }
    
    content_parts = []
    for doc in documents:
        content_parts.append(f"- **{get_doc_title(doc)}** ({doc.filename})")
    content = "\n".join(content_parts)
    
    toolcall_msg = {"role": "assistant_toolcall", "name": "workspace.list_docs", "arguments": {}}
    if use_think:
        toolcall_msg["think"] = rng.choice(tpl["think_list"])
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(tpl["list_q"])},
            toolcall_msg,
            {"role": "toolresult", "name": "workspace.list_docs", "content": tool_result},
            {"role": "assistant", "content": rng.choice(tpl["list_a"]).format(count=len(documents), content=content)},
        ],
        "language": lang,
    }


def generate_get_doc_example(
    documents: List,
    max_chars: int,
    rng: random.Random,
    lang: str = "en",
    use_think: bool = False,
) -> Optional[Dict]:
    """Generate a workspace.get_doc conversation example."""
    if not documents:
        return None
    
    tpl = pick_templates(lang)
    doc = rng.choice(documents)
    doc_id = get_doc_id(doc)
    
    tool_result = {
        "doc_id": doc_id,
        "title": get_doc_title(doc),
        "text": truncate_text(doc.text, max_chars),
        "length": len(doc.text),
    }
    
    content = truncate_text(doc.text, max_chars)
    
    toolcall_msg = {"role": "assistant_toolcall", "name": "workspace.get_doc", "arguments": {"doc_id": doc_id}}
    if use_think:
        toolcall_msg["think"] = rng.choice(tpl["think_getdoc"]).format(filename=doc.filename)
    
    answer_msg = {"role": "assistant", "content": rng.choice(tpl["get_doc_a"]).format(filename=doc.filename, content=content)}
    answer_msg["cite"] = doc.filename
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(tpl["get_doc_q"]).format(filename=doc.filename)},
            toolcall_msg,
            {"role": "toolresult", "name": "workspace.get_doc", "content": tool_result},
            answer_msg,
        ],
        "language": lang,
    }


def generate_summarize_example(
    documents: List,
    max_chars: int,
    rng: random.Random,
    lang: str = "en",
    use_think: bool = False,
) -> Optional[Dict]:
    """Generate a workspace.summarize conversation example."""
    if not documents:
        return None
    
    tpl = pick_templates(lang)
    doc = rng.choice(documents)
    doc_id = get_doc_id(doc)
    
    lines = doc.text.split("\n")
    summary_lines = []
    char_count = 0
    max_summary = 500
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if char_count + len(line) > max_summary:
            break
        summary_lines.append(line)
        char_count += len(line)
    
    summary = " ".join(summary_lines)
    if len(summary) < 50:
        summary = doc.text[:500] + "..."
    
    tool_result = {"summary": summary}
    
    toolcall_msg = {"role": "assistant_toolcall", "name": "workspace.summarize", "arguments": {"doc_id": doc_id}}
    if use_think:
        toolcall_msg["think"] = rng.choice(tpl["think_summarize"]).format(filename=doc.filename)
    
    answer_msg = {"role": "assistant", "content": rng.choice(tpl["summarize_a"]).format(filename=doc.filename, summary=summary)}
    answer_msg["cite"] = doc.filename
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(tpl["summarize_q"]).format(filename=doc.filename)},
            toolcall_msg,
            {"role": "toolresult", "name": "workspace.summarize", "content": tool_result},
            answer_msg,
        ],
        "language": lang,
    }


def generate_multi_step_example(
    documents: List,
    chunks: List,
    topics: List[str],
    max_chars: int,
    rng: random.Random,
    lang: str = "en",
    use_think: bool = False,
) -> Optional[Dict]:
    """Generate a multi-tool conversation example (search -> get_doc -> answer)."""
    if not topics or not chunks or not documents:
        return None
    
    tpl = pick_templates(lang)
    topic = rng.choice(topics)
    
    topic_lower = topic.lower()
    relevant_docs = [d for d in documents if topic_lower in d.text.lower()]
    
    if not relevant_docs:
        return None
    
    doc = rng.choice(relevant_docs)
    doc_id = get_doc_id(doc)
    
    search_result = {
        "documents": [{"doc_id": doc_id, "text": truncate_text(doc.text, 200), "source": {"filename": doc.filename}}],
        "total": 1
    }
    
    get_doc_result = {
        "doc_id": doc_id,
        "title": get_doc_title(doc),
        "text": truncate_text(doc.text, max_chars),
        "length": len(doc.text),
    }
    
    content = f"Based on searching for '{topic}' and reading {doc.filename}:\n\n{truncate_text(doc.text, max_chars)}"
    
    search_toolcall = {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": topic, "top_k": 3}}
    getdoc_toolcall = {"role": "assistant_toolcall", "name": "workspace.get_doc", "arguments": {"doc_id": doc_id}}
    if use_think:
        search_toolcall["think"] = rng.choice(tpl["think_multi_step"]).format(topic=topic)
    
    answer_msg = {"role": "assistant", "content": content, "cite": doc.filename}
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(tpl["multi_step_q"]).format(topic=topic)},
            search_toolcall,
            {"role": "toolresult", "name": "workspace.search", "content": search_result},
            getdoc_toolcall,
            {"role": "toolresult", "name": "workspace.get_doc", "content": get_doc_result},
            answer_msg,
        ],
        "language": lang,
    }


def generate_no_results_example(
    rng: random.Random,
    lang: str = "en",
    use_think: bool = False,
) -> Dict:
    """Generate a search-with-no-results example."""
    tpl = pick_templates(lang)
    fake_topics = [
        "quantum computing", "blockchain", "mars exploration",
        "ancient history", "underwater photography", "jazz music",
        "rocket science", "opera singing", "medieval architecture",
        "deep sea diving", "radio astronomy", "origami",
        "Quantenphysik", "Mittelalter", "Weltraumforschung",
        "Philosophie", "Meeresbiologie", "Kunstgeschichte",
    ]
    topic = rng.choice(fake_topics)
    
    tool_result = {"documents": [], "total": 0}
    
    toolcall_msg = {"role": "assistant_toolcall", "name": "workspace.search", "arguments": {"query": topic, "top_k": 3}}
    if use_think:
        toolcall_msg["think"] = rng.choice(tpl["think_search"]).format(topic=topic)
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": rng.choice(tpl["search_q"]).format(topic=topic)},
            toolcall_msg,
            {"role": "toolresult", "name": "workspace.search", "content": tool_result},
            {"role": "assistant", "content": rng.choice(tpl["no_results_a"]).format(topic=topic)},
        ],
        "language": lang,
    }


def generate_no_tool_example(
    rng: random.Random,
    lang: str = "en",
    use_think: bool = False,
) -> Dict:
    """Generate a NO_TOOL episode: answer directly without calling any tool."""
    tpl = pick_templates(lang)
    qa_pair = rng.choice(tpl["no_tool_qa"])
    question, answer = qa_pair
    
    assistant_msg = {"role": "assistant", "content": answer}
    if use_think:
        assistant_msg["think"] = rng.choice(tpl["think_no_tool"])
    
    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": question},
            assistant_msg,
        ],
        "language": lang,
    }


# ============================================================
# Contrastive / Negative Tool Examples (B5)
# Wrong-tool-then-correction episodes teach the model to
# distinguish between tools and recover from bad choices.
# ============================================================

WRONG_TOOL_SCENARIOS_EN = [
    {
        "user_q": "Summarize {filename} for me.",
        "wrong_tool": "workspace.search",
        "wrong_args": {"query": "{filename}", "top_k": 3},
        "wrong_think": "The user wants a summary. I'll search for the document first.",
        "correction_think": "Wait, the user asked me to summarize a specific file. I should use workspace.summarize directly instead of searching.",
        "correct_tool": "workspace.summarize",
        "correct_args_key": "doc_id",
    },
    {
        "user_q": "What documents do we have?",
        "wrong_tool": "workspace.search",
        "wrong_args": {"query": "all documents", "top_k": 10},
        "wrong_think": "The user wants to see all documents. Let me search for them.",
        "correction_think": "I used the wrong tool. The user wants a listing of all documents, not a search. I should use workspace.list_docs.",
        "correct_tool": "workspace.list_docs",
        "correct_args_key": None,
    },
    {
        "user_q": "Show me the full text of {filename}.",
        "wrong_tool": "workspace.summarize",
        "wrong_args_key": "doc_id",
        "wrong_think": "The user wants to see the document. Let me summarize it.",
        "correction_think": "Actually, the user wants the full text, not a summary. I need workspace.get_doc to retrieve the complete content.",
        "correct_tool": "workspace.get_doc",
        "correct_args_key": "doc_id",
    },
    {
        "user_q": "Search for {topic} in our documents.",
        "wrong_tool": "workspace.list_docs",
        "wrong_args": {},
        "wrong_think": "Let me see what documents we have about this topic.",
        "correction_think": "Listing all documents won't help find specific information about {topic}. I should use workspace.search with the topic as query.",
        "correct_tool": "workspace.search",
        "correct_args_key": "query",
    },
]

WRONG_TOOL_SCENARIOS_DE = [
    {
        "user_q": "Fasse {filename} fuer mich zusammen.",
        "wrong_tool": "workspace.search",
        "wrong_args": {"query": "{filename}", "top_k": 3},
        "wrong_think": "Der Benutzer moechte eine Zusammenfassung. Ich suche zuerst nach dem Dokument.",
        "correction_think": "Moment, der Benutzer hat mich gebeten, eine bestimmte Datei zusammenzufassen. Ich sollte workspace.summarize direkt verwenden statt zu suchen.",
        "correct_tool": "workspace.summarize",
        "correct_args_key": "doc_id",
    },
    {
        "user_q": "Welche Dokumente haben wir?",
        "wrong_tool": "workspace.search",
        "wrong_args": {"query": "alle Dokumente", "top_k": 10},
        "wrong_think": "Der Benutzer moechte alle Dokumente sehen. Ich suche danach.",
        "correction_think": "Ich habe das falsche Tool verwendet. Der Benutzer moechte eine Auflistung, keine Suche. Ich sollte workspace.list_docs verwenden.",
        "correct_tool": "workspace.list_docs",
        "correct_args_key": None,
    },
    {
        "user_q": "Zeig mir den vollstaendigen Text von {filename}.",
        "wrong_tool": "workspace.summarize",
        "wrong_args_key": "doc_id",
        "wrong_think": "Der Benutzer moechte das Dokument sehen. Ich fasse es zusammen.",
        "correction_think": "Der Benutzer moechte den vollen Text, keine Zusammenfassung. Ich brauche workspace.get_doc fuer den kompletten Inhalt.",
        "correct_tool": "workspace.get_doc",
        "correct_args_key": "doc_id",
    },
    {
        "user_q": "Suche nach {topic} in unseren Dokumenten.",
        "wrong_tool": "workspace.list_docs",
        "wrong_args": {},
        "wrong_think": "Mal sehen, welche Dokumente wir zu diesem Thema haben.",
        "correction_think": "Alle Dokumente aufzulisten hilft nicht, spezifische Informationen zu {topic} zu finden. Ich sollte workspace.search verwenden.",
        "correct_tool": "workspace.search",
        "correct_args_key": "query",
    },
]


def generate_wrong_tool_example(
    documents: List,
    topics: List[str],
    rng: random.Random,
    lang: str = "en",
) -> Optional[Dict]:
    """Generate a contrastive episode: wrong tool call -> correction -> right tool call.

    The model first picks the wrong tool, sees an unhelpful result, then
    self-corrects with a think block explaining the mistake, and finally
    uses the correct tool.  This teaches tool-selection discrimination.
    """
    scenarios = WRONG_TOOL_SCENARIOS_DE if lang == "de" else WRONG_TOOL_SCENARIOS_EN
    scenario = rng.choice(scenarios)

    # We need a document and a topic for template filling
    if not documents:
        return None
    doc = rng.choice(documents)
    doc_id = get_doc_id(doc)
    topic = rng.choice(topics) if topics else doc.filename

    fmt = {"filename": doc.filename, "topic": topic, "doc_id": doc_id}

    user_q = scenario["user_q"].format(**fmt)

    # -- Wrong tool call --
    wrong_args = scenario.get("wrong_args", {})
    if not wrong_args and scenario.get("wrong_args_key"):
        wrong_args = {scenario["wrong_args_key"]: doc_id}
    # Format string values
    wrong_args = {k: v.format(**fmt) if isinstance(v, str) else v for k, v in wrong_args.items()}

    wrong_toolcall = {
        "role": "assistant_toolcall",
        "name": scenario["wrong_tool"],
        "arguments": wrong_args,
        "think": scenario["wrong_think"].format(**fmt),
    }

    # Simulate an unhelpful / empty result from the wrong tool
    if scenario["wrong_tool"] == "workspace.search":
        wrong_result_content = {"documents": [], "total": 0}
    elif scenario["wrong_tool"] == "workspace.list_docs":
        wrong_result_content = {
            "documents": [{"doc_id": get_doc_id(d), "title": get_doc_title(d), "filename": d.filename} for d in documents[:3]],
            "total": len(documents),
        }
    elif scenario["wrong_tool"] == "workspace.summarize":
        wrong_result_content = {"summary": truncate_text(doc.text, 300)}
    else:
        wrong_result_content = {}

    wrong_result = {
        "role": "toolresult",
        "name": scenario["wrong_tool"],
        "content": wrong_result_content,
    }

    # -- Correct tool call (self-correction) --
    correct_args = {}
    if scenario["correct_args_key"] == "doc_id":
        correct_args = {"doc_id": doc_id}
    elif scenario["correct_args_key"] == "query":
        correct_args = {"query": topic, "top_k": 3}

    correct_toolcall = {
        "role": "assistant_toolcall",
        "name": scenario["correct_tool"],
        "arguments": correct_args,
        "think": scenario["correction_think"].format(**fmt),
    }

    # Simulate a helpful result from the correct tool
    if scenario["correct_tool"] == "workspace.search":
        correct_result_content = {
            "documents": [{"chunk_id": "c1", "text": truncate_text(doc.text, 300), "source": {"filename": doc.filename}}],
            "total": 1,
        }
    elif scenario["correct_tool"] == "workspace.list_docs":
        correct_result_content = {
            "documents": [{"doc_id": get_doc_id(d), "title": get_doc_title(d), "filename": d.filename} for d in documents],
            "total": len(documents),
        }
    elif scenario["correct_tool"] == "workspace.get_doc":
        correct_result_content = {"doc_id": doc_id, "title": get_doc_title(doc), "text": truncate_text(doc.text, 800), "length": len(doc.text)}
    elif scenario["correct_tool"] == "workspace.summarize":
        correct_result_content = {"summary": truncate_text(doc.text, 400)}
    else:
        correct_result_content = {}

    correct_result = {
        "role": "toolresult",
        "name": scenario["correct_tool"],
        "content": correct_result_content,
    }

    # -- Final answer based on correct tool --
    tpl = pick_templates(lang)
    if scenario["correct_tool"] == "workspace.search":
        answer_text = rng.choice(tpl["search_a"]).format(topic=topic, content=truncate_text(doc.text, 300))
    elif scenario["correct_tool"] == "workspace.list_docs":
        listing = "\n".join(f"- **{get_doc_title(d)}** ({d.filename})" for d in documents)
        answer_text = rng.choice(tpl["list_a"]).format(count=len(documents), content=listing)
    elif scenario["correct_tool"] == "workspace.get_doc":
        answer_text = rng.choice(tpl["get_doc_a"]).format(filename=doc.filename, content=truncate_text(doc.text, 800))
    elif scenario["correct_tool"] == "workspace.summarize":
        answer_text = rng.choice(tpl["summarize_a"]).format(filename=doc.filename, summary=truncate_text(doc.text, 400))
    else:
        answer_text = truncate_text(doc.text, 600)

    answer_msg = {"role": "assistant", "content": answer_text, "cite": doc.filename}

    return {
        "system": rng.choice(SYSTEM_MESSAGES),
        "messages": [
            {"role": "user", "content": user_q},
            wrong_toolcall,
            wrong_result,
            correct_toolcall,
            correct_result,
            answer_msg,
        ],
        "language": lang,
        "_contrastive": True,
    }


def main():
    args = parse_args()
    
    from core.banner import print_banner
    print_banner("MyPT Agentic SFT", "Phase 5: Single-Step Tool-Calling Data Generator")
    
    docs_dir = args.docs_dir or os.path.join(args.workspace_dir, "docs")
    
    if not os.path.exists(docs_dir):
        print(f"Error: Documents directory not found: {docs_dir}")
        print(f"Create {docs_dir}/ and add .txt or .md files.")
        sys.exit(1)
    
    print(f"\n  Language: {args.language} (DE ratio: {args.de_ratio})")
    think_desc = 'OFF'
    if args.include_think:
        think_desc = f'ON (omit_ratio={args.think_omit_ratio:.0%})'
    print(f"  Think tags: {think_desc}")
    print(f"  NO_TOOL ratio: {args.no_tool_ratio:.0%}")
    
    print(f"\nLoading documents from: {docs_dir}")
    loader = DocumentLoader()
    documents = loader.load_directory(docs_dir)
    
    if not documents:
        print("Error: No documents found. Add .txt or .md files to the docs directory.")
        sys.exit(1)
    
    print(f"  Loaded {len(documents)} documents")
    for doc in documents[:5]:
        print(f"    - {doc.filename}: {doc.num_chars} chars")
    if len(documents) > 5:
        print(f"    ... and {len(documents) - 5} more")
    
    print("\nChunking documents...")
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")
    
    print("\nExtracting topics...")
    topics = extract_topics_from_documents(documents)
    print(f"  Found {len(topics)} topics")
    if args.verbose:
        for t in topics[:10]:
            print(f"    - {t}")
    
    rng = random.Random(args.seed)
    
    print(f"\nGenerating {args.num_examples} synthetic examples...")
    
    # Adjusted weights: NO_TOOL and CONTRASTIVE get their own slices
    contrastive_ratio = 0.08  # 8% of episodes are wrong-tool-then-correction
    tool_budget = 1.0 - args.no_tool_ratio - contrastive_ratio
    pattern_weights = [
        ("search",       0.28 * tool_budget),
        ("list_docs",    0.12 * tool_budget),
        ("get_doc",      0.18 * tool_budget),
        ("summarize",    0.14 * tool_budget),
        ("multi_step",   0.15 * tool_budget),
        ("no_results",   0.13 * tool_budget),
        ("no_tool",      args.no_tool_ratio),
        ("contrastive",  contrastive_ratio),
    ]
    
    examples = []
    pattern_counts = {p[0]: 0 for p in pattern_weights}
    attempts = 0
    max_attempts = args.num_examples * 10
    
    while len(examples) < args.num_examples and attempts < max_attempts:
        attempts += 1
        
        if args.language == "mixed":
            lang = "de" if rng.random() < args.de_ratio else "en"
        else:
            lang = args.language
        
        # Dynamic think omission: when include_think is on, randomly omit
        # the think block from a fraction of episodes so the model learns
        # that thinking is optional and context-dependent.
        use_think = args.include_think
        if use_think and args.think_omit_ratio > 0:
            use_think = rng.random() > args.think_omit_ratio
        
        r = rng.random()
        cumulative = 0
        pattern_name = pattern_weights[0][0]
        for pname, pweight in pattern_weights:
            cumulative += pweight
            if r <= cumulative:
                pattern_name = pname
                break
        
        example = None
        if pattern_name == "search":
            example = generate_search_example(documents, chunks, topics, args.max_doc_chars, rng, lang, use_think)
        elif pattern_name == "list_docs":
            example = generate_list_docs_example(documents, rng, lang, use_think)
        elif pattern_name == "get_doc":
            example = generate_get_doc_example(documents, args.max_doc_chars, rng, lang, use_think)
        elif pattern_name == "summarize":
            example = generate_summarize_example(documents, args.max_doc_chars, rng, lang, use_think)
        elif pattern_name == "multi_step":
            example = generate_multi_step_example(documents, chunks, topics, args.max_doc_chars, rng, lang, use_think)
        elif pattern_name == "no_results":
            example = generate_no_results_example(rng, lang, use_think)
        elif pattern_name == "no_tool":
            example = generate_no_tool_example(rng, lang, use_think)
        elif pattern_name == "contrastive":
            example = generate_wrong_tool_example(documents, topics, rng, lang)
        
        if example:
            examples.append(example)
            pattern_counts[pattern_name] += 1
        
        if args.verbose and len(examples) % 500 == 0:
            print(f"  Generated {len(examples)} examples...")
    
    rng.shuffle(examples)
    
    print(f"\nSaving to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"  Saved {len(examples)} examples")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Phase 5 synthetic dataset generated!")
    print("=" * 60)
    
    print(f"\nPattern distribution:")
    for pattern, count in sorted(pattern_counts.items()):
        pct = count / len(examples) * 100 if examples else 0
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    lang_counts = {"en": 0, "de": 0}
    for ep in examples:
        lang_counts[ep.get("language", "en")] += 1
    print(f"\nLanguage distribution:")
    for lang, count in sorted(lang_counts.items()):
        pct = count / len(examples) * 100 if examples else 0
        print(f"  {lang}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput: {args.output}")
    print(f"Total examples: {len(examples)}")
    
    print(f"\nNext step - prepare SFT dataset:")
    print(f"  python scripts/sft/prepare_tool_sft.py \\")
    print(f"      --input {args.output} \\")
    print(f"      --output_dir data/sft_phase5_toolcall")
    
    print(f"\nThen train:")
    print(f"  python train.py --model_name phase5_toolcall \\")
    print(f"      --dataset_dir data/sft_phase5_toolcall \\")
    print(f"      --config_file configs/sft/phase5_simple_toolcall.json \\")
    print(f"      --init_from_model checkpoints/phase4_multiturn")


if __name__ == "__main__":
    main()

