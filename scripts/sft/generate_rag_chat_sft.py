#!/usr/bin/env python3
"""
Generate RAG Chat SFT episodes for Phase 3.

Creates single-turn Q&A episodes that teach the model to:
1. Answer from provided context passages (user_context)
2. Use <myPT_think> for reasoning before answering
3. Use <myPT_cite> to attribute answers to source documents
4. Handle NO_CONTEXT cases (answer from general knowledge)

Patterns:
    CONTEXT_ANSWER (60%):       user_context + answer + cite
    CONTEXT_THINK_ANSWER (25%): user_context + think + answer + cite
    NO_CONTEXT (15%):           no context, general knowledge answer

Input: Workspace docs from workspace/docs/ (markdown files)
Output: JSONL compatible with prepare_chat_sft.py

Usage:
    python scripts/sft/generate_rag_chat_sft.py \\
        --docs_dir workspace/docs \\
        --output data/sft_phase3_intermediate/rag_chat.jsonl

    python scripts/sft/generate_rag_chat_sft.py \\
        --docs_dir workspace/docs \\
        --output data/sft_phase3_intermediate/rag_chat.jsonl \\
        --num_examples 3000 --language mixed --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.system_prompts import CHAT_SYSTEM_PROMPTS


# =============================================================================
# Follow-up question templates (for multi-turn context episodes)
# =============================================================================

FOLLOWUP_QUESTIONS_EN = [
    "Can you elaborate on that?",
    "What else does the text say about this?",
    "Are there any exceptions or caveats mentioned?",
    "How does that relate to the rest of the document?",
    "Could you go into more detail?",
    "What's the practical implication of that?",
    "Is there anything else relevant in the passage?",
    "Why is that important?",
    "Can you give me an example from the text?",
    "So what does that mean in practice?",
]

FOLLOWUP_QUESTIONS_DE = [
    "Kannst du das naeher erlaeutern?",
    "Was sagt der Text sonst noch dazu?",
    "Werden Ausnahmen oder Einschraenkungen erwaehnt?",
    "Wie haengt das mit dem Rest des Dokuments zusammen?",
    "Koenntest du mehr ins Detail gehen?",
    "Was ist die praktische Bedeutung davon?",
    "Gibt es noch etwas Relevantes in der Passage?",
    "Warum ist das wichtig?",
    "Kannst du ein Beispiel aus dem Text geben?",
    "Was bedeutet das also in der Praxis?",
]

# =============================================================================
# Insufficient-context refusal templates
# =============================================================================

INSUFFICIENT_CONTEXT_EN = [
    "The provided context doesn't contain enough information about {topic} to give a complete answer. Based on what's available, {partial}",
    "I can only partially answer this. The text mentions {partial}, but doesn't cover {topic} in detail.",
    "The context doesn't directly address {topic}. However, from what's provided: {partial}",
    "There isn't specific information about {topic} in the passage. The closest related information is: {partial}",
]

INSUFFICIENT_CONTEXT_DE = [
    "Der bereitgestellte Kontext enthaelt nicht genug Informationen zu {topic} fuer eine vollstaendige Antwort. Basierend auf dem Verfuegbaren: {partial}",
    "Ich kann das nur teilweise beantworten. Der Text erwaehnt {partial}, geht aber nicht detailliert auf {topic} ein.",
    "Der Kontext behandelt {topic} nicht direkt. Aus dem Bereitgestellten ergibt sich jedoch: {partial}",
    "Es gibt keine spezifischen Informationen zu {topic} in der Passage. Die naechste verwandte Information ist: {partial}",
]

# =============================================================================
# Fact extraction templates (directive-style questions targeting specifics)
# =============================================================================

EXTRACTION_QUESTIONS_EN = [
    "List the key terms mentioned in the text about {topic}.",
    "What steps or stages are described for {topic}?",
    "Extract the main definitions from this passage about {topic}.",
    "What specific numbers, dates, or metrics are mentioned about {topic}?",
    "Name the technologies or tools mentioned in relation to {topic}.",
    "What comparisons or contrasts does the text make about {topic}?",
]

EXTRACTION_QUESTIONS_DE = [
    "Liste die wichtigsten Begriffe auf, die im Text zu {topic} erwaehnt werden.",
    "Welche Schritte oder Phasen werden fuer {topic} beschrieben?",
    "Extrahiere die wichtigsten Definitionen aus dieser Passage zu {topic}.",
    "Welche konkreten Zahlen, Daten oder Metriken werden zu {topic} erwaehnt?",
    "Nenne die Technologien oder Tools, die in Bezug auf {topic} erwaehnt werden.",
    "Welche Vergleiche oder Kontraste macht der Text zu {topic}?",
]

# =============================================================================
# Question Templates (EN)
# =============================================================================

CONTEXT_QUESTIONS_EN = [
    # -- Neutral / polite --
    "What does the document say about {topic}?",
    "Explain {topic} based on the provided information.",
    "Summarize the key points about {topic}.",
    "What is {topic}?",
    "How does {topic} work?",
    "Can you explain {topic} from the context?",
    "What are the main aspects of {topic}?",
    "Tell me about {topic}.",
    "Describe {topic} based on the given text.",
    "What information is provided about {topic}?",
    "According to the text, what is {topic}?",
    "Give me details about {topic}.",
    "What do we know about {topic}?",
    "Based on the context, explain {topic}.",
    "Help me understand {topic}.",
    # -- Casual / conversational --
    "So what's the deal with {topic}?",
    "Can you break down {topic} for me?",
    "I'm curious about {topic} -- what does the text say?",
    "Hey, what does it say about {topic}?",
    # -- Imperative / direct --
    "Just give me the key facts about {topic}.",
    "Walk me through {topic} step by step.",
    "Extract the most important info about {topic}.",
    # -- Task-oriented --
    "I need to understand {topic}. What does the source say?",
    "I'm writing about {topic}. What does the context provide?",
    "Give me bullet points on {topic} from the text.",
    # -- Analytical / deeper --
    "What are the advantages and limitations of {topic}?",
    "Why is {topic} relevant based on what's written here?",
    "What problem does {topic} solve according to the text?",
    "How does the passage characterize {topic}?",
    # -- Specific extraction --
    "What specific details does the text mention about {topic}?",
]

CONTEXT_QUESTIONS_DE = [
    # -- Neutral / hoeflich --
    "Was sagt das Dokument ueber {topic}?",
    "Erklaere {topic} anhand der bereitgestellten Informationen.",
    "Fasse die wichtigsten Punkte zu {topic} zusammen.",
    "Was ist {topic}?",
    "Wie funktioniert {topic}?",
    "Kannst du {topic} aus dem Kontext erklaeren?",
    "Was sind die wichtigsten Aspekte von {topic}?",
    "Erzaehl mir ueber {topic}.",
    "Beschreibe {topic} basierend auf dem gegebenen Text.",
    "Welche Informationen werden ueber {topic} bereitgestellt?",
    "Laut dem Text, was ist {topic}?",
    "Gib mir Details zu {topic}.",
    "Was wissen wir ueber {topic}?",
    "Erklaere {topic} basierend auf dem Kontext.",
    "Hilf mir {topic} zu verstehen.",
    # -- Umgangssprachlich --
    "Was hat es mit {topic} auf sich?",
    "Kannst du mir {topic} aufschluesseln?",
    "Ich bin neugierig auf {topic} -- was steht im Text?",
    "Hey, was steht da ueber {topic}?",
    # -- Direkt / imperativ --
    "Gib mir einfach die wichtigsten Fakten zu {topic}.",
    "Fuehr mich Schritt fuer Schritt durch {topic}.",
    "Extrahiere die wichtigsten Infos zu {topic}.",
    # -- Aufgabenorientiert --
    "Ich muss {topic} verstehen. Was sagt die Quelle?",
    "Ich schreibe ueber {topic}. Was liefert der Kontext?",
    "Gib mir Stichpunkte zu {topic} aus dem Text.",
    # -- Analytisch / tiefer --
    "Was sind die Vorteile und Einschraenkungen von {topic}?",
    "Warum ist {topic} relevant laut dem Geschriebenen?",
    "Welches Problem loest {topic} laut dem Text?",
    "Wie charakterisiert die Passage {topic}?",
    # -- Spezifische Extraktion --
    "Welche konkreten Details erwaehnt der Text zu {topic}?",
]

NO_CONTEXT_QUESTIONS_EN = [
    "What is {topic}?",
    "Explain {topic} briefly.",
    "How does {topic} work?",
    "Tell me about {topic}.",
    "What do you know about {topic}?",
    "Can you describe {topic}?",
    "Give me a short explanation of {topic}.",
    "What is the purpose of {topic}?",
    "Define {topic} in simple terms.",
    "Why is {topic} important?",
    "What are the basics of {topic}?",
    "How would you explain {topic} to a beginner?",
    "What should I know about {topic}?",
    "Can you give me an overview of {topic}?",
    "What role does {topic} play?",
]

NO_CONTEXT_QUESTIONS_DE = [
    "Was ist {topic}?",
    "Erklaere {topic} kurz.",
    "Wie funktioniert {topic}?",
    "Erzaehl mir ueber {topic}.",
    "Was weisst du ueber {topic}?",
    "Kannst du {topic} beschreiben?",
    "Gib mir eine kurze Erklaerung von {topic}.",
    "Was ist der Zweck von {topic}?",
    "Definiere {topic} in einfachen Worten.",
    "Warum ist {topic} wichtig?",
    "Was sind die Grundlagen von {topic}?",
    "Wie wuerdest du {topic} einem Anfaenger erklaeren?",
    "Was sollte ich ueber {topic} wissen?",
    "Kannst du mir einen Ueberblick ueber {topic} geben?",
    "Welche Rolle spielt {topic}?",
]

# =============================================================================
# Think Templates
# =============================================================================

THINK_TEMPLATES_EN = [
    "The context explains {topic}. I should summarize the key points.",
    "Let me analyze what the document says about {topic}.",
    "The user wants to know about {topic}. The context has relevant information.",
    "I need to extract the important details about {topic} from the provided text.",
    "The passage covers {topic}. I'll provide a concise answer.",
    "Looking at the context, it describes {topic} in detail.",
    "The question is about {topic}. Let me find the relevant parts in the context.",
    "I see the context mentions {topic}. I should focus on the most important information.",
    "The user is asking about {topic}. The provided text addresses this directly.",
    "Let me think about how to best explain {topic} based on what the document says.",
    "The context contains information about {topic}. I need to organize my answer clearly.",
    "This is a question about {topic}. The relevant information is in the passage.",
]

THINK_TEMPLATES_DE = [
    "Der Kontext erklaert {topic}. Ich sollte die wichtigsten Punkte zusammenfassen.",
    "Lass mich analysieren, was das Dokument ueber {topic} sagt.",
    "Der Benutzer moechte etwas ueber {topic} wissen. Der Kontext hat relevante Informationen.",
    "Ich muss die wichtigen Details ueber {topic} aus dem bereitgestellten Text extrahieren.",
    "Die Passage behandelt {topic}. Ich gebe eine praegnante Antwort.",
    "Im Kontext wird {topic} im Detail beschrieben.",
    "Die Frage bezieht sich auf {topic}. Ich suche die relevanten Teile im Kontext.",
    "Ich sehe, der Kontext erwaehnt {topic}. Ich sollte mich auf die wichtigsten Informationen konzentrieren.",
    "Der Benutzer fragt nach {topic}. Der bereitgestellte Text behandelt dies direkt.",
    "Lass mich ueberlegen, wie ich {topic} basierend auf dem Dokument am besten erklaeren kann.",
    "Der Kontext enthaelt Informationen zu {topic}. Ich muss meine Antwort klar strukturieren.",
    "Das ist eine Frage zu {topic}. Die relevanten Informationen stehen in der Passage.",
]

# =============================================================================
# Answer Templates
# =============================================================================

ANSWER_PREFIX_EN = [
    "Based on the provided information, {answer}",
    "{answer}",
    "According to the document, {answer}",
    "The text explains that {answer}",
    "{answer}",
    "From the context: {answer}",
    "The document states that {answer}",
    "As described in the text, {answer}",
    "Here is what the context says: {answer}",
    "{answer}",
    "To summarize the key information: {answer}",
    "The provided passage indicates that {answer}",
]

ANSWER_PREFIX_DE = [
    "Basierend auf den bereitgestellten Informationen, {answer}",
    "{answer}",
    "Laut dem Dokument, {answer}",
    "Der Text erklaert, dass {answer}",
    "{answer}",
    "Aus dem Kontext: {answer}",
    "Das Dokument besagt, dass {answer}",
    "Wie im Text beschrieben, {answer}",
    "Folgendes sagt der Kontext: {answer}",
    "{answer}",
    "Um die wichtigsten Informationen zusammenzufassen: {answer}",
    "Die bereitgestellte Passage zeigt, dass {answer}",
]

NO_CONTEXT_ANSWER_PREFIX_EN = [
    "{answer}",
    "In general, {answer}",
    "Simply put, {answer}",
    "{answer}",
    "To put it briefly, {answer}",
    "In short, {answer}",
    "Here is what I know: {answer}",
    "{answer}",
    "The short answer is: {answer}",
]

NO_CONTEXT_ANSWER_PREFIX_DE = [
    "{answer}",
    "Im Allgemeinen, {answer}",
    "Einfach gesagt, {answer}",
    "{answer}",
    "Kurz gefasst, {answer}",
    "Kurz gesagt, {answer}",
    "Folgendes weiss ich dazu: {answer}",
    "{answer}",
    "Die kurze Antwort lautet: {answer}",
]


# =============================================================================
# General Knowledge Topics (for NO_CONTEXT pattern)
# =============================================================================

GENERAL_TOPICS_EN = [
    ("Python", "Python is a high-level programming language known for its clear syntax and readability. It supports multiple programming paradigms including object-oriented, procedural, and functional programming."),
    ("machine learning", "Machine learning is a subset of artificial intelligence where systems learn patterns from data without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning."),
    ("a neural network", "A neural network is a computational model inspired by biological neurons. It consists of layers of interconnected nodes that process information and learn to recognize patterns in data."),
    ("version control", "Version control is a system for tracking changes to files over time. Git is the most widely used version control system, enabling collaboration and maintaining a complete history of changes."),
    ("an API", "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other. APIs define the methods and data formats for requesting and exchanging information."),
    ("a database", "A database is an organized collection of structured data stored electronically. Databases use management systems (DBMS) like PostgreSQL or SQLite to store, retrieve, and manage data efficiently."),
    ("encryption", "Encryption is the process of converting readable data into an encoded format that can only be decoded with the correct key. It protects sensitive information during storage and transmission."),
    ("a transformer model", "A transformer is a neural network architecture that uses self-attention mechanisms to process sequences in parallel. Introduced in 2017, transformers are the foundation of modern language models like GPT and BERT."),
    ("tokenization", "Tokenization is the process of breaking text into smaller units called tokens. In NLP, tokens can be words, subwords, or characters. BPE (Byte Pair Encoding) is a common tokenization algorithm."),
    ("loss masking", "Loss masking is a training technique where certain tokens are excluded from the loss computation. In SFT, only assistant-generated tokens contribute to the loss, while system and user tokens are masked."),
    ("cloud computing", "Cloud computing delivers computing services like storage, processing, and networking over the internet. Major providers include AWS, Azure, and Google Cloud. It enables scalable, on-demand access to resources."),
    ("HTTP", "HTTP (Hypertext Transfer Protocol) is the foundation of data communication on the web. It defines how messages are formatted and transmitted between clients and servers, using methods like GET, POST, PUT, and DELETE."),
    ("a hash function", "A hash function maps input data of arbitrary size to a fixed-size output. Hash functions are used in data structures (hash tables), cryptography, and data integrity verification. Good hash functions minimize collisions."),
    ("recursion", "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. Every recursive function needs a base case to prevent infinite recursion. Common examples include factorial and tree traversal."),
    ("REST APIs", "REST (Representational State Transfer) is an architectural style for designing web APIs. RESTful APIs use HTTP methods, are stateless, and organize resources through URIs. They typically exchange data in JSON format."),
    ("containerization", "Containerization packages an application with its dependencies into isolated units called containers. Docker is the most popular container platform. Containers are lighter than virtual machines and ensure consistent environments."),
    ("gradient descent", "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function. It computes the gradient (direction of steepest increase) and moves in the opposite direction. Learning rate controls the step size."),
    ("a linked list", "A linked list is a data structure where elements (nodes) are connected by pointers. Each node stores data and a reference to the next node. Linked lists allow efficient insertion and deletion but have slower random access than arrays."),
    ("regular expressions", "Regular expressions (regex) are patterns used to match and manipulate text. They support wildcards, quantifiers, character classes, and grouping. Regex is used in search, validation, and text processing across most programming languages."),
    ("TCP/IP", "TCP/IP is the fundamental communication protocol suite of the internet. TCP (Transmission Control Protocol) ensures reliable, ordered data delivery, while IP (Internet Protocol) handles addressing and routing packets between hosts."),
]

GENERAL_TOPICS_DE = [
    ("Python", "Python ist eine hoeherwertige Programmiersprache, die fuer ihre klare Syntax und Lesbarkeit bekannt ist. Sie unterstuetzt mehrere Programmierparadigmen einschliesslich objektorientierter, prozeduraler und funktionaler Programmierung."),
    ("maschinelles Lernen", "Maschinelles Lernen ist ein Teilgebiet der kuenstlichen Intelligenz, bei dem Systeme Muster aus Daten lernen, ohne explizit programmiert zu werden. Gaengige Ansaetze sind ueberwachtes Lernen, unueberwachtes Lernen und Reinforcement Learning."),
    ("ein neuronales Netz", "Ein neuronales Netz ist ein Rechenmodell, das von biologischen Neuronen inspiriert ist. Es besteht aus Schichten verbundener Knoten, die Informationen verarbeiten und lernen, Muster in Daten zu erkennen."),
    ("Versionskontrolle", "Versionskontrolle ist ein System zur Nachverfolgung von Aenderungen an Dateien ueber die Zeit. Git ist das am weitesten verbreitete Versionskontrollsystem und ermoeglicht Zusammenarbeit und eine vollstaendige Aenderungshistorie."),
    ("eine API", "Eine API (Application Programming Interface) ist ein Satz von Regeln, der es verschiedenen Softwareanwendungen ermoeglicht, miteinander zu kommunizieren. APIs definieren die Methoden und Datenformate fuer den Informationsaustausch."),
    ("eine Datenbank", "Eine Datenbank ist eine organisierte Sammlung strukturierter Daten, die elektronisch gespeichert werden. Datenbanken verwenden Verwaltungssysteme wie PostgreSQL oder SQLite zum effizienten Speichern und Abrufen von Daten."),
    ("Verschluesselung", "Verschluesselung ist der Prozess, lesbare Daten in ein codiertes Format umzuwandeln, das nur mit dem richtigen Schluessel decodiert werden kann. Sie schuetzt sensible Informationen bei Speicherung und Uebertragung."),
    ("ein Transformer-Modell", "Ein Transformer ist eine neuronale Netzwerkarchitektur, die Self-Attention-Mechanismen verwendet, um Sequenzen parallel zu verarbeiten. Transformer sind die Grundlage moderner Sprachmodelle wie GPT und BERT."),
    ("Tokenisierung", "Tokenisierung ist der Prozess, Text in kleinere Einheiten namens Tokens aufzuteilen. In der NLP koennen Tokens Woerter, Teilwoerter oder Zeichen sein. BPE (Byte Pair Encoding) ist ein gaengiger Tokenisierungsalgorithmus."),
    ("Loss Masking", "Loss Masking ist eine Trainingstechnik, bei der bestimmte Tokens von der Verlustberechnung ausgeschlossen werden. Bei SFT tragen nur vom Assistenten generierte Tokens zum Verlust bei, waehrend System- und Benutzertokens maskiert werden."),
    ("Cloud Computing", "Cloud Computing stellt Computerdienste wie Speicher, Verarbeitung und Netzwerke ueber das Internet bereit. Grosse Anbieter sind AWS, Azure und Google Cloud. Es ermoeglicht skalierbaren, bedarfsgerechten Zugang zu Ressourcen."),
    ("HTTP", "HTTP (Hypertext Transfer Protocol) ist die Grundlage der Datenkommunikation im Web. Es definiert, wie Nachrichten zwischen Clients und Servern formatiert und uebertragen werden, mit Methoden wie GET, POST, PUT und DELETE."),
    ("eine Hashfunktion", "Eine Hashfunktion bildet Eingabedaten beliebiger Groesse auf eine Ausgabe fester Groesse ab. Hashfunktionen werden in Datenstrukturen, Kryptographie und zur Integritaetspruefung verwendet. Gute Hashfunktionen minimieren Kollisionen."),
    ("Rekursion", "Rekursion ist eine Programmiertechnik, bei der eine Funktion sich selbst aufruft, um kleinere Instanzen desselben Problems zu loesen. Jede rekursive Funktion braucht einen Basisfall. Haeufige Beispiele sind Fakultaet und Baumdurchlauf."),
    ("REST-APIs", "REST (Representational State Transfer) ist ein Architekturstil fuer Web-APIs. RESTful APIs nutzen HTTP-Methoden, sind zustandslos und organisieren Ressourcen ueber URIs. Sie tauschen Daten typischerweise im JSON-Format aus."),
    ("Containerisierung", "Containerisierung verpackt eine Anwendung mit ihren Abhaengigkeiten in isolierte Einheiten namens Container. Docker ist die beliebteste Container-Plattform. Container sind leichter als virtuelle Maschinen und gewaehrleisten konsistente Umgebungen."),
    ("Gradientenabstieg", "Gradientenabstieg ist ein Optimierungsalgorithmus, der Parameter iterativ anpasst, um eine Verlustfunktion zu minimieren. Er berechnet den Gradienten und bewegt sich in die entgegengesetzte Richtung. Die Lernrate steuert die Schrittgroesse."),
    ("eine verkettete Liste", "Eine verkettete Liste ist eine Datenstruktur, bei der Elemente (Knoten) durch Zeiger verbunden sind. Jeder Knoten speichert Daten und eine Referenz zum naechsten Knoten. Sie erlaubt effizientes Einfuegen, hat aber langsameren Direktzugriff als Arrays."),
    ("regulaere Ausdruecke", "Regulaere Ausdruecke (Regex) sind Muster zum Abgleichen und Bearbeiten von Text. Sie unterstuetzen Platzhalter, Quantifizierer, Zeichenklassen und Gruppierungen. Regex wird in Suche, Validierung und Textverarbeitung in den meisten Programmiersprachen verwendet."),
    ("TCP/IP", "TCP/IP ist die grundlegende Kommunikationsprotokollsuite des Internets. TCP (Transmission Control Protocol) gewaehrleistet zuverlaessige, geordnete Datenzustellung, waehrend IP (Internet Protocol) die Adressierung und das Routing von Paketen uebernimmt."),
]


# =============================================================================
# Document Processing
# =============================================================================

def load_workspace_docs(docs_dir: str) -> List[Dict[str, Any]]:
    """Load markdown docs from workspace, extract content and metadata."""
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"Warning: docs directory not found: {docs_dir}")
        return docs

    for fpath in sorted(docs_path.glob("*.md")):
        text = fpath.read_text(encoding='utf-8', errors='replace')
        if len(text.strip()) < 50:
            continue

        title = fpath.stem.replace("_", " ").replace("-", " ").title()
        first_line = text.split("\n")[0].strip().lstrip("#").strip()
        if first_line and len(first_line) < 80:
            title = first_line

        docs.append({
            "filename": fpath.name,
            "title": title,
            "text": text,
            "topics": extract_topics(text, fpath.stem),
            "passages": extract_passages(text),
        })

    return docs


def extract_topics(text: str, stem: str) -> List[str]:
    """Extract meaningful topic phrases from document text."""
    topics = set()
    name = stem.replace("_", " ").replace("-", " ")
    if len(name) >= 3:
        topics.add(name)

    for line in text.split("\n")[:50]:
        line = line.strip()
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            if 3 <= len(heading) <= 60:
                topics.add(heading)

    return list(topics)


def extract_passages(text: str, min_len: int = 80, max_len: int = 500) -> List[str]:
    """Extract coherent passages (paragraph-level) from document text."""
    passages = []
    paragraphs = text.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if para.startswith("#"):
            continue
        if para.startswith("```"):
            continue
        if para.startswith("|") or para.startswith("---"):
            continue

        clean = para.replace("\n", " ").strip()
        if min_len <= len(clean) <= max_len:
            passages.append(clean)
        elif len(clean) > max_len:
            passages.append(clean[:max_len].rsplit(" ", 1)[0] + "...")

    return passages


def make_answer_from_passage(passage: str, rng: random.Random) -> str:
    """Create a concise answer by extracting/rephrasing from a passage."""
    sentences = []
    for s in passage.replace("\n", " ").split("."):
        s = s.strip()
        if len(s) > 15:
            sentences.append(s + ".")

    if not sentences:
        return passage[:200]

    n = min(len(sentences), rng.randint(1, 3))
    selected = sentences[:n]
    return " ".join(selected)


# =============================================================================
# Episode Generation
# =============================================================================

def generate_context_answer(
    doc: Dict, rng: random.Random, lang: str, use_think: bool = False
) -> Optional[Dict]:
    """Generate a CONTEXT_ANSWER or CONTEXT_THINK_ANSWER episode."""
    if not doc["passages"] or not doc["topics"]:
        return None

    passage = rng.choice(doc["passages"])
    topic = rng.choice(doc["topics"])

    if lang == "de":
        question = rng.choice(CONTEXT_QUESTIONS_DE).format(topic=topic)
        answer_raw = make_answer_from_passage(passage, rng)
        answer = rng.choice(ANSWER_PREFIX_DE).format(answer=answer_raw)
    else:
        question = rng.choice(CONTEXT_QUESTIONS_EN).format(topic=topic)
        answer_raw = make_answer_from_passage(passage, rng)
        answer = rng.choice(ANSWER_PREFIX_EN).format(answer=answer_raw)

    msg_user = {"role": "user", "content": question, "context": passage}
    msg_assistant = {"role": "assistant", "content": answer, "cite": doc["filename"]}

    if use_think:
        if lang == "de":
            think = rng.choice(THINK_TEMPLATES_DE).format(topic=topic)
        else:
            think = rng.choice(THINK_TEMPLATES_EN).format(topic=topic)
        msg_assistant["think"] = think

    return {
        "system": rng.choice(CHAT_SYSTEM_PROMPTS),
        "messages": [msg_user, msg_assistant],
        "language": lang,
    }


def generate_multiturn_context(
    doc: Dict, rng: random.Random, lang: str, use_think: bool = False
) -> Optional[Dict]:
    """Generate a multi-turn RAG episode: question -> answer -> follow-up -> answer."""
    if not doc["passages"] or not doc["topics"] or len(doc["passages"]) < 2:
        return None

    passage = rng.choice(doc["passages"])
    topic = rng.choice(doc["topics"])

    if lang == "de":
        question = rng.choice(CONTEXT_QUESTIONS_DE).format(topic=topic)
        answer_raw = make_answer_from_passage(passage, rng)
        answer1 = rng.choice(ANSWER_PREFIX_DE).format(answer=answer_raw)
        followup = rng.choice(FOLLOWUP_QUESTIONS_DE)
    else:
        question = rng.choice(CONTEXT_QUESTIONS_EN).format(topic=topic)
        answer_raw = make_answer_from_passage(passage, rng)
        answer1 = rng.choice(ANSWER_PREFIX_EN).format(answer=answer_raw)
        followup = rng.choice(FOLLOWUP_QUESTIONS_EN)

    # Second answer draws from a different passage in the same doc
    other_passages = [p for p in doc["passages"] if p != passage]
    passage2 = rng.choice(other_passages) if other_passages else passage
    answer2_raw = make_answer_from_passage(passage2, rng)
    if lang == "de":
        answer2 = rng.choice(ANSWER_PREFIX_DE).format(answer=answer2_raw)
    else:
        answer2 = rng.choice(ANSWER_PREFIX_EN).format(answer=answer2_raw)

    msg_user1 = {"role": "user", "content": question, "context": passage}
    msg_assistant1 = {"role": "assistant", "content": answer1, "cite": doc["filename"]}
    msg_user2 = {"role": "user", "content": followup}
    msg_assistant2 = {"role": "assistant", "content": answer2, "cite": doc["filename"]}

    if use_think:
        think = (rng.choice(THINK_TEMPLATES_DE) if lang == "de"
                 else rng.choice(THINK_TEMPLATES_EN)).format(topic=topic)
        msg_assistant1["think"] = think

    return {
        "system": rng.choice(CHAT_SYSTEM_PROMPTS),
        "messages": [msg_user1, msg_assistant1, msg_user2, msg_assistant2],
        "language": lang,
    }


def generate_insufficient_context(
    doc: Dict, rng: random.Random, lang: str, use_think: bool = False
) -> Optional[Dict]:
    """Generate an episode where the context only partially answers the question.

    Teaches the model to be honest about gaps rather than hallucinating.
    """
    if not doc["passages"] or not doc["topics"]:
        return None

    passage = rng.choice(doc["passages"])
    topic = rng.choice(doc["topics"])
    sentences = [s.strip() + "." for s in passage.replace("\n", " ").split(".")
                 if len(s.strip()) > 15]
    if not sentences:
        return None
    partial = sentences[0]

    if lang == "de":
        question = rng.choice(CONTEXT_QUESTIONS_DE).format(topic=topic)
        answer = rng.choice(INSUFFICIENT_CONTEXT_DE).format(topic=topic, partial=partial)
    else:
        question = rng.choice(CONTEXT_QUESTIONS_EN).format(topic=topic)
        answer = rng.choice(INSUFFICIENT_CONTEXT_EN).format(topic=topic, partial=partial)

    msg_user = {"role": "user", "content": question, "context": passage}
    msg_assistant = {"role": "assistant", "content": answer}

    if use_think:
        msg_assistant["think"] = (
            f"Der Kontext erwaehnt {topic}, liefert aber keine vollstaendige Antwort. "
            "Ich sollte ehrlich sagen, was der Text hergibt."
            if lang == "de" else
            f"The context mentions {topic} but doesn't fully answer the question. "
            "I should be honest about what the text provides."
        )

    return {
        "system": rng.choice(CHAT_SYSTEM_PROMPTS),
        "messages": [msg_user, msg_assistant],
        "language": lang,
    }


def generate_extraction(
    doc: Dict, rng: random.Random, lang: str, use_think: bool = False
) -> Optional[Dict]:
    """Generate a fact-extraction episode with directive-style questions."""
    if not doc["passages"] or not doc["topics"]:
        return None

    passage = rng.choice(doc["passages"])
    topic = rng.choice(doc["topics"])

    if lang == "de":
        question = rng.choice(EXTRACTION_QUESTIONS_DE).format(topic=topic)
        answer_raw = make_answer_from_passage(passage, rng)
        answer = rng.choice(ANSWER_PREFIX_DE).format(answer=answer_raw)
    else:
        question = rng.choice(EXTRACTION_QUESTIONS_EN).format(topic=topic)
        answer_raw = make_answer_from_passage(passage, rng)
        answer = rng.choice(ANSWER_PREFIX_EN).format(answer=answer_raw)

    msg_user = {"role": "user", "content": question, "context": passage}
    msg_assistant = {"role": "assistant", "content": answer, "cite": doc["filename"]}

    if use_think:
        think = (rng.choice(THINK_TEMPLATES_DE) if lang == "de"
                 else rng.choice(THINK_TEMPLATES_EN)).format(topic=topic)
        msg_assistant["think"] = think

    return {
        "system": rng.choice(CHAT_SYSTEM_PROMPTS),
        "messages": [msg_user, msg_assistant],
        "language": lang,
    }


def generate_no_context(rng: random.Random, lang: str) -> Dict:
    """Generate a NO_CONTEXT episode (general knowledge, no user_context)."""
    if lang == "de":
        topic, answer_text = rng.choice(GENERAL_TOPICS_DE)
        question = rng.choice(NO_CONTEXT_QUESTIONS_DE).format(topic=topic)
        answer = rng.choice(NO_CONTEXT_ANSWER_PREFIX_DE).format(answer=answer_text)
    else:
        topic, answer_text = rng.choice(GENERAL_TOPICS_EN)
        question = rng.choice(NO_CONTEXT_QUESTIONS_EN).format(topic=topic)
        answer = rng.choice(NO_CONTEXT_ANSWER_PREFIX_EN).format(answer=answer_text)

    return {
        "system": rng.choice(CHAT_SYSTEM_PROMPTS),
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "language": lang,
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate RAG Chat SFT episodes (Phase 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--docs_dir", type=str, default="workspace/docs",
                        help="Directory with workspace markdown docs (default: workspace/docs)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--num_examples", type=int, default=2000,
                        help="Number of episodes to generate (default: 2000)")
    parser.add_argument("--language", type=str, default="mixed",
                        choices=["en", "de", "mixed"],
                        help="Language: en, de, or mixed (default: mixed)")
    parser.add_argument("--de_ratio", type=float, default=0.40,
                        help="Fraction of DE episodes when language=mixed (default: 0.40)")
    parser.add_argument("--context_answer_ratio", type=float, default=0.40,
                        help="Fraction CONTEXT_ANSWER (default: 0.40)")
    parser.add_argument("--context_think_ratio", type=float, default=0.20,
                        help="Fraction CONTEXT_THINK_ANSWER (default: 0.20)")
    parser.add_argument("--multiturn_ratio", type=float, default=0.12,
                        help="Fraction MULTITURN context episodes (default: 0.12)")
    parser.add_argument("--extraction_ratio", type=float, default=0.10,
                        help="Fraction EXTRACTION directive questions (default: 0.10)")
    parser.add_argument("--insufficient_ratio", type=float, default=0.08,
                        help="Fraction INSUFFICIENT_CONTEXT honest-refusal (default: 0.08)")
    parser.add_argument("--no_context_ratio", type=float, default=0.10,
                        help="Fraction NO_CONTEXT general knowledge (default: 0.10)")
    parser.add_argument("--think_omit_ratio", type=float, default=0.15,
                        help="When in CONTEXT_THINK_ANSWER pattern, randomly omit the think "
                             "block from this fraction of episodes to teach the model that "
                             "thinking is optional (default: 0.15, i.e. 15%% of think-eligible "
                             "episodes skip the think block)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 60)
    print("  MyPT Phase 3: RAG Chat SFT Generator")
    print("=" * 60)

    docs = load_workspace_docs(args.docs_dir)
    if not docs:
        print(f"Error: No markdown documents found in {args.docs_dir}")
        sys.exit(1)

    docs_with_content = [d for d in docs if d["passages"] and d["topics"]]
    print(f"\n  Loaded {len(docs)} docs, {len(docs_with_content)} have usable passages")
    total_passages = sum(len(d["passages"]) for d in docs_with_content)
    total_topics = sum(len(d["topics"]) for d in docs_with_content)
    print(f"  Total passages: {total_passages}, topics: {total_topics}")

    if not docs_with_content:
        print("Error: No documents with extractable passages found.")
        sys.exit(1)

    episodes = []
    pattern_names = [
        "context_answer", "context_think_answer", "multiturn",
        "extraction", "insufficient", "no_context",
    ]
    pattern_counts = {p: 0 for p in pattern_names}
    # Build cumulative thresholds from ratios
    pattern_thresholds = [
        ("context_answer",       args.context_answer_ratio),
        ("context_think_answer", args.context_think_ratio),
        ("multiturn",            args.multiturn_ratio),
        ("extraction",           args.extraction_ratio),
        ("insufficient",         args.insufficient_ratio),
        ("no_context",           args.no_context_ratio),
    ]
    attempts = 0
    max_attempts = args.num_examples * 5

    while len(episodes) < args.num_examples and attempts < max_attempts:
        attempts += 1

        if args.language == "mixed":
            lang = "de" if rng.random() < args.de_ratio else "en"
        else:
            lang = args.language

        r = rng.random()
        cumulative = 0
        pattern = "context_answer"
        for pname, pweight in pattern_thresholds:
            cumulative += pweight
            if r <= cumulative:
                pattern = pname
                break

        ep = None
        if pattern == "context_answer":
            ep = generate_context_answer(rng.choice(docs_with_content), rng, lang, use_think=False)
        elif pattern == "context_think_answer":
            # Dynamic think omission: randomly omit the think block from a
            # fraction of think-eligible episodes so the model learns that
            # thinking is optional and context-dependent.
            actually_think = rng.random() > args.think_omit_ratio
            ep = generate_context_answer(rng.choice(docs_with_content), rng, lang, use_think=actually_think)
        elif pattern == "multiturn":
            actually_think = rng.random() > args.think_omit_ratio
            ep = generate_multiturn_context(rng.choice(docs_with_content), rng, lang, use_think=actually_think)
        elif pattern == "extraction":
            actually_think = rng.random() > args.think_omit_ratio
            ep = generate_extraction(rng.choice(docs_with_content), rng, lang, use_think=actually_think)
        elif pattern == "insufficient":
            actually_think = rng.random() > args.think_omit_ratio
            ep = generate_insufficient_context(rng.choice(docs_with_content), rng, lang, use_think=actually_think)
        elif pattern == "no_context":
            ep = generate_no_context(rng, lang)

        if ep:
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

    print(f"\n  Next step:")
    print(f"    python scripts/sft/prepare_chat_sft.py \\")
    print(f"        --input {args.output} \\")
    print(f"        --output_dir data/sft_phase3_chat \\")
    print(f"        --enable_packing --pack_block_size 1024")


if __name__ == "__main__":
    main()
