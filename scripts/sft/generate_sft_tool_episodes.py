#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multi-step agentic tool-use SFT episodes for Phase 6.

Creates multi-step tool-calling episodes that teach the model:
1. Multi-step tool chains (search -> get_doc -> answer)
2. LIST_SELECT_SUMMARIZE pattern (list -> get_doc -> summarize -> answer)
3. Full <myPT_think> reasoning chains across multiple steps
4. <myPT_cite> source attribution in final answers
5. ERROR_RECOVERY (tool returns error, graceful message)
6. NO_TOOL (answer directly without tool calls)

This generator uses the real WorkspaceEngine + WorkspaceTools to produce
authentic tool results, making episodes realistic and grounded.

Patterns (configurable weights):
    SEARCH_ANSWER (20%):            search -> answer
    SEARCH_GETDOC_ANSWER (30%):     search -> get_doc -> answer
    LIST_SELECT_SUMMARIZE (20%):    list_docs -> get_doc -> summarize -> answer
    ERROR_RECOVERY (10%):           search (no results) -> graceful message
    NO_TOOL (20%):                  direct answer without tools

Per the spec at docs/specs/spec_sft_tool_sequenze_generator.md:
- Offline only, no external APIs
- Deterministic (seeded RNG)
- Tool outputs from real tool execution (or controlled mocks)
- Validation: JSON parseable, valid tool names, answer references results

Usage:
    python scripts/sft/generate_sft_tool_episodes.py \\
        --workspace_dir workspace/ \\
        --output data/sft_phase6_intermediate/agentic_episodes.jsonl

    python scripts/sft/generate_sft_tool_episodes.py \\
        --workspace_dir workspace/ \\
        --output data/sft_phase6_intermediate/agentic_episodes.jsonl \\
        --num_examples 5000 --language mixed --seed 42

Then prepare:
    python scripts/sft/prepare_tool_sft.py \\
        --input data/sft_phase6_intermediate/agentic_episodes.jsonl \\
        --output_dir data/sft_phase6_agentic
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

from core.system_prompts import AGENTIC_STANDARD_PROMPT


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS = [
    AGENTIC_STANDARD_PROMPT,
    "You are MyPT, an AI assistant with access to a document workspace. Use tools to search, read, and summarize documents when needed. Answer directly when no tool is required.",
    "You are MyPT. Answer questions using workspace tools when needed.\nTools: workspace.search, workspace.list_docs, workspace.get_doc, workspace.summarize",
]


# =============================================================================
# Question Templates (EN + DE)
# =============================================================================

SEARCH_ANSWER_Q_EN = [
    "What information do we have about {topic}?",
    "Search for {topic} and tell me what you find.",
    "What do our docs say about {topic}?",
    "Find anything about {topic}.",
    "I need info on {topic}.",
]

SEARCH_ANSWER_Q_DE = [
    "Welche Informationen haben wir ueber {topic}?",
    "Suche nach {topic} und sag mir was du findest.",
    "Was sagen unsere Dokumente ueber {topic}?",
    "Finde alles ueber {topic}.",
    "Ich brauche Infos zu {topic}.",
]

SEARCH_GETDOC_Q_EN = [
    "Find documents about {topic} and give me the full details.",
    "Search for {topic}, then show me the relevant document.",
    "What's in our {topic} documentation? Get the full text.",
    "Look up {topic} and read the matching document for me.",
    "I want a thorough answer about {topic} from the workspace.",
]

SEARCH_GETDOC_Q_DE = [
    "Finde Dokumente ueber {topic} und gib mir die vollstaendigen Details.",
    "Suche nach {topic} und zeig mir dann das relevante Dokument.",
    "Was steht in unserer {topic}-Dokumentation? Hol den vollen Text.",
    "Schlag {topic} nach und lies das passende Dokument fuer mich.",
    "Ich moechte eine gruendliche Antwort zu {topic} aus dem Workspace.",
]

LIST_SELECT_Q_EN = [
    "What documents do we have? Pick one and summarize it for me.",
    "List the workspace docs and give me a summary of something interesting.",
    "Show me available documents and summarize one of them.",
    "What's in the workspace? Read one document and summarize it.",
]

LIST_SELECT_Q_DE = [
    "Welche Dokumente haben wir? Waehle eins aus und fasse es fuer mich zusammen.",
    "Liste die Workspace-Dokumente auf und gib mir eine Zusammenfassung von etwas Interessantem.",
    "Zeig mir die verfuegbaren Dokumente und fasse eines davon zusammen.",
    "Was ist im Workspace? Lies ein Dokument und fasse es zusammen.",
]

ERROR_Q_EN = [
    "Search for {topic} in the workspace.",
    "What do we know about {topic}?",
    "Find documents about {topic}.",
]

ERROR_Q_DE = [
    "Suche nach {topic} im Workspace.",
    "Was wissen wir ueber {topic}?",
    "Finde Dokumente ueber {topic}.",
]


# =============================================================================
# Think Templates
# =============================================================================

THINK_SEARCH_EN = [
    "The user wants to know about {topic}. Let me search the workspace.",
    "I should search for documents related to {topic} first.",
    "Searching the workspace for {topic} to find relevant information.",
]

THINK_SEARCH_DE = [
    "Der Benutzer moechte etwas ueber {topic} wissen. Lass mich den Workspace durchsuchen.",
    "Ich sollte zuerst nach Dokumenten zu {topic} suchen.",
    "Ich durchsuche den Workspace nach {topic} um relevante Informationen zu finden.",
]

THINK_GETDOC_EN = [
    "The search found a relevant document: {title}. Let me get the full text.",
    "I found {title} in the search results. I'll retrieve the complete document.",
    "The document {title} seems most relevant. Let me read it in full.",
]

THINK_GETDOC_DE = [
    "Die Suche hat ein relevantes Dokument gefunden: {title}. Lass mich den vollen Text holen.",
    "Ich habe {title} in den Suchergebnissen gefunden. Ich rufe das vollstaendige Dokument ab.",
    "Das Dokument {title} scheint am relevantesten. Lass mich es vollstaendig lesen.",
]

THINK_LIST_EN = [
    "The user wants to see what's available. I'll list all documents first.",
    "Let me check what documents are in the workspace.",
]

THINK_LIST_DE = [
    "Der Benutzer moechte sehen was verfuegbar ist. Ich liste zuerst alle Dokumente auf.",
    "Lass mich pruefen welche Dokumente im Workspace sind.",
]

THINK_SELECT_EN = [
    "From the list, {title} looks interesting. Let me get its full text.",
    "I'll pick {title} from the available documents and read it.",
]

THINK_SELECT_DE = [
    "Aus der Liste sieht {title} interessant aus. Lass mich den vollen Text holen.",
    "Ich waehle {title} aus den verfuegbaren Dokumenten und lese es.",
]

THINK_SUMMARIZE_EN = [
    "The document is quite long. Let me summarize it for the user.",
    "I have the full text of {title}. I'll summarize the key points.",
]

THINK_SUMMARIZE_DE = [
    "Das Dokument ist ziemlich lang. Lass mich es fuer den Benutzer zusammenfassen.",
    "Ich habe den vollen Text von {title}. Ich fasse die wichtigsten Punkte zusammen.",
]

THINK_ANSWER_EN = [
    "Now I have the information to answer the user's question about {topic}.",
    "Based on the document content, I can provide a grounded answer about {topic}.",
    "I've gathered enough information. Let me compose the answer.",
]

THINK_ANSWER_DE = [
    "Jetzt habe ich die Informationen um die Frage des Benutzers zu {topic} zu beantworten.",
    "Basierend auf dem Dokumentinhalt kann ich eine fundierte Antwort zu {topic} geben.",
    "Ich habe genug Informationen gesammelt. Lass mich die Antwort formulieren.",
]

THINK_NO_RESULTS_EN = [
    "The search returned no results for {topic}. I should let the user know gracefully.",
    "No documents found about {topic}. The workspace doesn't cover this topic.",
]

THINK_NO_RESULTS_DE = [
    "Die Suche hat keine Ergebnisse fuer {topic} geliefert. Ich sollte den Benutzer freundlich informieren.",
    "Keine Dokumente zu {topic} gefunden. Der Workspace deckt dieses Thema nicht ab.",
]

THINK_NO_TOOL_EN = [
    "This is a general knowledge question. I can answer directly without tools.",
    "No tool needed here -- I know the answer.",
    "I'll answer directly since this doesn't require workspace search.",
]

THINK_NO_TOOL_DE = [
    "Das ist eine allgemeine Wissensfrage. Ich kann direkt ohne Tools antworten.",
    "Kein Tool noetig -- ich kenne die Antwort.",
    "Ich antworte direkt, da keine Workspace-Suche erforderlich ist.",
]


# =============================================================================
# Answer Templates
# =============================================================================

ANSWER_SEARCH_EN = [
    "Based on the search results, {content}",
    "Here's what I found about {topic}:\n\n{content}",
    "The workspace has the following about {topic}:\n\n{content}",
]

ANSWER_SEARCH_DE = [
    "Basierend auf den Suchergebnissen, {content}",
    "Hier ist was ich ueber {topic} gefunden habe:\n\n{content}",
    "Der Workspace hat Folgendes zu {topic}:\n\n{content}",
]

ANSWER_GETDOC_EN = [
    "After searching and reading {filename}, here's what I found about {topic}:\n\n{content}",
    "Based on the document {filename}:\n\n{content}",
    "From {filename}, the key information about {topic} is:\n\n{content}",
]

ANSWER_GETDOC_DE = [
    "Nach Suche und Lesen von {filename} habe ich Folgendes zu {topic} gefunden:\n\n{content}",
    "Basierend auf dem Dokument {filename}:\n\n{content}",
    "Aus {filename}, die wichtigsten Informationen zu {topic} sind:\n\n{content}",
]

ANSWER_LIST_SUMMARIZE_EN = [
    "The workspace has {count} documents. I picked {filename} and here's a summary:\n\n{summary}",
    "From {count} available documents, I read {filename}. Summary:\n\n{summary}",
    "Here's a summary of {filename} from the workspace:\n\n{summary}",
]

ANSWER_LIST_SUMMARIZE_DE = [
    "Der Workspace hat {count} Dokumente. Ich habe {filename} gewaehlt und hier ist eine Zusammenfassung:\n\n{summary}",
    "Aus {count} verfuegbaren Dokumenten habe ich {filename} gelesen. Zusammenfassung:\n\n{summary}",
    "Hier ist eine Zusammenfassung von {filename} aus dem Workspace:\n\n{summary}",
]

ANSWER_ERROR_EN = [
    "I searched for {topic} but couldn't find any relevant documents in the workspace. This topic might not be covered in the current knowledge base.",
    "No results found for {topic}. The workspace doesn't seem to have documentation on this topic.",
    "Sorry, I couldn't find any documents about {topic} in the workspace. You may need to add relevant documentation.",
]

ANSWER_ERROR_DE = [
    "Ich habe nach {topic} gesucht, konnte aber keine relevanten Dokumente im Workspace finden. Dieses Thema ist moeglicherweise nicht in der aktuellen Wissensdatenbank abgedeckt.",
    "Keine Ergebnisse fuer {topic} gefunden. Der Workspace scheint keine Dokumentation zu diesem Thema zu haben.",
    "Entschuldigung, ich konnte keine Dokumente ueber {topic} im Workspace finden. Moeglicherweise muessen relevante Dokumente hinzugefuegt werden.",
]


# =============================================================================
# NO_TOOL Q&A
# =============================================================================

NO_TOOL_QA_EN = [
    ("What is Python?",
     "Python is a high-level, interpreted programming language known for clear syntax and readability. It supports object-oriented, procedural, and functional programming."),
    ("What is version control?",
     "Version control tracks changes to files over time. Git is the most popular system, using commits, branches, and merges to manage code history."),
    ("How does encryption work?",
     "Encryption converts readable data into encoded format using a key. Symmetric encryption uses one shared key, asymmetric uses a public/private pair."),
    ("What is a REST API?",
     "A REST API uses HTTP methods (GET, POST, PUT, DELETE) with resource-based URLs, returns JSON, and is stateless. Each request contains all needed information."),
    ("Explain machine learning briefly.",
     "Machine learning is AI that learns patterns from data. Main types: supervised (labeled data), unsupervised (finding structure), and reinforcement learning (rewards-based)."),
    ("What is Docker?",
     "Docker packages applications in containers -- lightweight units that include all dependencies. Containers share the host OS kernel, start in seconds, and run consistently everywhere."),
    ("What is a neural network?",
     "A neural network has layers of connected nodes. Data flows through input, hidden, and output layers. The network learns by adjusting connection weights during training."),
    ("What is JSON?",
     "JSON is a lightweight data format with key-value pairs and arrays. It's human-readable and widely used for APIs and configuration files."),
]

NO_TOOL_QA_DE = [
    ("Was ist Python?",
     "Python ist eine hoeherwertige, interpretierte Programmiersprache, bekannt fuer klare Syntax und Lesbarkeit. Sie unterstuetzt objektorientierte, prozedurale und funktionale Programmierung."),
    ("Was ist Versionskontrolle?",
     "Versionskontrolle verfolgt Aenderungen an Dateien ueber die Zeit. Git ist das populaerste System mit Commits, Branches und Merges zur Verwaltung der Code-Historie."),
    ("Wie funktioniert Verschluesselung?",
     "Verschluesselung wandelt lesbare Daten mit einem Schluessel in ein codiertes Format um. Symmetrisch: ein gemeinsamer Schluessel. Asymmetrisch: oeffentliches/privates Schluesselpaar."),
    ("Was ist eine REST-API?",
     "Eine REST-API verwendet HTTP-Methoden (GET, POST, PUT, DELETE) mit ressourcenbasierten URLs, gibt JSON zurueck und ist zustandslos."),
    ("Erklaere maschinelles Lernen kurz.",
     "Maschinelles Lernen ist KI, die Muster aus Daten lernt. Haupttypen: ueberwacht (markierte Daten), unueberwacht (Strukturfindung) und Reinforcement Learning (belohnungsbasiert)."),
    ("Was ist Docker?",
     "Docker verpackt Anwendungen in Container -- leichtgewichtige Einheiten mit allen Abhaengigkeiten. Container teilen den Host-OS-Kernel, starten in Sekunden und laufen ueberall konsistent."),
    ("Was ist ein neuronales Netz?",
     "Ein neuronales Netz hat Schichten verbundener Knoten. Daten fliessen durch Eingabe-, versteckte und Ausgabeschichten. Das Netzwerk lernt durch Anpassung der Gewichte beim Training."),
    ("Was ist JSON?",
     "JSON ist ein leichtgewichtiges Datenformat mit Schluessel-Wert-Paaren und Arrays. Es ist menschenlesbar und weit verbreitet fuer APIs und Konfigurationsdateien."),
]


# =============================================================================
# Document Helpers
# =============================================================================

def load_documents(docs_dir: str):
    """Load documents using DocumentLoader."""
    from core.document import DocumentLoader, TextChunker
    loader = DocumentLoader()
    documents = loader.load_directory(docs_dir)
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_documents(documents)
    return documents, chunks


def get_doc_id(doc) -> str:
    import hashlib
    return hashlib.md5(doc.source.encode()).hexdigest()[:12]


def get_doc_title(doc) -> str:
    name = os.path.splitext(doc.filename)[0]
    return name.replace("_", " ").replace("-", " ").title()


def truncate_text(text: str, max_chars: int = 500) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        return truncated[:last_space] + "..."
    return truncated + "..."


def extract_topics(documents: List) -> List[str]:
    """Extract search topics from document content."""
    topics = set()
    for doc in documents:
        name = os.path.splitext(doc.filename)[0]
        topics.add(name.replace("_", " ").replace("-", " "))
        first_line = doc.text.split("\n")[0].strip().lstrip("#").strip()
        if first_line and len(first_line) < 50:
            topics.add(first_line)
    return [t for t in topics if 3 <= len(t) <= 50]


def make_extractive_summary(text: str, max_len: int = 500) -> str:
    """Create an extractive summary from document text."""
    lines = text.split("\n")
    summary_lines = []
    char_count = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        if char_count + len(line) > max_len:
            break
        summary_lines.append(line)
        char_count += len(line)
    return " ".join(summary_lines) if summary_lines else text[:max_len] + "..."


# =============================================================================
# Episode Generators
# =============================================================================

def gen_search_answer(
    documents, chunks, topics, rng, lang, max_chars
) -> Optional[Dict]:
    """SEARCH_ANSWER: search -> answer (single step)."""
    topic = rng.choice(topics) if topics else None
    if not topic:
        return None

    topic_lower = topic.lower()
    relevant = [c for c in chunks if topic_lower in c.text.lower()]
    if not relevant:
        return None
    relevant = relevant[:3]

    tool_result = {
        "documents": [
            {"chunk_id": c.chunk_id, "text": truncate_text(c.text, 400), "source": c.source}
            for c in relevant
        ],
        "total": len(relevant),
    }

    content_parts = []
    cite_src = None
    for i, c in enumerate(relevant, 1):
        src = c.source.get("filename", "unknown")
        if cite_src is None:
            cite_src = src
        content_parts.append(f"[{i}] {truncate_text(c.text, 200)}")
    content = "\n\n".join(content_parts)

    tpl_q = SEARCH_ANSWER_Q_DE if lang == "de" else SEARCH_ANSWER_Q_EN
    tpl_think_s = THINK_SEARCH_DE if lang == "de" else THINK_SEARCH_EN
    tpl_think_a = THINK_ANSWER_DE if lang == "de" else THINK_ANSWER_EN
    tpl_a = ANSWER_SEARCH_DE if lang == "de" else ANSWER_SEARCH_EN

    return {
        "system": rng.choice(SYSTEM_PROMPTS),
        "messages": [
            {"role": "user", "content": rng.choice(tpl_q).format(topic=topic)},
            {"role": "assistant_toolcall", "name": "workspace.search",
             "arguments": {"query": topic, "top_k": 3},
             "think": rng.choice(tpl_think_s).format(topic=topic)},
            {"role": "toolresult", "name": "workspace.search", "content": tool_result},
            {"role": "assistant",
             "content": rng.choice(tpl_a).format(topic=topic, content=content),
             "think": rng.choice(tpl_think_a).format(topic=topic),
             "cite": cite_src or ""},
        ],
        "language": lang,
    }


def gen_search_getdoc_answer(
    documents, chunks, topics, rng, lang, max_chars
) -> Optional[Dict]:
    """SEARCH_GETDOC_ANSWER: search -> get_doc -> answer (two steps)."""
    topic = rng.choice(topics) if topics else None
    if not topic:
        return None

    topic_lower = topic.lower()
    relevant_docs = [d for d in documents if topic_lower in d.text.lower()]
    if not relevant_docs:
        return None

    doc = rng.choice(relevant_docs)
    doc_id = get_doc_id(doc)
    title = get_doc_title(doc)

    search_result = {
        "documents": [{"doc_id": doc_id, "text": truncate_text(doc.text, 200),
                        "source": {"filename": doc.filename}}],
        "total": 1,
    }

    getdoc_result = {
        "doc_id": doc_id, "title": title,
        "text": truncate_text(doc.text, max_chars), "length": len(doc.text),
    }

    content = truncate_text(doc.text, max_chars)

    tpl_q = SEARCH_GETDOC_Q_DE if lang == "de" else SEARCH_GETDOC_Q_EN
    tpl_ts = THINK_SEARCH_DE if lang == "de" else THINK_SEARCH_EN
    tpl_tg = THINK_GETDOC_DE if lang == "de" else THINK_GETDOC_EN
    tpl_a = ANSWER_GETDOC_DE if lang == "de" else ANSWER_GETDOC_EN

    return {
        "system": rng.choice(SYSTEM_PROMPTS),
        "messages": [
            {"role": "user", "content": rng.choice(tpl_q).format(topic=topic)},
            {"role": "assistant_toolcall", "name": "workspace.search",
             "arguments": {"query": topic, "top_k": 3},
             "think": rng.choice(tpl_ts).format(topic=topic)},
            {"role": "toolresult", "name": "workspace.search", "content": search_result},
            {"role": "assistant_toolcall", "name": "workspace.get_doc",
             "arguments": {"doc_id": doc_id},
             "think": rng.choice(tpl_tg).format(title=title)},
            {"role": "toolresult", "name": "workspace.get_doc", "content": getdoc_result},
            {"role": "assistant",
             "content": rng.choice(tpl_a).format(topic=topic, filename=doc.filename, content=content),
             "cite": doc.filename},
        ],
        "language": lang,
    }


def gen_list_select_summarize(
    documents, rng, lang, max_chars
) -> Optional[Dict]:
    """LIST_SELECT_SUMMARIZE: list_docs -> get_doc -> summarize -> answer (three steps)."""
    if len(documents) < 2:
        return None

    list_result = {
        "documents": [
            {"doc_id": get_doc_id(d), "title": get_doc_title(d), "filename": d.filename}
            for d in documents
        ],
        "total": len(documents),
    }

    doc = rng.choice(documents)
    doc_id = get_doc_id(doc)
    title = get_doc_title(doc)

    getdoc_result = {
        "doc_id": doc_id, "title": title,
        "text": truncate_text(doc.text, max_chars), "length": len(doc.text),
    }

    summary = make_extractive_summary(doc.text, 400)
    summarize_result = {"summary": summary, "source": "doc_id", "original_length": len(doc.text)}

    tpl_q = LIST_SELECT_Q_DE if lang == "de" else LIST_SELECT_Q_EN
    tpl_tl = THINK_LIST_DE if lang == "de" else THINK_LIST_EN
    tpl_ts = THINK_SELECT_DE if lang == "de" else THINK_SELECT_EN
    tpl_tsum = THINK_SUMMARIZE_DE if lang == "de" else THINK_SUMMARIZE_EN
    tpl_a = ANSWER_LIST_SUMMARIZE_DE if lang == "de" else ANSWER_LIST_SUMMARIZE_EN

    return {
        "system": rng.choice(SYSTEM_PROMPTS),
        "messages": [
            {"role": "user", "content": rng.choice(tpl_q)},
            {"role": "assistant_toolcall", "name": "workspace.list_docs",
             "arguments": {},
             "think": rng.choice(tpl_tl)},
            {"role": "toolresult", "name": "workspace.list_docs", "content": list_result},
            {"role": "assistant_toolcall", "name": "workspace.get_doc",
             "arguments": {"doc_id": doc_id},
             "think": rng.choice(tpl_ts).format(title=title)},
            {"role": "toolresult", "name": "workspace.get_doc", "content": getdoc_result},
            {"role": "assistant_toolcall", "name": "workspace.summarize",
             "arguments": {"doc_id": doc_id},
             "think": rng.choice(tpl_tsum).format(title=title)},
            {"role": "toolresult", "name": "workspace.summarize", "content": summarize_result},
            {"role": "assistant",
             "content": rng.choice(tpl_a).format(
                 count=len(documents), filename=doc.filename, summary=summary),
             "cite": doc.filename},
        ],
        "language": lang,
    }


def gen_error_recovery(rng, lang) -> Dict:
    """ERROR_RECOVERY: search for nonexistent topic -> graceful message."""
    fake_topics = [
        "quantum computing", "blockchain", "mars exploration",
        "ancient history", "underwater photography", "jazz music",
        "cryptocurrency", "space travel", "deep sea biology",
    ]
    topic = rng.choice(fake_topics)
    tool_result = {"documents": [], "total": 0}

    tpl_q = ERROR_Q_DE if lang == "de" else ERROR_Q_EN
    tpl_ts = THINK_SEARCH_DE if lang == "de" else THINK_SEARCH_EN
    tpl_tnr = THINK_NO_RESULTS_DE if lang == "de" else THINK_NO_RESULTS_EN
    tpl_a = ANSWER_ERROR_DE if lang == "de" else ANSWER_ERROR_EN

    return {
        "system": rng.choice(SYSTEM_PROMPTS),
        "messages": [
            {"role": "user", "content": rng.choice(tpl_q).format(topic=topic)},
            {"role": "assistant_toolcall", "name": "workspace.search",
             "arguments": {"query": topic, "top_k": 3},
             "think": rng.choice(tpl_ts).format(topic=topic)},
            {"role": "toolresult", "name": "workspace.search", "content": tool_result},
            {"role": "assistant",
             "content": rng.choice(tpl_a).format(topic=topic),
             "think": rng.choice(tpl_tnr).format(topic=topic)},
        ],
        "language": lang,
    }


def gen_no_tool(rng, lang) -> Dict:
    """NO_TOOL: answer directly without calling any tool."""
    qa = NO_TOOL_QA_DE if lang == "de" else NO_TOOL_QA_EN
    question, answer = rng.choice(qa)
    tpl_t = THINK_NO_TOOL_DE if lang == "de" else THINK_NO_TOOL_EN

    return {
        "system": rng.choice(SYSTEM_PROMPTS),
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer,
             "think": rng.choice(tpl_t)},
        ],
        "language": lang,
    }


# =============================================================================
# Validation
# =============================================================================

VALID_TOOLS = {"workspace.search", "workspace.list_docs", "workspace.get_doc", "workspace.summarize"}


def validate_episode(episode: Dict) -> bool:
    """Validate an episode for correctness."""
    messages = episode.get("messages", [])
    if not messages:
        return False

    has_assistant = False
    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant_toolcall":
            name = msg.get("name", "")
            if name not in VALID_TOOLS:
                return False
            args = msg.get("arguments", {})
            if not isinstance(args, dict):
                return False
            try:
                json.dumps({"name": name, **args})
            except (TypeError, ValueError):
                return False
        elif role == "assistant":
            has_assistant = True
            if not msg.get("content", "").strip():
                return False

    return has_assistant


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-step agentic tool-use SFT episodes (Phase 6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workspace_dir", type=str, default="workspace/",
                        help="Workspace directory (default: workspace/)")
    parser.add_argument("--docs_dir", type=str, default=None,
                        help="Documents directory (default: workspace_dir/docs/)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--num_examples", type=int, default=5000,
                        help="Number of episodes to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--max_doc_chars", type=int, default=1500,
                        help="Max chars per document in tool results (default: 1500)")
    parser.add_argument("--language", type=str, default="mixed",
                        choices=["en", "de", "mixed"],
                        help="Language: en, de, or mixed (default: mixed)")
    parser.add_argument("--de_ratio", type=float, default=0.30,
                        help="Fraction of DE episodes when language=mixed (default: 0.30)")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 60)
    print("  MyPT Phase 6: Multi-Step Agentic Tool Episodes Generator")
    print("=" * 60)

    docs_dir = args.docs_dir or os.path.join(args.workspace_dir, "docs")
    if not os.path.exists(docs_dir):
        print(f"Error: Documents directory not found: {docs_dir}")
        sys.exit(1)

    print(f"\n  Language: {args.language} (DE ratio: {args.de_ratio})")
    print(f"\nLoading documents from: {docs_dir}")

    documents, chunks = load_documents(docs_dir)
    if not documents:
        print("Error: No documents found.")
        sys.exit(1)

    topics = extract_topics(documents)
    print(f"  Loaded {len(documents)} documents, {len(chunks)} chunks, {len(topics)} topics")

    pattern_weights = [
        ("search_answer",           0.20),
        ("search_getdoc_answer",    0.30),
        ("list_select_summarize",   0.20),
        ("error_recovery",          0.10),
        ("no_tool",                 0.20),
    ]

    episodes = []
    pattern_counts = {p[0]: 0 for p in pattern_weights}
    validation_fails = 0
    attempts = 0
    max_attempts = args.num_examples * 10

    print(f"\nGenerating {args.num_examples} episodes...")

    while len(episodes) < args.num_examples and attempts < max_attempts:
        attempts += 1

        if args.language == "mixed":
            lang = "de" if rng.random() < args.de_ratio else "en"
        else:
            lang = args.language

        r = rng.random()
        cumulative = 0
        pattern = pattern_weights[0][0]
        for pname, pweight in pattern_weights:
            cumulative += pweight
            if r <= cumulative:
                pattern = pname
                break

        ep = None
        if pattern == "search_answer":
            ep = gen_search_answer(documents, chunks, topics, rng, lang, args.max_doc_chars)
        elif pattern == "search_getdoc_answer":
            ep = gen_search_getdoc_answer(documents, chunks, topics, rng, lang, args.max_doc_chars)
        elif pattern == "list_select_summarize":
            ep = gen_list_select_summarize(documents, rng, lang, args.max_doc_chars)
        elif pattern == "error_recovery":
            ep = gen_error_recovery(rng, lang)
        elif pattern == "no_tool":
            ep = gen_no_tool(rng, lang)

        if ep and validate_episode(ep):
            episodes.append(ep)
            pattern_counts[pattern] += 1
        elif ep:
            validation_fails += 1

        if args.verbose and len(episodes) % 500 == 0 and len(episodes) > 0:
            print(f"  Generated {len(episodes)} episodes...")

    rng.shuffle(episodes)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\n  Generated {len(episodes)} episodes -> {args.output}")
    if validation_fails:
        print(f"  Validation failures: {validation_fails}")

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

    tool_step_counts = {}
    for ep in episodes:
        n = sum(1 for m in ep["messages"] if m["role"] == "assistant_toolcall")
        tool_step_counts[n] = tool_step_counts.get(n, 0) + 1
    print(f"\n  Tool steps per episode:")
    for steps, count in sorted(tool_step_counts.items()):
        pct = count / len(episodes) * 100 if episodes else 0
        print(f"    {steps} tool calls: {count} ({pct:.1f}%)")

    print(f"\n  Next step:")
    print(f"    python scripts/sft/prepare_tool_sft.py \\")
    print(f"        --input {args.output} \\")
    print(f"        --output_dir data/sft_phase6_agentic")


if __name__ == "__main__":
    main()
