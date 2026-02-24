#!/usr/bin/env python3
"""
Convert HuggingFace SFT datasets to myPT JSONL format.

Supports multiple dataset formats and maps them all to our canonical
JSONL structure (system/user/assistant roles) compatible with
prepare_chat_sft.py and prepare_tool_sft.py.

Supported datasets:
    - OpenAssistant/oasst2           (conversation trees, multilingual)
    - allenai/Dolci-Instruct-SFT     (messages format, multilingual)
    - allenai/Dolci-Instruct-SFT-Tool-Use  (XML function calling)
    - Open-Orca/SlimOrca             (system/question/response)
    - vicgalle/alpaca-gpt4           (instruction/input/output)
    - databricks/databricks-dolly-15k (instruction/context/response)
    - mayflowergmbh/*_de             (ShareGPT format, German)
    - avemio/German-RAG-SFT-*        (ShareGPT format, German RAG)
    - flozi00/german-function-calling (German function calling)

Output JSONL format (per docs/sft/TAG_NESTING_REFERENCE.md):
    {
        "system": "You are MyPT.",
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "...", "think": "...", "cite": "..."}
        ],
        "language": "en",
        "source": "oasst2"
    }

Usage:
    # OASST2 (DE+EN, multi-turn)
    python scripts/sft/convert_hf_dataset.py \\
        --dataset OpenAssistant/oasst2 \\
        --output data/sft_hf/oasst2.jsonl \\
        --languages en de \\
        --max_examples 20000

    # Dolci Tool-Use (function calling)
    python scripts/sft/convert_hf_dataset.py \\
        --dataset allenai/Dolci-Instruct-SFT-Tool-Use \\
        --output data/sft_hf/dolci_tools.jsonl \\
        --max_examples 10000

    # German alpaca
    python scripts/sft/convert_hf_dataset.py \\
        --dataset mayflowergmbh/alpaca-gpt4_de \\
        --output data/sft_hf/alpaca_de.jsonl \\
        --max_examples 10000
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.system_prompts import CONVERSATION_SYSTEM_PROMPT, AGENTIC_STANDARD_PROMPT

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# OPTIONAL FIELD EXTRACTION (context / think / cite)
# =============================================================================

CONTEXT_KEYS = (
    "context", "contexts", "passage", "passages",
    "document", "documents", "evidence", "retrieved_context",
)
THINK_KEYS = (
    "think", "reasoning", "rationale", "analysis", "scratchpad", "cot",
)
CITE_KEYS = (
    "cite", "citation", "citations", "reference", "references",
    "source_url", "source_urls", "url", "urls", "doc_id", "document_id",
)


def _normalize_field_text(value: Any) -> str:
    """Normalize common field value types into compact text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, dict):
        # Common structures: {"text": "..."} / {"content": "..."} / {"id": "..."}
        for k in ("text", "content", "id", "title", "url", "source", "citation"):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        parts = [_normalize_field_text(v) for v in value]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        # Cites are usually more readable line-separated than JSON arrays.
        return " | ".join(parts)
    return str(value).strip()


def _extract_first_text(obj: Dict[str, Any], keys: Iterable[str]) -> str:
    """Return first non-empty normalized value among candidate keys."""
    for k in keys:
        if k in obj:
            v = _normalize_field_text(obj.get(k))
            if v:
                return v
    return ""


def _build_user_message(content: str, raw_msg: Optional[Dict[str, Any]] = None, raw_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build canonical user message and preserve retrieval context when available."""
    msg = {"role": "user", "content": content}
    user_context = ""
    if raw_msg:
        user_context = _extract_first_text(raw_msg, CONTEXT_KEYS)
    if (not user_context) and raw_row:
        user_context = _extract_first_text(raw_row, CONTEXT_KEYS)
    if user_context:
        msg["context"] = user_context
    return msg


def _build_assistant_message(
    content: str,
    raw_msg: Optional[Dict[str, Any]] = None,
    raw_row: Optional[Dict[str, Any]] = None,
    add_think: bool = False,
) -> Dict[str, Any]:
    """Build canonical assistant message with optional think/cite extraction."""
    msg = {"role": "assistant", "content": content}

    think = ""
    cite = ""
    if raw_msg:
        think = _extract_first_text(raw_msg, THINK_KEYS)
        cite = _extract_first_text(raw_msg, CITE_KEYS)
    if raw_row:
        if not think:
            think = _extract_first_text(raw_row, THINK_KEYS)
        if not cite:
            cite = _extract_first_text(raw_row, CITE_KEYS)

    # Keep optional explicit chain-of-thought only when present in source,
    # or when user explicitly requested extraction mode.
    if think and think.strip() != content.strip():
        msg["think"] = think
    elif add_think and len(content) > 220:
        # Conservative fallback: only if explicit add_think is requested.
        # We avoid synthetic CoT generation; this is just a weak splitter.
        for sep in ("\n\n", ". "):
            if sep in content:
                left, right = content.split(sep, 1)
                left = left.strip()
                if 30 <= len(left) <= 240:
                    msg["think"] = left
                    msg["content"] = right.strip()
                    break

    if cite:
        msg["cite"] = cite

    return msg


# =============================================================================
# DATASET-SPECIFIC PARSERS
# =============================================================================

def parse_oasst2(dataset, languages: List[str], max_examples: int, add_think: bool) -> List[dict]:
    """Parse OpenAssistant/oasst2 conversation trees.
    
    OASST2 has message trees. We extract linear conversation paths
    by following the highest-rated reply at each depth level.
    """
    episodes = []
    
    # Build message lookup: parent_id -> list of children
    messages_by_id = {}
    children_by_parent = {}
    
    for split in ['train', 'validation']:
        if split not in dataset:
            continue
        for row in dataset[split]:
            msg_id = row.get('message_id', '')
            parent_id = row.get('parent_id', None)
            lang = row.get('lang', 'en')
            
            if languages and lang not in languages:
                continue
            
            messages_by_id[msg_id] = row
            if parent_id:
                children_by_parent.setdefault(parent_id, []).append(msg_id)
    
    # Find root messages (no parent = conversation starters)
    roots = [mid for mid, msg in messages_by_id.items() 
             if msg.get('parent_id') is None and msg.get('role') == 'prompter']
    
    for root_id in roots:
        if len(episodes) >= max_examples:
            break
        
        # Walk the tree, picking the best-rated child at each level
        path = []
        current_id = root_id
        
        while current_id:
            msg = messages_by_id.get(current_id)
            if not msg:
                break
            path.append(msg)
            
            # Get children, pick highest rated
            kids = children_by_parent.get(current_id, [])
            if not kids:
                break
            
            best_kid = max(kids, key=lambda k: messages_by_id.get(k, {}).get('rank', 999))
            current_id = best_kid
        
        # Convert path to messages list
        if len(path) < 2:
            continue
        
        lang = path[0].get('lang', 'en')
        messages = []
        for msg in path:
            role = "user" if msg.get('role') == 'prompter' else "assistant"
            content = msg.get('text', '').strip()
            if not content:
                continue
            if role == "user":
                messages.append(_build_user_message(content, raw_msg=msg))
            else:
                messages.append(_build_assistant_message(content, raw_msg=msg, add_think=add_think))
        
        if len(messages) >= 2:
            episodes.append({
                "system": CONVERSATION_SYSTEM_PROMPT,
                "messages": messages,
                "language": lang,
                "source": "oasst2"
            })
    
    return episodes


def parse_dolci_instruct(dataset, languages: List[str], max_examples: int, add_think: bool) -> List[dict]:
    """Parse allenai/Dolci-Instruct-SFT (messages format)."""
    episodes = []
    
    for row in dataset.get('train', []):
        if len(episodes) >= max_examples:
            break
        
        messages_raw = row.get('messages', [])
        if not messages_raw:
            continue
        
        messages = []
        system_text = CONVERSATION_SYSTEM_PROMPT
        
        for msg in messages_raw:
            role = msg.get('role', '')
            content = msg.get('content', '').strip()
            if not content:
                continue
            
            if role == 'system':
                system_text = content
            elif role == 'user':
                messages.append(_build_user_message(content, raw_msg=msg, raw_row=row))
            elif role == 'assistant':
                messages.append(_build_assistant_message(content, raw_msg=msg, raw_row=row, add_think=add_think))
        
        if len(messages) >= 2:
            episodes.append({
                "system": system_text,
                "messages": messages,
                "language": "en",
                "source": "dolci_instruct"
            })
    
    return episodes


def parse_dolci_tool_use(dataset, languages: List[str], max_examples: int, add_think: bool) -> List[dict]:
    """Parse allenai/Dolci-Instruct-SFT-Tool-Use (XML function calling).
    
    Dolci uses XML format for function signatures and calls:
        <function_calls><invoke name="func"><parameter name="p">val</parameter></invoke></function_calls>
    
    We map this to our JSON toolcall format.
    """
    episodes = []
    
    for row in dataset.get('train', []):
        if len(episodes) >= max_examples:
            break
        
        messages_raw = row.get('messages', [])
        if not messages_raw:
            continue
        
        messages = []
        system_text = AGENTIC_STANDARD_PROMPT
        
        for msg in messages_raw:
            role = msg.get('role', '')
            content = msg.get('content', '').strip()
            function_calls = msg.get('function_calls', None)
            
            if role == 'system':
                system_text = content
            elif role == 'user':
                messages.append(_build_user_message(content, raw_msg=msg, raw_row=row))
            elif role == 'assistant':
                if function_calls:
                    # Parse XML function call to JSON
                    parsed = _parse_xml_function_call(function_calls)
                    if parsed:
                        messages.append({
                            "role": "assistant_toolcall",
                            "name": parsed["name"],
                            "arguments": parsed["arguments"]
                        })
                    else:
                        messages.append({"role": "assistant", "content": content or str(function_calls)})
                elif content:
                    messages.append(_build_assistant_message(content, raw_msg=msg, raw_row=row, add_think=add_think))
            elif role == 'tool':
                messages.append({
                    "role": "toolresult",
                    "name": msg.get('name', 'unknown'),
                    "content": content
                })
        
        if len(messages) >= 2:
            episodes.append({
                "system": system_text,
                "messages": messages,
                "language": "en",
                "source": "dolci_tool_use"
            })
    
    return episodes


def _parse_xml_function_call(xml_text: str) -> Optional[dict]:
    """Parse XML function call to name + arguments dict."""
    if not xml_text or not isinstance(xml_text, str):
        return None
    
    name_match = re.search(r'<invoke\s+name="([^"]+)"', xml_text)
    if not name_match:
        return None
    
    name = name_match.group(1)
    arguments = {}
    
    for param_match in re.finditer(r'<parameter\s+name="([^"]+)">([^<]*)</parameter>', xml_text):
        param_name = param_match.group(1)
        param_value = param_match.group(2).strip()
        arguments[param_name] = param_value
    
    return {"name": name, "arguments": arguments}


def parse_alpaca_format(dataset, languages: List[str], max_examples: int, 
                        add_think: bool, source_name: str = "alpaca") -> List[dict]:
    """Parse Alpaca-style datasets (instruction/input/output).
    
    Works for: vicgalle/alpaca-gpt4, mayflowergmbh/alpaca-gpt4_de, 
    LEL-A/translated_german_alpaca, etc.
    """
    episodes = []
    
    for row in dataset.get('train', []):
        if len(episodes) >= max_examples:
            break
        
        instruction = row.get('instruction', '').strip()
        input_text = row.get('input', '').strip()
        output = row.get('output', '').strip()
        
        if not instruction or not output:
            continue
        
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\n{input_text}"
        
        lang = "de" if "_de" in source_name or "german" in source_name.lower() else "en"
        
        messages = [
            _build_user_message(user_content, raw_row=row),
            _build_assistant_message(output, raw_row=row, add_think=add_think),
        ]
        
        episodes.append({
            "system": CONVERSATION_SYSTEM_PROMPT,
            "messages": messages,
            "language": lang,
            "source": source_name
        })
    
    return episodes


def parse_orca_format(dataset, languages: List[str], max_examples: int, add_think: bool) -> List[dict]:
    """Parse Orca-style datasets (system_prompt/question/response).
    
    Works for: Open-Orca/SlimOrca, jphme/slimorca_dedup_german_experimental.
    """
    episodes = []
    
    for row in dataset.get('train', []):
        if len(episodes) >= max_examples:
            break
        
        system = row.get('system_prompt', row.get('system', '')).strip()
        question = row.get('question', row.get('human', row.get('instruction', ''))).strip()
        response = row.get('response', row.get('gpt', row.get('output', ''))).strip()
        
        if not question or not response:
            continue
        
        if not system:
            system = CONVERSATION_SYSTEM_PROMPT
        
        msg_entry = _build_assistant_message(response, raw_row=row, add_think=add_think)
        
        episodes.append({
            "system": system,
            "messages": [
                _build_user_message(question, raw_row=row),
                msg_entry
            ],
            "language": _detect_language(question),
            "source": "slimorca"
        })
    
    return episodes


def parse_sharegpt_format(dataset, languages: List[str], max_examples: int, 
                          add_think: bool, source_name: str = "sharegpt") -> List[dict]:
    """Parse ShareGPT-style datasets (conversations list with from/value).
    
    Works for: many German datasets, avemio/German-RAG-SFT-*, 
    sroecker/aya_german-sharegpt, etc.
    """
    episodes = []
    
    for row in dataset.get('train', []):
        if len(episodes) >= max_examples:
            break
        
        conversations = row.get('conversations', row.get('messages', []))
        if not conversations:
            continue
        
        messages = []
        system_text = CONVERSATION_SYSTEM_PROMPT
        
        for msg in conversations:
            sender = msg.get('from', msg.get('role', ''))
            value = msg.get('value', msg.get('content', '')).strip()
            if not value:
                continue
            
            if sender in ('system', 'System'):
                system_text = value
            elif sender in ('human', 'user', 'Human', 'User'):
                messages.append(_build_user_message(value, raw_msg=msg, raw_row=row))
            elif sender in ('gpt', 'assistant', 'GPT', 'Assistant', 'bot'):
                messages.append(_build_assistant_message(value, raw_msg=msg, raw_row=row, add_think=add_think))
        
        if len(messages) >= 2:
            lang = _detect_language(messages[0].get('content', ''))
            if "german" in source_name.lower() or "_de" in source_name:
                lang = "de"
            
            if languages and lang not in languages:
                continue
            
            episodes.append({
                "system": system_text,
                "messages": messages,
                "language": lang,
                "source": source_name
            })
    
    return episodes


def parse_dolly(dataset, languages: List[str], max_examples: int, add_think: bool) -> List[dict]:
    """Parse databricks/databricks-dolly-15k."""
    episodes = []
    
    for row in dataset.get('train', []):
        if len(episodes) >= max_examples:
            break
        
        instruction = row.get('instruction', '').strip()
        context = row.get('context', '').strip()
        response = row.get('response', '').strip()
        
        if not instruction or not response:
            continue
        
        user_content = instruction
        user_context = None
        if context:
            user_context = context
        
        msg = _build_user_message(user_content, raw_row=row)
        if user_context:
            msg["context"] = user_context
        
        episodes.append({
            "system": CONVERSATION_SYSTEM_PROMPT,
            "messages": [
                msg,
                _build_assistant_message(response, raw_row=row, add_think=add_think),
            ],
            "language": "en",
            "source": "dolly"
        })
    
    return episodes


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _detect_language(text: str) -> str:
    """Simple heuristic language detection (EN vs DE)."""
    german_words = {'und', 'der', 'die', 'das', 'ist', 'ein', 'eine', 'nicht',
                    'mit', 'sich', 'auf', 'für', 'werden', 'kann', 'sind',
                    'haben', 'auch', 'oder', 'aber', 'nach', 'wie', 'über'}
    words = set(text.lower().split()[:50])
    german_count = len(words & german_words)
    return "de" if german_count >= 3 else "en"


def filter_episodes(episodes: List[dict], min_response_tokens: int = 5,
                    max_response_tokens: int = 800) -> List[dict]:
    """Filter episodes by quality heuristics."""
    filtered = []
    for ep in episodes:
        messages = ep.get("messages", [])
        if len(messages) < 2:
            continue
        
        # Check last assistant message
        last_assistant = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break
        
        if not last_assistant:
            continue
        
        # Approximate token count (words / 0.75)
        word_count = len(last_assistant.split())
        approx_tokens = int(word_count / 0.75)
        
        if approx_tokens < min_response_tokens:
            continue
        if approx_tokens > max_response_tokens:
            continue
        
        # Skip garbled text (high ratio of non-ascii/non-german chars)
        printable_ratio = sum(1 for c in last_assistant if c.isprintable()) / max(len(last_assistant), 1)
        if printable_ratio < 0.9:
            continue
        
        filtered.append(ep)
    
    return filtered


def summarize_optional_fields(episodes: List[dict]) -> Dict[str, int]:
    """Count how often optional fields are present in converted episodes."""
    stats = {
        "user_messages": 0,
        "assistant_messages": 0,
        "user_with_context": 0,
        "assistant_with_think": 0,
        "assistant_with_cite": 0,
    }
    for ep in episodes:
        for msg in ep.get("messages", []):
            role = msg.get("role")
            if role == "user":
                stats["user_messages"] += 1
                if str(msg.get("context", "")).strip():
                    stats["user_with_context"] += 1
            elif role == "assistant":
                stats["assistant_messages"] += 1
                if str(msg.get("think", "")).strip():
                    stats["assistant_with_think"] += 1
                if str(msg.get("cite", "")).strip():
                    stats["assistant_with_cite"] += 1
    return stats


# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_PARSERS = {
    "OpenAssistant/oasst2": parse_oasst2,
    "allenai/Dolci-Instruct-SFT": parse_dolci_instruct,
    "allenai/Dolci-Instruct-SFT-Tool-Use": parse_dolci_tool_use,
    "Open-Orca/SlimOrca": parse_orca_format,
    "databricks/databricks-dolly-15k": parse_dolly,
    "vicgalle/alpaca-gpt4": lambda ds, l, m, t: parse_alpaca_format(ds, l, m, t, "alpaca_gpt4"),
    "mayflowergmbh/alpaca-gpt4_de": lambda ds, l, m, t: parse_alpaca_format(ds, l, m, t, "alpaca_gpt4_de"),
    "LEL-A/translated_german_alpaca": lambda ds, l, m, t: parse_alpaca_format(ds, l, m, t, "german_alpaca"),
    "jphme/slimorca_dedup_german_experimental": parse_orca_format,
    "mayflowergmbh/ultra-chat_de": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "ultra_chat_de"),
    "mayflowergmbh/dolphin_de": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "dolphin_de"),
    "LeoLM/OpenSchnabeltier": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "openschnabeltier"),
    "avemio/German-RAG-SFT-ShareGPT-HESSIAN-AI": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "german_rag_sft"),
    "flozi00/german-function-calling": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "german_function_calling"),
    "mayflowergmbh/wiki_qa_de": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "wiki_qa_de"),
    "DiscoResearch/germanrag": lambda ds, l, m, t: parse_sharegpt_format(ds, l, m, t, "germanrag"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace SFT datasets to myPT JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported datasets:\n" + "\n".join(f"  - {k}" for k in DATASET_PARSERS.keys())
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset path (e.g., OpenAssistant/oasst2)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file path")
    parser.add_argument("--languages", nargs="*", default=["en", "de"],
                        help="Languages to keep (default: en de)")
    parser.add_argument("--max_examples", type=int, default=50000,
                        help="Maximum number of episodes to extract (default: 50000)")
    parser.add_argument("--add_think", action="store_true",
                        help="Wrap reasoning/explanation in <myPT_think> tags")
    parser.add_argument("--min_response_tokens", type=int, default=5,
                        help="Min approximate response tokens (default: 5)")
    parser.add_argument("--max_response_tokens", type=int, default=800,
                        help="Max approximate response tokens (default: 800, fits 1024 context)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset/config name (if applicable)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip quality filtering")
    args = parser.parse_args()
    
    # Check if dataset is supported
    if args.dataset not in DATASET_PARSERS:
        print(f"Error: Unknown dataset '{args.dataset}'")
        print(f"Supported datasets:")
        for k in DATASET_PARSERS.keys():
            print(f"  - {k}")
        sys.exit(1)
    
    # Load dataset from HuggingFace
    print(f"Loading dataset: {args.dataset}")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        sys.exit(1)
    
    load_kwargs = {"trust_remote_code": True}
    if args.subset:
        load_kwargs["name"] = args.subset
    
    dataset = load_dataset(args.dataset, **load_kwargs)
    
    # Parse using the appropriate handler
    print(f"Parsing with handler for: {args.dataset}")
    parse_fn = DATASET_PARSERS[args.dataset]
    episodes = parse_fn(dataset, args.languages, args.max_examples, args.add_think)
    
    print(f"  Parsed: {len(episodes)} episodes")
    
    # Filter for quality
    if not args.no_filter:
        before = len(episodes)
        episodes = filter_episodes(
            episodes,
            min_response_tokens=args.min_response_tokens,
            max_response_tokens=args.max_response_tokens,
        )
        print(f"  After filtering: {len(episodes)} episodes ({before - len(episodes)} removed)")
    
    # Language breakdown
    lang_counts = {}
    for ep in episodes:
        lang = ep.get("language", "?")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print(f"  Languages: {lang_counts}")

    # Optional field coverage (for prepare_chat_sft --enable_rag_tags)
    opt = summarize_optional_fields(episodes)
    print(
        "  Optional field coverage: "
        f"user.context={opt['user_with_context']}/{opt['user_messages']}, "
        f"assistant.think={opt['assistant_with_think']}/{opt['assistant_messages']}, "
        f"assistant.cite={opt['assistant_with_cite']}/{opt['assistant_messages']}"
    )
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + '\n')
    
    print(f"\n  Output: {output_path}")
    print(f"  Total episodes: {len(episodes)}")
    
    # Show samples
    print(f"\nSample episodes:")
    import random
    random.seed(args.seed)
    samples = random.sample(episodes, min(3, len(episodes)))
    for i, ep in enumerate(samples):
        msgs = ep.get("messages", [])
        lang = ep.get("language", "?")
        src = ep.get("source", "?")
        user_msg = msgs[0].get("content", "")[:80] if msgs else "?"
        asst_msg = msgs[-1].get("content", "")[:80] if len(msgs) > 1 else "?"
        print(f"  [{i+1}] lang={lang} src={src}")
        print(f"      U: {user_msg}...")
        print(f"      A: {asst_msg}...")


if __name__ == "__main__":
    main()
