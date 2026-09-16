#!/usr/bin/env python3
"""Write held-out capability eval JSONL for Phases 4–6.

Phase 4 hard gate lives in sft_eval_suite.multiturn_coherence.
These files are:
  - data/eval_ood/phase4_multiturn_ood.jsonl  (warning, novel phrasing)
  - data/eval_capability/toolcall_basic.jsonl (P5 hard, file bucket)
  - data/eval_capability/agentic_chain.jsonl  (P6 hard, file bucket)
  - data/eval_capability/docid_bind.jsonl     (P6.2 hard, rank-1 id bind)
  - data/eval_capability/toolresult_ground.jsonl (P6.3 hard, copy facts from get_doc)
  - data/eval_capability/search_miss.jsonl    (P6.3 hard, empty catalog → abstain)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.system_prompts import AGENTIC_STANDARD_PROMPT, CHAT_SYSTEM_PROMPT


def _chat(history_html: str) -> str:
    return f"<myPT_system>{CHAT_SYSTEM_PROMPT}</myPT_system>{history_html}<myPT_assistant>"


def _agent(history_html: str) -> str:
    return f"<myPT_system>{AGENTIC_STANDARD_PROMPT}</myPT_system>{history_html}<myPT_assistant>"


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} -> {path}")


def main() -> None:
    p4_ood = [
        {
            "name": "ood_pet_followup",
            "category": "followup",
            "prompt": _chat(
                "<myPT_user>Please remember: my bird is called Quorin.</myPT_user>"
                "<myPT_assistant>I will remember Quorin.</myPT_assistant>"
                "<myPT_user>Say the bird's name again.</myPT_user>"
            ),
            "expected_contains": ["Quorin"],
        },
        {
            "name": "ood_id_after_aside",
            "category": "followup",
            "prompt": _chat(
                "<myPT_user>Store this id: PLM-220.</myPT_user>"
                "<myPT_assistant>Stored PLM-220.</myPT_assistant>"
                "<myPT_user>Ok. Name a primary color.</myPT_user>"
                "<myPT_assistant>Red</myPT_assistant>"
                "<myPT_user>Repeat the id I stored.</myPT_user>"
            ),
            "expected_contains": ["PLM-220"],
        },
        {
            "name": "ood_switch_fruit",
            "category": "topic_switch",
            "prompt": _chat(
                "<myPT_user>We were talking about Quorin the bird.</myPT_user>"
                "<myPT_assistant>Yes, Quorin.</myPT_assistant>"
                "<myPT_user>Different topic — what color is grass?</myPT_user>"
            ),
            "expected_contains": ["green"],
        },
        {
            "name": "ood_de_followup",
            "category": "language_hold",
            "prompt": _chat(
                "<myPT_user>Meine Stadt heißt Vellmar.</myPT_user>"
                "<myPT_assistant>Du wohnst in Vellmar.</myPT_assistant>"
                "<myPT_user>In welcher Stadt wohne ich?</myPT_user>"
            ),
            "expected_contains": ["Vellmar"],
        },
    ]
    write_jsonl(PROJECT_ROOT / "data" / "eval_ood" / "phase4_multiturn_ood.jsonl", p4_ood)

    toolcall = [
        {
            "name": "tc_search_onboarding",
            "category": "search",
            "prompt": _agent(
                "<myPT_user>Search the workspace for the onboarding checklist.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.search"],
        },
        {
            "name": "tc_list_docs",
            "category": "list",
            "prompt": _agent(
                "<myPT_user>List all workspace documents.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.list_docs"],
        },
        {
            "name": "tc_get_doc",
            "category": "get",
            "prompt": _agent(
                "<myPT_user>Get the full text of document doc_alpha.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.get_doc"],
        },
        {
            "name": "tc_summarize",
            "category": "summarize",
            "prompt": _agent(
                "<myPT_user>Summarize document doc_alpha.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.summarize"],
        },
        {
            "name": "tc_summarize_relay",
            "category": "summarize_relay",
            "prompt": _agent(
                "<myPT_user>Summarize INSTALL.md.</myPT_user>"
                "<myPT_assistant><myPT_toolcall>{\"name\": \"workspace.summarize\", \"doc_id\": \"install_md\"}</myPT_toolcall></myPT_assistant>"
                "<myPT_toolresult>{\"name\": \"workspace.summarize\", \"summary\": "
                "\"Install covers prerequisites, setup, and verify. Install Python 3.11 and rsync first.\"}"
                "</myPT_toolresult>"
                "<myPT_user>Give me that summary.</myPT_user>"
            ),
            "expected_contains": ["prerequisites", "setup"],
        },
    ]
    write_jsonl(PROJECT_ROOT / "data" / "eval_capability" / "toolcall_basic.jsonl", toolcall)

    chain = [
        {
            "name": "chain_search_then_get",
            "category": "search_get",
            "prompt": _agent(
                "<myPT_user>Find the vacation policy and open the matching document.</myPT_user>"
                "<myPT_assistant><myPT_toolcall>{\"name\": \"workspace.search\", \"query\": \"vacation policy\"}</myPT_toolcall></myPT_assistant>"
                "<myPT_toolresult>{\"name\": \"workspace.search\", \"results\": [{\"doc_id\": \"hr_vac_01\", \"title\": \"Vacation Policy\"}]}</myPT_toolresult>"
                "<myPT_user>Continue with the next tool call.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.get_doc", "hr_vac_01"],
        },
        {
            "name": "chain_list_then_get",
            "category": "list_get",
            "prompt": _agent(
                "<myPT_user>See what docs exist, then read net_spec.</myPT_user>"
                "<myPT_assistant><myPT_toolcall>{\"name\": \"workspace.list_docs\"}</myPT_toolcall></myPT_assistant>"
                "<myPT_toolresult>{\"name\": \"workspace.list_docs\", \"docs\": [\"net_spec\", \"doc_alpha\"]}</myPT_toolresult>"
                "<myPT_user>Now get net_spec.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.get_doc", "net_spec"],
        },
        {
            "name": "chain_get_then_summarize",
            "category": "get_summarize",
            "prompt": _agent(
                "<myPT_user>Read doc_alpha and then summarize it.</myPT_user>"
                "<myPT_assistant><myPT_toolcall>{\"name\": \"workspace.get_doc\", \"doc_id\": \"doc_alpha\"}</myPT_toolcall></myPT_assistant>"
                "<myPT_toolresult>{\"name\": \"workspace.get_doc\", \"doc_id\": \"doc_alpha\", \"text\": \"Python is a programming language.\"}</myPT_toolresult>"
                "<myPT_user>Summarize that document.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.summarize"],
        },
        {
            "name": "chain_search_then_summarize",
            "category": "search_summarize",
            "prompt": _agent(
                "<myPT_user>Search for TCP handshake then summarize the hit.</myPT_user>"
                "<myPT_assistant><myPT_toolcall>{\"name\": \"workspace.search\", \"query\": \"TCP handshake\"}</myPT_toolcall></myPT_assistant>"
                "<myPT_toolresult>{\"name\": \"workspace.search\", \"results\": [{\"doc_id\": \"net_spec\", \"title\": \"TCP handshake\"}]}</myPT_toolresult>"
                "<myPT_user>Summarize net_spec.</myPT_user>"
            ),
            "expected_contains": ["<myPT_toolcall>", "workspace.summarize"],
        },
    ]
    write_jsonl(PROJECT_ROOT / "data" / "eval_capability" / "agentic_chain.jsonl", chain)

    def _catalog(hits, pretty=False):
        payload = {"documents": hits, "total": len(hits)}
        if pretty:
            body = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            body = json.dumps(payload, ensure_ascii=False)
        return body

    def _hit(rank, title, filename, snippet, doc_id):
        return {
            "rank": rank,
            "title": title,
            "filename": filename,
            "snippet": snippet,
            "doc_id": doc_id,
        }

    gold_a = "a1b2c3d4e5f6"
    dist_b = "f6e5d4c3b2a1"
    dist_c = "0123456789ab"
    gold_d = "9f8e7d6c5b4a"
    dist_e = "4a5b6c7d8e9f"
    gold_f = "111122223333"
    dist_g = "aaaabbbbcccc"
    dist_h = "deadbeef0001"

    hits_2 = [
        _hit(1, "CoverLetter", "cover_letter.pdf",
             "I would be glad to discuss the role. Kind regards, Ada Onboarding", gold_a),
        _hit(2, "VocabSize", "VOCAB_SIZE_EXPLAINED.md",
             "GPT-2 BPE tokenization uses vocab_size 50304 in the small config example.", dist_b),
    ]
    hits_3 = [
        _hit(1, "VacationPolicy", "vacation_policy.md",
             "Employees receive 25 days of paid vacation per calendar year.", gold_d),
        _hit(2, "CheckpointFormat", "CHECKPOINT_FORMAT.md",
             "from core.checkpoint import CheckpointManager ckpt_manager = CheckpointManager", dist_e),
        _hit(3, "InstallGuide", "INSTALL.md",
             "Install Python 3.11 and rsync before running the webapp locally.", dist_c),
    ]
    hits_tcp = [
        _hit(1, "NetSpec", "net_spec.md",
             "TCP handshake is SYN, SYN-ACK, ACK between client and server.", gold_f),
        _hit(2, "DocAlpha", "doc_alpha.md",
             "Python is a programming language used for scripting and ML.", dist_g),
    ]
    hits_hr = [
        _hit(1, "OnboardingChecklist", "onboarding_checklist.md",
             "Day one: laptop, accounts, and the onboarding checklist in HR.", "0ff1cec0ffee"),
        _hit(2, "HighContext", "HIGH_CONTEXT_CONFIGS.md",
             "rope_scale 4.0 extends context from 1024 to 4096 tokens.", dist_h),
        _hit(3, "CliRefactor", "CLI_REFACTORING.md",
             "The CLI was split into train, eval, and webapp entry points.", dist_c),
    ]

    def _bind_prompt(user_q, search_q, think, catalog):
        return _agent(
            f"<myPT_user>{user_q}</myPT_user>"
            f"<myPT_assistant><myPT_think>{think}</myPT_think>"
            f"<myPT_toolcall>{{\"name\": \"workspace.search\", \"query\": \"{search_q}\"}}"
            f"</myPT_toolcall></myPT_assistant>"
            f"<myPT_toolresult>{catalog}</myPT_toolresult>"
        )

    docid_bind = [
        {
            "name": "bind_2hit_compact_get",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "who wrote the cover letter?",
                "cover letter author",
                "The user wants to know about the author. Let me search the workspace.",
                _catalog(hits_2, pretty=False),
            ),
            "expected_all": ["<myPT_toolcall>", "workspace.get_doc", gold_a],
            "forbidden_contains": [dist_b],
        },
        {
            "name": "bind_2hit_pretty_get",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Find the cover letter and open it.",
                "cover letter",
                "The user wants to know about the author. Let me search the workspace.",
                _catalog(hits_2, pretty=True),
            ),
            "expected_all": ["<myPT_toolcall>", gold_a],
            "forbidden_contains": [dist_b],
        },
        {
            "name": "bind_3hit_compact_get",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "What is the vacation policy? Read the matching document.",
                "vacation policy",
                "The user wants to know about vacation. Let me search the workspace.",
                _catalog(hits_3, pretty=False),
            ),
            "expected_all": ["<myPT_toolcall>", gold_d],
            "forbidden_contains": [dist_e, dist_c],
        },
        {
            "name": "bind_3hit_pretty_get",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Open the vacation policy from search.",
                "vacation policy",
                "I'll search the workspace.",
                _catalog(hits_3, pretty=True),
            ),
            "expected_all": ["<myPT_toolcall>", gold_d],
            "forbidden_contains": [dist_e],
        },
        {
            "name": "bind_2hit_summarize",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Search for TCP handshake then summarize the hit.",
                "TCP handshake",
                "The user wants TCP handshake. Let me search the workspace.",
                _catalog(hits_tcp, pretty=False),
            ),
            "expected_all": ["<myPT_toolcall>", gold_f],
            "forbidden_contains": [dist_g],
        },
        {
            "name": "bind_generic_think_2hit",
            "category": "docid_bind",
            "prompt": _agent(
                "<myPT_user>who is the cover-letter author?</myPT_user>"
                "<myPT_assistant><myPT_think>The user wants to know about the author. "
                "Let me search the workspace.</myPT_think>"
                "<myPT_toolcall>{\"name\": \"workspace.search\", \"query\": \"cover letter author\"}"
                "</myPT_toolcall></myPT_assistant>"
                f"<myPT_toolresult>{_catalog(hits_2, pretty=True)}</myPT_toolresult>"
            ),
            "expected_all": [gold_a],
            "forbidden_contains": [dist_b],
        },
        {
            "name": "bind_onboarding_3hit",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Search the workspace for the onboarding checklist and read it.",
                "onboarding checklist",
                "The user wants the onboarding checklist. Let me search the workspace.",
                _catalog(hits_hr, pretty=False),
            ),
            "expected_all": ["<myPT_toolcall>", "0ff1cec0ffee"],
            "forbidden_contains": [dist_h, dist_c],
        },
        {
            "name": "bind_pretty_3hit_onboarding",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Find the onboarding checklist.",
                "onboarding checklist",
                "I'll search the workspace.",
                _catalog(hits_hr, pretty=True),
            ),
            "expected_all": ["0ff1cec0ffee"],
            "forbidden_contains": [dist_h],
        },
        {
            "name": "bind_tcp_pretty",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Look up TCP handshake and retrieve the spec.",
                "TCP handshake",
                "The user wants to know about TCP. Let me search the workspace.",
                _catalog(hits_tcp, pretty=True),
            ),
            "expected_all": ["<myPT_toolcall>", gold_f],
            "forbidden_contains": [dist_g],
        },
        {
            "name": "bind_cover_compact_summarize_ok",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Summarize the cover letter after searching.",
                "cover letter",
                "The user wants the cover letter. Let me search the workspace.",
                _catalog(hits_2, pretty=False),
            ),
            "expected_all": [gold_a],
            "forbidden_contains": [dist_b],
        },
        {
            "name": "bind_vacation_no_rank2",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "Read the vacation policy document.",
                "vacation",
                "I'll search the workspace.",
                _catalog(hits_3, pretty=False),
            ),
            "expected_all": [gold_d],
            "forbidden_contains": [dist_e, dist_c],
        },
        {
            "name": "bind_cover_pretty_no_vocab",
            "category": "docid_bind",
            "prompt": _bind_prompt(
                "who is Ada Onboarding?",
                "Ada Onboarding",
                "The user wants to know about Ada. Let me search the workspace.",
                _catalog(hits_2, pretty=True),
            ),
            "expected_all": [gold_a],
            "forbidden_contains": [dist_b],
        },
    ]
    write_jsonl(PROJECT_ROOT / "data" / "eval_capability" / "docid_bind.jsonl", docid_bind)

    def _getdoc_json(doc_id, title, text, pretty=False):
        payload = {"doc_id": doc_id, "title": title, "text": text, "length": len(text)}
        if pretty:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return json.dumps(payload, ensure_ascii=False)

    def _ground_prompt(user_q, search_q, think_search, catalog, think_get, doc_id, getdoc_json):
        return _agent(
            f"<myPT_user>{user_q}</myPT_user>"
            f"<myPT_assistant><myPT_think>{think_search}</myPT_think>"
            f"<myPT_toolcall>{{\"name\": \"workspace.search\", \"query\": \"{search_q}\"}}"
            f"</myPT_toolcall></myPT_assistant>"
            f"<myPT_toolresult>{catalog}</myPT_toolresult>"
            f"<myPT_assistant><myPT_think>{think_get}</myPT_think>"
            f"<myPT_toolcall>{{\"name\": \"workspace.get_doc\", \"doc_id\": \"{doc_id}\"}}"
            f"</myPT_toolcall></myPT_assistant>"
            f"<myPT_toolresult>{getdoc_json}</myPT_toolresult>"
        )

    forbid_template = [
        "grounded answer",
        "Based on the document content",
        "I'll generate a concise answer",
    ]
    qthorn_text = (
        "Quindlethorpe is a 19th-century brass foundry in Wexmere. "
        "Its hallmark is the coiled-hare stamp on every finished plate."
    )
    vellmar_text = (
        "Ada Onboarding lives at Kieferngasse 12 in Vellmar. "
        "She leads the night-shift millwright crew at Quindlethorpe."
    )
    synack_text = (
        "TCP handshake is SYN, SYN-ACK, ACK between client and server. "
        "The Wexmere lab traces this on port 4177 of the coiled-hare gateway."
    )
    vacation_text = (
        "Employees receive 25 days of paid vacation per calendar year. "
        "Vellmar site adds three bridging days around the coiled-hare festival."
    )
    onboarding_text = (
        "Day one: laptop, accounts, and the onboarding checklist in HR. "
        "Quindlethorpe assigns a millwright buddy before noon."
    )
    german_text = (
        "Die Kantonspolizei Wexmere sichert forensische Daten in Quindlethorpe. "
        "Ansprechpartnerin ist Ada Onboarding, Kieferngasse 12."
    )

    toolresult_ground = [
        {
            "name": "ground_foundry_compact",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "Who or what is Quindlethorpe?",
                "Quindlethorpe",
                "The user wants to know about Quindlethorpe. Let me search the workspace.",
                _catalog(hits_2, pretty=False),
                "I'll retrieve the document content.",
                gold_a,
                _getdoc_json(gold_a, "CoverLetter", qthorn_text, pretty=False),
            ),
            "expected_all": ["Quindlethorpe", "Wexmere", "coiled-hare"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_foundry_pretty",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "What do our documents say about Quindlethorpe?",
                "Quindlethorpe",
                "I should search for documents related to Quindlethorpe first.",
                _catalog(hits_2, pretty=True),
                "I'll retrieve the document content.",
                gold_a,
                _getdoc_json(gold_a, "CoverLetter", qthorn_text, pretty=True),
            ),
            "expected_all": ["brass foundry", "Wexmere"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_ada_address",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "Who is Ada Onboarding?",
                "Ada Onboarding",
                "The user wants to know about Ada. Let me search the workspace.",
                _catalog(hits_2, pretty=True),
                "I'll retrieve the document content.",
                gold_a,
                _getdoc_json(gold_a, "CoverLetter", vellmar_text, pretty=True),
            ),
            "expected_all": ["Kieferngasse", "Vellmar", "millwright"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_tcp_wexmere",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "What do the docs say about the TCP handshake?",
                "TCP handshake",
                "The user wants TCP handshake. Let me search the workspace.",
                _catalog(hits_tcp, pretty=False),
                "I'll retrieve the document content.",
                gold_f,
                _getdoc_json(gold_f, "NetSpec", synack_text, pretty=False),
            ),
            "expected_all": ["SYN-ACK", "4177", "coiled-hare"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_vacation_festival",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "Read the vacation policy and tell me the facts.",
                "vacation policy",
                "The user wants to know about vacation. Let me search the workspace.",
                _catalog(hits_3, pretty=True),
                "I'll retrieve the document content.",
                gold_d,
                _getdoc_json(gold_d, "VacationPolicy", vacation_text, pretty=True),
            ),
            "expected_all": ["25 days", "Vellmar", "coiled-hare festival"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_onboarding_buddy",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "What does the onboarding checklist say?",
                "onboarding checklist",
                "I'll search the workspace.",
                _catalog(hits_hr, pretty=False),
                "I'll retrieve the document content.",
                "0ff1cec0ffee",
                _getdoc_json("0ff1cec0ffee", "OnboardingChecklist", onboarding_text, pretty=False),
            ),
            "expected_all": ["millwright buddy", "Quindlethorpe"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_de_kanton",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "Was sagen unsere Dokumente zur Kantonspolizei Wexmere?",
                "Kantonspolizei Wexmere",
                "Der Benutzer moechte etwas ueber Kantonspolizei wissen. Lass mich den Workspace durchsuchen.",
                _catalog(hits_2, pretty=True),
                "Ich hole den Dokumentinhalt.",
                gold_a,
                _getdoc_json(gold_a, "CoverLetter", german_text, pretty=True),
            ),
            "expected_all": ["Kantonspolizei", "Quindlethorpe", "Kieferngasse"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_generic_think_get",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "Quote what the document says about the foundry.",
                "foundry",
                "The user wants to know about the foundry. Let me search the workspace.",
                _catalog(hits_2, pretty=False),
                "I'll retrieve the document content.",
                gold_a,
                _getdoc_json(gold_a, "CoverLetter", qthorn_text, pretty=False),
            ),
            "expected_all": ["Quindlethorpe", "Wexmere"],
            "forbidden_contains": ["grounded answer", "Based on the document content"],
        },
        {
            "name": "ground_pretty_ada_crew",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "After you retrieve Ada Onboarding, quote what the document says.",
                "Ada Onboarding",
                "I should search for documents related to Ada first.",
                _catalog(hits_2, pretty=True),
                "I'll retrieve the document content.",
                gold_a,
                _getdoc_json(gold_a, "CoverLetter", vellmar_text, pretty=False),
            ),
            "expected_all": ["night-shift", "Quindlethorpe"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_tcp_pretty",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "Read the workspace file about TCP and tell me the facts.",
                "TCP handshake",
                "The user wants to know about TCP. Let me search the workspace.",
                _catalog(hits_tcp, pretty=True),
                "I'll retrieve the document content.",
                gold_f,
                _getdoc_json(gold_f, "NetSpec", synack_text, pretty=True),
            ),
            "expected_all": ["port 4177", "Wexmere"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_vacation_compact",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "What do our documents say about vacation?",
                "vacation",
                "I'll search the workspace.",
                _catalog(hits_3, pretty=False),
                "I'll retrieve the document content.",
                gold_d,
                _getdoc_json(gold_d, "VacationPolicy", vacation_text, pretty=False),
            ),
            "expected_all": ["25 days", "bridging days"],
            "forbidden_contains": forbid_template,
        },
        {
            "name": "ground_onboarding_pretty",
            "category": "toolresult_ground",
            "prompt": _ground_prompt(
                "Who or what is the onboarding checklist about?",
                "onboarding checklist",
                "The user wants the onboarding checklist. Let me search the workspace.",
                _catalog(hits_hr, pretty=True),
                "I'll retrieve the document content.",
                "0ff1cec0ffee",
                _getdoc_json("0ff1cec0ffee", "OnboardingChecklist", onboarding_text, pretty=True),
            ),
            "expected_all": ["laptop", "millwright buddy"],
            "forbidden_contains": forbid_template,
        },
    ]
    write_jsonl(PROJECT_ROOT / "data" / "eval_capability" / "toolresult_ground.jsonl", toolresult_ground)

    empty_cat = json.dumps({"documents": [], "total": 0}, ensure_ascii=False)
    empty_pretty = json.dumps({"documents": [], "total": 0}, ensure_ascii=False, indent=2)

    def _miss_prompt(user_q, search_q, think, catalog):
        return _agent(
            f"<myPT_user>{user_q}</myPT_user>"
            f"<myPT_assistant><myPT_think>{think}</myPT_think>"
            f"<myPT_toolcall>{{\"name\": \"workspace.search\", \"query\": \"{search_q}\"}}"
            f"</myPT_toolcall></myPT_assistant>"
            f"<myPT_toolresult>{catalog}</myPT_toolresult>"
        )

    miss_forbid = ["workspace.get_doc", "INSTALL.md", "DATA_SOURCES_CONFIG"]
    # Skill: empty catalog → abstain, no get_doc. Exact "don't have enough
    # information" is not required — "no documents found" / "doesn't contain"
    # in think or body is the same behavior (1 Sep 2026: piedpiper_pretty,
    # zorblax were false fails).
    miss_abstain = [
        "don't have enough information",
        "not enough information",
        "no documents found",
        "no matching documents",
        "doesn't contain",
        "does not contain",
        "nothing in the catalog",
        "unknown",
        "keine passenden",
        "nicht genug",
    ]
    search_miss = [
        {
            "name": "miss_piedpiper_compact",
            "category": "search_miss",
            "prompt": _miss_prompt(
                'search for "piedpiper"',
                "piedpiper",
                "The user wants to know about piedpiper. Let me search the workspace.",
                empty_cat,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": miss_forbid,
        },
        {
            "name": "miss_piedpiper_pretty",
            "category": "search_miss",
            "prompt": _miss_prompt(
                "Search the workspace for Piedpiper.",
                "Piedpiper",
                "I should search for documents related to Piedpiper first.",
                empty_pretty,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": miss_forbid,
        },
        {
            "name": "miss_emanuela",
            "category": "search_miss",
            "prompt": _miss_prompt(
                "who is emanuela Kyritsis?",
                "emanuela Kyritsis",
                "The user wants to know about emanuela Kyritsis. Let me search the workspace.",
                empty_cat,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": miss_forbid,
        },
        {
            "name": "miss_zorblax",
            "category": "search_miss",
            "prompt": _miss_prompt(
                "Who or what is Zorblax Nine?",
                "Zorblax Nine",
                "I'll search the workspace.",
                empty_pretty,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": ["workspace.get_doc"],
        },
        {
            "name": "miss_no_getdoc",
            "category": "search_miss",
            "prompt": _miss_prompt(
                "Read the workspace file about Piedpiper and tell me the facts.",
                "Piedpiper",
                "The user wants to know about Piedpiper. Let me search the workspace.",
                empty_cat,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": ["workspace.get_doc"],
        },
        {
            "name": "miss_kyritsis_pretty",
            "category": "search_miss",
            "prompt": _miss_prompt(
                "What do our documents say about Emanuela Kyritsis?",
                "Emanuela Kyritsis",
                "I should search for documents related to Emanuela Kyritsis first.",
                empty_pretty,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": miss_forbid,
        },
        {
            "name": "miss_empty_total",
            "category": "search_miss",
            "prompt": _miss_prompt(
                "Look up coiled-hare foundry and retrieve the spec.",
                "coiled-hare foundry",
                "I'll search the workspace.",
                empty_cat,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": ["workspace.get_doc"],
        },
        {
            "name": "miss_no_install",
            "category": "search_miss",
            "prompt": _miss_prompt(
                "search for piedpiper",
                "piedpiper",
                "The user wants to know about piedpiper. Let me search the workspace.",
                empty_pretty,
            ),
            "expected_contains": miss_abstain,
            "forbidden_contains": ["Installation Guide", "INSTALL.md"],
        },
    ]
    write_jsonl(PROJECT_ROOT / "data" / "eval_capability" / "search_miss.jsonl", search_miss)


if __name__ == "__main__":
    main()
