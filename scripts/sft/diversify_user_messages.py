#!/usr/bin/env python3
"""
Diversify user messages in SFT goldset to reduce templated patterns.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDSET_PATH = PROJECT_ROOT / "data" / "sft_conversation_goldset" / "mypt_phase3a_gold_en_v2.jsonl"

# Diverse rewrites for user messages - reducing template overlap
REWRITES = {
    # Idempotency (0001-0003) - different phrasings for same topic
    '0001': 'Walk me through idempotency with concrete code examples.',
    '0002': 'Give me a quick bullet-point rundown on idempotency - I know the basics.',
    '0003': 'Idempotency in 200 words or less - what matters for distributed systems?',
    
    # Event sourcing (0004-0006)
    '0004': 'What exactly is event sourcing? Show me with real examples.',
    '0005': 'Event sourcing - hit me with 10 key points I should know.',
    '0006': 'Break down event sourcing briefly. When would I actually use it?',
    
    # Zero trust (0007-0009)
    '0007': 'How does zero trust networking actually work in practice?',
    '0008': 'Zero trust architecture - give me the TL;DR in bullet form.',
    '0009': 'Summarize zero trust for someone implementing microservices.',
    
    # Rate limiting (0010-0012)
    '0010': 'Rate limiting implementation patterns - what are my options?',
    '0011': 'Quick overview of rate limiting strategies and when to use each.',
    '0012': 'I need to add rate limiting to my API. What should I know?',
    
    # Backpressure (0013-0015)
    '0013': 'What is backpressure and how do I handle it in async systems?',
    '0014': 'Backpressure patterns - give me the essential points.',
    '0015': 'My message queue is backing up. Explain backpressure concepts.',
    
    # Structured logging (0016-0018)
    '0016': 'Why should I switch to structured logging? Show me examples.',
    '0017': 'Structured logging best practices - bullet points please.',
    '0018': 'Convince me to use structured logging over print statements.',
    
    # Diff-in-diff (0019-0021)
    '0019': 'How does difference-in-differences work for A/B analysis?',
    '0020': 'Diff-in-diff methodology - key points for a data engineer.',
    '0021': 'When should I use diff-in-diff instead of a standard A/B test?',
    
    # Calibration (0022-0024)
    '0022': 'My ML model probabilities seem off. Explain calibration.',
    '0023': 'Model calibration techniques - what do I need to know?',
    '0024': 'Why does ML calibration matter and how do I fix poor calibration?',
    
    # Vector similarity (0025-0027)
    '0025': 'Building a semantic search system - explain vector similarity.',
    '0026': 'Vector search fundamentals - give me the essentials.',
    '0027': 'FAISS, Annoy, HNSW - when do I use which for vector search?',
    
    # Prompt injection (0028-0030)
    '0028': 'My LLM app is vulnerable to prompt injection. What defenses work?',
    '0029': 'Prompt injection defense strategies - actionable checklist.',
    '0030': 'How serious is prompt injection and what can I actually do about it?',
    
    # SQL queries (0070-0073) - make each unique
    '0070': 'Need a query for top 10 most active users by logins in the last month. Tables: users(id,email), events(id,user_id,type,ts)',
    '0071': 'SQL to aggregate event counts by type over the past week? Schema: events(id,user_id,type,ts)',
    '0072': 'How do I find orphaned users with zero events? users(id,email), events(id,user_id,type,ts)',
    '0073': 'Query for daily login trends from my events table? events(id,user_id,type,ts)',
    
    # Error debugging (0031, 0034, 0037, 0040) - different phrasing
    '0031': '''Getting IndexError on line 10. Why would items[0] fail?

Traceback (most recent call last):
  File "app.py", line 10, in <module>
    x = items[0]
IndexError: list index out of range''',
    
    '0034': "TypeError: object of type 'NoneType' has no len() - what causes this?",
    
    '0037': '''Permission denied writing to /var/log/mypt/audit.log - what are my options?

PermissionError: [Errno 13] Permission denied: '/var/log/mypt/audit.log' ''',
    
    '0040': '''SQLite keeps saying "database is locked". Running multiple processes.

sqlite3.OperationalError: database is locked''',
    
    # JSON schema (0074-0076)
    '0074': 'Draft a JSON schema for support tickets with proper validation.',
    '0075': 'Need a schema for audit log entries. What fields and validations?',
    '0076': 'Design a JSON schema for ML training run metadata.',
    
    # Code review (0085-0086)
    '0085': '''Quick code review? Is this add function robust enough?

```python
def add(a,b):
    return a+b
```''',
    
    '0086': '''Check this file loader for resource leaks:

```python
def load(path):
    f=open(path)
    return f.read()
```''',
    
    # Security checklists (0090-0092)
    '0090': 'Hardening my FastAPI app that only runs locally. What should I check?',
    '0091': 'Best practices for storing API keys on disk? Security-focused.',
    '0092': 'My RAG pipeline might be vulnerable to prompt injection. Prevention tips?',
    
    # Python questions - diversify
    'python_001': 'Best way to load a JSON file in Python?',
    'python_002': 'I need to dedupe a list but keep the original order. Python solution?',
    'python_003': 'Quick way to hit a REST API with Python? GET request.',
    'python_004': "Exception handling in Python - what's the right pattern?",
    'python_005': 'Setting up CLI arguments in a Python script - argparse basics?',
    'python_006': 'Date/time manipulation in Python keeps confusing me. Basics?',
    'python_007': 'Writing data to CSV in Python - show me the clean way.',
    
    # Java questions - diversify
    'java_001': 'Reading a text file line by line in Java - modern approach?',
    'java_002': 'Parsing JSON in Java - Jackson or something else?',
    'java_003': "Making HTTP calls from Java 11+ - what's the current best practice?",
    'java_004': 'Null handling in Java - Optional vs defensive checks?',
    'java_005': 'Java Streams for filtering and mapping a collection?',
    'java_006': 'Minimal Spring Boot REST endpoint - show me the setup.',
    'java_007': 'Writing a JUnit 5 test - basic structure?',
    
    # Bash questions - diversify
    'bash_001': 'Finding files by name recursively in Linux?',
    'bash_002': 'Searching inside files for a pattern - grep or something better?',
    'bash_003': 'Checking disk space on Linux - quick commands?',
    'bash_004': 'Running a long command in background without it dying?',
    'bash_005': 'Bash script that takes arguments - template?',
    'bash_006': 'awk for text processing - basic patterns?',
    'bash_007': 'System monitoring commands on Linux - CPU, memory, processes?',
    'bash_008': 'Setting up a scheduled job with cron?',
    
    # Product requirements - diversify
    '0043': 'Spec out API key authentication for me. What are the acceptance criteria?',
    '0045': 'Product requirement: admin vs user roles. What should we implement?',
}


def main():
    # Read the JSONL file
    with open(GOLDSET_PATH, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    episodes = [json.loads(l) for l in lines]
    
    # Apply rewrites
    updated = 0
    for ep in episodes:
        ep_id = ep.get('context', '').replace('episode_id: ', '')
        if ep_id in REWRITES:
            # Update the first user message
            if ep['messages'] and ep['messages'][0]['role'] == 'user':
                ep['messages'][0]['content'] = REWRITES[ep_id]
                updated += 1
    
    # Save
    with open(GOLDSET_PATH, 'w', encoding='utf-8') as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + '\n')
    
    print(f'Updated {updated} user messages with diverse phrasing')
    print(f'  File: {GOLDSET_PATH}')


if __name__ == '__main__':
    main()

