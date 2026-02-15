#!/usr/bin/env python3
"""
Generate synthetic SFT conversations using Claude/GPT-4 API.

Creates domain-specific Q&A pairs for IT security, Swiss law, etc.
Output is in the JSONL format ready for prepare_chat_sft.py.

Cost estimate: ~5K conversations ≈ $30-50 with Claude Haiku/GPT-3.5

Usage:
    python scripts/generate_synthetic_sft.py --output data/synthetic_sft.jsonl --count 5000
"""

import argparse
import json
import os
import time
import random
from pathlib import Path

# Topic templates for domain-specific Q&A
TOPICS = {
    "it_security": [
        "Explain {concept} in simple terms.",
        "What are the risks of {threat}?",
        "How do I protect against {attack}?",
        "Best practices for {security_topic}?",
        "What is the difference between {a} and {b}?",
        "Step-by-step guide to {task}.",
        "Common mistakes when {action}?",
        "How does {technology} work?",
    ],
    "swiss_law": [
        "What does {law_article} say about {topic}?",
        "Explain {legal_concept} in simple terms.",
        "What are my rights regarding {situation}?",
        "How does Swiss law handle {case_type}?",
        "Difference between {a} and {b} in Swiss law?",
        "What are the penalties for {offense}?",
        "How to {legal_action} in Switzerland?",
    ],
    "general_assistant": [
        "Summarize {topic}.",
        "Give me the key points about {subject}.",
        "Explain {concept} like I'm a beginner.",
        "What should I know about {topic}?",
        "Quick overview of {subject}?",
    ]
}

IT_SECURITY_CONCEPTS = [
    "SQL injection", "XSS attacks", "CSRF", "buffer overflow", "ransomware",
    "phishing", "social engineering", "zero-day exploits", "DDoS attacks",
    "man-in-the-middle attacks", "password hashing", "encryption", "TLS/SSL",
    "firewalls", "intrusion detection", "penetration testing", "vulnerability scanning",
    "secure coding", "input validation", "authentication", "authorization",
    "OAuth", "JWT tokens", "API security", "container security", "cloud security",
    "network segmentation", "VPNs", "endpoint protection", "SIEM", "incident response"
]

SWISS_LAW_TOPICS = [
    "data protection (DSG)", "employment contracts", "rental law", "contract law",
    "inheritance law", "marriage and divorce", "criminal procedure", "civil procedure",
    "company formation", "tax obligations", "social security", "health insurance",
    "liability law", "intellectual property", "privacy rights", "consumer protection"
]


def load_api_key():
    """Load API key from .env or environment."""
    # Try .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.strip().split("=", 1)[1]
                if line.startswith("OPENAI_API_KEY="):
                    return ("openai", line.strip().split("=", 1)[1])
    
    # Try environment
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", os.environ["ANTHROPIC_API_KEY"])
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", os.environ["OPENAI_API_KEY"])
    
    return None


def generate_with_anthropic(prompt: str, api_key: str) -> str:
    """Generate response using Claude API."""
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    message = client.messages.create(
        model="claude-3-haiku-20240307",  # Cheapest, fast
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return message.content[0].text


def generate_with_openai(prompt: str, api_key: str) -> str:
    """Generate response using OpenAI API."""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Cheapest
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def create_generation_prompt(topic_type: str, user_question: str, language: str) -> str:
    """Create prompt for the API to generate a helpful response."""
    
    lang_instruction = "Respond in German." if language == "de" else "Respond in English."
    
    return f"""You are MyPT, a helpful offline assistant specializing in IT security and Swiss law.
    
A user asks: "{user_question}"

Provide a helpful, accurate, and concise response (2-4 sentences for simple questions, up to a paragraph for complex ones).
{lang_instruction}
Be practical and direct. Don't use phrases like "As an AI" or "I cannot".
Just answer the question helpfully."""


def generate_question(topic_type: str, language: str) -> str:
    """Generate a random question for the given topic type."""
    templates = TOPICS[topic_type]
    template = random.choice(templates)
    
    if topic_type == "it_security":
        concept = random.choice(IT_SECURITY_CONCEPTS)
        question = template.format(
            concept=concept, threat=concept, attack=concept,
            security_topic=concept, technology=concept, task=concept,
            action=concept, a=random.choice(IT_SECURITY_CONCEPTS),
            b=random.choice(IT_SECURITY_CONCEPTS)
        )
    elif topic_type == "swiss_law":
        topic = random.choice(SWISS_LAW_TOPICS)
        question = template.format(
            law_article="Swiss law", topic=topic, legal_concept=topic,
            situation=topic, case_type=topic, offense=topic,
            legal_action=topic, a=random.choice(SWISS_LAW_TOPICS),
            b=random.choice(SWISS_LAW_TOPICS)
        )
    else:
        concept = random.choice(IT_SECURITY_CONCEPTS + SWISS_LAW_TOPICS)
        question = template.format(topic=concept, subject=concept, concept=concept)
    
    # German translation for some questions
    if language == "de":
        # Simple keyword replacements for German flavor
        question = question.replace("Explain", "Erkläre")
        question = question.replace("What are", "Was sind")
        question = question.replace("How do I", "Wie kann ich")
        question = question.replace("What is the difference", "Was ist der Unterschied")
    
    return question


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SFT data")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--count", type=int, default=1000, help="Number of conversations")
    parser.add_argument("--batch_size", type=int, default=50, help="Save every N conversations")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("SYNTHETIC SFT DATA GENERATOR")
    print("=" * 60)
    
    # Load API key
    api_info = load_api_key()
    if api_info is None:
        print("\n❌ No API key found!")
        print("   Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env or environment")
        return
    
    provider, api_key = api_info
    print(f"\nUsing {provider.upper()} API")
    print(f"Target: {args.count} conversations")
    print(f"Output: {args.output}")
    
    generate_fn = generate_with_anthropic if provider == "anthropic" else generate_with_openai
    
    # Prepare output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    conversations = []
    topic_weights = [("it_security", 0.5), ("swiss_law", 0.3), ("general_assistant", 0.2)]
    lang_weights = [("en", 0.5), ("de", 0.5)]
    
    print("\nGenerating...")
    
    for i in range(args.count):
        try:
            # Random topic and language
            topic_type = random.choices(
                [t[0] for t in topic_weights],
                weights=[t[1] for t in topic_weights]
            )[0]
            language = random.choices(
                [l[0] for l in lang_weights],
                weights=[l[1] for l in lang_weights]
            )[0]
            
            # Generate question
            user_question = generate_question(topic_type, language)
            
            # Generate response via API
            prompt = create_generation_prompt(topic_type, user_question, language)
            assistant_response = generate_fn(prompt, api_key)
            
            # Create conversation entry
            conv = {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_response}
                ],
                "context": {
                    "episode_id": f"synthetic_{i:05d}",
                    "language": language,
                    "topic": topic_type
                }
            }
            conversations.append(conv)
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{args.count}] Generated ({topic_type}, {language})")
            
            # Save periodically
            if (i + 1) % args.batch_size == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for conv in conversations:
                        f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                print(f"  💾 Saved {len(conversations)} conversations")
            
            # Rate limiting
            time.sleep(args.delay)
            
        except Exception as e:
            print(f"  ⚠️ Error at {i}: {e}")
            time.sleep(2)  # Back off on error
    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Done! Generated {len(conversations)} conversations")
    print(f"   Output: {output_path}")
    print(f"\nNext steps:")
    print(f"   1. python scripts/prepare_chat_sft.py --input {output_path} --output data/sft_synthetic")
    print(f"   2. Train with the new dataset")


if __name__ == "__main__":
    main()
