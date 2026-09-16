#!/usr/bin/env python3
"""
DeepL Translation Script for SFT Goldset
Translates English user/assistant messages to German using DeepL API.

Usage:
    # Default (uses data/temp/ directory):
    python scripts/translate_deepl.py
    
    # Custom directory:
    python scripts/translate_deepl.py --dir data/sft_run2_minimal_qa
    
    # Resume from last position:
    python scripts/translate_deepl.py --resume

Input file format (from extract_for_translation.py):
    === episode_id ===
    Message content here...
    
    === multi_turn_episode ===
    First message...
    <<<MSG_SEP>>>
    Second message...

Requirements:
    - Create a .env file in project root with: DEEPL_API_KEY=your_key_here
    - pip install requests python-dotenv

Get your free API key at: https://www.deepl.com/pro-api
Free tier: 500,000 characters/month
"""

import os
import sys
import time
import re
import argparse
import requests
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("Warning: python-dotenv not installed. Reading API key from environment only.")
    print("Install with: pip install python-dotenv")


# Configuration
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"  # Free API endpoint
DEEPL_API_URL_PRO = "https://api.deepl.com/v2/translate"   # Pro API endpoint

# Default directory (can be overridden via --dir)
DEFAULT_DIR = PROJECT_ROOT / "data" / "temp"

# Files to translate (input_en, output_de)
FILES_TO_TRANSLATE = [
    ("user_messages_en.txt", "user_messages_de.txt"),
    ("assistant_messages_en.txt", "assistant_messages_de.txt"),
]

# Rate limiting - increased for stability
DELAY_BETWEEN_REQUESTS = 1.5  # seconds (increased from 0.5)
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds before retry


def get_api_key() -> str:
    """Get DeepL API key from environment."""
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        print("\n" + "="*60)
        print("ERROR: DEEPL_API_KEY not found!")
        print("="*60)
        print("\nPlease create a .env file in the project root with:")
        print("  DEEPL_API_KEY=your_api_key_here")
        print("\nGet your free API key at: https://www.deepl.com/pro-api")
        print("Free tier includes 500,000 characters/month")
        print("="*60 + "\n")
        sys.exit(1)
    return api_key


def detect_api_type(api_key: str) -> str:
    """Detect if API key is free or pro based on suffix."""
    # Free API keys end with ":fx"
    if api_key.strip().endswith(":fx"):
        return DEEPL_API_URL
    return DEEPL_API_URL_PRO


def clean_text_for_api(text: str) -> str:
    """Clean text to avoid API errors."""
    # Remove null bytes and other problematic characters
    text = text.replace('\x00', '')
    # Normalize whitespace but preserve newlines
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Remove any control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


def translate_text(text: str, api_key: str, api_url: str, target_lang: str = "DE") -> tuple[Optional[str], str]:
    """
    Translate text using DeepL API with retry logic.
    Returns (translated_text, error_message). Error is empty string on success.
    """
    if not text.strip():
        return text, ""
    
    # Clean text before sending
    clean_text = clean_text_for_api(text)
    if not clean_text:
        return "", ""
    
    headers = {
        "Authorization": f"DeepL-Auth-Key {api_key}",
        "Content-Type": "application/json",
    }
    
    # Simplified data - removed XML tag handling which can cause issues
    data = {
        "text": [clean_text],
        "target_lang": target_lang,
        "source_lang": "EN",
        "preserve_formatting": True,
    }
    
    last_error = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["translations"][0]["text"], ""
            
        except requests.exceptions.HTTPError as e:
            last_error = str(e)
            if response.status_code == 456:
                return None, "QUOTA_EXCEEDED"
            elif response.status_code == 403:
                return None, "UNAUTHORIZED"
            elif response.status_code == 400:
                # Bad request - try to get more details
                try:
                    error_detail = response.json().get("message", "Unknown")
                except:
                    error_detail = response.text[:200]
                last_error = f"400 Bad Request: {error_detail}"
                # Don't retry 400 errors - they're usually content issues
                break
            elif response.status_code == 429:
                # Rate limited - wait longer and retry
                wait_time = RETRY_DELAY * (attempt + 1) * 2
                print(f"\n   ⏳ Rate limited, waiting {wait_time}s...", end="", flush=True)
                time.sleep(wait_time)
                continue
            elif response.status_code >= 500:
                # Server error - retry with backoff
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"\n   ⏳ Server error, retry in {wait_time}s...", end="", flush=True)
                time.sleep(wait_time)
                continue
            else:
                break
                
        except requests.exceptions.Timeout:
            last_error = "Request timeout"
            wait_time = RETRY_DELAY * (attempt + 1)
            print(f"\n   ⏳ Timeout, retry in {wait_time}s...", end="", flush=True)
            time.sleep(wait_time)
            continue
            
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            wait_time = RETRY_DELAY * (attempt + 1)
            print(f"\n   ⏳ Network error, retry in {wait_time}s...", end="", flush=True)
            time.sleep(wait_time)
            continue
    
    return None, last_error


def parse_message_file(filepath: Path) -> list[tuple[str, str]]:
    """Parse message file into list of (episode_id, content) tuples."""
    content = filepath.read_text(encoding="utf-8")
    
    episodes = []
    current_id = None
    current_lines = []
    
    for line in content.split("\n"):
        if line.startswith("=== ") and line.endswith(" ==="):
            # Save previous episode
            if current_id is not None:
                episodes.append((current_id, "\n".join(current_lines).strip()))
            # Start new episode
            current_id = line[4:-4]  # Extract ID between === and ===
            current_lines = []
        else:
            current_lines.append(line)
    
    # Don't forget the last episode
    if current_id is not None:
        episodes.append((current_id, "\n".join(current_lines).strip()))
    
    return episodes


def load_existing_translations(output_file: Path) -> dict[str, str]:
    """Load already translated episodes for resume capability."""
    if not output_file.exists():
        return {}
    
    existing = {}
    episodes = parse_message_file(output_file)
    for ep_id, content in episodes:
        if content.strip():  # Only count non-empty translations
            existing[ep_id] = content
    return existing


def translate_file(input_file: Path, output_file: Path, api_key: str, api_url: str, resume: bool = False) -> bool:
    """Translate an entire message file."""
    print(f"\n📄 Translating: {input_file.name}")
    
    episodes = parse_message_file(input_file)
    print(f"   Found {len(episodes)} episodes")
    
    # Load existing translations for resume
    existing = {}
    if resume:
        existing = load_existing_translations(output_file)
        if existing:
            print(f"   📂 Found {len(existing)} existing translations (resuming)")
    
    translated_episodes = []
    total_chars = 0
    skipped = 0
    
    for i, (ep_id, content) in enumerate(episodes):
        # Skip empty content
        if not content.strip():
            translated_episodes.append((ep_id, ""))
            continue
        
        # Check if already translated (resume mode)
        if ep_id in existing:
            translated_episodes.append((ep_id, existing[ep_id]))
            skipped += 1
            continue
        
        # Show progress
        print(f"   [{i+1}/{len(episodes)}] {ep_id}...", end=" ", flush=True)
        
        # Translate with retry
        translated, error = translate_text(content, api_key, api_url)
        
        if translated is None:
            print(f"❌ FAILED")
            print(f"\n   Error: {error}")
            if error not in ("QUOTA_EXCEEDED", "UNAUTHORIZED"):
                # Show content preview for debugging
                preview = content[:100].replace('\n', ' ')
                print(f"   Content preview: {preview}...")
            
            # Save progress so far
            save_translated_file(output_file, translated_episodes)
            print(f"\n⚠️  Partial progress saved to {output_file}")
            print(f"   Translated {len(translated_episodes) - skipped}/{len(episodes)} episodes")
            if skipped:
                print(f"   (Plus {skipped} already translated)")
            print(f"\n💡 Tip: Run with --resume to continue from here")
            return False
        
        translated_episodes.append((ep_id, translated))
        total_chars += len(content)
        print(f"✓ ({len(content)} chars)")
        
        # Rate limiting - save periodically
        if (i + 1) % 50 == 0:
            save_translated_file(output_file, translated_episodes)
            print(f"   💾 Progress saved ({i+1}/{len(episodes)})")
        
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    # Save complete file
    save_translated_file(output_file, translated_episodes)
    print(f"   ✅ Complete! Translated {total_chars:,} characters")
    if skipped:
        print(f"   (Skipped {skipped} already translated)")
    return True


def save_translated_file(output_file: Path, episodes: list[tuple[str, str]]):
    """Save translated episodes to file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for ep_id, content in episodes:
            f.write(f"=== {ep_id} ===\n")
            f.write(content + "\n\n")


def check_usage(api_key: str, api_url: str):
    """Check remaining API usage."""
    usage_url = api_url.replace("/translate", "/usage")
    headers = {"Authorization": f"DeepL-Auth-Key {api_key}"}
    
    try:
        response = requests.get(usage_url, headers=headers, timeout=10)
        response.raise_for_status()
        usage = response.json()
        
        used = usage.get("character_count", 0)
        limit = usage.get("character_limit", 500000)
        remaining = limit - used
        pct_used = (used / limit) * 100 if limit > 0 else 0
        
        print(f"\n📊 DeepL API Usage:")
        print(f"   Used: {used:,} / {limit:,} characters ({pct_used:.1f}%)")
        print(f"   Remaining: {remaining:,} characters")
        
        return remaining
    except Exception as e:
        print(f"   Could not check usage: {e}")
        return None


def estimate_chars(input_dir: Path) -> int:
    """Estimate total characters to translate."""
    total = 0
    for en_file, _ in FILES_TO_TRANSLATE:
        filepath = input_dir / en_file
        if filepath.exists():
            total += len(filepath.read_text(encoding="utf-8"))
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Translate SFT goldset to German using DeepL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Default (uses data/temp/):
    python scripts/translate_deepl.py
    
    # Custom directory:
    python scripts/translate_deepl.py --dir data/sft_run2_minimal_qa
    
    # Resume interrupted translation:
    python scripts/translate_deepl.py --resume
"""
    )
    parser.add_argument("--resume", action="store_true", help="Resume from existing translations")
    parser.add_argument("--dir", type=str, default=None,
                        help="Directory containing user_messages_en.txt and assistant_messages_en.txt "
                             "(default: data/temp)")
    args = parser.parse_args()
    
    # Determine input/output directory
    work_dir = Path(args.dir) if args.dir else DEFAULT_DIR
    if not work_dir.is_absolute():
        work_dir = PROJECT_ROOT / work_dir
    
    print("="*60)
    print("🌐 DeepL Translation Script for SFT Goldset")
    print("   English → German (DE)")
    print(f"   Directory: {work_dir}")
    if args.resume:
        print("   Mode: RESUME (skipping already translated)")
    print("="*60)
    
    # Get API key
    api_key = get_api_key()
    api_url = detect_api_type(api_key)
    
    api_type = "FREE" if "api-free" in api_url else "PRO"
    print(f"\n✓ API Key loaded ({api_type} tier)")
    
    # Check usage
    remaining = check_usage(api_key, api_url)
    
    # Estimate required characters
    estimated = estimate_chars(work_dir)
    print(f"\n📝 Estimated characters to translate: {estimated:,}")
    
    if remaining is not None and estimated > remaining:
        print(f"\n⚠️  WARNING: You may not have enough quota!")
        print(f"   Required: ~{estimated:,} chars, Remaining: {remaining:,} chars")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("   Aborted.")
            return
    
    # Translate files
    print("\n" + "-"*60)
    success = True
    
    for en_file, de_file in FILES_TO_TRANSLATE:
        input_path = work_dir / en_file
        output_path = work_dir / de_file
        
        if not input_path.exists():
            print(f"\n⚠️  Skipping {en_file} (not found at {input_path})")
            continue
        
        if not translate_file(input_path, output_path, api_key, api_url, resume=args.resume):
            success = False
            break
    
    # Final status
    print("\n" + "="*60)
    if success:
        print("✅ Translation complete!")
        print(f"\nOutput files:")
        for _, de_file in FILES_TO_TRANSLATE:
            print(f"   - {work_dir / de_file}")
        print(f"\nNext step: Run recombine script to create German episodes:")
        print(f"   python scripts/recombine_translations.py \\")
        print(f"       --original <your_original.jsonl> \\")
        print(f"       --user_translated {work_dir / 'user_messages_de.txt'} \\")
        print(f"       --assistant_translated {work_dir / 'assistant_messages_de.txt'} \\")
        print(f"       --output <output_de.jsonl>")
    else:
        print("⚠️  Translation incomplete - run with --resume to continue")
    print("="*60)


if __name__ == "__main__":
    main()
