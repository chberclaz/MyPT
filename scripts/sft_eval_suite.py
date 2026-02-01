#!/usr/bin/env python3
"""
SFT Evaluation Suite - Mandatory gates after each training phase.

Runs deterministic generation (temperature=0) on a fixed prompt set and
evaluates pass/fail on 4 buckets:

A) format_strict:
   - Output must be within proper tags and stop correctly
   - No malformed tags

B) echo_basic:
   - "Say Hello" → "Hello"
   - "Repeat: Hello world" → "Hello world"

C) anti_echo:
   - "What is 'Blurpix'?" → must NOT output "Blurpix"
   - Model should answer "Unknown." or similar

D) regression_basic:
   - Simple math: 5+7 → 12
   - Simple fact: capital of Germany → Berlin
   - Must not collapse to constant token for all inputs

Usage:
    python scripts/sft_eval_suite.py --model phase3a1_alpha
    python scripts/sft_eval_suite.py --model phase3a1_beta --output results.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import load_model
from core.system_prompts import CONVERSATION_SYSTEM_PROMPT

# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

SYSTEM_PROMPT = CONVERSATION_SYSTEM_PROMPT

def make_prompt(user_content: str) -> str:
    """Create a full prompt with system and user tags."""
    return f"<myPT_system>{SYSTEM_PROMPT}</myPT_system><myPT_user>{user_content}</myPT_user><myPT_assistant>"


# Bucket A: Format Strict
FORMAT_PROMPTS = [
    ("format_say_ok", make_prompt("Say OK."), None),  # Any valid response
    ("format_say_hello", make_prompt("Say Hello."), None),
    ("format_status", make_prompt("Status?"), None),
    ("format_ready", make_prompt("Ready?"), None),
    ("format_german", make_prompt("Sag Hallo."), None),
]

# Bucket B: Echo Basic
ECHO_PROMPTS = [
    ("echo_hello", make_prompt("Say Hello."), "Hello"),
    ("echo_ok", make_prompt("Say OK."), "OK"),
    ("echo_yes", make_prompt("Say yes."), "yes"),
    ("echo_repeat", make_prompt("Repeat: Hello world"), "Hello world"),
    ("echo_colon", make_prompt("Echo: Testing"), "Testing"),
    ("echo_german", make_prompt("Sag Hallo."), "Hallo"),
    ("echo_banana", make_prompt("Say Banana."), "Banana"),
    ("echo_number", make_prompt("Say 42."), "42"),
]

# Bucket C: Anti-Echo (model should NOT copy the quoted text)
ANTI_ECHO_PROMPTS = [
    ("anti_blurpix", make_prompt('What is "Blurpix"?'), "Blurpix"),  # Should NOT contain
    ("anti_zanthor", make_prompt('Is "Zanthor" a real word?'), "Zanthor"),
    ("anti_gibberish", make_prompt('What does "Quexling" mean?'), "Quexling"),
    ("anti_meaning", make_prompt('Does "Flimzap" have a meaning?'), "Flimzap"),
]

# Bucket D: Regression Basic (simple knowledge, must not all be same answer)
REGRESSION_PROMPTS = [
    ("reg_math_5_7", make_prompt("What is 5 + 7?"), "12"),
    ("reg_math_2_2", make_prompt("What is 2 + 2?"), "4"),
    ("reg_math_10_3", make_prompt("What is 10 - 3?"), "7"),
    ("reg_capital_de", make_prompt("Capital of Germany?"), "Berlin"),
    ("reg_capital_fr", make_prompt("Capital of France?"), "Paris"),
    ("reg_yes_no_water", make_prompt("Is water wet?"), "Yes"),
    ("reg_yes_no_fire", make_prompt("Is fire cold?"), "No"),
]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def extract_response(generated: str) -> str:
    """Extract the response content from generated text."""
    # Remove leading/trailing whitespace
    text = generated.strip()
    
    # Remove closing tag if present
    if "</myPT_assistant>" in text:
        text = text.split("</myPT_assistant>")[0]
    if "<myPT_eot>" in text:
        text = text.split("<myPT_eot>")[0]
    
    return text.strip()


def check_format_strict(generated: str) -> Tuple[bool, str]:
    """Check if output has valid format (no malformed tags, proper structure)."""
    text = generated.strip()
    
    # Check for malformed tags
    malformed_patterns = [
        r"<myPT_[a-z]*[^>]*<",  # Nested or unclosed tags
        r"<myPT_(?!system|user|assistant|eot|assistant_context)[a-z]+>",  # Invalid tag names
        r"</myPT_(?!system|user|assistant|eot|assistant_context)[a-z]+>",
    ]
    
    for pattern in malformed_patterns:
        if re.search(pattern, text):
            return False, f"Malformed tag pattern: {pattern}"
    
    # Check that response doesn't contain system or user tags (it shouldn't generate those)
    if "<myPT_system>" in text or "<myPT_user>" in text:
        return False, "Generated system/user tags (should only generate assistant content)"
    
    # Check it eventually stops (has closing tag or is reasonably short)
    if len(text) > 500 and "</myPT_assistant>" not in text and "<myPT_eot>" not in text:
        return False, "Response too long without closing tag"
    
    return True, "OK"


def check_echo(generated: str, expected: str) -> Tuple[bool, str]:
    """Check if response matches expected echo target."""
    response = extract_response(generated)
    
    # Remove trailing punctuation for comparison
    response_clean = response.rstrip(".!?,;:")
    expected_clean = expected.rstrip(".!?,;:")
    
    # Case-insensitive comparison for now (can be stricter later)
    if response_clean.lower() == expected_clean.lower():
        return True, f"Matched: '{response}'"
    
    # Also accept if response starts with expected (allows some tolerance)
    if response_clean.lower().startswith(expected_clean.lower()):
        return True, f"Starts with expected: '{response}'"
    
    return False, f"Expected '{expected}', got '{response}'"


def check_anti_echo(generated: str, forbidden: str) -> Tuple[bool, str]:
    """Check that response does NOT contain the forbidden string (anti-echo)."""
    response = extract_response(generated)
    
    if forbidden.lower() in response.lower():
        return False, f"Response contains forbidden '{forbidden}': '{response}'"
    
    # Check for expected anti-echo responses
    anti_responses = ["unknown", "no", "nein", "unbekannt", "i don't know", "not a word"]
    if any(ar in response.lower() for ar in anti_responses):
        return True, f"Correct anti-echo response: '{response}'"
    
    # If it doesn't contain forbidden AND is short, likely OK
    if len(response) < 50:
        return True, f"Short response without forbidden text: '{response}'"
    
    return True, f"Did not contain forbidden text: '{response}'"


def check_regression(generated: str, expected: str) -> Tuple[bool, str]:
    """Check if response contains the expected answer."""
    response = extract_response(generated)
    
    # Check if expected is in response
    if expected.lower() in response.lower():
        return True, f"Contains expected '{expected}': '{response}'"
    
    # For numbers, also check if the number appears
    if expected.isdigit() and expected in response:
        return True, f"Contains number '{expected}': '{response}'"
    
    return False, f"Expected '{expected}', got '{response}'"


def check_no_collapse(results: List[str]) -> Tuple[bool, str]:
    """Check that regression results aren't all the same (mode collapse detection)."""
    unique_responses = set(extract_response(r).lower()[:20] for r in results)
    
    if len(unique_responses) <= 1 and len(results) > 2:
        return False, f"Mode collapse detected: all responses are '{list(unique_responses)[0] if unique_responses else 'empty'}'"
    
    return True, f"Diverse responses: {len(unique_responses)} unique"


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(
    model_name: str,
    max_new_tokens: int = 64,
    verbose: bool = False,
) -> Dict:
    """Run full evaluation suite on a model.
    
    Returns dict with per-bucket results and overall pass/fail.
    """
    print(f"Loading model: {model_name}")
    model = load_model(model_name)
    tokenizer = model.tokenizer
    config = model.config
    model.eval()
    
    import torch
    device = next(model.parameters()).device
    
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "buckets": {},
        "summary": {},
    }
    
    def generate(prompt: str) -> str:
        """Generate response with deterministic settings."""
        with torch.no_grad():
            # model.generate expects a string prompt, returns string output
            output = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,  # Greedy
                top_k=0,  # Disabled (greedy)
                top_p=1.0,
                repetition_penalty=1.0,
            )
        
        # Output includes prompt, strip it to get just the generated part
        if output.startswith(prompt):
            return output[len(prompt):]
        return output
    
    # Bucket A: Format Strict
    print("\n📋 Bucket A: Format Strict")
    bucket_a = {"passed": 0, "failed": 0, "details": []}
    
    for name, prompt, _ in FORMAT_PROMPTS:
        gen = generate(prompt)
        passed, reason = check_format_strict(gen)
        bucket_a["details"].append({"name": name, "passed": passed, "reason": reason, "generated": gen[:100]})
        if passed:
            bucket_a["passed"] += 1
            if verbose:
                print(f"  ✅ {name}: {reason}")
        else:
            bucket_a["failed"] += 1
            print(f"  ❌ {name}: {reason}")
    
    results["buckets"]["format_strict"] = bucket_a
    
    # Bucket B: Echo Basic
    print("\n📋 Bucket B: Echo Basic")
    bucket_b = {"passed": 0, "failed": 0, "details": []}
    
    for name, prompt, expected in ECHO_PROMPTS:
        gen = generate(prompt)
        passed, reason = check_echo(gen, expected)
        bucket_b["details"].append({"name": name, "passed": passed, "reason": reason, "expected": expected, "generated": gen[:100]})
        if passed:
            bucket_b["passed"] += 1
            if verbose:
                print(f"  ✅ {name}: {reason}")
        else:
            bucket_b["failed"] += 1
            print(f"  ❌ {name}: {reason}")
    
    results["buckets"]["echo_basic"] = bucket_b
    
    # Bucket C: Anti-Echo
    print("\n📋 Bucket C: Anti-Echo")
    bucket_c = {"passed": 0, "failed": 0, "details": []}
    
    for name, prompt, forbidden in ANTI_ECHO_PROMPTS:
        gen = generate(prompt)
        passed, reason = check_anti_echo(gen, forbidden)
        bucket_c["details"].append({"name": name, "passed": passed, "reason": reason, "forbidden": forbidden, "generated": gen[:100]})
        if passed:
            bucket_c["passed"] += 1
            if verbose:
                print(f"  ✅ {name}: {reason}")
        else:
            bucket_c["failed"] += 1
            print(f"  ❌ {name}: {reason}")
    
    results["buckets"]["anti_echo"] = bucket_c
    
    # Bucket D: Regression Basic
    print("\n📋 Bucket D: Regression Basic")
    bucket_d = {"passed": 0, "failed": 0, "details": []}
    regression_outputs = []
    
    for name, prompt, expected in REGRESSION_PROMPTS:
        gen = generate(prompt)
        regression_outputs.append(gen)
        passed, reason = check_regression(gen, expected)
        bucket_d["details"].append({"name": name, "passed": passed, "reason": reason, "expected": expected, "generated": gen[:100]})
        if passed:
            bucket_d["passed"] += 1
            if verbose:
                print(f"  ✅ {name}: {reason}")
        else:
            bucket_d["failed"] += 1
            print(f"  ❌ {name}: {reason}")
    
    # Check for mode collapse
    collapse_passed, collapse_reason = check_no_collapse(regression_outputs)
    bucket_d["collapse_check"] = {"passed": collapse_passed, "reason": collapse_reason}
    if not collapse_passed:
        print(f"  ⚠️  Mode collapse: {collapse_reason}")
    
    results["buckets"]["regression_basic"] = bucket_d
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    total_passed = 0
    total_failed = 0
    bucket_results = []
    
    for bucket_name, bucket_data in results["buckets"].items():
        passed = bucket_data["passed"]
        failed = bucket_data["failed"]
        total = passed + failed
        pct = (passed / total * 100) if total > 0 else 0
        
        status = "✅ PASS" if failed == 0 else "❌ FAIL"
        print(f"  {bucket_name}: {passed}/{total} ({pct:.0f}%) {status}")
        
        total_passed += passed
        total_failed += failed
        bucket_results.append({
            "bucket": bucket_name,
            "passed": passed,
            "failed": failed,
            "total": total,
            "pass_rate": pct,
            "status": "PASS" if failed == 0 else "FAIL",
        })
    
    overall_pct = (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0
    overall_pass = total_failed == 0
    
    print(f"\n  OVERALL: {total_passed}/{total_passed + total_failed} ({overall_pct:.0f}%)")
    print(f"  STATUS: {'✅ ALL PASS' if overall_pass else '❌ SOME FAILURES'}")
    
    results["summary"] = {
        "total_passed": total_passed,
        "total_failed": total_failed,
        "overall_pass_rate": overall_pct,
        "overall_status": "PASS" if overall_pass else "FAIL",
        "buckets": bucket_results,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SFT Evaluation Suite")
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all results, not just failures")
    args = parser.parse_args()
    
    print("="*60)
    print("  SFT EVALUATION SUITE")
    print("="*60)
    
    results = run_evaluation(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )
    
    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")
    
    # Exit with appropriate code
    sys.exit(0 if results["summary"]["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
