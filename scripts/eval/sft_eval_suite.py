#!/usr/bin/env python3
"""
SFT Evaluation Suite - Mandatory gates after each training phase.

Runs deterministic generation (temperature=0) on a fixed prompt set and
evaluates pass/fail on 5 buckets:

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

E) operators (EXACT MATCH) - TRUE ABSTRACTION TEST:
   - COPY: "Give back: X" → "X"
   - WRAP: "Frame with brackets: X" → "[X]"
   - EXTRACT: 'Strip the quotes from "X"' → "X"
   - Uses NOVEL templates NEVER seen in training OR validation
   - This tests TRUE abstraction, not template memorization
   - Tests both single-word and multi-word (2-4 words) payloads
   - Reports breakdown by operator AND by word count

Usage:
    # Full suite with system prompt
    python scripts/sft_eval_suite.py --model phase3a1_alpha
    
    # Operators only, no system prompt (for packed operator models)
    python scripts/sft_eval_suite.py --model phase3a_operator_packed --operators_only --no_system_prompt
    
    # Save detailed results
    python scripts/sft_eval_suite.py --model phase3a1_beta --output results.json -v

Options:
    --no_system_prompt  Omit system prompt (for operator models trained without it)
    --operators_only    Only run Bucket E (operators), skip A-D
    --verbose, -v       Show all results, not just failures
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

# Global flag for no-system-prompt mode (set from CLI)
_NO_SYSTEM_PROMPT = False

def make_prompt(user_content: str, no_system: bool = None) -> str:
    """Create a full prompt with optional system and user tags.
    
    Args:
        user_content: The user message content
        no_system: If True, skip system prompt (for operator training).
                   If None, uses global _NO_SYSTEM_PROMPT setting.
    """
    use_no_system = no_system if no_system is not None else _NO_SYSTEM_PROMPT
    if use_no_system:
        return f"<myPT_user>{user_content}</myPT_user><myPT_assistant>"
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

# Bucket E: Operators (EXACT MATCH - tests abstract operator learning)
# Uses NOVEL templates (NEVER seen in training or validation) to test TRUE abstraction
# Includes both single-word and multi-word payloads
#
# IMPORTANT: These templates are deliberately DIFFERENT from both:
#   - TRAIN templates (e.g., "Repeat exactly:", "Copy:", "Echo:")
#   - VAL templates (e.g., "Parrot this:", "Mirror", "Reproduce")
# This tests whether the model learned the ABSTRACT operator, not template memorization.

def build_operator_prompts(use_val_templates: bool = False):
    """Build operator prompts dynamically (respects _NO_SYSTEM_PROMPT).
    
    Args:
        use_val_templates: If True, use VAL templates (closer to training, easier).
                          If False, use NOVEL templates (harder, tests true abstraction).
    """
    prompts = []
    
    if use_val_templates:
        # =====================================================================
        # VAL TEMPLATES - VARIED DELIMITERS (backticks, quotes, pipes)
        # Tests if model learned abstract "delimited content" concept
        # =====================================================================
        
        # COPY - VAL templates (varied delimiters)
        prompts.append(("op_copy_1w_1", make_prompt("Parrot this: `xylophone`"), "COPY", "xylophone"))
        prompts.append(("op_copy_1w_2", make_prompt("Return verbatim: 'quantum'"), "COPY", "quantum"))
        prompts.append(("op_copy_1w_3", make_prompt("Parrot |nebula|"), "COPY", "nebula"))
        prompts.append(("op_copy_1w_4", make_prompt("Mirror `twilight`"), "COPY", "twilight"))
        prompts.append(("op_copy_1w_5", make_prompt("Reproduce 'cascade'"), "COPY", "cascade"))
        prompts.append(("op_copy_2w_1", make_prompt("Parrot this: `blue ocean`"), "COPY", "blue ocean"))
        prompts.append(("op_copy_2w_2", make_prompt("Mirror |dark forest|"), "COPY", "dark forest"))
        prompts.append(("op_copy_3w_1", make_prompt("Return verbatim: 'red mountain peak'"), "COPY", "red mountain peak"))
        prompts.append(("op_copy_3w_2", make_prompt("Reproduce `golden sunset sky`"), "COPY", "golden sunset sky"))
        prompts.append(("op_copy_4w_1", make_prompt("Output exactly |the river and valley|"), "COPY", "the river and valley"))
        
        # WRAP - VAL templates (varied delimiters)
        prompts.append(("op_wrap_1w_1", make_prompt("Put in square brackets: `phoenix`"), "WRAP", "[phoenix]"))
        prompts.append(("op_wrap_1w_2", make_prompt("Wrap with []: 'marble'"), "WRAP", "[marble]"))
        prompts.append(("op_wrap_1w_3", make_prompt("Add [] around |crystal|"), "WRAP", "[crystal]"))
        prompts.append(("op_wrap_1w_4", make_prompt("Put brackets around `thunder`"), "WRAP", "[thunder]"))
        prompts.append(("op_wrap_1w_5", make_prompt("Put in square brackets: 'horizon'"), "WRAP", "[horizon]"))
        prompts.append(("op_wrap_2w_1", make_prompt("Add [] around `silver moonlight`"), "WRAP", "[silver moonlight]"))
        prompts.append(("op_wrap_2w_2", make_prompt("Put brackets around |wild river|"), "WRAP", "[wild river]"))
        prompts.append(("op_wrap_3w_1", make_prompt("Put in square brackets: 'ancient stone bridge'"), "WRAP", "[ancient stone bridge]"))
        prompts.append(("op_wrap_3w_2", make_prompt("Wrap with []: `cold winter night`"), "WRAP", "[cold winter night]"))
        prompts.append(("op_wrap_4w_1", make_prompt("Add [] around |the forest and lake|"), "WRAP", "[the forest and lake]"))
        
        # EXTRACT - VAL templates (quotes already provide boundaries)
        prompts.append(("op_extract_1w_1", make_prompt('What\'s inside "emerald"?'), "EXTRACT", "emerald"))
        prompts.append(("op_extract_1w_2", make_prompt('Output the quoted content: "silver"'), "EXTRACT", "silver"))
        prompts.append(("op_extract_1w_3", make_prompt('Get the text from "dragon"'), "EXTRACT", "dragon"))
        prompts.append(("op_extract_1w_4", make_prompt('Return what\'s in "galaxy"'), "EXTRACT", "galaxy"))
        prompts.append(("op_extract_1w_5", make_prompt('What\'s inside "zenith"?'), "EXTRACT", "zenith"))
        prompts.append(("op_extract_2w_1", make_prompt('Output the quoted content: "deep space"'), "EXTRACT", "deep space"))
        prompts.append(("op_extract_2w_2", make_prompt('Get the text from "bright star"'), "EXTRACT", "bright star"))
        prompts.append(("op_extract_3w_1", make_prompt('What\'s inside "northern winter sky"?'), "EXTRACT", "northern winter sky"))
        prompts.append(("op_extract_3w_2", make_prompt('Return what\'s in "eternal summer dawn"'), "EXTRACT", "eternal summer dawn"))
        prompts.append(("op_extract_4w_1", make_prompt('Output the quoted content: "the mountain and river"'), "EXTRACT", "the mountain and river"))
    
    else:
        # =====================================================================
        # NOVEL TEMPLATES - completely new phrasings never seen in train OR val
        # VARIED DELIMITERS - tests if model learned abstract "delimited content"
        # =====================================================================
        
        # COPY - single word (novel templates, varied delimiters)
        prompts.append(("op_copy_1w_1", make_prompt("Give back: `xylophone`"), "COPY", "xylophone"))
        prompts.append(("op_copy_1w_2", make_prompt("Send this unchanged: 'quantum'"), "COPY", "quantum"))
        prompts.append(("op_copy_1w_3", make_prompt("What I say is |nebula|"), "COPY", "nebula"))
        prompts.append(("op_copy_1w_4", make_prompt("Relay: `twilight`"), "COPY", "twilight"))
        prompts.append(("op_copy_1w_5", make_prompt("Pass through 'cascade'"), "COPY", "cascade"))
        
        # COPY - multi-word (novel templates, varied delimiters)
        prompts.append(("op_copy_2w_1", make_prompt("Transmit exactly: `blue ocean`"), "COPY", "blue ocean"))
        prompts.append(("op_copy_2w_2", make_prompt("Feed back |dark forest|"), "COPY", "dark forest"))
        prompts.append(("op_copy_3w_1", make_prompt("Bounce back: 'red mountain peak'"), "COPY", "red mountain peak"))
        prompts.append(("op_copy_3w_2", make_prompt("Return unchanged `golden sunset sky`"), "COPY", "golden sunset sky"))
        prompts.append(("op_copy_4w_1", make_prompt("Spit out |the river and valley|"), "COPY", "the river and valley"))
        
        # WRAP - single word (novel templates, varied delimiters)
        prompts.append(("op_wrap_1w_1", make_prompt("Frame with brackets: `phoenix`"), "WRAP", "[phoenix]"))
        prompts.append(("op_wrap_1w_2", make_prompt("Box this in []: 'marble'"), "WRAP", "[marble]"))
        prompts.append(("op_wrap_1w_3", make_prompt("Sandwich between [ and ]: |crystal|"), "WRAP", "[crystal]"))
        prompts.append(("op_wrap_1w_4", make_prompt("Place brackets on `thunder`"), "WRAP", "[thunder]"))
        prompts.append(("op_wrap_1w_5", make_prompt("Give me 'horizon' with [] around it"), "WRAP", "[horizon]"))
        
        # WRAP - multi-word (novel templates, varied delimiters)
        prompts.append(("op_wrap_2w_1", make_prompt("Package in brackets `silver moonlight`"), "WRAP", "[silver moonlight]"))
        prompts.append(("op_wrap_2w_2", make_prompt("Return |wild river| inside []"), "WRAP", "[wild river]"))
        prompts.append(("op_wrap_3w_1", make_prompt("Bookend with []: 'ancient stone bridge'"), "WRAP", "[ancient stone bridge]"))
        prompts.append(("op_wrap_3w_2", make_prompt("`cold winter night` but in brackets"), "WRAP", "[cold winter night]"))
        prompts.append(("op_wrap_4w_1", make_prompt("I need |the forest and lake| wrapped in []"), "WRAP", "[the forest and lake]"))
        
        # EXTRACT - single word (novel templates - quotes already provide boundaries)
        prompts.append(("op_extract_1w_1", make_prompt('Strip the quotes from "emerald"'), "EXTRACT", "emerald"))
        prompts.append(("op_extract_1w_2", make_prompt('Unquote: "silver"'), "EXTRACT", "silver"))
        prompts.append(("op_extract_1w_3", make_prompt('Remove quotation marks from "dragon"'), "EXTRACT", "dragon"))
        prompts.append(("op_extract_1w_4", make_prompt('Give me just the word in "galaxy"'), "EXTRACT", "galaxy"))
        prompts.append(("op_extract_1w_5", make_prompt('Dequote "zenith"'), "EXTRACT", "zenith"))
        
        # EXTRACT - multi-word (novel templates)
        prompts.append(("op_extract_2w_1", make_prompt('Take off the quotes: "deep space"'), "EXTRACT", "deep space"))
        prompts.append(("op_extract_2w_2", make_prompt('Peel away the quotes from "bright star"'), "EXTRACT", "bright star"))
        prompts.append(("op_extract_3w_1", make_prompt('The quoted text in "northern winter sky" is?'), "EXTRACT", "northern winter sky"))
        prompts.append(("op_extract_3w_2", make_prompt('Without quotes: "eternal summer dawn"'), "EXTRACT", "eternal summer dawn"))
        prompts.append(("op_extract_4w_1", make_prompt('Bare text from "the mountain and river"'), "EXTRACT", "the mountain and river"))
    
    return prompts

# Will be populated at runtime
OPERATOR_PROMPTS = []


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


def check_operator_exact(generated: str, expected: str, operator: str) -> Tuple[bool, str]:
    """Check EXACT MATCH for operator transformations.
    
    This is stricter than echo - the output must match exactly (case-sensitive).
    """
    response = extract_response(generated)
    
    # Remove trailing period if model added it
    response_clean = response.rstrip(".")
    expected_clean = expected.rstrip(".")
    
    # EXACT match (case-sensitive for operators)
    if response_clean == expected_clean:
        return True, f"EXACT MATCH: '{response}'"
    
    # Also accept with trailing period
    if response == expected or response == expected + ".":
        return True, f"EXACT MATCH (with period): '{response}'"
    
    return False, f"Expected '{expected}', got '{response}'"


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def run_evaluation(
    model_name: str,
    max_new_tokens: int = 64,
    verbose: bool = False,
    operators_only: bool = False,
) -> Dict:
    """Run full evaluation suite on a model.
    
    Args:
        model_name: Name of model to evaluate
        max_new_tokens: Max tokens to generate per prompt
        verbose: Show all results (not just failures)
        operators_only: If True, only run Bucket E (operators), skip A-D.
                        Use for models trained only on operator tasks.
    
    Returns dict with per-bucket results and overall pass/fail.
    """
    print(f"Loading model: {model_name}")
    print(f"  System prompt: {'OMITTED' if _NO_SYSTEM_PROMPT else 'included'}")
    print(f"  Mode: {'operators only' if operators_only else 'full suite'}")
    
    model = load_model(model_name)
    tokenizer = model.tokenizer
    config = model.config
    model.eval()
    
    import torch
    device = next(model.parameters()).device
    
    results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "no_system_prompt": _NO_SYSTEM_PROMPT,
        "operators_only": operators_only,
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
    
    # Skip Buckets A-D if operators_only mode
    if not operators_only:
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
    
    # Bucket E: Operators (EXACT MATCH)
    print("\n📋 Bucket E: Operators (EXACT MATCH)")
    bucket_e = {
        "passed": 0, 
        "failed": 0, 
        "details": [], 
        "by_operator": {"COPY": {"passed": 0, "failed": 0}, "WRAP": {"passed": 0, "failed": 0}, "EXTRACT": {"passed": 0, "failed": 0}},
        "by_word_count": {"1w": {"passed": 0, "failed": 0}, "2w": {"passed": 0, "failed": 0}, "3w": {"passed": 0, "failed": 0}, "4w": {"passed": 0, "failed": 0}},
    }
    
    for name, prompt, operator, expected in OPERATOR_PROMPTS:
        gen = generate(prompt)
        passed, reason = check_operator_exact(gen, expected, operator)
        bucket_e["details"].append({"name": name, "passed": passed, "reason": reason, "operator": operator, "expected": expected, "generated": gen[:100]})
        bucket_e["by_operator"][operator]["passed" if passed else "failed"] += 1
        
        # Track by word count (extract from test name like "op_copy_2w_1")
        for wc in ["1w", "2w", "3w", "4w"]:
            if f"_{wc}_" in name:
                bucket_e["by_word_count"][wc]["passed" if passed else "failed"] += 1
                break
        
        if passed:
            bucket_e["passed"] += 1
            if verbose:
                print(f"  ✅ {name} [{operator}]: {reason}")
        else:
            bucket_e["failed"] += 1
            print(f"  ❌ {name} [{operator}]: {reason}")
    
    # Per-operator summary
    print(f"\n  Operator breakdown:")
    for op in ["COPY", "WRAP", "EXTRACT"]:
        op_passed = bucket_e["by_operator"][op]["passed"]
        op_failed = bucket_e["by_operator"][op]["failed"]
        op_total = op_passed + op_failed
        op_pct = (op_passed / op_total * 100) if op_total > 0 else 0
        status = "✅" if op_failed == 0 else "❌"
        print(f"    {op}: {op_passed}/{op_total} ({op_pct:.0f}%) {status}")
    
    # Per word-count summary
    print(f"\n  Word count breakdown:")
    for wc in ["1w", "2w", "3w", "4w"]:
        wc_passed = bucket_e["by_word_count"][wc]["passed"]
        wc_failed = bucket_e["by_word_count"][wc]["failed"]
        wc_total = wc_passed + wc_failed
        if wc_total > 0:
            wc_pct = (wc_passed / wc_total * 100)
            status = "✅" if wc_failed == 0 else "❌"
            print(f"    {wc}: {wc_passed}/{wc_total} ({wc_pct:.0f}%) {status}")
    
    results["buckets"]["operators"] = bucket_e
    
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
    global _NO_SYSTEM_PROMPT, OPERATOR_PROMPTS
    
    parser = argparse.ArgumentParser(description="SFT Evaluation Suite")
    parser.add_argument("--model", type=str, required=True, help="Model name to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max tokens to generate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all results, not just failures")
    parser.add_argument("--no_system_prompt", action="store_true",
                        help="Omit system prompt (for operator-only models trained without system prompt)")
    parser.add_argument("--operators_only", action="store_true",
                        help="Only run Bucket E (operators), skip A-D")
    parser.add_argument("--use_val_templates", action="store_true",
                        help="Use VAL templates (easier) instead of NOVEL templates (harder). "
                             "Helps diagnose if failure is due to extreme template novelty.")
    args = parser.parse_args()
    
    # Set global no-system flag BEFORE building prompts
    _NO_SYSTEM_PROMPT = args.no_system_prompt
    
    # Rebuild operator prompts with current settings
    OPERATOR_PROMPTS = build_operator_prompts(use_val_templates=args.use_val_templates)
    
    if args.use_val_templates:
        print("  Using VAL templates (easier - closer to training distribution)")
    else:
        print("  Using NOVEL templates (harder - tests true abstraction)")
    
    print("="*60)
    print("  SFT EVALUATION SUITE")
    print("="*60)
    
    results = run_evaluation(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
        operators_only=args.operators_only,
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
