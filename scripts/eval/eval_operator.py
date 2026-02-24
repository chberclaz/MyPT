#!/usr/bin/env python3
"""
Evaluate operator learning with EXACT-MATCH metrics.

This script tests whether the model learned abstract operators by:
1. Using validation examples with DIFFERENT templates than training
2. Checking EXACT match of the expected transformation
3. Reporting per-operator accuracy

Usage:
    python scripts/eval_operator.py --model phase3a_operator -v
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core import load_model


def normalize_output(text: str) -> str:
    """Normalize output for comparison."""
    # Strip whitespace and common punctuation at end
    text = text.strip()
    # Remove trailing period if present (model might add it)
    if text.endswith('.'):
        text = text[:-1]
    # Remove closing tag if present
    if '</myPT_assistant>' in text:
        text = text.split('</myPT_assistant>')[0]
    return text.strip()


def exact_match(generated: str, expected: str) -> bool:
    """Check if generated output exactly matches expected."""
    gen_norm = normalize_output(generated)
    exp_norm = normalize_output(expected)
    return gen_norm == exp_norm


def run_evaluation(
    model_name: str,
    val_file: str = None,
    max_new_tokens: int = 50,
    verbose: bool = False,
) -> dict:
    """
    Run exact-match evaluation on operator dataset.
    
    Returns dict with per-operator accuracy and overall stats.
    """
    
    # Load model
    print(f"Loading model: {model_name}")
    model = load_model(model_name)
    model.eval()
    
    # Find validation file
    if val_file is None:
        val_file = Path("data/sft_operator/operator_val.jsonl")
    else:
        val_file = Path(val_file)
    
    if not val_file.exists():
        print(f"ERROR: Validation file not found: {val_file}")
        print("Run: python scripts/generate_operator_dataset.py first")
        return None
    
    # Load validation episodes
    episodes = []
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            episodes.append(json.loads(line))
    
    print(f"Loaded {len(episodes)} validation episodes")
    print()
    
    # Track results
    results = {
        "total": 0,
        "correct": 0,
        "by_operator": defaultdict(lambda: {"total": 0, "correct": 0, "examples": []}),
    }
    
    # Evaluate each episode
    for i, episode in enumerate(episodes):
        meta = episode["_meta"]
        operator = meta["operator"]
        expected = meta["expected"]
        payload = meta["payload"]
        
        # Build prompt
        system = episode["messages"][0]["content"]
        user = episode["messages"][1]["content"]
        
        prompt = f"<myPT_system>{system}</myPT_system><myPT_user>{user}</myPT_user><myPT_assistant>"
        
        # Generate
        import torch
        with torch.no_grad():
            output = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
        
        # Extract generated part
        generated = output[len(prompt):] if output.startswith(prompt) else output
        generated = normalize_output(generated)
        
        # Check exact match
        is_correct = exact_match(generated, expected)
        
        results["total"] += 1
        results["by_operator"][operator]["total"] += 1
        
        if is_correct:
            results["correct"] += 1
            results["by_operator"][operator]["correct"] += 1
        
        # Store example (limit to first 5 failures per operator)
        if not is_correct and len([e for e in results["by_operator"][operator]["examples"] if not e["correct"]]) < 5:
            results["by_operator"][operator]["examples"].append({
                "payload": payload,
                "expected": expected,
                "generated": generated,
                "correct": False,
            })
        elif is_correct and len([e for e in results["by_operator"][operator]["examples"] if e["correct"]]) < 2:
            results["by_operator"][operator]["examples"].append({
                "payload": payload,
                "expected": expected,
                "generated": generated,
                "correct": True,
            })
        
        # Progress
        if verbose and (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(episodes)}...")
    
    return results


def print_results(results: dict, verbose: bool = False):
    """Print evaluation results."""
    
    print("=" * 60)
    print("  OPERATOR EVALUATION RESULTS (EXACT MATCH)")
    print("=" * 60)
    print()
    
    # Per-operator results
    all_pass = True
    for operator in ["COPY", "WRAP", "EXTRACT"]:
        stats = results["by_operator"][operator]
        total = stats["total"]
        correct = stats["correct"]
        pct = (correct / total * 100) if total > 0 else 0
        
        status = "✅ PASS" if pct >= 90 else "❌ FAIL"
        if pct < 90:
            all_pass = False
        
        print(f"  {operator}: {correct}/{total} ({pct:.1f}%) {status}")
        
        # Show examples
        if verbose:
            for ex in stats["examples"]:
                mark = "✓" if ex["correct"] else "✗"
                print(f"    {mark} payload='{ex['payload']}'")
                print(f"      expected: '{ex['expected']}'")
                print(f"      got:      '{ex['generated']}'")
            print()
    
    # Overall
    total = results["total"]
    correct = results["correct"]
    pct = (correct / total * 100) if total > 0 else 0
    
    print()
    print("=" * 60)
    print(f"  OVERALL: {correct}/{total} ({pct:.1f}%)")
    
    if all_pass:
        print("  STATUS: ✅ ALL OPERATORS PASS (≥90%)")
        print()
        print("  The model has learned abstract operators!")
        print("  Safe to proceed with RAG/context-grounded training.")
    else:
        print("  STATUS: ❌ SOME OPERATORS FAIL (<90%)")
        print()
        print("  The model has NOT learned abstract operators.")
        print("  Check:")
        print("    1. Training coverage (should be 2-3.5x, not more)")
        print("    2. Dataset integrity (unique payloads, template split)")
        print("    3. If still failing, consider scaling model size")
    
    print("=" * 60)
    
    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate operator learning")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name to evaluate")
    parser.add_argument("--val_file", type=str, default=None,
                        help="Path to validation JSONL (default: data/sft_operator/operator_val.jsonl)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Max tokens to generate")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show example outputs")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    results = run_evaluation(
        model_name=args.model,
        val_file=args.val_file,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )
    
    if results is None:
        sys.exit(1)
    
    all_pass = print_results(results, verbose=args.verbose)
    
    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "total": results["total"],
            "correct": results["correct"],
            "accuracy": results["correct"] / results["total"] if results["total"] > 0 else 0,
            "by_operator": {
                op: {
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0,
                }
                for op, stats in results["by_operator"].items()
            },
            "all_pass": all_pass,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
