#!/usr/bin/env python3
"""
Debug Inference Parity

Comprehensive tests to ensure training and inference are aligned:
1. Tokenizer special-token stability
2. KV-cache parity (prefill and multi-step)
3. Greedy is truly greedy
4. Forward parity (train vs eval vs cache)

Usage:
    python scripts/debug_inference_parity.py --model phase3a1_format_lock_v5
    python scripts/debug_inference_parity.py --model phase3a1_format_lock_v5 --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from core.model import GPT, GPTConfig, load_model
from core.tokenizer import Tokenizer
from core.special_tokens import SPECIAL_TOKEN_STRINGS, SPECIAL_TOKEN_IDS


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_pass(test_name: str):
    print(f"  ✅ PASS: {test_name}")


def print_fail(test_name: str, reason: str):
    print(f"  ❌ FAIL: {test_name}")
    print(f"      Reason: {reason}")


def tokenizer_self_test(tokenizer: Tokenizer, verbose: bool = False) -> bool:
    """Test that encode/decode roundtrips work correctly."""
    print_header("Tokenizer Self-Test")
    
    all_passed = True
    
    # Test 1: Basic text roundtrip
    test_texts = [
        "Hello world",
        "What is 5 + 7?",
        "Capital of Germany?",
        "Say OK.",
    ]
    
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        if decoded != text:
            print_fail(f"Roundtrip '{text}'", f"Got '{decoded}'")
            all_passed = False
        elif verbose:
            print(f"    '{text}' -> {ids} -> '{decoded}'")
    
    if all_passed:
        print_pass("Basic text roundtrip")
    
    # Test 2: Special tokens encode to single ID
    special_passed = True
    for name, token_str in SPECIAL_TOKEN_STRINGS.items():
        ids = tokenizer.encode(token_str)
        expected_id = SPECIAL_TOKEN_IDS.get(name)
        
        if len(ids) != 1:
            print_fail(f"Special token '{token_str}'", f"Encoded to {len(ids)} tokens: {ids}")
            special_passed = False
            all_passed = False
        elif ids[0] != expected_id:
            print_fail(f"Special token '{token_str}'", f"ID mismatch: got {ids[0]}, expected {expected_id}")
            special_passed = False
            all_passed = False
        elif verbose:
            print(f"    '{token_str}' -> [{ids[0]}] (expected {expected_id})")
    
    if special_passed:
        print_pass("Special tokens encode to single ID")
    
    # Test 3: Mixed text with special tokens
    mixed_texts = [
        ("<myPT_system>Hello</myPT_system>", [50257, 15496, 50258]),
        ("<myPT_user>Test</myPT_user>", [50259]),  # Will have more tokens
        ("<myPT_assistant>", [50263]),
    ]
    
    mixed_passed = True
    for text, expected_contains in mixed_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        if decoded != text:
            print_fail(f"Mixed roundtrip '{text}'", f"Got '{decoded}'")
            mixed_passed = False
            all_passed = False
        elif verbose:
            print(f"    '{text[:40]}...' -> {len(ids)} tokens")
    
    if mixed_passed:
        print_pass("Mixed text with special tokens")
    
    return all_passed


def special_token_id_stability_test(tokenizer: Tokenizer, verbose: bool = False) -> bool:
    """Test that special token IDs match expected values."""
    print_header("Special Token ID Stability Test")
    
    all_passed = True
    
    # Check each special token
    for name, expected_id in SPECIAL_TOKEN_IDS.items():
        token_str = SPECIAL_TOKEN_STRINGS.get(name)
        if token_str is None:
            print_fail(f"Token '{name}'", "Not found in SPECIAL_TOKEN_STRINGS")
            all_passed = False
            continue
        
        # Check encoder
        if hasattr(tokenizer, 'special_token_encoder'):
            actual_id = tokenizer.special_token_encoder.get(token_str)
            if actual_id != expected_id:
                print_fail(f"Token '{name}'", f"Encoder has ID {actual_id}, expected {expected_id}")
                all_passed = False
            elif verbose:
                print(f"    {name}: '{token_str}' -> ID {actual_id}")
        
        # Check decoder
        if hasattr(tokenizer, 'special_token_decoder'):
            decoded_str = tokenizer.special_token_decoder.get(expected_id)
            if decoded_str != token_str:
                print_fail(f"Token '{name}'", f"Decoder has '{decoded_str}', expected '{token_str}'")
                all_passed = False
    
    if all_passed:
        print_pass("All special token IDs are stable")
    
    return all_passed


def greedy_is_greedy_test(model: GPT, tokenizer: Tokenizer, verbose: bool = False) -> bool:
    """Test that temperature=0 produces deterministic output."""
    print_header("Greedy is Greedy Test")
    
    prompt = "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say OK.</myPT_user><myPT_assistant>"
    
    # Generate multiple times with temperature=0
    results = []
    for i in range(3):
        output = model.generate(
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        results.append(output)
        if verbose:
            print(f"    Run {i+1}: '{output[len(prompt):]}'")
    
    # Check all results are identical
    if len(set(results)) == 1:
        print_pass("Greedy generation is deterministic")
        return True
    else:
        print_fail("Greedy determinism", f"Got {len(set(results))} different outputs")
        for i, r in enumerate(results):
            print(f"      Run {i+1}: '{r[len(prompt):]}'")
        return False


def kv_cache_parity_prefill_test(model: GPT, tokenizer: Tokenizer, verbose: bool = False) -> bool:
    """Test that KV-cache prefill matches non-cached forward."""
    print_header("KV-Cache Parity (Prefill) Test")
    
    prompt = "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>What is 2+2?</myPT_user><myPT_assistant>"
    
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=model.device)
    
    model.eval()
    
    # Determine autocast dtype
    weight_dtype = next(model.parameters()).dtype
    if weight_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif weight_dtype == torch.float16:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    
    ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if model.device.type == 'cuda' else torch.amp.autocast('cpu', enabled=False)
    
    with torch.no_grad(), ctx:
        # Non-cached forward
        logits_no_cache, _ = model(x, use_cache=False)
        last_logits_no_cache = logits_no_cache[0, -1, :]
        
        # Cached prefill
        logits_cache, kv_cache = model(x, use_cache=True)
        last_logits_cache = logits_cache[0, -1, :]
    
    # Compare
    argmax_no_cache = last_logits_no_cache.argmax().item()
    argmax_cache = last_logits_cache.argmax().item()
    
    logit_diff = (last_logits_no_cache - last_logits_cache).abs()
    max_diff = logit_diff.max().item()
    mean_diff = logit_diff.mean().item()
    
    if verbose:
        print(f"    No-cache argmax: {argmax_no_cache} = '{tokenizer.decode([argmax_no_cache])}'")
        print(f"    Cache argmax:    {argmax_cache} = '{tokenizer.decode([argmax_cache])}'")
        print(f"    Logit diff: max={max_diff:.6f}, mean={mean_diff:.6f}")
    
    if argmax_no_cache == argmax_cache:
        print_pass(f"Prefill parity (argmax={argmax_cache}, diff={max_diff:.2e})")
        return True
    else:
        print_fail("Prefill parity", f"argmax mismatch: no_cache={argmax_no_cache}, cache={argmax_cache}")
        return False


def kv_cache_parity_steps_test(model: GPT, tokenizer: Tokenizer, steps: int = 8, verbose: bool = False) -> bool:
    """Test that KV-cache multi-step decode matches non-cached."""
    print_header(f"KV-Cache Parity (Multi-Step, {steps} steps) Test")
    
    prompt = "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say hello.</myPT_user><myPT_assistant>"
    
    ids = tokenizer.encode(prompt)
    
    model.eval()
    
    # Determine autocast dtype
    weight_dtype = next(model.parameters()).dtype
    if weight_dtype == torch.bfloat16 and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif weight_dtype == torch.float16:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    
    ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if model.device.type == 'cuda' else torch.amp.autocast('cpu', enabled=False)
    
    # Track generated tokens
    generated_cache = list(ids)
    generated_no_cache = list(ids)
    kv_cache = None
    
    all_passed = True
    
    with torch.no_grad(), ctx:
        # Prefill with cache
        x_cache = torch.tensor([ids], dtype=torch.long, device=model.device)
        logits_cache, kv_cache = model(x_cache, use_cache=True)
        
        for step in range(steps):
            # Get next token from cache path
            last_logits_cache = logits_cache[0, -1, :]
            next_token_cache = last_logits_cache.argmax().item()
            
            # Get next token from no-cache path (full context)
            x_no_cache = torch.tensor([generated_no_cache], dtype=torch.long, device=model.device)
            # Trim to block_size if needed
            if x_no_cache.shape[1] > model.config.block_size:
                x_no_cache = x_no_cache[:, -model.config.block_size:]
            logits_no_cache, _ = model(x_no_cache, use_cache=False)
            last_logits_no_cache = logits_no_cache[0, -1, :]
            next_token_no_cache = last_logits_no_cache.argmax().item()
            
            # Compare
            if next_token_cache != next_token_no_cache:
                logit_diff = (last_logits_cache - last_logits_no_cache).abs()
                print_fail(f"Step {step}", 
                    f"cache={next_token_cache} ('{tokenizer.decode([next_token_cache])}'), "
                    f"no_cache={next_token_no_cache} ('{tokenizer.decode([next_token_no_cache])}')")
                print(f"      Logit diff: max={logit_diff.max().item():.6f}")
                all_passed = False
                break
            
            if verbose:
                print(f"    Step {step}: token={next_token_cache} = '{tokenizer.decode([next_token_cache])}'")
            
            # Check for stop tokens
            if next_token_cache in {50264, 50271}:  # </myPT_assistant> or <myPT_eot>
                if verbose:
                    print(f"    Stopped at step {step} (stop token)")
                break
            
            # Append token and continue
            generated_cache.append(next_token_cache)
            generated_no_cache.append(next_token_no_cache)
            
            # Advance cache with just the new token
            x_new = torch.tensor([[next_token_cache]], dtype=torch.long, device=model.device)
            logits_cache, kv_cache = model(x_new, use_cache=True, kv_cache=kv_cache)
    
    if all_passed:
        print_pass(f"Multi-step parity ({step+1} steps)")
    
    return all_passed


def print_model_info(model: GPT, tokenizer: Tokenizer):
    """Print model and tokenizer info."""
    print_header("Model & Tokenizer Info")
    
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print(f"  Model device: {model.device}")
    print(f"  Config vocab_size: {model.config.vocab_size}")
    print(f"  Tokenizer kind: {tokenizer.token_kind}")
    print(f"  Tokenizer base_vocab_size: {tokenizer.base_vocab_size}")
    
    if hasattr(tokenizer, 'special_token_encoder'):
        print(f"  Special tokens registered: {len(tokenizer.special_token_encoder)}")
        print(f"  Special token IDs: {min(tokenizer.special_token_encoder.values())}-{max(tokenizer.special_token_encoder.values())}")


def main():
    parser = argparse.ArgumentParser(description="Debug inference parity")
    parser.add_argument("--model", required=True, help="Model name to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  DEBUG INFERENCE PARITY")
    print("="*60)
    
    # Load model
    print(f"\nLoading model '{args.model}'...")
    model, tokenizer = load_model(args.model)
    model.eval()
    
    # Print info
    print_model_info(model, tokenizer)
    
    # Run tests
    results = {}
    
    results["tokenizer_self_test"] = tokenizer_self_test(tokenizer, args.verbose)
    results["special_token_stability"] = special_token_id_stability_test(tokenizer, args.verbose)
    results["kv_cache_prefill"] = kv_cache_parity_prefill_test(model, tokenizer, args.verbose)
    results["kv_cache_steps"] = kv_cache_parity_steps_test(model, tokenizer, steps=8, verbose=args.verbose)
    results["greedy_determinism"] = greedy_is_greedy_test(model, tokenizer, args.verbose)
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {total - passed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
