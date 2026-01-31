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
from contextlib import nullcontext

from core import load_model, GPT, Tokenizer
from core.special_tokens import SPECIAL_TOKEN_STRINGS


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
        expected_id = tokenizer.special_tokens.get(name)
        
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
    
    # Test 3: Mixed text with special tokens - roundtrip
    mixed_texts = [
        "<myPT_system>Hello</myPT_system>",
        "<myPT_user>Test</myPT_user>",
        "<myPT_assistant>OK.</myPT_assistant>",
        "<myPT_system>You are MyPT.</myPT_system><myPT_user>Say hi.</myPT_user><myPT_assistant>",
    ]
    
    mixed_passed = True
    for text in mixed_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        
        if decoded != text:
            print_fail(f"Mixed roundtrip", f"'{text}' -> '{decoded}'")
            mixed_passed = False
            all_passed = False
        elif verbose:
            print(f"    '{text[:50]}...' -> {len(ids)} tokens -> roundtrip OK")
    
    if mixed_passed:
        print_pass("Mixed text with special tokens roundtrip")
    
    return all_passed


def special_token_id_stability_test(tokenizer: Tokenizer, verbose: bool = False) -> bool:
    """Test that special token IDs are in expected range and consistent."""
    print_header("Special Token ID Stability Test")
    
    all_passed = True
    
    # Check IDs are in expected range (base_vocab_size to model_vocab_size)
    base = tokenizer.base_vocab_size
    model = tokenizer.model_vocab_size
    
    if verbose:
        print(f"    Base vocab size: {base}")
        print(f"    Model vocab size: {model}")
        print(f"    Special token range: {base} - {model-1}")
    
    for name, token_id in tokenizer.special_tokens.items():
        token_str = SPECIAL_TOKEN_STRINGS.get(name, "???")
        
        if token_id < base or token_id >= model:
            print_fail(f"Token '{name}'", f"ID {token_id} out of range [{base}, {model})")
            all_passed = False
        elif verbose:
            print(f"    {name}: '{token_str}' -> ID {token_id}")
        
        # Verify encoder/decoder consistency
        if tokenizer.special_token_encoder.get(token_str) != token_id:
            print_fail(f"Token '{name}'", "Encoder mismatch")
            all_passed = False
        if tokenizer.special_token_decoder.get(token_id) != token_str:
            print_fail(f"Token '{name}'", "Decoder mismatch")
            all_passed = False
    
    if all_passed:
        print_pass(f"All {len(tokenizer.special_tokens)} special token IDs are stable and consistent")
    
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
    """Test that KV-cache prefill matches non-cached forward.
    
    Uses EXACT same settings as model.generate():
    - fp16 autocast on CUDA (line 1166 in model.py)
    - cache_dtype = model weight dtype (line 1198)
    - FAST MODE preallocated kv_cache
    """
    print_header("KV-Cache Parity (Prefill) Test")
    
    prompt = "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>What is 2+2?</myPT_user><myPT_assistant>"
    
    ids = tokenizer.encode(prompt)
    device = model.config.device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    
    model.eval()
    
    # EXACT match to generate() line 1163-1166:
    # device_type = "cuda" if isinstance(device, str) and "cuda" in device else "cpu"
    # ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16) if device_type == "cuda" else nullcontext()
    device_type = "cuda" if isinstance(device, str) and "cuda" in device else "cpu"
    if device_type == "cuda":
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    else:
        ctx = nullcontext()
    
    # EXACT match to generate() line 1196-1204:
    # cache_dtype = next(self.parameters()).dtype
    nh = model.config.n_head
    hs = model.config.n_embd // model.config.n_head
    cache_dtype = next(model.parameters()).dtype
    
    kv_cache = []
    for _ in range(model.config.n_layer):
        k_cache = torch.empty((1, nh, model.config.block_size, hs), device=device, dtype=cache_dtype)
        v_cache = torch.empty((1, nh, model.config.block_size, hs), device=device, dtype=cache_dtype)
        kv_cache.append((k_cache, v_cache))
    
    if verbose:
        print(f"    device_type: {device_type}")
        print(f"    autocast dtype: torch.float16 (same as generate)")
        print(f"    cache_dtype: {cache_dtype} (from model weights)")
    
    with torch.no_grad(), ctx:
        # Non-cached forward (returns logits, loss, present_kv)
        logits_no_cache, _, _ = model(x, use_cache=False)
        last_logits_no_cache = logits_no_cache[0, -1, :]
        
        # Cached prefill (FAST MODE: use kv_cache parameter)
        logits_cache, _, present = model(x, use_cache=True, kv_cache=kv_cache, cache_pos=0)
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
    """Test that KV-cache multi-step decode matches non-cached.
    
    Uses EXACT same settings as model.generate():
    - fp16 autocast on CUDA (line 1166 in model.py)
    - cache_dtype = model weight dtype (line 1198)
    - FAST MODE preallocated kv_cache
    """
    print_header(f"KV-Cache Parity (Multi-Step, {steps} steps) Test")
    
    prompt = "<myPT_system>You are MyPT. Be concise: 1-2 sentences. Follow instructions exactly.</myPT_system><myPT_user>Say hello.</myPT_user><myPT_assistant>"
    
    ids = tokenizer.encode(prompt)
    device = model.config.device
    
    model.eval()
    
    # EXACT match to generate() line 1163-1166
    device_type = "cuda" if isinstance(device, str) and "cuda" in device else "cpu"
    if device_type == "cuda":
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    else:
        ctx = nullcontext()
    
    # EXACT match to generate() line 1196-1204
    nh = model.config.n_head
    hs = model.config.n_embd // model.config.n_head
    cache_dtype = next(model.parameters()).dtype
    
    kv_cache = []
    for _ in range(model.config.n_layer):
        k_cache = torch.empty((1, nh, model.config.block_size, hs), device=device, dtype=cache_dtype)
        v_cache = torch.empty((1, nh, model.config.block_size, hs), device=device, dtype=cache_dtype)
        kv_cache.append((k_cache, v_cache))
    cache_pos = 0
    
    # Track generated tokens
    generated_no_cache = list(ids)
    
    all_passed = True
    step = 0
    
    # Get stop token IDs from tokenizer (not hardcoded like generate() does!)
    stop_ids = {
        tokenizer.special_tokens.get("myPT_assistant_close"),
        tokenizer.special_tokens.get("myPT_eot"),
    }
    stop_ids = {s for s in stop_ids if s is not None}
    
    if verbose:
        print(f"    device_type: {device_type}")
        print(f"    cache_dtype: {cache_dtype}")
        print(f"    stop_ids: {stop_ids}")
    
    with torch.no_grad(), ctx:
        # Prefill with cache (FAST MODE)
        x_cache = torch.tensor([ids], dtype=torch.long, device=device)
        logits_cache, _, present = model(x_cache, use_cache=True, kv_cache=kv_cache, cache_pos=cache_pos)
        kv_cache, cache_pos = present
        
        for step in range(steps):
            # Get next token from cache path
            last_logits_cache = logits_cache[0, -1, :]
            next_token_cache = last_logits_cache.argmax().item()
            
            # Get next token from no-cache path (full context)
            x_no_cache = torch.tensor([generated_no_cache], dtype=torch.long, device=device)
            # Trim to block_size if needed
            if x_no_cache.shape[1] > model.config.block_size:
                x_no_cache = x_no_cache[:, -model.config.block_size:]
            logits_no_cache, _, _ = model(x_no_cache, use_cache=False)
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
            if next_token_cache in stop_ids:
                if verbose:
                    print(f"    Stopped at step {step} (stop token)")
                break
            
            # Append token to no-cache context
            generated_no_cache.append(next_token_cache)
            
            # Advance cache with just the new token (FAST MODE)
            x_new = torch.tensor([[next_token_cache]], dtype=torch.long, device=device)
            logits_cache, _, present = model(x_new, use_cache=True, kv_cache=kv_cache, cache_pos=cache_pos)
            kv_cache, cache_pos = present
    
    if all_passed:
        print_pass(f"Multi-step parity ({step+1} steps)")
    
    return all_passed


def print_model_info(model: GPT, tokenizer: Tokenizer):
    """Print model and tokenizer info."""
    print_header("Model & Tokenizer Info")
    
    print(f"  Model dtype: {next(model.parameters()).dtype}")
    print(f"  Model device: {model.config.device}")
    print(f"  Config vocab_size: {model.config.vocab_size}")
    print(f"  Tokenizer kind: {tokenizer.token_kind}")
    print(f"  Tokenizer base_vocab_size: {tokenizer.base_vocab_size}")
    
    if hasattr(tokenizer, 'special_token_encoder'):
        print(f"  Special tokens registered: {len(tokenizer.special_token_encoder)}")
        if tokenizer.special_token_encoder:
            min_id = min(tokenizer.special_token_encoder.values())
            max_id = max(tokenizer.special_token_encoder.values())
            print(f"  Special token IDs: {min_id}-{max_id}")


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
    model = load_model(args.model)
    tokenizer = model.tokenizer
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
