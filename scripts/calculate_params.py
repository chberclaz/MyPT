"""
Calculate the number of parameters in a GPT model.

This script helps you understand how model size relates to architecture parameters.

Usage:
    python scripts/calculate_params.py --n_layer 16 --n_embd 768 --n_head 12
    python scripts/calculate_params.py --config_file configs/150M.json
    python scripts/calculate_params.py --interactive
"""

import argparse
import json
import os


def calculate_gpt_params(n_layer, n_embd, n_head, vocab_size, block_size, bias=False):
    """
    Calculate the exact number of parameters in a GPT model.
    
    Args:
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        n_head: Number of attention heads
        vocab_size: Vocabulary size
        block_size: Context length (for position embeddings)
        bias: Whether to use bias in layers
    
    Returns:
        Dictionary with detailed parameter breakdown
    """
    params = {}
    
    # Token embeddings: vocab_size × n_embd
    params['token_embeddings'] = vocab_size * n_embd
    
    # Position embeddings: block_size × n_embd
    params['position_embeddings'] = block_size * n_embd
    
    # Per-layer parameters
    head_size = n_embd // n_head
    
    # Initialize layer params
    params['attention_per_layer'] = 0
    params['feedforward_per_layer'] = 0
    params['layernorm_per_layer'] = 0
    
    # Attention parameters per layer
    # Q, K, V projections: 3 × (n_embd × n_embd)
    attn_qkv = 3 * n_embd * n_embd
    if bias:
        attn_qkv += 3 * n_embd
    
    # Output projection: n_embd × n_embd
    attn_out = n_embd * n_embd
    if bias:
        attn_out += n_embd
    
    params['attention_per_layer'] = attn_qkv + attn_out
    
    # Feed-forward parameters per layer
    # First linear: n_embd × (4 × n_embd)
    ff_1 = n_embd * (4 * n_embd)
    if bias:
        ff_1 += 4 * n_embd
    
    # Second linear: (4 × n_embd) × n_embd
    ff_2 = (4 * n_embd) * n_embd
    if bias:
        ff_2 += n_embd
    
    params['feedforward_per_layer'] = ff_1 + ff_2
    
    # Layer norm parameters per layer (2 layer norms)
    # Each layer norm has 2 × n_embd (gamma and beta)
    params['layernorm_per_layer'] = 2 * 2 * n_embd  # 2 layer norms per layer
    
    # Total per layer
    params['total_per_layer'] = (
        params['attention_per_layer'] + 
        params['feedforward_per_layer'] + 
        params['layernorm_per_layer']
    )
    
    # All layers
    params['all_layers'] = params['total_per_layer'] * n_layer
    
    # Final layer norm: 2 × n_embd
    params['final_layernorm'] = 2 * n_embd
    
    # Language model head (tied with token embeddings usually, but we count it)
    # n_embd × vocab_size
    params['lm_head'] = n_embd * vocab_size
    
    # Total parameters
    params['total'] = (
        params['token_embeddings'] +
        params['position_embeddings'] +
        params['all_layers'] +
        params['final_layernorm'] +
        params['lm_head']
    )
    
    return params


def format_number(num):
    """Format a number with commas and in M/B notation"""
    if num >= 1e9:
        return f"{num:,} ({num/1e9:.2f}B)"
    elif num >= 1e6:
        return f"{num:,} ({num/1e6:.2f}M)"
    elif num >= 1e3:
        return f"{num:,} ({num/1e3:.2f}K)"
    else:
        return str(num)


def print_breakdown(params, n_layer, n_embd, n_head, vocab_size, block_size, bias):
    """Print detailed parameter breakdown"""
    print("=" * 70)
    print("GPT Model Parameter Breakdown")
    print("=" * 70)
    print()
    print("Architecture:")
    print(f"  n_layer     : {n_layer}")
    print(f"  n_embd      : {n_embd}")
    print(f"  n_head      : {n_head}")
    print(f"  vocab_size  : {vocab_size:,}")
    print(f"  block_size  : {block_size}")
    print(f"  bias        : {bias}")
    print()
    print("=" * 70)
    print("Parameter Breakdown:")
    print("=" * 70)
    print()
    print("Embeddings:")
    print(f"  Token embeddings    : {format_number(params['token_embeddings'])}")
    print(f"  Position embeddings : {format_number(params['position_embeddings'])}")
    print()
    print(f"Per Transformer Layer ({n_layer} layers):")
    print(f"  Attention           : {format_number(params['attention_per_layer'])}")
    print(f"  Feed-forward        : {format_number(params['feedforward_per_layer'])}")
    print(f"  Layer norms         : {format_number(params['layernorm_per_layer'])}")
    print(f"  Total per layer     : {format_number(params['total_per_layer'])}")
    print()
    print(f"All {n_layer} Layers        : {format_number(params['all_layers'])}")
    print()
    print("Final Components:")
    print(f"  Final layer norm    : {format_number(params['final_layernorm'])}")
    print(f"  LM head             : {format_number(params['lm_head'])}")
    print()
    print("=" * 70)
    print(f"TOTAL PARAMETERS      : {format_number(params['total'])}")
    print("=" * 70)
    print()
    
    # Show memory estimate
    # FP32: 4 bytes per param
    # FP16: 2 bytes per param
    mem_fp32 = params['total'] * 4 / (1024**3)  # GB
    mem_fp16 = params['total'] * 2 / (1024**3)  # GB
    
    print("Estimated Memory (model weights only):")
    print(f"  FP32 (4 bytes/param) : {mem_fp32:.2f} GB")
    print(f"  FP16 (2 bytes/param) : {mem_fp16:.2f} GB")
    print()
    print("Note: Training requires 3-4x more memory (gradients, optimizer states)")
    print(f"  Training estimate (FP32): {mem_fp32 * 3.5:.2f} - {mem_fp32 * 4:.2f} GB")
    print()


def print_formula():
    """Print the parameter calculation formula"""
    print("=" * 70)
    print("GPT Parameter Calculation Formula")
    print("=" * 70)
    print()
    print("Total Parameters = Embeddings + Transformer Layers + LM Head")
    print()
    print("Breakdown:")
    print()
    print("1. Embeddings:")
    print("   - Token embeddings:    vocab_size × n_embd")
    print("   - Position embeddings: block_size × n_embd")
    print()
    print("2. Each Transformer Layer:")
    print("   - Attention:")
    print("     • Q, K, V projections: 3 × (n_embd × n_embd)")
    print("     • Output projection:   n_embd × n_embd")
    print("     • Total attention:     4 × n_embd²")
    print()
    print("   - Feed-forward:")
    print("     • First layer:  n_embd × (4 × n_embd)")
    print("     • Second layer: (4 × n_embd) × n_embd")
    print("     • Total FF:     8 × n_embd²")
    print()
    print("   - Layer norms: 2 × (2 × n_embd) = 4 × n_embd")
    print()
    print("   Total per layer ≈ 12 × n_embd² + 4 × n_embd")
    print()
    print("3. All Layers: n_layer × (12 × n_embd² + 4 × n_embd)")
    print()
    print("4. Final layer norm: 2 × n_embd")
    print()
    print("5. LM Head: n_embd × vocab_size")
    print()
    print("=" * 70)
    print()
    print("Simplified Approximation:")
    print("  Total ≈ 12 × n_layer × n_embd² + 2 × vocab_size × n_embd")
    print()
    print("Note: block_size affects position embeddings (usually negligible)")
    print("=" * 70)
    print()


def interactive_mode():
    """Interactive parameter calculator"""
    print()
    print("=" * 70)
    print("Interactive GPT Parameter Calculator")
    print("=" * 70)
    print()
    
    try:
        n_layer = int(input("Number of layers (e.g., 16): "))
        n_embd = int(input("Embedding dimension (e.g., 768): "))
        n_head = int(input("Number of attention heads (e.g., 12): "))
        vocab_size = int(input("Vocabulary size (50304 for GPT-2, or custom): ") or "50304")
        block_size = int(input("Block size / context length (e.g., 256 or 1024): ") or "256")
        bias_input = input("Use bias? (y/n, default n): ").lower()
        bias = bias_input == 'y'
        
        print()
        params = calculate_gpt_params(n_layer, n_embd, n_head, vocab_size, block_size, bias)
        print_breakdown(params, n_layer, n_embd, n_head, vocab_size, block_size, bias)
        
    except (ValueError, KeyboardInterrupt):
        print("\nInvalid input or cancelled.")
        return


def main():
    parser = argparse.ArgumentParser(
        description="Calculate GPT model parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate from architecture params
  python scripts/calculate_params.py --n_layer 16 --n_embd 768 --n_head 12
  
  # Calculate from config file
  python scripts/calculate_params.py --config_file configs/150M.json
  
  # Interactive mode
  python scripts/calculate_params.py --interactive
  
  # Show formula
  python scripts/calculate_params.py --show_formula
        """
    )
    
    parser.add_argument("--n_layer", type=int, help="Number of transformer layers")
    parser.add_argument("--n_embd", type=int, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=50304, help="Vocabulary size (default: 50304)")
    parser.add_argument("--block_size", type=int, default=256, help="Context length (default: 256)")
    parser.add_argument("--bias", action="store_true", help="Use bias in layers")
    parser.add_argument("--config_file", type=str, help="Load architecture from config file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--show_formula", action="store_true", help="Show parameter calculation formula")
    
    args = parser.parse_args()
    
    if args.show_formula:
        print_formula()
        return
    
    if args.interactive:
        interactive_mode()
        return
    
    if args.config_file:
        if not os.path.exists(args.config_file):
            print(f"Error: Config file not found: {args.config_file}")
            return
        
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        
        n_layer = config['n_layer']
        n_embd = config['n_embd']
        n_head = config['n_head']
        vocab_size = config.get('vocab_size', 50304)
        block_size = config.get('block_size', 256)
        bias = config.get('bias', False)
        
        print(f"Loaded config from: {args.config_file}")
        if 'name' in config:
            print(f"Config name: {config['name']}")
        if 'description' in config:
            print(f"Description: {config['description']}")
        print()
        
    elif args.n_layer and args.n_embd and args.n_head:
        n_layer = args.n_layer
        n_embd = args.n_embd
        n_head = args.n_head
        vocab_size = args.vocab_size
        block_size = args.block_size
        bias = args.bias
    else:
        parser.print_help()
        return
    
    # Validate
    if n_embd % n_head != 0:
        print(f"Error: n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        return
    
    # Calculate
    params = calculate_gpt_params(n_layer, n_embd, n_head, vocab_size, block_size, bias)
    print_breakdown(params, n_layer, n_embd, n_head, vocab_size, block_size, bias)


if __name__ == "__main__":
    main()

