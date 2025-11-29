"""
Display available model configurations and their parameter counts.

Usage:
    python scripts/show_configs.py
    python scripts/show_configs.py --config_file configs/150M.json
"""

import argparse
import json
import os
from pathlib import Path


def calculate_params(config):
    """
    Calculate approximate number of parameters for a GPT model.
    
    Args:
        config: Dict with n_layer, n_embd, n_head, vocab_size, block_size, bias
    
    Returns:
        Total parameter count
    """
    n_layer = config['n_layer']
    n_embd = config['n_embd']
    n_head = config['n_head']
    vocab_size = config['vocab_size']
    block_size = config['block_size']
    bias = config.get('bias', False)
    
    # Token + position embeddings
    params = vocab_size * n_embd + block_size * n_embd
    
    # Each transformer layer has:
    # - Multi-head attention (Q, K, V projections + output projection)
    # - Feed-forward (2 linear layers)
    # - 2 layer norms
    
    for _ in range(n_layer):
        # Multi-head attention
        head_size = n_embd // n_head
        # Q, K, V projections
        params += 3 * (n_embd * head_size + (head_size if bias else 0))
        # Output projection
        params += n_embd * n_embd + (n_embd if bias else 0)
        
        # Feed-forward network (4x expansion)
        params += n_embd * (4 * n_embd) + (4 * n_embd if bias else 0)
        params += (4 * n_embd) * n_embd + (n_embd if bias else 0)
        
        # Layer norms (2 per layer, no bias by default)
        params += 2 * n_embd  # gamma
        params += 2 * n_embd  # beta
    
    # Final layer norm
    params += n_embd  # gamma
    params += n_embd  # beta
    
    # Language model head
    params += n_embd * vocab_size
    
    return params


def format_params(num_params):
    """Format parameter count in human-readable form"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def show_config(config_path):
    """Display a single config file with parameter count"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    name = config.pop('name', Path(config_path).stem)
    desc = config.pop('description', 'No description')
    
    params = calculate_params(config)
    
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"File: {config_path}")
    print(f"{'='*70}")
    print(f"Description: {desc}")
    print(f"\nArchitecture:")
    print(f"  Layers      : {config['n_layer']}")
    print(f"  Heads       : {config['n_head']}")
    print(f"  Embedding   : {config['n_embd']}")
    print(f"  Block size  : {config['block_size']}")
    print(f"  Vocab size  : {config['vocab_size']}")
    print(f"  Dropout     : {config['dropout']}")
    print(f"  Bias        : {config.get('bias', False)}")
    print(f"\nParameters:")
    print(f"  Total       : {params:,} ({format_params(params)})")
    print(f"  Batch size  : {config['batch_size']}")
    print(f"\nUsage:")
    print(f"  python train.py --config_file {config_path} --model_name my_model --input_file input.txt")
    print(f"{'='*70}")


def show_all_configs(configs_dir="configs"):
    """Display all available configurations"""
    if not os.path.exists(configs_dir):
        print(f"No configs directory found at: {configs_dir}")
        return
    
    config_files = sorted([f for f in os.listdir(configs_dir) if f.endswith('.json')])
    
    if not config_files:
        print(f"No config files found in: {configs_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Available Model Configurations")
    print(f"{'='*70}\n")
    
    configs_info = []
    
    for config_file in config_files:
        config_path = os.path.join(configs_dir, config_file)
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        name = config.get('name', Path(config_file).stem)
        desc = config.get('description', 'No description')
        params = calculate_params(config)
        
        configs_info.append({
            'file': config_file,
            'name': name,
            'desc': desc,
            'params': params,
            'n_layer': config['n_layer'],
            'n_head': config['n_head'],
            'n_embd': config['n_embd'],
        })
    
    # Display summary table
    print(f"{'File':<20} {'Name':<15} {'Params':<12} {'Layers':<8} {'Heads':<8} {'Embed':<8}")
    print(f"{'-'*20} {'-'*15} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    
    for info in configs_info:
        print(f"{info['file']:<20} {info['name']:<15} {format_params(info['params']):<12} "
              f"{info['n_layer']:<8} {info['n_head']:<8} {info['n_embd']:<8}")
    
    print(f"\n{'='*70}")
    print(f"\nTo use a config:")
    print(f"  python train.py --config_file configs/<filename> --model_name <name> --input_file <data>")
    print(f"\nTo view details:")
    print(f"  python scripts/show_configs.py --config_file configs/<filename>")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Show available model configurations")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Show details for specific config file")
    parser.add_argument("--configs_dir", type=str, default="configs",
                        help="Directory containing config files")
    
    args = parser.parse_args()
    
    if args.config_file:
        show_config(args.config_file)
    else:
        show_all_configs(args.configs_dir)


if __name__ == "__main__":
    main()

