"""
MyPT Core - A minimal GPT implementation with modular architecture.

This package provides a clean, educational implementation of GPT (Generative Pre-trained Transformer)
with a focus on modularity, maintainability, and ease of understanding.

Key Features:
    - Clean transformer architecture with multi-head attention
    - Two tokenization modes: GPT-2 BPE and character-level
    - Model owns its training logic (PyTorch Lightning style)
    - JSON-based checkpoint format for robustness
    - Easy fine-tuning and transfer learning

Quick Start:
    >>> from core import GPT, GPTConfig, GPTDataLoader, CheckpointManager
    >>> 
    >>> # Create a model
    >>> config = GPTConfig(n_layer=6, n_head=6, n_embd=384)
    >>> model = GPT(config)
    >>> 
    >>> # Train
    >>> model.fit(data_loader, optimizer, max_iters=1000, checkpoint_dir="checkpoints/my_model")
    >>> 
    >>> # Generate
    >>> output = model.generate("Hello", max_new_tokens=100)

Public API:
    Model & Config:
        GPT: Main GPT model class
        GPTConfig: Model architecture configuration
    
    Data:
        GPTDataLoader: Data loading and batching
    
    Tokenization:
        Tokenizer: Handles GPT-2 BPE and character-level tokenization
    
    Checkpoints:
        CheckpointManager: Checkpoint management and model initialization
"""

# Version info
__version__ = "0.2.0"
__author__ = "MyPT Contributors"

# Core model classes
from .model import GPT, GPTConfig, CausalSelfAttention, FeedForward, Block

# Tokenization
from .tokenizer import Tokenizer

# Data loading
from .data_loader import GPTDataLoader
from .episode_data_loader import GPTEpisodeDataLoader, is_episode_indexed_dataset

# Checkpoint management
from .checkpoint import CheckpointManager

# Generator
from .generator import Generator

# Banner
from .banner import (
    print_banner,
    print_section,
    get_banner_string,
    banner_train,
    banner_generate,
    banner_webapp,
    banner_workspace,
    banner_dataset,
    ROBOT_HEAD,
)

# Training utilities
from .training_utils import (
    calculate_dataset_coverage,
    print_coverage_analysis,
    calculate_episode_coverage,
    print_episode_coverage_analysis,
    estimate_training_time,
    print_training_estimates,
)

# Dataset lineage / provenance
from .dataset_lineage import (
    iso_now,
    count_jsonl_rows,
    load_lineage_for_input,
    merge_lineage,
    write_lineage_sidecar,
)

# System prompts (for training scripts and inference)
from .system_prompts import (
    CONVERSATION_SYSTEM_PROMPT,
    DEFAULT_CONVERSATION_PROMPT,
    AGENTIC_COMPACT_PROMPT,
    AGENTIC_STANDARD_PROMPT,
    AGENTIC_VERBOSE_PROMPT,
    AGENTIC_SYSTEM_PROMPT,
    DEFAULT_AGENTIC_PROMPT,
)


# Public API - explicitly define what gets imported with "from core import *"
__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Model & Architecture
    'GPT',
    'GPTConfig',
    'Head',
    'MultiHeadAttention',
    'FeedForward',
    'Block',
    
    # Tokenization
    'Tokenizer',
    
    # Data
    'GPTDataLoader',
    'GPTEpisodeDataLoader',
    'is_episode_indexed_dataset',
    
    # Checkpoints
    'CheckpointManager',
    
    # Generator
    'Generator',
    
    # Training utilities
    'calculate_dataset_coverage',
    'print_coverage_analysis',
    'calculate_episode_coverage',
    'print_episode_coverage_analysis',
    'estimate_training_time',
    'print_training_estimates',
    'iso_now',
    'count_jsonl_rows',
    'load_lineage_for_input',
    'merge_lineage',
    'write_lineage_sidecar',
    
    # Banner
    'print_banner',
    'print_section',
    'get_banner_string',
    'banner_train',
    'banner_generate',
    'banner_webapp',
    'banner_workspace',
    'banner_dataset',
    'ROBOT_HEAD',
    
    # System Prompts
    'CONVERSATION_SYSTEM_PROMPT',
    'DEFAULT_CONVERSATION_PROMPT',
    'AGENTIC_COMPACT_PROMPT',
    'AGENTIC_STANDARD_PROMPT',
    'AGENTIC_VERBOSE_PROMPT',
    'AGENTIC_SYSTEM_PROMPT',
    'DEFAULT_AGENTIC_PROMPT',
]


# Convenience functions for common operations
def create_model(
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    block_size: int = 256,
    dropout: float = 0.2,
    bias: bool = False,
    tokenization: str = "gpt2",
    device: str | None = None
) -> GPT:
    """
    Create a GPT model with sensible defaults.
    
    Args:
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        block_size: Context length (max sequence length)
        dropout: Dropout rate
        bias: Use bias in layers (GPT-2 style if True)
        tokenization: Tokenizer type ('gpt2' or 'char')
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        Initialized GPT model
    
    Example:
        >>> model = create_model(n_layer=8, n_head=8, n_embd=512)
    """
    import torch
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = GPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        dropout=dropout,
        bias=bias,
        vocab_size=50304,  # Will be adjusted if char tokenization
        device=device
    )
    
    tokenizer = Tokenizer(config, tokenization)
    model = GPT(config, tokenizer=tokenizer)
    model.to(device)
    
    return model


def load_model(model_name: str, base_dir: str = "checkpoints", load_dtype: str | None = None) -> GPT:
    """
    Load a trained model from checkpoint.
    Automatically detects new JSON format or legacy format.
    
    Args:
        model_name: Name of the model (e.g., 'dante', 'shakespeare')
        base_dir: Base checkpoints directory
        load_dtype: Optional dtype to load model as ('fp32', 'fp16', 'bf16').
                   If None, uses smart defaults:
                   - bf16 checkpoint on older GPU → auto-converts to fp32
                   - fp16/fp32 checkpoints → keeps original dtype
    
    Returns:
        Loaded GPT model ready for generation
    
    Example:
        >>> model = load_model("dante")  # Smart defaults
        >>> model = load_model("dante", load_dtype="fp16")  # Force fp16
        >>> output = model.generate("Nel mezzo del cammin", max_new_tokens=100)
    """
    return CheckpointManager.load_for_inference(model_name, base_dir=base_dir, load_dtype=load_dtype)


def get_model_info(model_name: str, base_dir: str = "checkpoints") -> dict:
    """
    Get information about a model checkpoint without loading the full model.
    
    Args:
        model_name: Name of the model
        base_dir: Base checkpoints directory
    
    Returns:
        Dictionary with model information (config, tokenizer, training state)
    
    Example:
        >>> info = get_model_info("dante")
        >>> print(f"Model has {info['config']['n_layer']} layers")
    """
    import os
    import json
    import torch
    
    ckpt_manager = CheckpointManager(model_name, base_dir)
    
    info = {}
    
    # Try new format
    if ckpt_manager.exists_new_format():
        config_path = os.path.join(ckpt_manager.checkpoint_dir, "config.json")
        with open(config_path, 'r') as f:
            info['config'] = json.load(f)
        
        tokenizer_path = os.path.join(ckpt_manager.checkpoint_dir, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'r') as f:
                info['tokenizer'] = json.load(f)
        
        training_state_path = os.path.join(ckpt_manager.checkpoint_dir, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, 'r') as f:
                info['training_state'] = json.load(f)
        
        info['format'] = 'json'
    
    # Try legacy format
    else:
        for filename in ["latest.pt", "final.pt"]:
            if ckpt_manager.exists_legacy_format(filename):
                # Legacy checkpoints contain dicts with config/tokenizer
                checkpoint = torch.load(
                    ckpt_manager.get_path(filename),
                    map_location='cpu',
                    weights_only=False
                )
                info['config'] = checkpoint.get('config', {})
                info['tokenizer'] = checkpoint.get('tokenizer', {})
                info['training_state'] = {
                    'step': checkpoint.get('step', None)
                }
                info['format'] = 'legacy'
                break
    
    return info


# Add these convenience functions to __all__
__all__.extend(['create_model', 'load_model', 'get_model_info'])

