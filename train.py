# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import argparse
from core import (
    GPTConfig, 
    GPTDataLoader, 
    CheckpointManager, 
    get_model_info,
    calculate_dataset_coverage,
    print_coverage_analysis,
)


def parse_args():
    """Parse command-line arguments for training"""
    parser = argparse.ArgumentParser(description="Train a GPT model")
    
    # Model/data config
    parser.add_argument("--model_name", type=str, default="default",
                        help="Name of the model/checkpoint set (e.g. dante, shakespeare)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to training text file (for in-memory mode, small datasets)")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to sharded dataset directory (for large datasets, created with prepare_dataset.py)")
    parser.add_argument("--tokenization", type=str, default="gpt2",
                        choices=["gpt2", "char"],
                        help="Tokenizer type: gpt2 or char")
    parser.add_argument("--init_from_model", type=str, default=None,
                        help="Optional: model_name to initialize weights from (e.g. dante_base)")
    
    # Configuration file (alternative to individual params)
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to JSON config file (e.g. configs/150M.json). "
                             "If specified, overrides individual architecture params.")
    
    # Training hyperparameters
    parser.add_argument("--max_iters", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--eval_iters", type=int, default=200,
                        help="Number of iterations for evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for optimizer")
    
    # Model architecture (ignored if --config_file is specified)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--block_size", type=int, default=256,
                        help="Context length (block size)")
    parser.add_argument("--n_embd", type=int, default=384,
                        help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--bias", action="store_true",
                        help="Use bias in Linear/LayerNorm like GPT-2")
    
    return parser.parse_args()


def main():
    """Main training entry point"""
    args = parse_args()
    
    # Load config from file or use CLI arguments
    if args.config_file:
        print(f"Loading configuration from: {args.config_file}")
        import json
        import os
        
        if not os.path.exists(args.config_file):
            raise FileNotFoundError(f"Config file not found: {args.config_file}")
        
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Extract description if present
        config_name = config_dict.pop("name", None)
        config_desc = config_dict.pop("description", None)
        
        # Create config from file
        # Set device if not in config
        if "device" not in config_dict:
            import torch
            config_dict["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        config = GPTConfig(**config_dict)
        
        if config_name:
            print(f"Configuration: {config_name}")
        if config_desc:
            print(f"Description: {config_desc}")
    else:
        # Use CLI arguments
        config = GPTConfig(
            batch_size=args.batch_size,
            block_size=args.block_size,
            vocab_size=50304,  # Will be adjusted for char-level if needed
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            bias=args.bias,
        )
    
    print(f"========== Training Configuration ==========")
    print(f"Model name: {args.model_name}")
    print(f"Input file: {args.input_file}")
    print(f"Tokenization: {args.tokenization}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Learning rate: {args.learning_rate}")
    if args.init_from_model:
        print(f"Initializing from: {args.init_from_model}")
    print()
    
    # Check if model already exists (for resume info)
    ckpt_manager = CheckpointManager(args.model_name)
    if ckpt_manager.exists():
        print("========== Existing Model Detected ==========")
        print("Will resume training from existing checkpoint")
        try:
            info = get_model_info(args.model_name)
            print(f"Format: {info.get('format', 'unknown')}")
            if 'training_state' in info and info['training_state'].get('step'):
                print(f"Last training step: {info['training_state']['step']}")
            if 'config' in info:
                print(f"Architecture: {info['config'].get('n_layer', '?')} layers, "
                      f"{info['config'].get('n_embd', '?')} embd dim")
        except Exception:
            print("(Could not load full model info)")
        print()
    
    # Read training text (only for in-memory mode)
    text = None
    if args.input_file:
        print("Loading training data (in-memory mode)...")
        text = GPTDataLoader.read_text(args.input_file)
        print(f"Loaded {len(text):,} characters")
        print(f"Approximate tokens (char-level): {len(text):,}")
        print(f"Approximate tokens (GPT-2): ~{len(text)//4:,}")
    
    # Initialize model (handles resume / init_from / fresh)
    print("\nInitializing model...")
    model, optimizer, start_step = ckpt_manager.initialize_for_training(
        config=config,
        tokenization=args.tokenization,
        input_text=text,
        learning_rate=args.learning_rate,
        init_from_model=args.init_from_model
    )
    
    print(f"\n========== Model Configuration ==========")
    print(f"Architecture: {model.config.n_layer} layers × {model.config.n_head} heads")
    print(f"Embedding dimension: {model.config.n_embd}")
    print(f"Context length: {model.config.block_size}")
    print(f"Vocabulary size: {model.config.vocab_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {model.config.device}")
    print(f"Starting from step: {start_step}")
    print()
    
    # Prepare data
    print("Preparing data...")
    if args.dataset_dir:
        # Sharded mode: data_loader will memory-map shards on demand
        data_loader = GPTDataLoader(model.config, model.tokenizer, dataset_dir=args.dataset_dir)
        
        # Get total tokens from metadata
        import json
        import os
        metadata_path = os.path.join(args.dataset_dir, "dataset_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            total_tokens = metadata.get('total_tokens', 0)
        else:
            total_tokens = None
    else:
        # In-memory mode: load and tokenize entire text
        data_loader = GPTDataLoader(model.config, model.tokenizer)
        data_loader.prepare_data(text)
        total_tokens = len(data_loader.train_data) + len(data_loader.val_data)
    
    # Analyze dataset coverage
    if total_tokens:
        coverage = calculate_dataset_coverage(
            max_iters=args.max_iters,
            batch_size=model.config.batch_size,
            block_size=model.config.block_size,
            total_tokens=total_tokens
        )
        print_coverage_analysis(coverage, args.max_iters)
        
        # Ask user to confirm if coverage is very low
        if coverage['coverage_ratio'] < 1.0:
            print("⚠️  Your model will not see the entire dataset!")
            response = input("Continue anyway? (y/n): ").lower().strip()
            if response != 'y' and response != 'yes':
                print("Training cancelled. Adjust --max_iters and try again.")
                return
            print()
    
    # Train the model (model trains itself!)
    print("========== Starting Training ==========")
    print(f"Training from step {start_step} to {args.max_iters}")
    print(f"Checkpoints: {ckpt_manager.checkpoint_dir}")
    print(f"Format: JSON-based (model.pt + config.json + tokenizer.json + ...)")
    print(f"Evaluation every {args.eval_interval} steps")
    print()
    
    model.fit(
        data_loader=data_loader,
        optimizer=optimizer,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_dir=ckpt_manager.checkpoint_dir,
        start_step=start_step,
        learning_rate=args.learning_rate
    )
    
    print("\n========== Training Complete ==========")
    print(f"✅ Model saved to: {ckpt_manager.checkpoint_dir}")
    print()
    print("Checkpoint files:")
    print("  📄 model.pt           - Model weights")
    print("  📄 config.json        - Architecture configuration")
    print("  📄 tokenizer.json     - Vocabulary")
    print("  📄 training_state.json - Training progress")
    print("  📄 optimizer.pt       - Optimizer state")
    print()
    print("Next steps:")
    print(f"  Generate: python generate.py --model_name {args.model_name} --prompt 'Your prompt'")
    print(f"  Inspect:  python scripts/inspect_model.py --model_name {args.model_name}")


if __name__ == "__main__":
    main()
