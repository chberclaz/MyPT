# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import argparse
from core.model import GPT, GPTConfig
from core.tokenizer import Tokenizer
from core.data_loader import GPTDataLoader
from core.checkpoint import CheckpointManager


def parse_args():
    """Parse command-line arguments for training"""
    parser = argparse.ArgumentParser(description="Train a GPT model")
    
    # Model/data config
    parser.add_argument("--model_name", type=str, default="default",
                        help="Name of the model/checkpoint set (e.g. dante, shakespeare)")
    parser.add_argument("--input_file", type=str, default="input_dante.txt",
                        help="Path to training text file")
    parser.add_argument("--tokenization", type=str, default="gpt2",
                        choices=["gpt2", "char"],
                        help="Tokenizer type: gpt2 or char")
    parser.add_argument("--init_from_model", type=str, default=None,
                        help="Optional: model_name to initialize weights from (e.g. dante_base)")
    
    # Training hyperparameters
    parser.add_argument("--max_iters", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--eval_interval", type=int, default=50,
                        help="Evaluate every N steps")
    parser.add_argument("--eval_iters", type=int, default=200,
                        help="Number of iterations for evaluation")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for optimizer")
    
    # Model architecture
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
    
    # Initial config (may be overridden if resuming from checkpoint)
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
    
    # Read training text
    print("Loading training data...")
    text = GPTDataLoader.read_text(args.input_file)
    print(f"Loaded {len(text)} characters")
    
    # Initialize model (handles resume / init_from / fresh)
    print("\nInitializing model...")
    ckpt_manager = CheckpointManager(args.model_name)
    model, optimizer, start_step = ckpt_manager.initialize_for_training(
        config=config,
        tokenization=args.tokenization,
        input_text=text,
        learning_rate=args.learning_rate,
        init_from_model=args.init_from_model
    )
    
    print(f"Model configuration:")
    print(model.config)
    print()
    
    # Prepare data
    print("Preparing data...")
    data_loader = GPTDataLoader(model.config, model.tokenizer)
    data_loader.prepare_data(text)
    
    # Train the model (model trains itself!)
    print("\n========== Starting Training ==========")
    model.fit(
        data_loader=data_loader,
        optimizer=optimizer,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        checkpoint_dir=ckpt_manager.checkpoint_dir,
        start_step=start_step
    )
    
    print("\n========== Training Complete ==========")


if __name__ == "__main__":
    main()
