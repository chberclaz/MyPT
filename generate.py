import argparse
from core import load_model, get_model_info, Generator


def parse_args():
    """Parse command-line arguments for generation"""
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model")
    
    parser.add_argument("--model_name", type=str, default="default",
                        help="Name of the model/checkpoint set to load (e.g. dante, shakespeare)")
    parser.add_argument("--legacy_checkpoint", type=str, default=None,
                        help="Optional: specific legacy checkpoint file (e.g. final.pt, latest.pt). "
                             "If not specified, will auto-detect new JSON format or legacy format.")
    parser.add_argument("--prompt", type=str, default="Die Nacht",
                        help="Prompt text to start generation from")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "qa"],
                        help="Generation mode: basic (text completion) or qa (question answering)")
    parser.add_argument("--show_info", action="store_true",
                        help="Show model info before generating")
    
    return parser.parse_args()


def main():
    """Main generation entry point"""
    args = parse_args()
    
    print(f"========== Generation Configuration ==========")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()
    
    # Optional: Show model info without loading (fast preview)
    if args.show_info:
        print("========== Model Info ==========")
        try:
            info = get_model_info(args.model_name)
            print(f"Format: {info.get('format', 'unknown')}")
            if 'config' in info:
                cfg = info['config']
                print(f"Layers: {cfg.get('n_layer', '?')}")
                print(f"Embedding dim: {cfg.get('n_embd', '?')}")
                print(f"Vocab size: {cfg.get('vocab_size', '?')}")
            if 'tokenizer' in info:
                print(f"Tokenizer: {info['tokenizer'].get('token_kind', 'unknown')}")
            if 'training_state' in info and info['training_state'].get('step'):
                print(f"Training step: {info['training_state']['step']}")
            print()
        except Exception as e:
            print(f"Could not load model info: {e}")
            print()
    
    # Load model using convenience function
    print(f"Loading model '{args.model_name}'...")
    
    # Handle legacy checkpoint parameter if specified
    if args.legacy_checkpoint:
        from core.checkpoint import CheckpointManager
        model = CheckpointManager.load_for_inference(
            args.model_name,
            legacy_filename=args.legacy_checkpoint
        )
    else:
        # Use convenience function (cleaner API)
        model = load_model(args.model_name)
    
    # Print model info
    print(f"✅ Model loaded successfully!")
    print(f"Tokenizer: {getattr(model.tokenizer, 'token_kind', 'unknown')}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Device: {model.config.device}")
    print()
    
    # Create generator
    gen = Generator(model)
    
    # Generate based on mode
    print(f"========== Generating ==========")
    
    if args.mode == "basic":
        print(f"Prompt: {args.prompt}\n")
        output = gen.generate(args.prompt, args.max_new_tokens)
        print(output)
    
    elif args.mode == "qa":
        question = args.prompt  # Use prompt as question
        print(f"Question: {question}\n")
        answer = gen.generate_qa(question, args.max_new_tokens)
        print(f"Answer: {answer}")
    
    print("\n========== Generation Complete ==========")
    
    # Optional: Additional Q&A example (can be removed or made conditional)
    if args.mode == "basic" and model.tokenizer.token_kind != "char":
        print("\n========== Q&A Example ==========")
        question = "How far is the moon?"
        print(f"Question: {question}")
        answer = gen.generate_qa(question, 150)
        print(f"Answer: {answer}")
        print()


if __name__ == "__main__":
    main()
