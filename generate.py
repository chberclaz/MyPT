import argparse
from core.checkpoint import CheckpointManager
from generator import Generator


def parse_args():
    """Parse command-line arguments for generation"""
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model")
    
    parser.add_argument("--model_name", type=str, default="default",
                        help="Name of the model/checkpoint set to load (e.g. dante, shakespeare)")
    parser.add_argument("--checkpoint", type=str, default="latest.pt",
                        help="Which checkpoint file to load (e.g. final.pt or latest.pt)")
    parser.add_argument("--prompt", type=str, default="Die Nacht",
                        help="Prompt text to start generation from")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "qa"],
                        help="Generation mode: basic (text completion) or qa (question answering)")
    
    return parser.parse_args()


def main():
    """Main generation entry point"""
    args = parse_args()
    
    print(f"========== Generation Configuration ==========")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Mode: {args.mode}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print()
    
    # Load model
    print(f"Loading model '{args.model_name}' from {args.checkpoint}...")
    model = CheckpointManager.load_for_inference(args.model_name, args.checkpoint)
    
    # Print model info
    print(f"Model loaded successfully!")
    print(f"Tokenizer: {getattr(model.tokenizer, 'token_kind', 'unknown')}")
    print(f"Vocab size: {model.config.vocab_size}")
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
    if args.mode == "basic":
        print("\n========== Q&A Example ==========")
        question = "How far is the moon?"
        print(f"Question: {question}")
        answer = gen.generate_qa(question, 150)
        print(f"Answer: {answer}")
        print()


if __name__ == "__main__":
    main()
