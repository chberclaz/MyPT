import argparse
import codecs
from core import load_model, get_model_info, Generator, banner_generate


def process_prompt(prompt: str) -> str:
    """
    Process prompt string to handle escape sequences.
    
    Converts literal \\n to actual newlines, \\t to tabs, etc.
    This is important for command-line prompts containing special characters.
    """
    # Use codecs.decode to handle common escape sequences
    # This converts \n -> newline, \t -> tab, \\ -> \, etc.
    try:
        return codecs.decode(prompt, 'unicode_escape')
    except UnicodeDecodeError:
        # If decode fails (e.g., malformed escapes), return original
        return prompt


def parse_args():
    """Parse command-line arguments for generation"""
    parser = argparse.ArgumentParser(description="Generate text using a trained GPT model")
    
    # Model selection
    parser.add_argument("--model_name", type=str, default="default",
                        help="Name of the model/checkpoint set to load (e.g. dante, shakespeare)")
    parser.add_argument("--legacy_checkpoint", type=str, default=None,
                        help="Optional: specific legacy checkpoint file (e.g. final.pt, latest.pt). "
                             "If not specified, will auto-detect new JSON format or legacy format.")
    
    # Basic generation options
    parser.add_argument("--prompt", type=str, default="Die Nacht",
                        help="Prompt text to start generation from")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--mode", type=str, default="basic",
                        choices=["basic", "qa", "creative", "factual", "code"],
                        help="Generation mode/preset")
    parser.add_argument("--show_info", action="store_true",
                        help="Show model info before generating")
    
    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.0=deterministic, 1.0=neutral, >1.0=creative)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K sampling (0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-P/nucleus sampling threshold (1.0=disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty (1.0=disabled, >1.0=penalize repeats)")
    
    # Presets override individual parameters
    parser.add_argument("--preset", type=str, default=None,
                        choices=["chat", "creative", "factual", "code", "deterministic"],
                        help="Use a preset configuration (overrides individual sampling params)")
    
    return parser.parse_args()


# Preset configurations
PRESETS = {
    "chat": {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.1},
    "creative": {"temperature": 1.0, "top_k": 100, "top_p": 0.95, "repetition_penalty": 1.2},
    "factual": {"temperature": 0.3, "top_k": 10, "top_p": 0.8, "repetition_penalty": 1.0},
    "code": {"temperature": 0.2, "top_k": 40, "top_p": 0.95, "repetition_penalty": 1.0},
    "deterministic": {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "repetition_penalty": 1.0},
}


def main():
    """Main generation entry point"""
    banner_generate()
    
    args = parse_args()
    
    # Apply preset if specified
    if args.preset:
        preset = PRESETS[args.preset]
        args.temperature = preset["temperature"]
        args.top_k = preset["top_k"]
        args.top_p = preset["top_p"]
        args.repetition_penalty = preset["repetition_penalty"]
    
    print(f"========== Generation Configuration ==========")
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-K: {args.top_k}")
    print(f"Top-P: {args.top_p}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    if args.preset:
        print(f"Preset: {args.preset}")
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
    
    # Process prompt to handle escape sequences (e.g., \n -> actual newlines)
    args.prompt = process_prompt(args.prompt)
    
    # Generate based on mode
    print(f"========== Generating ==========")
    
    if args.mode == "basic":
        print(f"Prompt: {args.prompt}\n")
        output = gen.generate(
            args.prompt, 
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        print(output)
    
    elif args.mode == "creative":
        print(f"Prompt: {args.prompt}\n")
        output = gen.generate_creative(args.prompt, args.max_new_tokens)
        print(output)
    
    elif args.mode == "factual":
        print(f"Prompt: {args.prompt}\n")
        output = gen.generate_factual(args.prompt, args.max_new_tokens)
        print(output)
    
    elif args.mode == "code":
        print(f"Prompt: {args.prompt}\n")
        output = gen.generate_code(args.prompt, args.max_new_tokens)
        print(output)
    
    elif args.mode == "qa":
        question = args.prompt  # Use prompt as question
        print(f"Question: {question}\n")
        # Q&A uses factual settings
        answer = gen.generate_factual(question, args.max_new_tokens)
        print(f"Answer: {answer}")
    
    print("\n========== Generation Complete ==========")


if __name__ == "__main__":
    main()
