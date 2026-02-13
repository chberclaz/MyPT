# prerequisites
# pytorch needs to be installed
# choco needs to be installed to install python
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
import argparse
import torch
import torch.nn as nn
from core import (
    GPTConfig, 
    GPTDataLoader,
    GPTEpisodeDataLoader,
    is_episode_indexed_dataset,
    CheckpointManager, 
    get_model_info,
    calculate_dataset_coverage,
    print_coverage_analysis,
    calculate_episode_coverage,
    print_episode_coverage_analysis,
    banner_train,
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
    parser.add_argument("--eval_dataset_dir", type=str, default=None,
                        help="Additional eval dataset for dual evaluation (e.g., general eval during domain adaptation)")
    
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
    parser.add_argument("--warmup_iters", type=float, default=0,
                        help="Learning rate warmup. Int for absolute steps (e.g., 1000) or "
                             "float 0-1 for fraction of max_iters (e.g., 0.05 for 5%%). "
                             "Recommended: 0.05-0.10 for large models (750M+)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm (default: 1.0). Set to 0 to disable.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay for AdamW optimizer (default: 0.1)")
    parser.add_argument("--use_amp", type=bool, default=True,
                        help="Enable automatic mixed precision (default: True)")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"],
                        help="AMP dtype: bf16 for A100/H100, fp16 for older GPUs (default: bf16)")
    
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
    parser.add_argument("--save_dtype", type=str, default=None,
                        choices=["fp32", "float32", "bf16", "bfloat16", "fp16", "float16"],
                        help="Dtype for saving checkpoints (default: use model dtype)")
    
    # Eval-time generation (for SFT training verification)
    parser.add_argument("--eval_prompts_file", type=str, default=None,
                        help="JSON file with eval prompts for training-time inference testing")
    parser.add_argument("--eval_max_new_tokens", type=int, default=64,
                        help="Max tokens to generate during eval (default: 64)")
    
    return parser.parse_args()


def main():
    """Main training entry point"""
    banner_train()
    
    args = parse_args()
    
    # Load config from file or use CLI arguments
    # Training hyperparameters from config file (can be overridden by CLI)
    config_training = {}
    
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
        
        # Extract training hyperparameters (not part of GPTConfig)
        training_keys = ["learning_rate", "max_iters", "eval_interval", "eval_iters", "warmup_iters", "grad_clip", "weight_decay", "use_amp", "amp_dtype", "eval_sets", "eval_seed", "log_file", "freeze_layers", "freeze_embeddings", "curriculum", "grad_accum_steps"]
        for key in training_keys:
            if key in config_dict:
                config_training[key] = config_dict.pop(key)
        
        # Create config from file
        # Set device if not in config
        if "device" not in config_dict:
            import torch
            config_dict["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Allow CLI override for save_dtype
        if args.save_dtype is not None:
            config_dict["save_dtype"] = args.save_dtype
        
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
            vocab_size=args.vocab_size,  # Will be adjusted for char-level if needed
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
            bias=args.bias,
            save_dtype=args.save_dtype,
        )
    
    # Resolve training hyperparameters: CLI overrides config file
    # Use argparse defaults to detect if user explicitly set a value
    parser_defaults = {'learning_rate': 3e-4, 'max_iters': 1000, 'eval_interval': 50, 
                       'eval_iters': 200, 'warmup_iters': 0, 'grad_clip': 1.0, 'weight_decay': 0.1,
                       'use_amp': True, 'amp_dtype': 'bf16'}
    
    def get_effective_value(arg_name, arg_value, config_training, defaults):
        """Get effective value: CLI override > config file > default"""
        if arg_value != defaults.get(arg_name):
            # User explicitly set this via CLI
            return arg_value
        elif arg_name in config_training:
            # Use config file value
            return config_training[arg_name]
        else:
            # Use CLI/default value
            return arg_value
    
    effective_learning_rate = get_effective_value('learning_rate', args.learning_rate, config_training, parser_defaults)
    effective_max_iters = get_effective_value('max_iters', args.max_iters, config_training, parser_defaults)
    effective_eval_interval = get_effective_value('eval_interval', args.eval_interval, config_training, parser_defaults)
    effective_eval_iters = get_effective_value('eval_iters', args.eval_iters, config_training, parser_defaults)
    effective_warmup_iters = get_effective_value('warmup_iters', args.warmup_iters, config_training, parser_defaults)
    effective_grad_clip = get_effective_value('grad_clip', args.grad_clip, config_training, parser_defaults)
    effective_weight_decay = get_effective_value('weight_decay', args.weight_decay, config_training, parser_defaults)
    effective_use_amp = get_effective_value('use_amp', args.use_amp, config_training, parser_defaults)
    effective_amp_dtype = get_effective_value('amp_dtype', args.amp_dtype, config_training, parser_defaults)
    effective_grad_accum_steps = int(config_training.get('grad_accum_steps', 1))
    
    # Auto-adjust amp_dtype based on GPU capability (if user didn't explicitly override)
    if effective_use_amp and args.amp_dtype is None and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8 and effective_amp_dtype == 'bf16':
            # Older GPU doesn't support native bf16, switch to fp16
            effective_amp_dtype = 'fp16'
            print(f"[Auto] GPU compute {capability[0]}.{capability[1]} - switching AMP from bf16 to fp16")
    
    print(f"========== Training Configuration ==========")
    print(f"Model name: {args.model_name}")
    print(f"Input file: {args.input_file}")
    print(f"Tokenization: {args.tokenization}")
    print(f"Max iterations: {effective_max_iters}")
    print(f"Learning rate: {effective_learning_rate}")
    print(f"Weight decay: {effective_weight_decay}")
    print(f"Gradient clip: {effective_grad_clip}")
    print(f"Warmup: {effective_warmup_iters}")
    print(f"Mixed precision: {effective_amp_dtype.upper() if effective_use_amp else 'disabled'}")
    if effective_grad_accum_steps > 1:
        print(f"Gradient accumulation: {effective_grad_accum_steps} micro-steps")
    if config_training.get('freeze_layers', 0) > 0:
        print(f"Freeze layers: {config_training['freeze_layers']} (of {config_training.get('n_layer', '?')})")
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
    dataset_tokenizer_state = None
    
    if args.input_file:
        print("Loading training data (in-memory mode)...")
        text = GPTDataLoader.read_text(args.input_file)
        print(f"Loaded {len(text):,} characters")
        print(f"Approximate tokens (char-level): {len(text):,}")
        print(f"Approximate tokens (GPT-2): ~{len(text)//4:,}")
    elif args.dataset_dir:
        # For sharded mode, load tokenizer state from dataset directory
        import json
        import os
        tokenizer_state_path = os.path.join(args.dataset_dir, "tokenizer_state.json")
        if os.path.exists(tokenizer_state_path):
            print(f"Loading tokenizer state from {args.dataset_dir}...")
            with open(tokenizer_state_path, 'r') as f:
                dataset_tokenizer_state = json.load(f)
            print(f"Tokenizer type: {dataset_tokenizer_state.get('token_kind', 'unknown')}")
        else:
            raise FileNotFoundError(
                f"Tokenizer state not found in dataset directory: {tokenizer_state_path}\n"
                f"Dataset must contain tokenizer_state.json created by prepare_dataset.py"
            )
    
    # Initialize model (handles resume / init_from / fresh)
    # Layer freezing is applied inside initialize_for_training, BEFORE the
    # optimizer is created, so saved optimizer state can be restored on resume.
    effective_freeze_layers = config_training.get('freeze_layers', 0)
    effective_freeze_embeddings = config_training.get('freeze_embeddings', False)
    
    print("\nInitializing model...")
    model, optimizer, start_step = ckpt_manager.initialize_for_training(
        config=config,
        tokenization=args.tokenization,
        input_text=text,
        learning_rate=effective_learning_rate,
        init_from_model=args.init_from_model,
        weight_decay=effective_weight_decay,
        dataset_tokenizer_state=dataset_tokenizer_state,
        freeze_layers=effective_freeze_layers,
        freeze_embeddings=effective_freeze_embeddings,
    )
    model = model.to(config.device)
    
    # ---- Layer Freezing Report ----
    if effective_freeze_layers > 0 or effective_freeze_embeddings:
        n_layer = model.config.n_layer
        effective_freeze_layers = min(effective_freeze_layers, n_layer)
        
        # Warn about tie_weights + freeze_embeddings conflict
        if effective_freeze_embeddings and getattr(model.config, 'tie_weights', False):
            print("  WARNING: tie_weights=True and freeze_embeddings=True means lm_head is also frozen!")
            print("  Consider setting freeze_embeddings=False with tie_weights=True.")
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        frozen = total - trainable
        
        print(f"\n========== Layer Freezing ==========")
        print(f"Frozen blocks: {effective_freeze_layers}/{n_layer} (blocks 0-{effective_freeze_layers - 1})")
        print(f"Trainable blocks: {n_layer - effective_freeze_layers} (blocks {effective_freeze_layers}-{n_layer - 1})")
        if effective_freeze_embeddings:
            print(f"Embeddings: FROZEN")
        else:
            print(f"Embeddings: trainable")
        print(f"ln_f + lm_head: trainable")
        print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
        print(f"Frozen: {frozen:,} ({100*frozen/total:.1f}%)")
        print()
    
    # Smart dtype selection based on GPU capability
    # - A100/H100 (compute 8.0+): Native bf16 support, use bf16 throughout
    # - RTX 30/40 (compute 8.6): bf16 supported, same pattern works
    # - RTX 20/GTX (compute < 8.0): Keep fp32 weights, rely on autocast for mixed precision
    training_dtype = 'fp32'  # default
    
    if torch.cuda.is_available() and 'cuda' in str(config.device):
        capability = torch.cuda.get_device_capability()
        compute_capability = capability[0] + capability[1] / 10
        
        if compute_capability >= 8.0:
            # Modern GPU with native bf16 support (A100, H100, RTX 30/40)
            training_dtype = 'bf16'
            print(f"GPU compute {capability[0]}.{capability[1]}: Using bf16 weights")
        elif compute_capability >= 7.0:
            # Older GPU (RTX 20 series, GTX 16 series) - use fp16 via autocast only
            training_dtype = 'fp32'
            print(f"GPU compute {capability[0]}.{capability[1]}: Using fp32 weights + fp16 autocast")
            print("  (Weights stay fp32, autocast handles mixed precision during forward/backward)")
        else:
            # Very old GPU - fp32 only
            training_dtype = 'fp32'
            print(f"GPU compute {capability[0]}.{capability[1]}: Using fp32 (no mixed precision)")
    else:
        print("CPU training: Using fp32")
    
    # Apply dtype conversion
    if training_dtype == 'bf16':
        model = model.to(dtype=torch.bfloat16)
    
    print(f"Model weights dtype: {next(model.parameters()).dtype}")
    
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
        print(f"Dataset directory: {args.dataset_dir}")
        print(f"Using loss masking: {model.config.use_loss_mask}")
        
        # Auto-detect dataset format: episode-indexed vs token-stream
        if is_episode_indexed_dataset(args.dataset_dir):
            # Episode-indexed format (SFT with episode boundaries)
            print(f"Detected: Episode-indexed dataset (SFT mode)")
            print(f"Sampling mode: {model.config.batch_sampling_mode}")
            print(f"Epoch seed: {model.config.epoch_seed} (for reproducibility)")
            # Config is the single source of truth for all episode loader parameters
            data_loader = GPTEpisodeDataLoader(
                model.config,
                model.tokenizer,
                dataset_dir=args.dataset_dir,
            )
        else:
            # Token-stream format (pre-training or legacy sharded)
            print(f"Detected: Token-stream dataset (sharded mode)")
            data_loader = GPTDataLoader(
                model.config, 
                model.tokenizer, 
                dataset_dir=args.dataset_dir,
                use_loss_mask=model.config.use_loss_mask
            )
        
        # Get dataset info from metadata
        import json
        import os
        metadata_path = os.path.join(args.dataset_dir, "dataset_metadata.json")
        total_tokens = None
        total_episodes = None
        is_episode_dataset = is_episode_indexed_dataset(args.dataset_dir)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            # Support both old and new metadata fields
            total_tokens = metadata.get('num_train_tokens', metadata.get('total_tokens', 0))
            if metadata.get('num_val_tokens'):
                total_tokens += metadata.get('num_val_tokens', 0)
            # Get episode counts for episode-indexed datasets
            if is_episode_dataset:
                total_episodes = metadata.get('num_train_episodes', 0)
        
        # For episode datasets, also try to get count from dataloader
        if is_episode_dataset and hasattr(data_loader, 'get_epoch_info'):
            epoch_info = data_loader.get_epoch_info('train')
            total_episodes = epoch_info.get('episodes_total', total_episodes)
    else:
        # In-memory mode: load and tokenize entire text
        is_episode_dataset = False
        total_episodes = None
        data_loader = GPTDataLoader(
            model.config, 
            model.tokenizer,
            use_loss_mask=model.config.use_loss_mask
        )
        data_loader.prepare_data(text)
        total_tokens = len(data_loader.train_data) + len(data_loader.val_data)
    
    # Analyze dataset coverage (different calculation for episode vs token datasets)
    if is_episode_dataset and total_episodes:
        # Episode-indexed dataset: coverage by episodes
        coverage = calculate_episode_coverage(
            max_iters=effective_max_iters,
            batch_size=model.config.batch_size,
            total_episodes=total_episodes
        )
        print_episode_coverage_analysis(coverage, effective_max_iters)
        
        # Ask user to confirm if coverage is very low
        if coverage['coverage_ratio'] < 1.0:
            print("⚠️  Your model will not see all episodes!")
            response = input("Continue anyway? (y/n): ").lower().strip()
            if response != 'y' and response != 'yes':
                print("Training cancelled. Adjust --max_iters and try again.")
                return
            print()
    elif total_tokens:
        # Token-stream dataset: coverage by tokens
        coverage = calculate_dataset_coverage(
            max_iters=effective_max_iters,
            batch_size=model.config.batch_size,
            block_size=model.config.block_size,
            total_tokens=total_tokens
        )
        print_coverage_analysis(coverage, effective_max_iters)
        
        # Ask user to confirm if coverage is very low
        if coverage['coverage_ratio'] < 1.0:
            print("⚠️  Your model will not see the entire dataset!")
            response = input("Continue anyway? (y/n): ").lower().strip()
            if response != 'y' and response != 'yes':
                print("Training cancelled. Adjust --max_iters and try again.")
                return
            print()
    
    # Build curriculum data loader schedule if specified in config
    data_loader_schedule = None
    curriculum_config = config_training.get('curriculum', None)
    
    if curriculum_config:
        import os as _os
        phases = curriculum_config.get('phases', [])
        if len(phases) >= 2:
            print()
            print("========== Curriculum Training ==========")
            print(f"Phases: {len(phases)}")
            
            data_loader_schedule = []
            for i, phase in enumerate(phases):
                phase_name = phase['name']
                phase_dir = phase['dataset_dir']
                until_iter = phase.get('until_iter', None)
                desc = phase.get('description', '')
                
                print(f"  Phase {i+1}: '{phase_name}'")
                print(f"    Dataset: {phase_dir}")
                if until_iter is not None:
                    print(f"    Until iter: {until_iter:,}")
                else:
                    print(f"    Until iter: end of training")
                if desc:
                    print(f"    {desc}")
                
                # Build a data loader for each phase
                if not _os.path.exists(phase_dir):
                    print(f"    [WARN] Dataset dir not found: {phase_dir}")
                    print(f"    Will fail at training time if not created before then.")
                
                phase_loader = GPTDataLoader(
                    model.config,
                    model.tokenizer,
                    dataset_dir=phase_dir,
                    use_loss_mask=model.config.use_loss_mask
                )
                
                # Schedule entry: (switch_at_iter, loader, phase_name)
                # Phase 0 starts at iter 0 (or start_step on resume)
                # Phase 1 starts at until_iter of phase 0, etc.
                if i == 0:
                    # First phase: use as the initial data_loader
                    data_loader = phase_loader
                    switch_iter = until_iter if until_iter is not None else effective_max_iters
                else:
                    # Subsequent phases: schedule the switch
                    data_loader_schedule.append((prev_switch_iter, phase_loader, phase_name))
                
                prev_switch_iter = until_iter if until_iter is not None else effective_max_iters
            
            # Handle resume: if start_step is past a phase boundary, 
            # use the correct phase's loader from the start
            if start_step > 0 and data_loader_schedule:
                for switch_iter, switch_loader, phase_name in data_loader_schedule:
                    if start_step >= switch_iter:
                        data_loader = switch_loader
                        print(f"\n  [RESUME] start_step={start_step} >= switch at {switch_iter}")
                        print(f"  Using '{phase_name}' data loader from the start")
            
            print()
    
    # Determine initial curriculum phase name
    initial_phase_name = None
    if curriculum_config:
        phases = curriculum_config.get('phases', [])
        if phases:
            initial_phase_name = phases[0]['name']
            # If resuming past a phase boundary, find the correct phase
            if start_step > 0:
                for phase in phases:
                    until = phase.get('until_iter')
                    if until is None or start_step < until:
                        initial_phase_name = phase['name']
                        break
    
    # Train the model (model trains itself!)
    print("========== Starting Training ==========")
    print(f"Training from step {start_step} to {effective_max_iters}")
    print(f"Checkpoints: {ckpt_manager.checkpoint_dir}")
    print(f"Format: JSON-based (model.pt + config.json + tokenizer.json + ...)")
    print(f"Evaluation every {effective_eval_interval} steps")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if effective_warmup_iters > 0:
        if isinstance(effective_warmup_iters, float) and effective_warmup_iters < 1:
            print(f"LR warmup: {effective_warmup_iters*100:.0f}% of training ({int(effective_warmup_iters * effective_max_iters)} steps)")
        else:
            print(f"LR warmup: {int(effective_warmup_iters)} steps")
    
    # Setup additional eval data loaders if eval_sets specified in config or CLI
    eval_data_loaders = None
    eval_sets_config = config_training.get('eval_sets', None)
    
    # Merge CLI --eval_dataset_dir with config eval_sets
    if args.eval_dataset_dir:
        if eval_sets_config is None:
            eval_sets_config = {}
        # CLI arg adds a "general" eval set (common use case for domain adaptation)
        eval_sets_config["general"] = args.eval_dataset_dir
    
    if eval_sets_config is not None:
        print()
        print("========== Additional Eval Sets ==========")
        eval_data_loaders = {}
        for eval_name, eval_path in eval_sets_config.items():
            print(f"Loading eval set '{eval_name}' from: {eval_path}")
            # Auto-detect dataset format for eval sets too
            if is_episode_indexed_dataset(eval_path):
                eval_loader = GPTEpisodeDataLoader(
                    model.config,
                    model.tokenizer,
                    dataset_dir=eval_path,
                )
            else:
                eval_loader = GPTDataLoader(
                    model.config,
                    model.tokenizer,
                    dataset_dir=eval_path,
                    eval_only=True
                )
            eval_data_loaders[eval_name] = eval_loader
        print()
    
    # Get optional eval_seed and log_file from config
    eval_seed = config_training.get('eval_seed', None)
    log_file = config_training.get('log_file', None)
    if log_file is not None:
        print(f"Training log: {log_file}")
    if eval_seed is not None:
        print(f"Eval RNG seed: {eval_seed}")
    
    print()
    print("Model param dtype:", next(model.parameters()).dtype)
    model.fit(
        data_loader=data_loader,
        optimizer=optimizer,
        max_iters=effective_max_iters,
        eval_interval=effective_eval_interval,
        eval_iters=effective_eval_iters,
        checkpoint_dir=ckpt_manager.checkpoint_dir,
        start_step=start_step,
        learning_rate=effective_learning_rate,
        warmup_iters=effective_warmup_iters,
        grad_clip=effective_grad_clip,
        use_amp=effective_use_amp,
        amp_dtype=effective_amp_dtype,
        eval_data_loaders=eval_data_loaders,
        log_file=log_file,
        eval_seed=eval_seed,
        config_file=args.config_file,
        dataset_dir=args.dataset_dir,
        eval_prompts_file=args.eval_prompts_file,
        eval_max_new_tokens=args.eval_max_new_tokens,
        data_loader_schedule=data_loader_schedule,
        grad_accum_steps=effective_grad_accum_steps,
        initial_phase=initial_phase_name,
    )
    
    print("\n========== Training Complete ==========")
    print(f"✅ Model saved to: {ckpt_manager.checkpoint_dir}")
    print()
    print("Checkpoint files:")
    print("  📄 model.pt            - Model weights")
    print("  📄 config.json         - Architecture configuration")
    print("  📄 tokenizer.json      - Vocabulary")
    print("  📄 training_state.json - Training progress + hyperparameters + provenance")
    print("  📄 optimizer.pt        - Optimizer state")
    if args.config_file:
        print("  📄 original_config.json - Copy of training config file")
    print()
    print("Next steps:")
    print(f"  Generate: python generate.py --model_name {args.model_name} --prompt 'Your prompt'")
    print(f"  Inspect:  python scripts/model/inspect_model.py --model_name {args.model_name}")


if __name__ == "__main__":
    main()
