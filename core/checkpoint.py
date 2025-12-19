import os
import torch
from .model import GPT, GPTConfig
from .tokenizer import Tokenizer

class CheckpointManager:
    """
    Manages checkpoint paths and model initialization 
    (resume / init_from / fresh).
    
    Supports both new JSON-based format and legacy single-file format.
    """
    def __init__(self, model_name, base_dir="checkpoints"):
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(base_dir, model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def get_path(self, filename="model.pt"):
        """Get full path to a checkpoint file"""
        return os.path.join(self.checkpoint_dir, filename)
    
    def exists_new_format(self):
        """
        Check if checkpoint exists in new JSON-based format.
        Required files: config.json, model.pt
        """
        config_path = os.path.join(self.checkpoint_dir, "config.json")
        model_path = os.path.join(self.checkpoint_dir, "model.pt")
        return os.path.exists(config_path) and os.path.exists(model_path)
    
    def exists_legacy_format(self, filename="latest.pt"):
        """Check if checkpoint exists in legacy single-file format"""
        return os.path.exists(self.get_path(filename))
    
    def exists(self):
        """Check if any checkpoint format exists"""
        return self.exists_new_format() or self.exists_legacy_format()
    
    def initialize_for_training(self, config, tokenization, input_text, 
                                learning_rate, init_from_model=None,
                                dataset_tokenizer_state=None):
        """
        Initialize model for training. Handles 3 cases:
        1. Resume from this model's checkpoint
        2. Initialize from another model (fine-tuning)
        3. Create fresh model
        
        Supports both new JSON-based format and legacy single-file format.
        
        Args:
            config: GPTConfig instance
            tokenization: 'gpt2' or 'char'
            input_text: Training text (for char vocab building, in-memory mode)
            learning_rate: Learning rate for optimizer
            init_from_model: Optional model name to initialize from
            dataset_tokenizer_state: Optional tokenizer state from dataset directory (sharded mode)
        
        Returns: (model, optimizer, start_step)
        """
        device = config.device
        
        # Case 1: RESUME FROM THIS MODEL'S CHECKPOINT
        if self.exists_new_format():
            print(f"Found checkpoint in new format at {self.checkpoint_dir}, resuming training.")
            model, _, start_step, optim_state = GPT.load(self.checkpoint_dir, map_location=device)
            optimizer = model.configure_optimizer(learning_rate, optim_state)
            return model, optimizer, start_step or 0
        
        elif self.exists_legacy_format("latest.pt"):
            print(f"Found legacy checkpoint, resuming training.")
            print("Note: Will save in new JSON format on next checkpoint.")
            legacy_path = self.get_path("latest.pt")
            model, _, start_step, optim_state = GPT.load_legacy(legacy_path, map_location=device)
            optimizer = model.configure_optimizer(learning_rate, optim_state)
            return model, optimizer, start_step or 0
        
        # Case 2: INIT FROM ANOTHER MODEL (Fine-tuning)
        elif init_from_model is not None:
            init_manager = CheckpointManager(init_from_model)
            
            # Try new format first
            if init_manager.exists_new_format():
                print(f"Initializing from base model '{init_from_model}' (new format)")
                model, base_tok_state, _, _ = GPT.load(init_manager.checkpoint_dir, map_location=device)
            
            # Fall back to legacy format
            elif init_manager.exists_legacy_format("final.pt"):
                print(f"Initializing from base model '{init_from_model}' (legacy format)")
                legacy_path = init_manager.get_path("final.pt")
                model, base_tok_state, _, _ = GPT.load_legacy(legacy_path, map_location=device)
            
            else:
                raise FileNotFoundError(
                    f"--init_from_model '{init_from_model}' specified, "
                    f"but no checkpoint found at {init_manager.checkpoint_dir}"
                )
            
            # Validate tokenization compatibility
            self._validate_tokenization(base_tok_state, tokenization, init_from_model, model, config)
            
            # Update model config with new training parameters
            # Keep architecture params (n_layer, n_embd, etc.) from base model
            # Update mutable training params from new config
            print(f"Updating model config with new training parameters...")
            model.config.batch_size = config.batch_size
            model.config.dropout = config.dropout
            model.config.use_loss_mask = config.use_loss_mask
            model.config.device = config.device
            
            # Update dropout in model layers if it changed
            if hasattr(model, 'blocks'):
                for block in model.blocks:
                    if hasattr(block, 'sa') and hasattr(block.sa, 'dropout'):
                        block.sa.dropout.p = config.dropout
                    if hasattr(block, 'fwd') and hasattr(block.fwd, 'net'):
                        for layer in block.fwd.net:
                            if isinstance(layer, torch.nn.Dropout):
                                layer.p = config.dropout
            
            print(f"  use_loss_mask: {model.config.use_loss_mask}")
            print(f"  dropout: {model.config.dropout}")
            print(f"  batch_size: {model.config.batch_size}")
            
            optimizer = model.configure_optimizer(learning_rate)
            return model, optimizer, 0  # start from step 0 for new training
        
        # Case 3: FRESH MODEL
        else:
            print("No checkpoint found. Starting from scratch.")
            print(f"Using config: {config}")
            
            # Build tokenizer
            tokenizer = Tokenizer(config, tokenization)
            
            # Load from dataset tokenizer state (sharded mode) or build from input_text (in-memory mode)
            if dataset_tokenizer_state is not None:
                print(f"Loading tokenizer from dataset...")
                tokenizer.set_state(dataset_tokenizer_state)
                if tokenization == 'char':
                    config.vocab_size = len(tokenizer.chars)
                else:
                    config.vocab_size = 50304
            elif tokenization == 'char':
                if input_text is None:
                    raise ValueError(
                        "For character-level tokenization, either input_text or dataset_tokenizer_state must be provided."
                    )
                tokenizer.build_char_vocab(input_text)
                config.vocab_size = len(tokenizer.chars)
            else:
                config.vocab_size = 50304
            
            model = GPT(config, tokenizer=tokenizer).to(device)
            optimizer = model.configure_optimizer(learning_rate)
            return model, optimizer, 0
    
    def _validate_tokenization(self, base_tok_state, requested_tokenization, 
                               init_from_model, model, config):
        """Validate tokenization compatibility when fine-tuning"""
        if base_tok_state is None:
            raise ValueError(
                f"Base model '{init_from_model}' has no tokenizer state. "
                "Cannot safely fine-tune."
            )
        
        base_kind = base_tok_state.get("token_kind", None)
        if base_kind is None:
            raise ValueError(f"Base model '{init_from_model}' has unknown tokenizer kind.")
        
        if base_kind != requested_tokenization:
            raise ValueError(
                f"Tokenization mismatch:\n"
                f"  Base model uses: {base_kind}\n"
                f"  Requested: {requested_tokenization}\n"
                f"Fine-tuning requires matching tokenizer."
            )
        
        # Handle char-level vocabulary
        if base_kind == 'char':
            base_chars = base_tok_state.get("chars", None)
            if base_chars is None:
                raise ValueError(
                    f"Base char-level model '{init_from_model}' has no 'chars' saved."
                )
            model.tokenizer.chars = base_chars
            config.vocab_size = len(base_chars)
            print(f"Using base model's char vocabulary (size: {len(base_chars)})")
    
    @staticmethod
    def load_for_inference(model_name, base_dir="checkpoints", legacy_filename=None, load_dtype=None):
        """
        Load model for generation/inference.
        Automatically detects new JSON format or legacy single-file format.
        
        Args:
            model_name: Name of the model
            base_dir: Base checkpoints directory
            legacy_filename: If specified, load this specific legacy file (e.g., "final.pt")
            load_dtype: Optional dtype to load model as ('fp32', 'fp16', 'bf16').
                       If None, uses smart defaults (auto-converts bf16 to fp32 on older GPUs).
        
        Returns:
            Loaded GPT model
        """
        ckpt_manager = CheckpointManager(model_name, base_dir)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Try new JSON format first
        if ckpt_manager.exists_new_format():
            print(f"Loading model '{model_name}' (new format)")
            model, _, _, _ = GPT.load(ckpt_manager.checkpoint_dir, map_location=device, load_dtype=load_dtype)
            return model
        
        # Fall back to legacy format
        if legacy_filename:
            legacy_path = ckpt_manager.get_path(legacy_filename)
        else:
            # Try common legacy filenames
            for filename in ["latest.pt", "final.pt"]:
                if ckpt_manager.exists_legacy_format(filename):
                    legacy_path = ckpt_manager.get_path(filename)
                    break
            else:
                raise FileNotFoundError(
                    f"No checkpoint found for model '{model_name}' at {ckpt_manager.checkpoint_dir}"
                )
        
        print(f"Loading model '{model_name}' from legacy checkpoint: {legacy_path}")
        model, _, _, _ = GPT.load_legacy(legacy_path, map_location=device)
        return model

