import os
import torch
from .model import GPT, GPTConfig
from .tokenizer import Tokenizer

class CheckpointManager:
    """
    Manages checkpoint paths and model initialization 
    (resume / init_from / fresh).
    """
    def __init__(self, model_name, base_dir="checkpoints"):
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(base_dir, model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def get_path(self, filename="latest.pt"):
        """Get full path to a checkpoint file"""
        return os.path.join(self.checkpoint_dir, filename)
    
    def exists(self, filename="latest.pt"):
        """Check if checkpoint exists"""
        return os.path.exists(self.get_path(filename))
    
    def initialize_for_training(self, config, tokenization, input_text, 
                                learning_rate, init_from_model=None):
        """
        Initialize model for training. Handles 3 cases:
        1. Resume from this model's checkpoint
        2. Initialize from another model (fine-tuning)
        3. Create fresh model
        
        Args:
            config: GPTConfig instance
            tokenization: 'gpt2' or 'char'
            input_text: Training text (for char vocab building)
            learning_rate: Learning rate for optimizer
            init_from_model: Optional model name to initialize from
        
        Returns: (model, optimizer, start_step)
        """
        device = config.device
        checkpoint_path = self.get_path("latest.pt")
        
        # Case 1: RESUME
        if self.exists("latest.pt"):
            print(f"Found checkpoint at {checkpoint_path}, resuming training.")
            model, _, start_step, optim_state = GPT.load(checkpoint_path, map_location=device)
            
            optimizer = model.configure_optimizer(learning_rate, optim_state)
            return model, optimizer, start_step
        
        # Case 2: INIT FROM ANOTHER MODEL (Fine-tuning)
        elif init_from_model is not None:
            init_manager = CheckpointManager(init_from_model)
            init_path = init_manager.get_path("final.pt")
            
            if not init_manager.exists("final.pt"):
                raise FileNotFoundError(
                    f"--init_from_model '{init_from_model}' specified, "
                    f"but {init_path} does not exist."
                )
            
            print(f"Initializing from base model '{init_from_model}' at {init_path}")
            model, base_tok_state, _, _ = GPT.load(init_path, map_location=device)
            
            # Validate tokenization compatibility
            self._validate_tokenization(base_tok_state, tokenization, init_from_model, model, config)
            
            optimizer = model.configure_optimizer(learning_rate)
            return model, optimizer, 0  # start from step 0 for new training
        
        # Case 3: FRESH MODEL
        else:
            print("No checkpoint found. Starting from scratch.")
            print(f"Using config: {config}")
            
            # Build tokenizer
            tokenizer = Tokenizer(config, tokenization)
            if tokenization == 'char':
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
    def load_for_inference(model_name, checkpoint="latest.pt", base_dir="checkpoints"):
        """
        Load model for generation/inference.
        
        Args:
            model_name: Name of the model
            checkpoint: Checkpoint filename (default: "latest.pt")
            base_dir: Base checkpoints directory
        
        Returns:
            Loaded GPT model
        """
        ckpt_manager = CheckpointManager(model_name, base_dir)
        path = ckpt_manager.get_path(checkpoint)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, _, _ = GPT.load(path, map_location=device)
        return model

