import torch

class Generator:
    """
    Wrapper for GPT model providing different generation modes.
    Encapsulates generation strategies (basic, Q&A, etc.)
    """
    def __init__(self, model):
        """
        Initialize generator with a trained model.
        
        Args:
            model: Trained GPT model instance
        """
        self.model = model
        self.model.eval()  # Set to eval mode
    
    def generate(
        self, 
        prompt, 
        max_new_tokens=100, 
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        stop_tokens=None
    ):
        """
        Basic text generation with sampling controls.
        
        Args:
            prompt: Starting text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (0.0=deterministic, 1.0=neutral)
            top_k: Only sample from top K tokens (0=disabled)
            top_p: Nucleus sampling threshold (1.0=disabled)
            repetition_penalty: Penalize repeated tokens (1.0=disabled)
            stop_tokens: Optional list of token IDs to stop generation on
        
        Returns:
            Generated text
        """
        with torch.no_grad():
            output = self.model.generate(
                prompt, 
                max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop_tokens=stop_tokens
            )
        return output
    
    def generate_creative(self, prompt, max_new_tokens=500):
        """Generate with creative/story settings."""
        return self.generate(
            prompt, max_new_tokens,
            temperature=1.0, top_k=100, top_p=0.95, repetition_penalty=1.2
        )
    
    def generate_factual(self, prompt, max_new_tokens=100):
        """Generate with factual/focused settings."""
        return self.generate(
            prompt, max_new_tokens,
            temperature=0.3, top_k=10, top_p=0.8, repetition_penalty=1.0
        )
    
    def generate_code(self, prompt, max_new_tokens=200):
        """Generate with code-focused settings."""
        return self.generate(
            prompt, max_new_tokens,
            temperature=0.2, top_k=40, top_p=0.95, repetition_penalty=1.0
        )
    
    

