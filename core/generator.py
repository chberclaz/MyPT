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
    
    def generate(self, prompt, max_new_tokens=100, temperature=1.0):
        """
        Basic text generation.
        
        Args:
            prompt: Starting text
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (not yet implemented in model)
        
        Returns:
            Generated text
        """
        with torch.no_grad():
            output = self.model.generate(prompt, max_new_tokens)
        return output
    
    

