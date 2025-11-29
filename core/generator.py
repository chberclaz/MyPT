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
    
    def generate_qa(self, question, max_new_tokens=150):
        """
        Q&A generation with special formatting.
        Assumes model was trained on <Question>/<Answer> format.
        
        Args:
            question: Question to ask
            max_new_tokens: Max tokens for answer
        
        Returns:
            Extracted answer text
        """
        # Format Q&A prompt
        prompt = f"<Question> {question} </Question>\n<Answer> "
        
        # Generate continuation
        with torch.no_grad():
            output = self.model.generate(prompt, max_new_tokens)
        
        # Extract answer
        answer = output.split("<Answer>")[-1]
        answer = answer.split("</Answer>")[0]
        return answer.strip()
    
    def generate_batch(self, prompts, max_new_tokens=100):
        """
        Generate for multiple prompts (sequential processing).
        
        Args:
            prompts: List of prompt strings
            max_new_tokens: Tokens to generate per prompt
        
        Returns:
            List of generated texts
        """
        return [self.generate(p, max_new_tokens) for p in prompts]

