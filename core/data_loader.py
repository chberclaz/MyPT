import torch

class GPTDataLoader:
    """
    Handles data loading, tokenization, and batching for GPT training.
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.train_data = None
        self.val_data = None
    
    @staticmethod
    def read_text(file_path):
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def prepare_data(self, text, train_ratio=0.9):
        """
        Tokenize text and split into train/val sets.
        
        Args:
            text: Input text to tokenize
            train_ratio: Fraction of data to use for training (default 0.9)
        """
        tokens = self.tokenizer.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        
        n = int(train_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
        print(f"Data prepared: {len(self.train_data)} train tokens, "
              f"{len(self.val_data)} val tokens")
    
    def get_batch(self, split='train'):
        """
        Generate a batch of data.
        
        Args:
            split: 'train' or 'val'
        
        Returns:
            Tuple of (x, y) tensors on device
        """
        batch_data = self.train_data if split == 'train' else self.val_data
        
        if batch_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        ix = torch.randint(len(batch_data) - self.config.block_size, 
                          (self.config.batch_size,))
        x = torch.stack([batch_data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([batch_data[i + 1:i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y

