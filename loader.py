class GPTDataLoader():
    def __init__(self, config):
        super().__init__()
        self.config=config

    def forward(self, x):
        return True
    
    @classmethod
    def read_textData(self, file):
        with open(file, 'r', encoding='utf-8')as f:
            text=f.read()
        return text
