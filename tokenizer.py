import tiktoken

class Tokenizer():

    def __init__(self, config, kind):
        super().__init__()
        self.config=config
        self.token_kind='char'
        self.chars=int
        self.enc=tiktoken.Encoding
        self.__set_encoding(kind)
        
    def forward(self, kind):
        self.__set_encoding(kind)
    
    def read_textData(self, file):
        with open(file, 'r', encoding='utf-8')as f:
            text=f.read()
        return text
    
    def encode(self,idx):
        if self.token_kind=='gpt2':
            return self.__encode_gpt2(idx)
        elif self.token_kind=='char':
            return self.__encode_char(idx)

    def decode(self,idx):
        if self.token_kind=='gpt2':
            return self.__decode_gpt2(idx)
        elif self.token_kind=='char':
            return self.__decode_char(idx)

    def __set_encoding(self,kind):
        self.token_kind=kind
        if self.token_kind == 'gpt2':
            self.enc = tiktoken.get_encoding(self.token_kind)
        if self.token_kind == 'char':
            self.token_kind='char'

    def __encode_gpt2(self,idx):
        return self.enc.encode_ordinary(idx)
    
    def __encode_char(self,idx):
        self.chars=sorted(list(set(idx)))
        stoi = {ch:i for i, ch in enumerate(self.chars) }
        return [stoi[c] for c in idx] #encoder: take a string, output a list of integer
    
    def __decode_gpt2(self,idx):
        return self.enc.decode(idx)
    
    def __decode_char(self,idx):
        itos = {i:ch for i, ch in enumerate(self.chars) }  
        s=''.join([itos[q] for q in idx]) #decoder: take a list of integers, output a string
        return s

