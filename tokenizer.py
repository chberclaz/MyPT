import tiktoken

class Tokenizer():

    def __init__(self, config, kind):
        super().__init__()
        self.config=config
        self.token_kind='char'
        self.chars=None
        self.enc=None
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
            # char moe: encunused, well build chars from corpus
            self.enc = None
    
    # ---- GPT2 ----
    def __encode_gpt2(self,idx):
        return self.enc.encode_ordinary(idx)

    def __decode_gpt2(self,idx):
        return self.enc.decode(idx)

    # ---- CHAR ----
    def build_char_vocab(self, text):
        #Build vocabulary from full training corpus text.
        self.chars = sorted(list(set(text)))

    def __encode_char(self,idx):
        if self.chars is None:
            raise ValueError("Char vocabulary not built. Call build_char_vocab(text) first.")
        stoi = {ch:i for i, ch in enumerate(self.chars) }
        return [stoi[c] for c in idx] #encoder: take a string, output a list of integer
    

    def __decode_char(self,idx):
        if self.chars is None:
            raise ValueError("Char vocabulary not built.")
        itos = {i: ch for i, ch in enumerate(self.chars)}
        return ''.join([itos[q] for q in idx])

    # ----- SERIALIZATION HELPERS -----
    def get_state(self):
        return {
            "token_kind": self.token_kind,
            "chars": self.chars,
        }


    @classmethod
    def from_state(cls, config, state):
        tok = cls(config, state["token_kind"])
        tok.chars = state.get("chars", None)
        return tok

