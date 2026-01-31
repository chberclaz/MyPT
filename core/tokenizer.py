import re
import tiktoken
from .special_tokens import SPECIAL_TOKEN_STRINGS


class Tokenizer():

    def __init__(self, config, kind):
        super().__init__()
        self.config = config
        self.token_kind = kind
        self.chars = None
        self.enc = None
        self.__set_encoding(kind)

        # Get actual vocabulary size from base tokenizer
        if self.token_kind == 'gpt2':
            self.base_vocab_size = self.enc.n_vocab  # 50257 for GPT-2
        else:
            # For char tokenization, will be set when build_char_vocab() is called
            self.base_vocab_size = None
        
        self.model_vocab_size = config.vocab_size
        
        # Special token mappings: string <-> id
        self.special_token_encoder = {}  # str -> id
        self.special_token_decoder = {}  # id -> str
        self.special_tokens = {}         # name -> id (for easy access by name)
        self._special_token_pattern = None  # Compiled regex for fast encoding
        
        # Register special tokens into reserved slots (after base vocab)
        if self.token_kind == 'gpt2':
            self._register_special_tokens()
        
        self.vocab_size = self.model_vocab_size
    
    def _register_special_tokens(self):
        """
        Register special tokens in reserved vocabulary slots.
        
        Tokens are read dynamically from SPECIAL_TOKEN_STRINGS (defined in special_tokens.py).
        To add new special tokens, simply edit special_tokens.py - no changes needed here.
        
        The regex pattern for fast encoding is also built dynamically from the same source.
        """
        next_id = self.base_vocab_size  # Start after base vocab (50257 for GPT-2)
        
        # Dynamically register all tokens defined in special_tokens.py
        for name, tok_str in SPECIAL_TOKEN_STRINGS.items():
            if next_id >= self.model_vocab_size:
                raise ValueError(
                    f"Not enough reserved vocab slots for special tokens! "
                    f"Need {len(SPECIAL_TOKEN_STRINGS)} slots, but only "
                    f"{self.model_vocab_size - self.base_vocab_size} available. "
                    f"Increase config.vocab_size (currently {self.model_vocab_size})."
                )
            
            self.special_token_encoder[tok_str] = next_id
            self.special_token_decoder[next_id] = tok_str
            self.special_tokens[name] = next_id
            next_id += 1
        
        # Build regex pattern dynamically from registered tokens
        # Pattern: (<system>|</system>|<user>|</user>|...) - matches any special token
        # This automatically includes any tokens added to special_tokens.py
        # IMPORTANT: Sort by descending length to prevent prefix-matching bugs
        # (regex alternation matches left-to-right, shorter tokens could match first)
        tokens_by_length = sorted(self.special_token_encoder.keys(), key=len, reverse=True)
        escaped_tokens = [re.escape(tok) for tok in tokens_by_length]
        self._special_token_pattern = re.compile('|'.join(escaped_tokens))
        
        print(f"Registered {len(self.special_tokens)} special tokens (IDs {self.base_vocab_size}-{next_id-1})")
        
    def forward(self, kind):
        self.__set_encoding(kind)
    
    def read_textData(self, file):
        with open(file, 'r', encoding='utf-8')as f:
            text=f.read()
        return text
    
    def encode(self, text):
        """Encode text to token IDs, handling special tokens."""
        if self.token_kind == 'gpt2':
            return self.__encode_gpt2_with_special(text)
        elif self.token_kind == 'char':
            return self.__encode_char(text)

    def decode(self, ids):
        """Decode token IDs to text, handling special tokens."""
        if self.token_kind == 'gpt2':
            return self.__decode_gpt2_with_special(ids)
        elif self.token_kind == 'char':
            return self.__decode_char(ids)
    
    def encode_ordinary(self, text):
        """Encode text without special token handling (raw encoding)."""
        if self.token_kind == 'gpt2':
            return self.enc.encode_ordinary(text)
        elif self.token_kind == 'char':
            return self.__encode_char(text)

    def __set_encoding(self,kind):
        self.token_kind=kind
        if self.token_kind == 'gpt2':
            self.enc = tiktoken.get_encoding(self.token_kind)
        if self.token_kind == 'char':
            # char moe: encunused, well build chars from corpus
            self.enc = None
    
    # ---- GPT2 ----
    def __encode_gpt2_with_special(self, text):
        """
        Encode text, replacing special token strings with their IDs.
        Uses compiled regex for fast special token detection (2-3x faster than iterative find).
        """
        if not self.special_token_encoder:
            return self.enc.encode_ordinary(text)
        
        result = []
        last_end = 0
        
        # Use compiled regex to find all special tokens in one pass
        for match in self._special_token_pattern.finditer(text):
            # Encode text before this match
            if match.start() > last_end:
                result.extend(self.enc.encode_ordinary(text[last_end:match.start()]))
            
            # Add the special token ID
            result.append(self.special_token_encoder[match.group()])
            last_end = match.end()
        
        # Encode any remaining text after the last special token
        if last_end < len(text):
            result.extend(self.enc.encode_ordinary(text[last_end:]))
        
        return result

    def __decode_gpt2_with_special(self, ids):
        """Decode token IDs, replacing special token IDs with their strings."""
        if not self.special_token_decoder:
            return self.enc.decode(ids)
        
        result = []
        current_chunk = []
        
        for token_id in ids:
            if token_id in self.special_token_decoder:
                # Decode accumulated regular tokens first
                if current_chunk:
                    result.append(self.enc.decode(current_chunk))
                    current_chunk = []
                # Add special token string
                result.append(self.special_token_decoder[token_id])
            else:
                current_chunk.append(token_id)
        
        # Decode any remaining regular tokens
        if current_chunk:
            result.append(self.enc.decode(current_chunk))
        
        return ''.join(result)

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

    # ----- SPECIAL TOKEN HELPERS -----
    def get_special_token_id(self, name):
        """Get the token ID for a special token by name (e.g., 'user_open')."""
        if name not in self.special_tokens:
            raise KeyError(f"Unknown special token: {name}. Available: {list(self.special_tokens.keys())}")
        return self.special_tokens[name]
    
    def get_special_token_string(self, name):
        """Get the string for a special token by name (e.g., '<user>')."""
        return SPECIAL_TOKEN_STRINGS.get(name)
    
    def is_special_token(self, token_id):
        """Check if a token ID is a special token."""
        return token_id in self.special_token_decoder

    # ----- SERIALIZATION HELPERS -----
    def get_state(self):
        return {
            "token_kind": self.token_kind,
            "chars": self.chars,
            "base_vocab_size": self.base_vocab_size,
        }
    
    def set_state(self, state):
        """Restore tokenizer state from a saved state dictionary."""
        self.token_kind = state["token_kind"]
        self.chars = state.get("chars", None)
        self.__set_encoding(self.token_kind)
        
        # Re-register special tokens if GPT-2
        if self.token_kind == 'gpt2':
            self.base_vocab_size = state.get("base_vocab_size", self.enc.n_vocab)
            self._register_special_tokens()

    @classmethod
    def from_state(cls, config, state):
        tok = cls(config, state["token_kind"])
        tok.chars = state.get("chars", None)
        return tok

