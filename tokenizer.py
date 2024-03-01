import os
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, DATA_CACHE_DIR, token_model_name):
        """
        Tokenizer class constructor
        input:  DATA_CACHE_DIR : directory of token model file
                token_model_name : file name of token model file
        """
        self.DATA_CACHE_DIR = DATA_CACHE_DIR
        tokenizer_path = os.path.join(DATA_CACHE_DIR, token_model_name)

        assert os.path.isfile(tokenizer_path), f"Tokenizer not found at {tokenizer_path}"

        self.sp = SentencePieceProcessor(tokenizer_path)

        self.vocab_size = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
    
    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        """
        encodes the input string
        input:  text(str) : text to be encoded
                bos(bool) : add bos_id to the output vector 
                eos(bool) : add eos_id to the output vector
        output: (list[int]) : vector of encoded string
        """
        encoded_token = self.sp.encode(text)
        if bos:
            encoded_token = [self.bos_id] + encoded_token
        if eos:
            encoded_token = encoded_token + [self.eos_id]
        return encoded_token
    
    def decode(self, encoded_token: list[int]) -> str:
        """
        decodes the input vector
        input:  encoded_token(list[int]) : vector to be decoded
        output: (str) : string of decoded vector
        """
        return self.sp.decode(encoded_token)

if __name__ == "__main__":
    tokenizer = Tokenizer("./data/", "token2048.model")
    print(tokenizer.encode("Hello world"))
