import os
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, tokenizer_path):
        """
        Tokenizer class constructor
        """
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

    def list_decode(self, encoded_token: list[int], print_bos=True, print_eos=True, print_pad=False) -> list[str]:
        str = []
        for token in encoded_token:
            if token == self.bos_id:
                if print_bos:
                    str.append("<bos>")
            elif token == self.eos_id:
                if print_eos:
                    str.append("<eos>")
            elif token == self.pad_id:
                if print_pad:
                    str.append("<pad>")
            else:
                str.append(self.sp.decode(token))
        return str


if __name__ == "__main__":
    tokenizer = Tokenizer(f"./data/token{12000}.model")
    tokens = tokenizer.encode("Hello world", bos=True, eos=True, pad=True, seq_len=32)
    print(tokens)
    print(tokenizer.decode(tokens))
    print(tokenizer.sp.EncodeAsPieces("Hello world"))
