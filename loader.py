import os
import torch

from tokenizer import Tokenizer
from model import Model, ModelArgs

from einops import rearrange

DATA_CACHE_DIR = "./data"

tokenizer = Tokenizer("./data/", "token2048.model")

args = ModelArgs(vocab_size=tokenizer.vocab_size)

model = Model(args)
model.cuda()


def pad_tokens(tokens, seq_len, pad_token):
    assert tokens.ndim == 1
    return torch.nn.functional.pad(tokens, (0, seq_len - tokens.size(-1)), value=pad_token)


if __name__ == "__main__":
    tokens_packed = torch.load(os.path.join(DATA_CACHE_DIR, "tiny_tokens.pt"), map_location="cuda")
    offsets_packed = torch.load(os.path.join(DATA_CACHE_DIR, "tiny_offsets.pt"), map_location="cuda")

    first_paragraph = tokens_packed[offsets_packed[0] : offsets_packed[1]]
    first_paragraph = first_paragraph.long()

    x = pad_tokens(first_paragraph[:-1], args.s, tokenizer.eos_id)
    y = pad_tokens(first_paragraph[1:], args.s, tokenizer.eos_id)

    # TODO: change to stack
    x = rearrange(x, "s -> 1 s")
    y = rearrange(y, "s -> 1 s")

    o, loss = model(x, y)
    print(f"{o=}")
    print(f"{o.dtype=}")
    print(f"{o.size()=}")
    print(f"{loss=}")
    print(f"{loss.dtype=}")
    print(f"{loss.size()=}")

    # first_paragraph_str = tokenizer.decode(first_paragraph.tolist())
    # print(f"{first_paragraph_str=}")
    # print(f"{first_paragraph=}")
    # print(f"{len(first_paragraph)=}")
    # print(f"{tokens_packed=}")
    # print(f"{tokens_packed.max()=}")
    # print(f"{offsets_packed=}")
    # print(f"{offsets_packed.max()=}")
