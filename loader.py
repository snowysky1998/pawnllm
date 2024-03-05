import os
import torch

from tokenizer import Tokenizer
from model import Model, ModelArgs

import time

from einops import rearrange

DATA_CACHE_DIR = "./data"

tokenizer = Tokenizer("./data/", "token2048.model")

model_args = ModelArgs(vocab_size=tokenizer.vocab_size)

model = Model(model_args)
model.cuda()

if __name__ == "__main__":
    tokens_packed = torch.load(os.path.join(DATA_CACHE_DIR, "tiny_tokens.pt"), map_location="cuda")
    offsets_packed = torch.load(os.path.join(DATA_CACHE_DIR, "tiny_offsets.pt"), map_location="cuda")

    first_paragraph = tokens_packed[offsets_packed[0] : offsets_packed[1]]
    first_paragraph = torch.nn.functional.pad(
        first_paragraph, (0, model_args.s - first_paragraph.size(-1)), value=tokenizer.eos_id
    )
    first_paragraph = rearrange(first_paragraph, "s -> 1 s")

    output = model(first_paragraph)
    # print(f"{output=}")
    # print(f"{output.dtype=}")
    # print(f"{output.size()=}")

    # first_paragraph_str = tokenizer.decode(first_paragraph.tolist())
    # print(f"{first_paragraph_str=}")
    # print(f"{first_paragraph=}")
    # print(f"{len(first_paragraph)=}")
    # print(f"{tokens_packed=}")
    # print(f"{tokens_packed.max()=}")
    # print(f"{offsets_packed=}")
    # print(f"{offsets_packed.max()=}")
