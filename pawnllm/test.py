import os
import torch
from dataclasses import dataclass
from einops import rearrange

from tokenizer import Tokenizer


@dataclass
class TrainArgs:
    data_dir: str = "../data"
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 2e-2
    checkpoint_steps: int = 10
    max_steps: int = 30000

train_args = TrainArgs()

tokenizer = Tokenizer(os.path.join(train_args.data_dir, f"token12000.model"))

o = torch.load("output.pt")
o = o.argmax(axis=-1)
o = rearrange(o, "(b s) -> b s", b=train_args.batch_size)
o_paragraph = o[42, :]
print(o_paragraph)
print(o_paragraph.size())

print(tokenizer.decode(o_paragraph.tolist()))


