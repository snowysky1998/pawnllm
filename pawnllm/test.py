import os
import torch
from einops import rearrange
from config import TrainArgs, ModelArgs
from tokenizer import Tokenizer
from model import Model


tokenizer = Tokenizer(os.path.join(train_args.data_dir, f"token12000.model"))
assert os.path.isfile(checkpoint_path) ,f"Checkpoint not found at {checkpoint_path}"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
o = torch.load(os.path.join(train_args.data_dir, "output.pt"))
o = o.argmax(axis=-1)
o = rearrange(o, "(b s) -> b s", b=train_args.batch_size)
o_paragraph = o[42, :]
print(o_paragraph)
print(o_paragraph.size())

print(tokenizer.decode(o_paragraph.tolist()))


