import os
import torch
from einops import rearrange
from config import TrainArgs, ModelArgs
from tokenizer import Tokenizer
from model import Model

train_args = TrainArgs()
args = ModelArgs
model = Model(args)
checkpoint_path = os.path.join(train_args.data_dir, "checkpoint10.pt")
assert os.path.isfile(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.cuda()
# model.compile()

tokenizer = Tokenizer(os.path.join(train_args.data_dir, f"token12000.model"))

x = torch.full((1, args.s), tokenizer.pad_id)
x[0, 0] = tokenizer.bos_id
x = x.long().cuda()

print(x)

for s in range(1, args.s):
    y = model(x)
    y[0, 0, 0] = float("-inf")
    y = y.argmax(axis=-1)
    x[0, s] = y
    print(f"{x=}")

# o = rearrange(o, "(b s) -> b s", b=train_args.batch_size)
# o_paragraph = o[42, :]
# print(o_paragraph)
# print(o_paragraph.size())

# print(tokenizer.decode(o_paragraph.tolist()))
