import os
import torch
from einops import rearrange
from config import TrainArgs, ModelArgs
from tokenizer import Tokenizer
from model import Model

train_args = TrainArgs()
args = ModelArgs
model = Model(args)
checkpoint_path = os.path.join(train_args.data_dir, "checkpoint1.pt")
assert os.path.isfile(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.cuda()
tokenizer = Tokenizer(os.path.join(train_args.data_dir, f"token12000.model"))

x = torch.full((1, args.s), tokenizer.pad_id)
x[0, 0] = tokenizer.bos_id
x = x.long().cuda()

for s in range(1, args.s):
    print(f"{x.size()=}")
    y = model(x)
    y = y.argmax(axis=-1)
    x[:, s] = y[:, s - 1]

print(tokenizer.decode(x.tolist()))
