import os
import torch
from dataclasses import dataclass

from tokenizer import Tokenizer
from model import Model
from config import ModelArgs, TrainArgs




train_args = TrainArgs()

args = ModelArgs()
model = Model(args)
model.cuda()
model.train()
model.compile()

tokenizer = Tokenizer(os.path.join(train_args.data_dir, f"token{args.vocab_size}.model"))
optimizer = torch.optim.Adam(model.parameters(), lr=10 * train_args.learning_rate, weight_decay=train_args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.max_steps, eta_min=train_args.learning_rate)
scaler = torch.cuda.amp.GradScaler()

if __name__ == "__main__":
    token_batches = torch.load(os.path.join(train_args.data_dir, f"tiny_batches_s{args.s}.pt"))
    assert token_batches.ndim == 2
    num_batches, _ = token_batches.size()

    print(f"{token_batches.size()=}")
    dataset = torch.utils.data.TensorDataset(token_batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args.batch_size, shuffle=True, drop_last=True)

    step_start = 0
    step_end = train_args.max_steps

    for step in range(step_start, step_end):

        count = 0

        for batch, in dataloader:
            optimizer.zero_grad()

            batch = batch.long().cuda()
            x = batch[:, :-1]
            y = batch[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o, loss = model(x, y)
            
            loss = scaler.scale(loss)
            loss.backward()

            optimizer.step()

            print(f"{count}\t:loss={loss.item()}")

            count += 1

            if count == 4000:
                torch.save(o, "output.pt")
                break

        scheduler.step()

        break




































# print(f"{o=}")
# print(f"{o.dtype=}")
# print(f"{o.size()=}")
# print(f"{loss=}")
# print(f"{loss.dtype=}")
# print(f"{loss.size()=}")

# first_paragraph = batch[0, :]
# first_paragraph_str = tokenizer.decode(first_paragraph.tolist())
# print(f"{first_paragraph=}")
# print(f"{first_paragraph_str=}")
# print(f"{len(first_paragraph)=}")

# TODO
# train
# - shuffle and batch paragraphs, and split eval and validation
# - write model eval and validation
# - loss function, backward optimizer, gradient, improve mixed precision
# - training loop checkpoint code, and logging
# - hyperparameter tuning
# - perplexity analysis (how close are the tokens?)

# - what do you do if the model doesn't converge?
# https://developers.google.com/machine-learning/testing-debugging/metrics/interpretic
