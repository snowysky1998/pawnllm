import os
import torch
from dataclasses import dataclass

from tokenizer import Tokenizer
from model import Model
from config import ModelArgs, TrainArgs
from tqdm import tqdm
import argparse



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

checkpoint_path = os.path.join(train_args.data_dir, "checkpoint.tar")

if __name__ == "__main__":
    parser =  argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", help="start training form checkpoint", action="store_true")
    parser_args = parser.parse_args()

    if parser_args.checkpoint:
        # train from checkpoint
        print(f"Training from checkpoint")
        print(f"Loading checkpoint")
        assert os.path.isfile(checkpoint_path) ,f"Checkpoint not found at {checkpoint_path}"
        print(f"Checkpoint found")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step_start = checkpoint["epoch"] + 1
        loss = checkpoint["loss"]
        
        model.train()
        model.compile()
        print(f"Loading checkpoint done")
        print(f"Checkpoint final_loss={loss.item()}")
        print(f"Training start from step={step_start}")
    else:
        # train from zero
        print(f"Training from zero")
        step_start = 0
        print(f"Training start from step={step_start}")

    print(f"Loading dataset")
    token_batches = torch.load(os.path.join(train_args.data_dir, f"tiny_batches_s{args.s}.pt"))
    assert token_batches.ndim == 2
    num_batches, _ = token_batches.size()

    print(f"{token_batches.size()=}")
    dataset = torch.utils.data.TensorDataset(token_batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args.batch_size, shuffle=True, drop_last=True)

    step_end = train_args.max_steps
    print(f"Train start")
    for step in range(step_start, step_end):
        for batch, in tqdm(dataloader, total=len(dataloader), desc=f"step:{step}", leave=False):
            optimizer.zero_grad()

            batch = batch.long().cuda()
            x = batch[:, :-1]
            y = batch[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o, loss = model(x, y)
            
            loss = scaler.scale(loss)
            loss.backward()

            optimizer.step()

            # print(f"{count}\t:loss={loss.item()}")
        print(f"step:{step}\tfinal_loss={loss.item()}")
        scheduler.step()

        # checkpoint
        if (step + 1) % train_args.checkpoint_steps == 0:
            print(f"Saving checkpoint")
            # saving model data
            torch.save({
                "epoch": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": loss,
            }, checkpoint_path)
            # saving output?
            torch.save(o, "output.pt")
            # break

        # break




































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
