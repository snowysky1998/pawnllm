import os
import torch
from dataclasses import dataclass
from tqdm import tqdm
import argparse

from tokenizer import Tokenizer
from model import Model
from config import ModelArgs, TrainArgs
from logger import discord_log


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    print(f"Loading checkpoint")

    assert os.path.isfile(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"] + 1


def main(checkpoint):
    train_args = TrainArgs()
    args = ModelArgs()

    model = Model(args)
    tokenizer = Tokenizer(os.path.join(train_args.data_dir, f"token{args.vocab_size}.model"))
    optimizer = torch.optim.Adam(model.parameters(), lr=10 * train_args.learning_rate, weight_decay=train_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_args.max_steps, eta_min=train_args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    step_start = 1
    if checkpoint:
        step_start = load_checkpoint(os.path.join(train_args.data_dir, checkpoint), model, optimizer, scheduler)

    model.train()
    model.cuda()
    model.compile()

    print(f"{step_start=}")

    print(f"Loading dataset")
    token_batches = torch.load(os.path.join(train_args.data_dir, f"tiny_batches_s{args.s}.pt"))
    assert token_batches.ndim == 2
    num_batches, _ = token_batches.size()
    print(f"{token_batches.size()=}")
    # token_batches = token_batches[:1024*128, :]
    print(f"{token_batches.size()=}")

    dataset = torch.utils.data.TensorDataset(token_batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args.batch_size, shuffle=True, drop_last=True)

    for step in range(step_start, train_args.max_steps + 1):
        print(f"Starting {step=}")

        total_loss = torch.tensor(0.0)
        for (batch,) in tqdm(dataloader, total=len(dataloader), desc=f"{step=}", leave=False):
            optimizer.zero_grad()

            batch = batch.long().cuda()
            x = batch[:, :-1]
            y = batch[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o, loss = model(x, y)

            total_loss += loss.cpu().detach().clone()

            loss = scaler.scale(loss)
            loss.backward()

            optimizer.step()

        scheduler.step()

        # checkpoint
        avg_loss = total_loss.item() / len(dataloader)
        print(f"{step=} {avg_loss=}")
        if step % train_args.checkpoint_steps == 0:
            discord_log(f"```{step=} {avg_loss=:.4f}```")
            torch.save(
                {
                    "epoch": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(train_args.data_dir, f"checkpoint{step}.pt"),
            )
            torch.save(o, "./data/output.pt")


# TODO
# train
# - split eval and validation
# - write model eval and validation
# - perplexity analysis (how close are the tokens?)
# - early termination

# DONE
# - hyperparameter tuning
# - loss function, backward optimizer, gradient, improve mixed precision
# - shuffle and batch paragraphs
# - training loop checkpoint code, and logging

# TODO
# args
# data_dir
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default=None, help="checkpoint path", type=str)
    parser_args = parser.parse_args()

    main(parser_args.checkpoint)
