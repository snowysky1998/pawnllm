import os
import torch
from dataclasses import dataclass
from tqdm import tqdm
import argparse

from tokenizer import Tokenizer
from model import Model
from config import ModelArgs, TrainArgs

train_args = TrainArgs()
args = ModelArgs()

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    print(f"Loading checkpoint")

    assert os.path.isfile(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()
    model.to(device)
    if device == "cuda":
        model.compile()
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"] + 1


def main(checkpoint, max_steps, limit_batches):
    print(f"Using device: {device}")
    model = Model(args)
    model.train()
    model.to(device)
    if device == "cuda":
        model.compile()

    optimizer = torch.optim.Adam(model.parameters(), lr=10 * train_args.learning_rate, weight_decay=train_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=train_args.learning_rate)
    scaler = torch.amp.GradScaler(device, init_scale=train_args.grad_init_scale, growth_interval=train_args.grad_growth_steps, enabled=device == "cuda")

    step_start = 1
    if checkpoint:
        step_start = load_checkpoint(os.path.join(train_args.data_dir, checkpoint), model, optimizer, scheduler)

    print(f"{step_start=}")

    print(f"Loading dataset")
    token_batches = torch.load(os.path.join(train_args.data_dir, f"tiny_batches_s{args.s}.pt"))
    assert token_batches.ndim == 2
    if limit_batches:
        token_batches = token_batches[: limit_batches * train_args.batch_size]

    dataset = torch.utils.data.TensorDataset(token_batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args.batch_size, shuffle=True, drop_last=True)

    for step in range(step_start, max_steps + 1):
        print(f"Starting {step=}")

        total_loss = torch.tensor(0.0)
        for (batch,) in tqdm(dataloader, total=len(dataloader), desc=f"{step=}", leave=False):
            batch = batch.long().to(device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                o, loss = model(x, y)

            total_loss += loss.cpu().detach().clone()

            scaler.scale(loss).backward()
            optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        scheduler.step()

        # checkpoint
        avg_loss = total_loss.item() / len(dataloader)
        print(f"{step=} {avg_loss=}")
        if step % train_args.checkpoint_steps == 0:
            torch.save(
                {
                    "epoch": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(train_args.data_dir, f"checkpoint{step}.pt"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default=None, help="checkpoint path", type=str)
    parser.add_argument("--max_steps", default=train_args.max_steps, help="number of epochs to train", type=int)
    parser.add_argument("--limit_batches", default=0, help="use only the first N batches per epoch (0 = all)", type=int)
    parser_args = parser.parse_args()

    main(parser_args.checkpoint, parser_args.max_steps, parser_args.limit_batches)
