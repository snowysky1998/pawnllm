import os
import torch
from einops import rearrange
from config import TrainArgs, ModelArgs
from tokenizer import Tokenizer
from model import Model
import argparse

train_args = TrainArgs()
args = ModelArgs()


def main(checkpoint, prompt):
    model = Model(args)
    checkpoint_path = os.path.join(train_args.data_dir, checkpoint)
    assert os.path.isfile(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.cuda()
    tokenizer = Tokenizer(os.path.join(train_args.data_dir, f"token12000.model"))

    # for name, tensor in model.named_parameters():
    #     print(f"{tensor.dtype} - {name} - {tuple(tensor.size())}")

    if prompt != None:
        prompt_token = torch.Tensor(tokenizer.encode(prompt, bos=False))
        prompt_token = rearrange(prompt_token, "len -> 1 len")
        tokens = prompt_token
        infer_start = prompt_token.size(dim=-1)
    else:
        tokens = torch.full((1, 1), tokenizer.bos_id)
        infer_start = 1

    tokens = tokens.long().cuda()

    for _ in range(infer_start, args.s):
        logits = model(tokens)
        logits = rearrange(logits, "b vocab_size -> b 1 vocab_size")

        next_token = logits.argmax(axis=-1)

        # print(tokenizer.list_decode(next_token.tolist()))
        tokens = torch.cat([tokens, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_id:
            break

    tokens = rearrange(tokens, "1 s -> s")
    print(tokenizer.list_decode(tokens.tolist(), print_eos=True))
    print(tokenizer.decode(tokens.tolist()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", default=None, help="checkpoint path", type=str)
    parser.add_argument("-p", "--prompt", default=None, help="prompt for inference", type=str)
    parser_args = parser.parse_args()
    main(parser_args.checkpoint, parser_args.prompt)
