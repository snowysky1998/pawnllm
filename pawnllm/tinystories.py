import datasets
import os
import warnings
from tqdm import tqdm
import sentencepiece as spm
from tokenizer import Tokenizer
from concurrent.futures import ProcessPoolExecutor
import torch
import argparse
from config import TokenArgs

token_args = TokenArgs()

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


def train_vocab():
    print(f"stage: train_vocab")
    os.makedirs(token_args.data_dir, exist_ok=True)
    tiny_file = os.path.join(token_args.data_dir, "tiny.txt")
    prefix = os.path.join(token_args.data_dir, f"token{token_args.vocab_size}")

    print(f"Writing temporary file {tiny_file}")
    dataset = datasets.load_dataset("roneneldan/TinyStories")
    text = dataset["train"]["text"]
    with open(tiny_file, "w", encoding="utf-8") as of:
        for paragraph in tqdm(text):
            of.write(paragraph + "\n\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

    spm.SentencePieceTrainer.train(
        input=tiny_file,
        vocab_size=token_args.vocab_size,
        model_type="bpe",
        model_prefix=prefix,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=1.0,
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
    )

    print(f"Trained tokenizer is in {prefix}.model")

    os.remove(tiny_file)
    print(f"Deleted {tiny_file}")

    print(f"Done")


def process_paragraph(paragraph_chunk, tqdm_position):
    tokenizer_path = os.path.join(token_args.data_dir, f"token{token_args.vocab_size}.model")
    tokenizer = Tokenizer(tokenizer_path)
    chunk = []
    for paragraph in tqdm(paragraph_chunk, position=tqdm_position, desc=f"{tqdm_position:2d}core", leave=False, total=len(paragraph_chunk)):
        paragraph = paragraph.replace("\n\n", "")
        chunk.append(tokenizer.encode(paragraph, bos=True, eos=True))
    return chunk


def pretokenize1():
    print(f"stage: pretokenize1")
    dataset = datasets.load_dataset("roneneldan/TinyStories")
    text = dataset["train"]["text"]
    print(f"Text obtained")

    text = text[: len(text) // 2]

    chunk_list = lambda l, n: [l[i : i + n] for i in range(0, len(l), n)]
    tqdm_position = lambda n: range(0, n)

    with ProcessPoolExecutor(max_workers=token_args.num_cpu_thread) as e:
        chunks = e.map(
            process_paragraph,
            chunk_list(text, len(text) // token_args.num_cpu_thread),
            tqdm_position(token_args.num_cpu_thread),
        )

    # os.system("clear")
    print(f"\nEncoding done")

    tokens_list = [tokens for chunk in chunks for tokens in chunk]

    tokens_packed = torch.tensor([token for tokens in tokens_list for token in tokens]).int()
    offsets_packed = torch.cumsum(torch.tensor([0] + [len(tokens) for tokens in tokens_list]), dim=0).int()

    print(f"Saving tokens as binary files(.pt)")
    torch.save(tokens_packed, os.path.join(token_args.data_dir, "tiny_tokens.pt"))
    torch.save(offsets_packed, os.path.join(token_args.data_dir, "tiny_offsets.pt"))
    print(f"Done")


def pretokenize2(seq_len):
    print(f"stage: pretokenize2")
    tokenizer_path = os.path.join(token_args.data_dir, f"token{token_args.vocab_size}.model")
    tokenizer = Tokenizer(tokenizer_path)
    tokens_packed = torch.load(os.path.join(token_args.data_dir, "tiny_tokens.pt"))
    offsets_packed = torch.load(os.path.join(token_args.data_dir, "tiny_offsets.pt"))
    print("loaded packed tensors")

    def pad_tokens(tokens, seq_len, pad_token):
        assert tokens.ndim == 1
        assert seq_len >= tokens.size(-1)
        return torch.nn.functional.pad(tokens, (0, seq_len - tokens.size(-1)), value=pad_token)

    num_batches = len(offsets_packed) - 1

    batches = torch.empty(num_batches, seq_len + 1, dtype=torch.short)

    num_batches_sliced = 0
    num_batches_padded = 0

    for b in tqdm(range(num_batches)):
        paragraph = tokens_packed[offsets_packed[b] : offsets_packed[b + 1]]
        if seq_len < paragraph.size(-1):
            num_batches_sliced += 1
            batches[b, :] = paragraph[: seq_len + 1].short()
        else:
            num_batches_padded += 1
            batches[b, :] = pad_tokens(paragraph, seq_len + 1, tokenizer.pad_id).short()

    print(f"{num_batches_sliced=}")
    print(f"{num_batches_padded=}")

    print(f"{batches.size()=}")
    print(f"{batches.dtype=}")
    torch.save(batches, os.path.join(token_args.data_dir, f"tiny_batches_s{seq_len}.pt"))
    print("saved padded tensors as batches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="all", choices=["all", "train_vocab", "pretokenize1", "pretokenize2"])
    parser.add_argument("--seq_len", type=int, default=token_args.default_seq_len)
    args = parser.parse_args()

    if args.stage == "all":
        train_vocab()
        pretokenize1()
        pretokenize2(args.seq_len)
    elif args.stage == "train_vocab":
        train_vocab()
    elif args.stage == "pretokenize1":
        pretokenize1()
    elif args.stage == "pretokenize2":
        pretokenize2(args.seq_len)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
