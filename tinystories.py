import datasets
import os
import warnings
from tqdm import tqdm
import sentencepiece as spm
from tokenizer import Tokenizer
from concurrent.futures import ProcessPoolExecutor
import torch
import argparse

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

DATA_CACHE_DIR = "./data"

PARAGRAPH_SIZE=1000000
VOCAB_SIZE=2048
NUM_CPU_THREAD=32

def train_vocab():
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    tiny_file = os.path.join(DATA_CACHE_DIR, "tiny.txt")
    prefix = os.path.join(DATA_CACHE_DIR, f"token{VOCAB_SIZE}")

    print(f"Writing temporary file {tiny_file}")
    dataset = datasets.load_dataset("roneneldan/TinyStories")
    text = dataset["train"]["text"]
    with open(tiny_file, "w", encoding="utf-8") as of:
        for paragraph in tqdm(text[:PARAGRAPH_SIZE]):
            of.write(paragraph + "\n\n")
    print(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")


    spm.SentencePieceTrainer.train(input=tiny_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=VOCAB_SIZE,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")

    print(f"Trained tokenizer is in {prefix}.model")

    os.remove(tiny_file)
    print(f"Deleted {tiny_file}")

    print(f"Done")

def process_paragraph(paragraph_chunk, tqdm_position):
    tokenizer = Tokenizer("./data/", "token2048.model")
    chunk = []
    for paragraph in tqdm(paragraph_chunk, position=tqdm_position, desc=f"{tqdm_position:2d}core", leave=None, total=len(paragraph_chunk)):
        paragraph = paragraph.replace("\n\n","")
        chunk.append(tokenizer.encode(paragraph))
    return chunk

def pretokenize():
    dataset = datasets.load_dataset("roneneldan/TinyStories")
    text = dataset["train"]["text"]
    print("text obtained")

    chunk_list = lambda l, n: [l[i:i+n] for i in range(0, len(l), n)]
    tqdm_position = lambda n: [i for i in range(0, n)]

    with ProcessPoolExecutor(max_workers=NUM_CPU_THREAD) as e:
        chunks = e.map(process_paragraph, chunk_list(text[:PARAGRAPH_SIZE], PARAGRAPH_SIZE//NUM_CPU_THREAD), tqdm_position(NUM_CPU_THREAD))
    print("encoding done")

    tokens_list = [tokens for chunk in chunks for tokens in chunk]

    tokens_packed = [token for tokens in tokens_list for token in tokens]
    tokens_packed = torch.tensor(tokens_packed).int()
    offsets_packed = torch.tensor([0] + [len(tokens) for tokens in tokens_list])
    offsets_packed = torch.cumsum(offsets_packed, dim=0)

    print(f"saving tokens as binary files(.pt)")
    torch.save(tokens_packed, os.path.join(DATA_CACHE_DIR, "tiny_tokens.pt"))
    torch.save(offsets_packed, os.path.join(DATA_CACHE_DIR, "tiny_offsets.pt"))
    print(f"done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["train_vocab", "pretokenize"])
    args = parser.parse_args()

    if args.stage == "train_vocab":
        train_vocab()
    elif args.stage == "pretokenize":
        pretokenize()
    else:
        raise ValueError(f"Unknown stage {args.stage}")
