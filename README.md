# PawnLLM

<img height="190" style="" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Chess_piece_-_White_pawn.JPG/134px-Chess_piece_-_White_pawn.JPG">

Minimal and educational LLM trained on the tinystories dataset in 421 lines of code.

## Dependencies:

- PyTorch

- einops

- tqdm

- datasets (for getting huggingface tinystories dataset)

- sentencepiece

## Instructions:

```
# install dependencies
pip install -r requirements.txt

# download dataset, train tokenizer, and pretokenize
python pawnllm/tinystories.py --stage train_vocab
python pawnllm/tinystories.py --stage pretokenize1
python pawnllm/tinystories.py --stage pretokenize2 --seq_len 512
# python pawnllm/tinystories.py --stage all

# train model on GPU with mixed precision
python pawnllm/train.py
# resume training from specific checkpoint
# python pawnllm/train.py -c checkpoint10.pt

# run inference with prompt
# python pawnllm/infer.py -c checkpoint15.pt
python pawnllm/infer.py -c checkpoint15.pt -p "Once upon a time, there was a big bear"
```

## Example output

```
$ python pawnllm/infer.py -c checkpoint_backup15.pt -p "Once upon a time, there was a big bear"
['<bos>', 'Once', 'upon', 'a', 'time', ',', 'there', 'was', 'a', 'big', 'bear', 'named', 'Benny', '.', 'Benny', 'loved', 'to', 'play', 'with', 'his', 'friends', '.', 'One', 'day', ',', 'Benny', "'", 's', 'friend', ',', 'a', 'little', 'bird', 'named', 'Lily', ',', 'came', 'to', 'visit', 'him', '.', 'Lily', 'was', 'very', 'happy', 'to', 'see', 'him', 'and', 'they', 'played', 'together', '.', 'A', 's', 'they', 'played', ',', 'Benny', 'saw', 'a', 'big', ',', 'scary', 'monster', '.', 'The', 'monster', 'was', 'scared', 'and', 'ran', 'away', '.', 'Benny', 'was', 'scared', 'and', 'didn', "'", 't', 'know', 'what', 'to', 'do', '.', 'Lily', 'told', 'him', 'that', 'he', 'was', 'not', 'safe', 'to', 'play', 'with', 'his', 'friends', '.', 'Benny', 'didn', "'", 't', 'know', 'what', 'to', 'do', '.', 'L', 'ater', 'that', 'day', ',', 'Benny', "'", 's', 'friend', 'came', 'to', 'visit', '.', 'Benny', 'saw', 'the', 'monster', 'and', 'said', ',', '"', 'I', "'", 'm', 'sorry', ',', 'Lily', '.', 'I', 'didn', "'", 't', 'mean', 'to', 'hurt', 'you', '."', 'Benny', 'forgave', 'him', 'and', 'they', 'played', 'together', '.', 'From', 'that', 'day', 'on', ',', 'Benny', 'and', 'Lily', 'became', 'good', 'friends', 'and', 'played', 'together', 'every', 'day', '.', '<eos>']
Once upon a time, there was a big bear named Benny. Benny loved to play with his friends. One day, Benny's friend, a little bird named Lily, came to visit him. Lily was very happy to see him and they played together.As they played, Benny saw a big, scary monster. The monster was scared and ran away. Benny was scared and didn't know what to do. Lily told him that he was not safe to play with his friends. Benny didn't know what to do.Later that day, Benny's friend came to visit. Benny saw the monster and said, "I'm sorry, Lily. I didn't mean to hurt you." Benny forgave him and they played together. From that day on, Benny and Lily became good friends and played together every day.
```

## Credits

[karpathy/llama2.c](https://github.com/karpathy/llama2.c)

[clabrugere/scratch-llm](https://github.com/clabrugere/scratch-llm)
