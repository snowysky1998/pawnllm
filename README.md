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
$ python pawnllm/infer.py -c checkpoint15.pt -p "Once upon a time, there was a big bear"
['<bos>', 'Once', 'upon', 'a', 'time', ',', 'there', 'was', 'a', 'big', 'bear', 'named', 'Benny', '.', 'Benny', 'loved', 'to', 'play', 'with', 'his', 'friends', '.', 'One', 'day', ',', 'Benny', "'", 's', 'friend', ',', 'a', 'little', 'bird', 'named', 'Lily', ',', 'came', 'to', 'visit', 'him', '.', 'Lily', 'was', 'very', 'happy', 'to', 'see', 'him', 'and', 'they', 'played', 'together', '.', 'A', 's', 'they', 'played', ',', 'Benny', 'saw', 'a', 'big', ',', 'scary', 'monster', '.', 'The', 'monster', 'was', 'scared', 'and', 'ran', 'away', '.', 'Benny', 'was', 'scared', 'and', 'didn', "'", 't', 'know', 'what', 'to', 'do', '.', 'Lily', 'told', 'him', 'that', 'he', 'was', 'not', 'safe', 'to', 'play', 'with', 'his', 'friends', '.', 'Benny', 'didn', "'", 't', 'know', 'what', 'to', 'do', '.', 'L', 'ater', 'that', 'day', ',', 'Benny', "'", 's', 'friend', 'came', 'to', 'visit', '.', 'Benny', 'saw', 'the', 'monster', 'and', 'said', ',', '"', 'I', "'", 'm', 'sorry', ',', 'Lily', '.', 'I', 'didn', "'", 't', 'mean', 'to', 'hurt', 'you', '."', 'Benny', 'forgave', 'him', 'and', 'they', 'played', 'together', '.', 'From', 'that', 'day', 'on', ',', 'Benny', 'and', 'Lily', 'became', 'good', 'friends', 'and', 'played', 'together', 'every', 'day', '.', '<eos>']
Once upon a time, there was a big bear named Benny. Benny loved to play with his friends. One day, Benny's friend, a little bird named Lily, came to visit him. Lily was very happy to see him and they played together.As they played, Benny saw a big, scary monster. The monster was scared and ran away. Benny was scared and didn't know what to do. Lily told him that he was not safe to play with his friends. Benny didn't know what to do.Later that day, Benny's friend came to visit. Benny saw the monster and said, "I'm sorry, Lily. I didn't mean to hurt you." Benny forgave him and they played together. From that day on, Benny and Lily became good friends and played together every day.
```

## Credits

[karpathy/llama2.c](https://github.com/karpathy/llama2.c)

[clabrugere/scratch-llm](https://github.com/clabrugere/scratch-llm)

## Chess policy/value transformer

A separate chess pipeline now trains a 5.08M-parameter, 64-square transformer on
`/mnt/archive/lichess/strength-filtered`. It predicts legal moves and win/draw/loss
probabilities, using compact streaming-prepared shards to bound host RAM use.

```sh
uv sync
uv run python -m pawnllm.chess_data --output data/chess-pilot-new --games 10000
uv run python -m pawnllm.chess_train --data data/chess-pilot-new --output runs/chess-pilot-new --steps 1000
uv run python -m pawnllm.chess_play --checkpoint runs/chess-pilot-new/last.pt
uv run python -m unittest discover -s tests -v
```

See [the pilot report](docs/chess-pilot.md) for measured RTX 4090 memory/throughput,
data sampling limits, checkpoint resume, evaluation, and match commands. The initial
500-step pilot reached 28.6% move accuracy on 1,575 held-out test positions; this is
an implementation baseline, with playing strength still to be established.

Play against the completed pilot in your terminal:

```sh
uv run python -m pawnllm.chess_play --checkpoint runs/chess-pilot/last.pt --human white
```

Use `--human black` to play Black. Enter moves such as `e4`, `Nf3`, or `e2e4`;
type `moves` to list legal moves, `resign` to resign, or `quit` to exit.

### Human move analysis with selectable Elo

```sh
uv run python -m pawnllm.chess_web --host 0.0.0.0
```

The browser UI now uses `runs/chess-human-calibrated/best.pt`, the refined 22.8M
rating-conditioned model with validation-fitted human-move probabilities. Open port 8000 on this machine.
Without `--host`, the server binds to localhost.
To use the original training checkpoint, pass `--checkpoint runs/chess-human/best.pt`.

- Set **White Elo** and **Black Elo** independently, from 600 to 2600.
- Move **both sides** yourself. The board never makes an automatic reply.
- The **top three likely human moves** appear as gold, teal, and purple arrows.
- The sidebar shows human-move probabilities and predicted game outcomes at the
  chosen ratings. These are not best-move recommendations or verified engine Elo.
- Inspect candidate lines up to three half-moves, or click move history. Use
  **Return to main line** to leave a variation; **Undo move** removes one half-move.
- Board orientation, promotion choices, and saved moves/ratings are supported.

No frontend build or extra dependencies are required. Use `--port 8001` or
`--device cpu` to change the defaults; stop with Ctrl+C. The selectable-rating UI
requires a rating-conditioned checkpoint and refuses older unconditioned models.

See [the human-model report](docs/chess-human.md) for preparation/training commands,
and [the probability evaluation](docs/human-probability-comparison.md) for the current
checkpoint, Elo-balanced log loss, calibration, Stockfish comparison, and reproduction commands.
The [official Maia comparison](docs/maia-probability-comparison.md) reports human-move
probability for PawnLLM, Maia-3, Stockfish, and uniform random on the same positions. The earlier
[enhanced-model comparison](docs/chess-enhanced.md) records the strength-filtered
experiment; its web launch instructions have been superseded by this analysis UI.
