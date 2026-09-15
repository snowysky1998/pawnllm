# Policy/value transformer pilot — 2026-09-15

## Outcome

The first pipeline runs on the local RTX 4090 within the 32 GB host RAM budget.
The 5,079,047-parameter model learns to predict held-out moves. This short run
establishes correctness and feasibility; it does not establish competitive engine strength.

| Measurement | Result |
| --- | --- |
| Optimizer steps | 500 |
| Microbatch / accumulation | 128 / 1 |
| Precision | BF16 on CUDA; FP32 parameters and losses |
| Validation positions | 1,722 |
| Validation move accuracy, initialization → step 500 | 7.43% → 27.35% |
| Validation policy cross-entropy | 3.320 → 2.636 |
| Test positions | 1,575 |
| Test move accuracy / top-three accuracy | 28.63% / 48.38% |
| Test WDL accuracy | 44.76% |
| Test largest outcome class | 44.13% |
| Uniform legal-move expected test accuracy | 6.00% |
| Training-loop throughput | ~11,500 positions/s |
| Peak PyTorch GPU allocation / reservation | 0.643 / 0.773 GiB |
| Peak trainer RSS | 1.75 GiB |
| Sampled trainer + loader worker PSS during resume | 1.84 GiB |
| Games against random legal moves | 2 wins, 6 draws, 0 losses; no ply-limit truncations |

The value head is only marginally above the test majority-class baseline. Improving
it needs more training and potentially better labels. Eight games against a random
opponent test the game loop; they provide no useful Elo estimate. No Stockfish
executable was installed, so the optional UCI-opponent path has not been exercised
against a real engine.

Throughput is the trainer's cumulative positions divided by loop wall time, including
intermediate validation/checkpoint time but excluding initial setup and validation,
preprocessing, and the final evaluation/save. This is a short run on small cached
shards, not a sustained archive-to-training throughput measurement. GPU figures are
PyTorch allocator measurements and exclude driver/context overhead. PSS counts
shared host pages proportionally; the 1.84 GiB figure is a sample from the separate
10-step resume check, not a continuous peak of the original run.

## Data

- Source: `/mnt/archive/lichess/strength-filtered`, all 91 monthly files.
- Pilot contains 9,614 games and 76,129 sampled positions: 72,832 train,
  1,722 validation, and 1,575 test. Compressed shards occupy about 5.9 MiB.
- Both players >= 2000, termination `Normal`, known outcome, and positions after
  the first 20 plies. Up to eight positions are sampled per accepted game.
- Speed counts: 3,334 blitz, 3,333 rapid, 2,947 classical. The scan budget was
  reached before the requested 10,000 games and equal speed quotas were filled.
- Monthly streams are interleaved from their beginnings. This spreads the pilot
  across months but is **not uniform sampling within a month**, nor a guarantee of
  equal speed proportions in every year. A larger experiment needs a more thorough
  sampling plan and per-year/speed evaluation.
- Game IDs select splits deterministically using BLAKE2b. All positions in the
  same game stay together, including if that ID appears in another dataset.
  Transpositions can still occur across splits; this is not a position-disjoint test.
- 2,919 sampled games have evaluation strings. These strings are counted but
  intentionally not consumed as training labels; per-ply alignment, missing entries,
  mate scores, and evaluation quality still need auditing.
- Targets are the played human move and the eventual game result from the current
  player's perspective. Resignations and subsequent human mistakes make results
  noisy estimates of position strength.

The pilot manifest records the original preprocessing arguments and file list.
Its original scan loop inspected 1,000,090 rows; the boundary was subsequently fixed
and tested so future runs respect `--max-scan` exactly.

## Architecture

A Chessformer-inspired encoder with eight layers, width 256, eight attention heads,
MLP expansion two, and eight historical board positions represented as features
of 64 square tokens. Boards and moves use the side-to-move perspective, consistently
across the history window. Missing history repeats the earliest available board.

GAB mixes 64 learned square-pair templates shared across layers, with mixtures
conditioned on pooled board features. This is a compact implementation inspired by
[Chessformer](https://arxiv.org/html/2605.19091v1), not an exact reproduction of its
training setup or every architectural detail.

The policy has 4,096 source/destination logits plus 256 promotion logits (including
all four promotion pieces). Legal-move masking applies during training and play.
The value head predicts loss, draw, and win probabilities. Training minimizes
policy cross-entropy plus 0.5 times value cross-entropy, with AdamW, gradient clipping,
and a linear learning-rate warmup.

Rule features include castling rights, en passant, the halfmove clock, and the
current position's repetition count. A finite history and repetition summary do
not encode every historical position needed for future repetition claims.
`python-chess` retains the full game state and enforces legality/terminal outcomes
in the match runner. FEN-only inference cannot recover missing game history.

## Files and commands

Run from the repository root:

```sh
uv sync

# Prepare a new pilot directory. Existing output directories are refused.
uv run python -m pawnllm.chess_data --output data/chess-pilot-new --games 10000

# Bounded training run; produces best.pt, last.pt, metrics.jsonl.
uv run python -m pawnllm.chess_train --data data/chess-pilot-new \
  --output runs/chess-pilot-new --steps 1000

# Resume optimizer/model state. --steps is the total desired step count.
uv run python -m pawnllm.chess_train --data data/chess-pilot-new \
  --output runs/chess-pilot-new --resume runs/chess-pilot-new/last.pt --steps 2000

# Evaluate the checkpoint from this completed pilot.
uv run python -m pawnllm.chess_train --resume runs/chess-pilot/last.pt --evaluate test

# Pick one legal move (optionally pass --fen).
uv run python -m pawnllm.chess_play --checkpoint runs/chess-pilot/last.pt

# Paired colors, shared eight-ply randomized openings, PGN output.
uv run python -m pawnllm.chess_play --checkpoint runs/chess-pilot/last.pt \
  --games 20 --pgn runs/random-matches.pgn

# Optional external UCI engine; replace path with an installed executable.
uv run python -m pawnllm.chess_play --checkpoint runs/chess-pilot/last.pt \
  --games 20 --engine /path/to/stockfish --nodes 1000 --pgn runs/engine-matches.pgn

uv run python -m unittest discover -s tests -v
```

The eight tests cover special moves and both colors, canonical board/history
encoding, repetition, pre-move targets, legal masks and a tiny overfit, streaming
preparation and multiworker loading, exact scan limits, and checkpoint round trips.
Real-data CUDA training, held-out evaluation, ten-step optimizer resume, and eight
complete games were also exercised.

Resume restores model/optimizer state and training RNG state but starts a newly
shuffled data pass. It does not restore the exact loader cursor. Checkpoints are
written atomically at evaluation boundaries. Data and checkpoints are ignored by
Git; the reproducible code and this report are tracked when committed.

## Next experiment

Prepare a larger, better distributed sample while retaining these memory bounds,
then run a longer fixed-compute baseline. Track value calibration, move accuracy
by year/speed, tactical performance, and matches against a fixed engine opponent.
Audit existing Stockfish labels before adding value distillation. Profile sustained
preprocessing throughput before increasing model size or purchasing compute.
