# Human move prediction with selectable ratings

## Current behavior

The web app is an analysis board. You control both White and Black; neither side
moves automatically. White and Black have separate rating selectors from 600 to
2600, in 100-point increments. Ratings refer to the Lichess blitz population used
for training, not a certified playing strength of the network.

The model predicts **what a human player would play**, given the board/history,
that player's rating, and the opponent's rating. It has not been trained to select
the objectively best move. The top three human-move predictions are drawn as gold,
teal, and purple arrows, matching the first three policy rows. The overlay rotates
with the board and does not intercept clicks. Promotion predictions sharing source
and destination are offset so that separate promotion arrows remain visible.

Changing either rating recalculates the probabilities without making a move. Both
ratings are preserved by color through every continuation, even though the model
inputs are ordered as side-to-move rating and opponent rating. Outcome probabilities
remain White win / draw / Black win, conditioned on those same player ratings.
Terminal results are exact. Continuations are greedy human-policy rollouts up to
three half-moves, not best-play search. Any continuation can be explored without
changing the main line. Undo removes one half-move, and return-to-main-line refreshes
its scores at the currently selected ratings. Moves and ratings survive refresh.

## Data and model

Source: `/mnt/archive/lichess/elo-conditioned/`, the all-rating archive.

This pilot accepted **25,000 blitz games** after scanning 321,101 rows across all
91 monthly files. Both players must be between 600 and 2600, the result must be
known, and termination must be `Normal`. There are equal quotas of 5,000 games in
five bands of the *mean* player rating: 600–999, 1000–1399, 1400–1799, 1800–2199,
and 2200–2600. This balances game averages, not the exact number of positions for
every individual rating or every year.

Eight positions are sampled per game, including openings:

| Split | Games | Positions |
| --- | ---: | ---: |
| Train | 24,018 | 192,144 |
| Validation | 502 | 4,016 |
| Test | 480 | 3,840 |

Splits are determined by game ID. Monthly archive streams are interleaved from
their beginnings, so this is a bounded pilot sample, not uniform sampling from the
entire archive. No engine evaluations are used. The retained format has no per-move
clock information, so time-pressure moves cannot be excluded individually.

The **22,810,439-parameter** model uses the enhanced square-token encoder (eight
layers, width 512, geometric and 2-D relative attention biases, Q/K normalization,
and SwiGLU feed-forward layers). A trainable nonlinear embedding of the two ratings
is added to each square token. Both ratings are normalized as `(rating - 1600)/1000`.
This provides real learned conditioning, rather than changing sampling temperature
or adding noise to a strong-player policy.

Training ran from scratch for **5,000 steps** with batch size 128, AdamW, BF16,
learning rate 0.0003, 100 warmup steps, and policy CE + 0.5 × value CE. Validation
selected the checkpoint at **step 4,500**. The final checkpoint had worse policy
loss, so it is not served. Training used ~2.083 GiB of peak PyTorch GPU allocation,
2.25 GiB reservation, and ~1.78 GiB of sampled host PSS at the final log; the loop
reported ~4,540 positions/sec. These are short-run allocator/sampled measurements,
not a continuous total-machine peak or archive-to-training throughput.

## Full held-out evaluation

| Metric | Validation (4,016 positions) | Test (3,840 positions) |
| --- | ---: | ---: |
| Exact human-move agreement | 36.68% | 36.02% |
| Human move among top three | 62.05% | 59.77% |
| Outcome accuracy | 59.94% | 59.82% |
| Policy cross-entropy | 2.1769 | 2.2362 |
| Value cross-entropy | 0.8106 | 0.7556 |

By **side-to-move rating**:

| Rating | Val positions | Val move agreement | Test positions | Test move agreement |
| --- | ---: | ---: | ---: | ---: |
| 600–999 | 848 | 33.84% | 646 | 36.07% |
| 1000–1399 | 800 | 36.63% | 733 | 36.15% |
| 1400–1799 | 880 | 40.11% | 829 | 37.39% |
| 1800–2199 | 901 | 36.85% | 847 | 38.72% |
| 2200–2600 | 587 | 35.43% | 785 | 31.46% |

As a conditioning check, replacing both actual ratings with 1500 on the same
validation positions increased policy cross-entropy from **2.1769 to 2.2066**, and
reduced move agreement from **36.68% to 36.38%**. Outcome accuracy fell from 59.94%
to 55.83%. This is an inference-time rating ablation, not a separately trained
unconditioned baseline or a statistical significance test.

These results use a different dataset and include openings, so they should not be
compared directly with the previous strength-filtered model's accuracy. The
selectors demonstrably affect the learned probabilities, but precise fidelity to
each requested rating has not been established. More games, seeds, held-out months,
and calibration checks are needed before claiming a faithful model of each level.

## Reproduce

```sh
uv run python -m pawnllm.chess_data \
  --source /mnt/archive/lichess/elo-conditioned \
  --output data/chess-human-new --rating-conditioned \
  --speeds blitz --min-elo 600 --min-ply 0 \
  --games 25000 --max-scan 3000000

uv run python -m pawnllm.chess_train --model human \
  --data data/chess-human-new --output runs/chess-human-new \
  --steps 5000 --eval-every 500 --log-every 500

uv run python -m pawnllm.chess_train --data data/chess-human-new \
  --resume runs/chess-human-new/best.pt --evaluate val --eval-batches 0

uv run python -m pawnllm.chess_web --host 0.0.0.0 \
  --checkpoint runs/chess-human-new/best.pt

uv run python -m unittest discover -s tests -v
```

Current local artifacts: `data/chess-human/manifest.json`,
`runs/chess-human/best.pt`, `runs/chess-human/metrics.jsonl`, and
`runs/chess-human/evaluation.json`. These generated files are ignored by Git.

The tests cover rating order during preparation and analysis, trainable conditioning
and checkpoint round trips, manual control of both colors, and existing chess rules
and value-perspective behavior. Browser checks also cover score changes with ratings,
three-arrow mapping and board flips, one-ply undo, variation persistence, saved
ratings, invalid rating rejection, responsive layout, and terminal arrow removal.
