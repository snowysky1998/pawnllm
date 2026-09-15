# Enhanced chess transformer and interactive analysis

> Historical experiment: the current web UI uses the [rating-conditioned human model](chess-human.md).
> The old unconditioned-checkpoint web commands below describe the earlier version.

## What changed

The original `pawnllm/model.py` is a Llama-style text model. Chess play uses the
separate board encoder in `chess_model.py`; neither chess preset uses that text model.

The new **enhanced** preset has 22,776,967 parameters, eight layers, width 512,
16 heads, and eight historical positions. It adds:

- SwiGLU gated feed-forward layers.
- Layer normalization on attention queries and keys.
- Learned two-dimensional relative attention biases for board-square displacements,
  in addition to the existing board-conditioned geometric attention bias (GAB).

Both presets retain legal source/destination policies, promotion support, and a
win/draw/loss value head. The original 5.08M preset and its checkpoints remain
loadable. This is a combined architecture/scale experiment; it does not isolate the
contribution of each architectural change.

## Controlled pilot comparison

Both models were trained from scratch for **2,000 optimizer steps** on the same
72,832 training positions, with the same seed, batch size 128, learning rate,
optimizer, value-loss weight, and validation schedule. Validation evaluated every
500 steps. Each model's checkpoint was selected using validation policy
cross-entropy + 0.5 × value cross-entropy, before evaluating the test split.

| Metric | Small control | Enhanced |
| --- | ---: | ---: |
| Parameters | 5.08M | 22.78M |
| Selected checkpoint step | 2,000 | 1,500 |
| Test move accuracy | 30.35% | 30.48% |
| Test top-three move accuracy | 53.97% | 52.89% |
| Test policy cross-entropy | 2.4472 | 2.4506 |
| Test WDL accuracy | 51.30% | 53.33% |
| Test value cross-entropy | 0.9403 | 0.9055 |
| Test value Brier score (lower is better) | 0.5669 | 0.5500 |
| Training-loop positions/sec | ~11,720 | ~4,630 |
| Peak PyTorch allocated GPU memory | 0.643 GiB | 2.077 GiB |
| Peak PyTorch reserved GPU memory | 0.773 GiB | 2.234 GiB |
| Sampled host PSS at final training log | 1.66 GiB | 1.77 GiB |

The test split contains 1,575 positions from held-out games. The enhanced model
improves value metrics in this run; move prediction is broadly comparable. This is
one seed on a very small sample, and the test split was also used in the earlier
pilot. These are exploratory results, not a statistically established improvement
or evidence that these probabilities are calibrated.

At the final 2,000-step checkpoint (rather than validation-selected checkpoints),
validation move accuracy was 31.0% for the small control and 29.9% for enhanced;
validation WDL accuracy was 49.7% and 51.2%, respectively. This illustrates why
additional capacity and more training steps do not guarantee improvement.

A 16-game policy-only match check gave enhanced **4 wins, 1 loss, and 11 draws**.
Games used paired colors with eight shared random opening plies, seed 73, a maximum
of 200 additional plies, and automatic draw claims. No games hit the ply limit.
This small check does not support a reliable Elo estimate. Both models still need
much more training and stronger evaluation.

The GPU figures are PyTorch allocator measurements, and host PSS is sampled,
including loader workers. Throughput uses the training-loop timer, excluding initial
setup/evaluation and final evaluation/save; it is not sustained archive-to-training
throughput. The same prefix-sampling limitations described in
[the initial pilot report](chess-pilot.md) apply.

Local artifacts:

- `runs/chess-enhanced/best.pt` — enhanced checkpoint served in the UI.
- `runs/chess-enhanced/last.pt` — final enhanced checkpoint.
- `runs/chess-pilot-control/best.pt` — comparison checkpoint.
- `runs/chess-comparison.json` — test metrics.
- `runs/chess-enhanced/versus-pilot.pgn` and `versus-pilot.json` — match records.
- Each run's `metrics.jsonl` — validation curves and resource measurements.

## Interactive policy/value UI

```sh
uv run python -m pawnllm.chess_web \
  --host 0.0.0.0 --checkpoint runs/chess-enhanced/best.pt
```

Open port 8000 on this machine. The server continues to bind to localhost by default
unless `--host` is passed; the enhanced checkpoint must be explicitly selected.

The sidebar shows:

- **Position value:** White win / draw / Black win probabilities for the displayed
  board. The perspective stays White even when you play Black or inspect a Black move.
- **Move policy:** the top eight legal moves with probabilities normalized across
  *all* legal moves, not renormalized over the displayed eight. The side to move and
  displayed share of policy mass are labeled.
- **Policy continuations:** choose one, two, or three half-moves. Each row starts
  with its candidate move, then follows the network's highest-policy reply. The
  terminal displayed value is evaluated at the end of that line. It is **White's
  expected points**, P(White win) + 0.5 × P(draw), not a centipawn score.
- **Interactive exploration:** hover a move to highlight its squares, click a move
  or any continuation step to preview that position, and inspect the updated policy
  and value there. Board clicks within a preview explore further. “Back” steps
  backward and “Return to game” restores the live board. Exploring never changes
  or persists the actual game. Click a move in game history to inspect earlier
  positions, including the position before an engine reply.

The UI values come from this human-game-trained model, not Stockfish. Terminal
results override network estimates with exact outcomes. Branches stop at terminal
positions. The continuations are bounded greedy policy rollouts, **not minimax or
MCTS search**; increasing preview length does not change engine play, which still
uses the highest-policy legal move. Batched sibling evaluations keep the workload
small: one root batch plus at most three batches of eight positions per request.

The opening position was excluded from policy training by the pilot's first-20-plies
filter, so its opening probabilities can be especially unreliable.

## Reproduce training and validation

Use fresh output directories:

```sh
uv run python -m pawnllm.chess_train --model enhanced --steps 2000 \
  --eval-every 500 --log-every 250 --output runs/chess-enhanced-new
uv run python -m pawnllm.chess_train --model pilot --steps 2000 \
  --eval-every 500 --log-every 250 --output runs/chess-control-new
uv run python -m pawnllm.chess_train --resume runs/chess-enhanced-new/best.pt --evaluate test
uv run python -m unittest discover -s tests -v
```

The test suite covers backward-compatible checkpoints, enhanced-layer gradients,
legal policy normalization, White/Black value perspective, exact terminal results,
legal bounded rollouts, and unchanged source boards. Browser checks exercise score
rendering, hover highlights, depth changes, variation/history inspection, return to
live play, saved games, takebacks, colors, terminal responses, and mobile overflow.
