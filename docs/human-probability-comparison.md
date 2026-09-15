# Human move probability evaluation

The primary score is **human-move negative log likelihood (NLL, natural-log units)**,
with the five Elo bands weighted equally. Lower is better. Actual recorded human
moves are the targets, including mistakes. Top-three agreement remains a secondary
metric. The game-outcome head is separate and does not determine checkpoint selection.

## Changes and training

- Human training now defaults to `macro-policy` checkpoint selection: average policy
  NLL across the five rating bands. Other models retain the joint policy/value default.
- Per-band validation output includes NLL and top-three agreement.
- Continued the original step-4500 checkpoint for 1,500 steps at learning rate 3e-5
  (previously 3e-4). Batch 128; same training data and value-loss weight 0.5.
- Selected **step 5500** using validation macro NLL, before computing test results.
- Fit temperature and uniform-tail probability calibration independently for each
  rating band using validation only. Both PawnLLM and the Stockfish baseline receive
  the same calibration family. Search: 81 temperatures and 22 tail mixtures from 0
  through 0.95. Temperature ranges: 0.25–4 for model logits, 10–5000 for centipawns.
  The tail grid was expanded because a validation optimum hit the initial boundary.
- Exported `runs/chess-human-calibrated/best.pt` with the refined weights and fitted
  calibration. The analysis UI uses this checkpoint by default. Calibration uses
  the active player's rating and all legal moves before selecting display candidates.
  Outcome probabilities are unchanged by policy calibration.
- Raw training/evaluation logits stay uncalibrated; `pawnllm.human_benchmark` reports
  calibrated and uncalibrated metrics explicitly.

The additional training took about 46 seconds inside the training loop, including
periodic evaluations/checkpoint writes. Peak CUDA allocation was 2.08 GiB; sampled
trainer-plus-workers host memory was about 2.07 GiB. The original dataset remains
192,144 training positions from 24,018 games, 4,016 validation positions from 502
games, and 3,840 test positions from 480 games. No new games were added.

## Overall test results

NLL in this table is **equal-weight mean across rating bands**. Top-three accuracy,
Brier score and calibration error are position-weighted. Calibration error is
10-bin top-label expected calibration error (ECE); it does not establish calibration
of every individual move or of the value head.

| System | N | Macro NLL ↓ | Top 3 ↑ | Policy Brier ↓ | Top-label ECE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original PawnLLM, raw | 3,840 | 2.2361 | 59.77% | 0.7831 | 4.13% |
| Original PawnLLM, calibrated | 3,840 | 2.2324 | 59.77% | 0.7818 | 3.46% |
| Refined PawnLLM, raw | 3,840 | 2.2425 | 59.14% | 0.7839 | 6.74% |
| Refined PawnLLM, calibrated | 3,840 | 2.2135 | 59.14% | 0.7760 | 2.88% |
| Stockfish + fitted probability model | 3,840 | 2.3482 | 63.39% | 0.8065 | 12.10% |
| Refined, both inputs fixed at 1500 (raw) | 3,840 | 2.2583 | 60.16% | 0.7930 | 9.14% |

Fine-tuning alone did not improve test NLL. Fine-tuning plus calibration improved
probability quality, while top-three accuracy fell slightly. This is not an
across-the-board improvement. Equal calibration of the original model is included
so that the effect of refinement is not confused with adding calibration.

## Per-rating validation and test comparison

Each cell contains **N / NLL / top-three agreement**. Validation was used for model
selection and calibration, so its results are fitted rather than unbiased estimates.

### Validation

| Rating | Original, raw | Refined, calibrated | Stockfish + calibration |
| --- | ---: | ---: | ---: |
| 600-999 | 848 / 2.273 / 61.32% | 848 / 2.221 / 57.67% | 848 / 2.584 / 54.72% |
| 1000-1399 | 800 / 2.188 / 61.75% | 800 / 2.155 / 61.12% | 800 / 2.463 / 60.75% |
| 1400-1799 | 880 / 2.104 / 62.05% | 880 / 2.080 / 62.73% | 880 / 2.326 / 65.91% |
| 1800-2199 | 901 / 2.087 / 65.26% | 901 / 2.044 / 64.59% | 901 / 2.260 / 65.70% |
| 2200-2600 | 587 / 2.269 / 58.60% | 587 / 2.226 / 58.94% | 587 / 2.052 / 72.23% |
| Overall | 4,016 / 2.184 / 62.05% | 4,016 / 2.145 / 61.21% | 4,016 / 2.337 / 63.40% |

### Test

| Rating | Original, raw | Refined, calibrated | Stockfish + calibration |
| --- | ---: | ---: | ---: |
| 600-999 | 646 / 2.193 / 59.75% | 646 / 2.191 / 59.29% | 646 / 2.456 / 59.60% |
| 1000-1399 | 733 / 2.239 / 59.48% | 733 / 2.201 / 58.94% | 733 / 2.465 / 58.94% |
| 1400-1799 | 829 / 2.203 / 61.76% | 829 / 2.183 / 61.04% | 829 / 2.372 / 62.61% |
| 1800-2199 | 847 / 2.171 / 62.22% | 847 / 2.157 / 61.75% | 847 / 2.243 / 66.00% |
| 2200-2600 | 785 / 2.374 / 55.29% | 785 / 2.336 / 54.39% | 785 / 2.206 / 68.66% |
| Overall | 3,840 / 2.236 / 59.77% | 3,840 / 2.214 / 59.14% | 3,840 / 2.348 / 63.39% |

## Openings and move-quality strata

NLL below is **position-weighted within each subset**. These are complementary
views of the full test set, not alternative subsets selected to declare a winner.
All quality categories are included. Engine quality labels are approximate, from
the same finite-budget all-legal search; they are not authoritative blunder judgments.
Mate scores map to ±100,000 centipawns for these categories.

| Test subset | N | Original NLL | Refined + calibrated NLL | Stockfish + calibration NLL |
| --- | ---: | ---: | ---: | ---: |
| First 20 half-moves | 1,522 | 2.089 | 2.033 | 2.432 |
| After first 20 half-moves | 2,318 | 2.333 | 2.332 | 2.281 |
| Engine loss <50 cp | 2,633 | 1.937 | 1.923 | 1.623 |
| Engine loss 50–199 cp | 804 | 2.878 | 2.813 | 3.418 |
| Engine loss ≥200 cp | 403 | 2.912 | 2.911 | 4.882 |

**Stockfish wins after the opening.** Its equal-band NLL is 2.289 versus 2.320 for
refined PawnLLM. PawnLLM's overall advantage is strongly influenced by openings;
it also assigns more probability to the human moves in the higher engine-loss
categories. This does not establish superiority on a Maia-style filtered benchmark.

## Paired uncertainty

95% percentile intervals use 2,000 bootstrap resamples of whole test games, preserving
all sampled positions within each game and pairing the systems. Negative differences
favor the first system. These intervals condition on the fitted models/calibrators;
they do not include training-seed or calibration-fit uncertainty.

| Macro NLL difference | Estimate | 95% interval |
| --- | ---: | ---: |
| refined_raw minus original_raw | 0.0064 | [-0.0135, 0.0259] |
| refined_calibrated minus original_calibrated | -0.0188 | [-0.0357, -0.0014] |
| refined_calibrated minus stockfish_calibrated | -0.1347 | [-0.1888, -0.0815] |
| refined_raw minus refined_fixed1500_raw | -0.0158 | [-0.0338, 0.0020] |

## Stockfish protocol

Stockfish 19, Skill Level 20, UCI_LimitStrength=false, one thread and 64 MiB hash
per engine, four workers, fresh game/hash for each position, no tablebases.
Every legal root move was evaluated with **100,000 total nodes per position**.
This run took about 144 seconds. It is a different search configuration from the
[earlier comparison](stockfish-comparison.md), which used 10,000 nodes and MultiPV=3.

The probability baseline is:

`(1 - mixture) * softmax(clip(centipawns, -2000, 2000) / temperature) + mixture / legal_count`

Parameters depend on the human rating band, fitted on validation only. This is a
**Stockfish-based probability model**, not native Stockfish human-move probabilities
or the UCI_Elo weakening feature. Other probability mappings could perform differently.
Full original game histories are restored for Stockfish; PawnLLM receives its eight
board snapshots plus rule features and ratings. Compute and history are not matched.

For the probability baseline, candidates are ranked by its probability distribution;
clipping and unequal final search depths can make this differ slightly from the
engine's reported MultiPV ordering. Native all-legal MultiPV top-three agreement was
63.65%; the fitted-distribution ranking gives 63.39%. The previous 10k/MultiPV=3 run
remains 65.34%; it is not overwritten or reinterpreted.

Model ranking uses the original action-space torch argmax/topk tie-breaking before
calibration. This matters for tied BF16 scores: breaking model ties in Stockfish
move order would bias accuracy. Calibration preserves ranking, and tests explicitly
check that the evaluator preserves the original tie behavior. The original model's
reported top-three accuracy still exactly matches its earlier 59.77% test result.

## Limitations

The test games are disjoint from training and validation, but their earlier results
were already inspected. This is an **exploratory comparison on the existing pilot**,
not a fresh confirmatory test. Model selection and calibration share validation.
There is no temporal holdout, no player-disjoint guarantee, and no per-move clock
information in the TSV archive. Within-month prefix sampling and Normal-termination
filtering limit representativeness. Repeated opening positions can occur across splits.
No claim is made about all 886 million archived games or a verified playing Elo.

## Reproduce

The existing verified reconstruction is described in the earlier comparison.
Use fresh output directories when repeating engine runs.

```sh
uv run python -m pawnllm.chess_train --data data/chess-human \
  --resume runs/chess-human/best.pt --output runs/chess-human-refined \
  --steps 6000 --lr 0.00003 --selection-metric macro-policy \
  --eval-every 250 --log-every 250

uv run python -m pawnllm.stockfish_benchmark run \
  --engine runs/stockfish/bin/stockfish/stockfish-linux-x86-64-universal \
  --output runs/stockfish/all-legal-100000 --nodes 100000 --workers 4 --all-legal

uv run python -m pawnllm.human_benchmark \
  --export-checkpoint runs/chess-human-calibrated/best.pt

uv run --with matplotlib python scripts/render_human_comparison.py
uv run python -m pawnllm.chess_web --host 0.0.0.0
```

Artifacts: `runs/human-comparison/summary.json` (metrics, fitted parameters, paired
intervals, checkpoint and input hashes), `scores.npz` (all legal scores and original
ranking hits), `comparison.png`, `comparison.pdf`. Engine candidates and raw scores
are in `runs/stockfish/all-legal-100000/results.jsonl`; settings and binary checksum
are in that directory's `summary.json`. Generated artifacts are ignored by Git.
