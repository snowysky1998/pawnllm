# Stockfish comparison: top-three human-move agreement

We measured Stockfish on **all 4,016 validation and 3,840 test positions** used for
the rating-conditioned human model. Both methods are scored against the actual
human move. There is no best-move correctness metric in this comparison.

## Results

N counts positions. Each accuracy cell contains **N (top-three agreement)**.

| Rating | PawnLLM train N | Validation: PawnLLM | Validation: Stockfish | Test: PawnLLM | Test: Stockfish |
| --- | ---: | ---: | ---: | ---: | ---: |
| 600–999 | 35,624 | 848 (61.32%) | 848 (56.49%) | 646 (59.75%) | 646 (58.67%) |
| 1000–1399 | 40,685 | 800 (61.75%) | 800 (62.25%) | 733 (59.48%) | 733 (62.62%) |
| 1400–1799 | 39,241 | 880 (62.05%) | 880 (64.66%) | 829 (61.76%) | 829 (66.10%) |
| 1800–2199 | 41,288 | 901 (65.26%) | 901 (68.48%) | 847 (62.22%) | 847 (66.59%) |
| 2200–2600 | 35,306 | 587 (58.60%) | 587 (73.25%) | 785 (55.29%) | 785 (71.21%) |
| **Overall** | **192,144** | **4,016 (62.05%)** | **4,016 (64.57%)** | **3,840 (59.77%)** | **3,840 (65.34%)** |

Stockfish is ahead by 5.57 percentage points on the test set overall. PawnLLM is
slightly ahead in the lowest rating band; Stockfish is ahead in the others. No
statistical significance claim is made for the individual bands. Multiple positions
come from each game, so position counts should not be treated as independent games.

## Exact-position verification

The original preparation was replayed using the same manifest, seed, filtering,
quotas, and sampling. Every field in all **7,856** held-out samples matched its
original saved array exactly: pieces, rules, legal actions, played action, result,
ratings, and ply. The reconstruction then retained the full original UCI history
and human target for each position. No held-out positions were omitted, duplicated,
or replaced. This preserves original side to move, castling/en passant information,
and repetition history when sending positions to Stockfish.

## Engine settings

The official Linux universal binary reported **Stockfish 19**. It was obtained from
[the official download page](https://stockfishchess.org/download/); its SHA-256 is
recorded in `runs/stockfish/nodes-10000/summary.json`.

- `MultiPV=3`: request three alternative root moves, not a three-ply search.
- `go nodes 10000`: one total node limit per position, shared across the candidates.
- `Threads=1`, `Hash=64` MiB per engine.
- `Skill Level=20`, `UCI_LimitStrength=false`: no weakening by the human rating band.
- No tablebases configured.
- A fresh `ucinewgame` for each position clears cross-position search state.
- Four independent engines process positions concurrently; each position still uses
  one search thread and the same node limit.

The actual node counts ranged from 490 to 10,062, with a median of 10,005: searches
can finish early or slightly overshoot the requested node limit. The median minimum
reported depth across the three alternatives was eight. Engine depth is variable;
the benchmark uses a node budget rather than a fixed depth or wall time.

The run completed in approximately 33 seconds after reconstruction. This runtime
is for the four-worker benchmark, not single-position inference latency. In positions
with fewer than three legal moves, all returned legal candidates are used.

## Interpretation

PawnLLM receives both actual player ratings and predicts human behavior without
search. Stockfish ranks engine moves using search and does not receive the ratings.
The rating bands describe the humans in the dataset; they are **not Stockfish Elo
settings**. This is not a comparison of compute-matched systems, nor a comparison
against Stockfish's weakened-play modes. Its limited search budget also does not
represent unlimited Stockfish analysis.

The experiment uses the pilot's held-out sample, including openings, and inherits
its within-month prefix-sampling limitations. It measures agreement with these
humans, not playing strength, Elo, or general performance on the entire archive.
Stockfish was not trained on our 192,144 training positions; that column describes
PawnLLM's training sample only.

## Reproduce

```sh
# Reconstruct positions and verify each held-out sample. Use a fresh output path
# when rerunning, because existing reconstruction/results are not overwritten.
uv run python -m pawnllm.stockfish_benchmark prepare \
  --data data/chess-human --output runs/stockfish-new

# Supply the official Stockfish 19 binary to reproduce this engine version.
uv run python -m pawnllm.stockfish_benchmark run \
  --positions runs/stockfish-new/positions.jsonl \
  --engine /path/to/stockfish-linux-x86-64-universal \
  --output runs/stockfish-new/nodes-10000 --nodes 10000 --workers 4
```

Current local artifacts:

- `runs/stockfish/positions.jsonl`: full histories and human targets.
- `runs/stockfish/positions-verification.json`: counts and reconstruction checksum.
- `runs/stockfish/nodes-10000/results.jsonl`: per-position candidate moves and matches.
- `runs/stockfish/nodes-10000/summary.json`: metrics, options, and binary checksum.
- `runs/stockfish/top3-comparison.png`: comparison image.

These generated files and the downloaded binary are ignored by Git.
