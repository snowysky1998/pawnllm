# Post-publication validation and test data

The reference Maia-3 23M checkpoint was uploaded on **2026-05-17 at 23:38:18 UTC**.
This holdout uses **July 2026 for validation and August 2026 for test**, so its games
were played after those exact weights were already public.

This is a stronger boundary than relying only on the paper's reported July 2025
training cutoff. It applies to the pinned weights, not to arbitrary future Maia releases.
Recurring positions and players can still overlap with training; the guarantee is
about the later game examples, not never-before-seen board states.

## Checkpoint evidence

- [Official weight-upload commit](https://huggingface.co/UofTCSSLab/Maia3-23M/commit/6975c77af86aa3f6c95d54a8da4e83e12119a4de)
- Revision: `6975c77af86aa3f6c95d54a8da4e83e12119a4de`
- File: `maia3-23m.pt`, 91,799,307 bytes
- SHA-256: `bce6cd1af5f0399ac7eed33fabb7a6a2ef6193662c2740f262bf93af7bfb3569`
- The uploaded file hash matches the local checkpoint used in the earlier comparison.
- The Hugging Face commit/file metadata is preserved in
  [maia3-publication.json](holdout-metadata/maia3-publication.json).

## Verified dataset

| Split | UTC game-start dates | Games | Positions | Positions per Elo band |
| --- | --- | ---: | ---: | ---: |
| Validation | July 1–31, 2026 | 5,000 | 5,000 | 1,000 |
| Test | August 1–31, 2026 | 10,000 | 10,000 | 2,000 |

Bands: 600–999, 1000–1399, 1400–1799, 1800–2199, 2200–2600. Both sets cover all
31 calendar days. Validation/test game overlap is zero; overlap with the prior
25,000-game dataset is zero. The full audit verified every saved position, including
its clocks and PGN history, and all artifact checksums.

We scanned 1,388,329 complete July games and 1,378,279 complete August games from
the sampled chunks. The source frames total 903,659,692 compressed bytes. The final
dataset, including complete selected PGNs and metadata, is about 21 MB. A resident
memory check during the build was below 1 GiB; all processing was on CPU.

The [frozen manifest](holdout-metadata/temporal-holdout-manifest.json) and
[verification report](holdout-metadata/temporal-holdout-verification.json) preserve the exact
provenance and results in the repository. A local `data/chess-human-temporal` symlink
points to the archive dataset for convenient access.

## Sampling and filters

The source is the official [Lichess standard-rated PGN archive](https://database.lichess.org/).
We fetch 96 independently compressed pzstd frames per month. Offsets are seeded and
spread across equal-size compressed-byte strata over each full monthly archive.
Each frame's byte range, source ETag, length, and SHA-256 are recorded. Decompression
verifies complete frames; partial games at frame boundaries are discarded.

This is a month-spread **chunk sample**, not a uniform
sample of every game in a month. Compressed-byte sampling can weight regions with
different compression ratios differently. The sample is appropriate for a reproducible
pilot holdout; it should not be represented as the entire monthly population.

Within the sampled chunks:

- Accept rated standard blitz games, both players rated 600–2600, with a known result.
- Exclude BOT titles; include normal and time-forfeit terminations, retaining only
  positions before time pressure. This avoids excluding a whole game solely because
  it eventually ended on time.
- Exclude all 25,000 games from the prior PawnLLM training, validation and test sets.
  Those IDs were reconstructed by replaying the original preparation and verifying
  every field of every saved sample against the original shards.
- Assign an active color by a seeded game-ID hash; balance five 400-point rating bands
  using that player's rating. Select games by a separate seeded hash order.
- Take **one position per game**, chosen uniformly from eligible moves for the assigned
  color. This prevents long games from contributing multiple correlated positions.
- Exclude the first 20 half-moves (10 full moves).
- Require known clocks up through the selected move. Stop eligibility at the first
  missing clock or first post-move clock below 30 seconds, including the crossing move.
  Later increments cannot make the game eligible again. Both pre-move clocks must
  also be at least 30 seconds.
- Check exact UTC timestamps against the month and pinned publication date.

The builder retains a hash-ranked candidate pool of six times each per-band quota
before parsing clocks, failing rather than silently under-filling if the pool is too
small. This bounds memory. All decoded tensors and metadata are stored on disk.

## Files and use

Dataset location: `/mnt/archive/lichess/temporal-holdout-2026/`.

- `val/*.npz`, `test/*.npz`: existing seven-array PawnLLM format, with actual ratings
  and ply indices. The empty `train/` directory intentionally contains no training data.
- `positions.jsonl`: exact full histories, human targets, timestamps and clock values;
  compatible with the Stockfish and Maia evaluators.
- `selected-games.pgn.zst`: complete selected games with original clock annotations.
- `games.json`: all reserved game IDs, split assignments and dates.
- `excluded-prior-games.json`: the prior 25,000 games excluded from this holdout.
- `manifest.json`: source provenance, filtering, counts, date coverage and file hashes.
- `verification.json`: full data audit; no model predictions are needed to verify data.

The [first current-model evaluation](temporal-scores.md) has now been run at the user’s request,
with frozen weights and calibration. The test set is therefore inspected, while the
dataset manifest preserves its original creation-time status.

**Keep test frozen.** Use validation for checkpoint selection and calibration. Evaluate
test only after a training decision is fixed. Do not put July/August 2026 into future
training data, and always exclude `games.json` as an additional guard.

The dataset was prepared and verified before any model inference or calibration
fitting on either new split. Quality audits inspect labels and data integrity;
they do not select examples based on a model's predictions.

## Reproduce

The fetch command resumes from independently verified cached frames. It checks the
archive ETag and exact HTTP byte ranges and respects rate-limit backoff. Use a fresh
output directory for a new build; existing datasets are not overwritten.

```sh
# Reconstruct prior-game exclusions; run in a fresh replay output directory.
uv run python scripts/export_prior_game_ids.py

uv run python -m pawnllm.temporal_holdout fetch --month 2026-07
uv run python -m pawnllm.temporal_holdout fetch --month 2026-08
uv run python -m pawnllm.temporal_holdout build --publication docs/holdout-metadata/maia3-publication.json
uv run python scripts/verify_temporal_holdout.py
```

The build defaults expect the saved publication evidence at
`runs/temporal-holdout/maia-publication.json`; copy the committed evidence file there
or pass `--publication docs/holdout-metadata/maia3-publication.json`.

Example future training-data preparation guard (this command has **not** been run):

```sh
uv run python -m pawnllm.chess_data \
  --source /mnt/archive/lichess/elo-conditioned \
  --output data/chess-human-next --rating-conditioned --speeds blitz --min-elo 600 \
  --before-month 2026-07 \
  --exclude-games /mnt/archive/lichess/temporal-holdout-2026/games.json
```

`--before-month` is exclusive. The exclusion list is checked before sampling, and its
checksum is recorded in the new training manifest.
