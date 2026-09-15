"""Evaluate one validation-selected checkpoint against frozen temporal baselines."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawnllm.human_benchmark import BANDS, bootstrap_macro, extract_model, losses, probabilities
from pawnllm.maia_benchmark import aggregate
from pawnllm.temporal_scores import data_inputs, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='/mnt/archive/lichess/temporal-holdout-2026')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--baseline', default='runs/temporal-scores')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    if output.exists():
        raise ValueError('Use a new output directory to preserve previous comparisons')
    records, legal, target, bands, verification = data_inputs(args)
    baseline = json.loads((Path(args.baseline) / 'summary.json').read_text())
    assert baseline['provenance']['dataset_verification']['manifest_sha256'] == verification['manifest_sha256']
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert state['args']['validation_manifest'] == json.loads((Path(args.data) / 'manifest.json').read_text())
    assert not state.get('policy_calibration'), 'This comparison expects raw native probabilities'
    training_manifest = state['manifest']
    del state
    scores, _, metadata = extract_model(args.checkpoint, args.data, records, legal)
    p = probabilities(scores)
    splits = np.array([r['split'] for r in records])
    games = np.array([r['game_id'] for r in records])
    with np.load(Path(args.baseline) / 'per-position-losses.npz') as previous:
        np.testing.assert_array_equal(previous['games'], games)
        np.testing.assert_array_equal(previous['bands'], bands)
        np.testing.assert_array_equal(previous['split'], splits)
        previous_loss = previous['pawnllm'].copy()
    result = dict(evaluated_at=datetime.now(timezone.utc).isoformat(), metadata=metadata,
                  dataset_manifest_sha256=verification['manifest_sha256'], training=training_manifest,
                  baseline_summary_sha256=sha(Path(args.baseline) / 'summary.json'),
                  protocol='Scaled PawnLLM uses native softmax, no calibration. Checkpoint selected by July macro NLL; August evaluated after selection. Other models retain frozen previous scores/calibration. Baseline August results had already been inspected.',
                  metrics=copy.deepcopy(baseline['metrics']), paired_nll_difference={})
    new_loss = losses(p, target)
    for split in ('val', 'test'):
        mask = splits == split
        result['metrics'][split]['pawnllm_previous'] = result['metrics'][split].pop('pawnllm')
        subsets = {'Overall': mask, **{label: mask & (bands == i) for i, label in enumerate(BANDS)}}
        result['metrics'][split]['pawnllm'] = {label: aggregate(p, target, bands, select) for label, select in subsets.items()}
        ci = bootstrap_macro(new_loss[mask], bands[mask], games[mask])
        result['metrics'][split]['pawnllm']['Overall']['probability_ci95'] = dict(low=float(np.exp(-ci['high'])), high=float(np.exp(-ci['low'])))
        difference = new_loss[mask] - previous_loss[mask]
        result['paired_nll_difference'][split] = dict(estimate=float(np.mean([difference[bands[mask] == b].mean() for b in range(5)])),
                ci95=bootstrap_macro(difference, bands[mask], games[mask]))
    output.mkdir(parents=True)
    np.savez_compressed(output / 'scores.npz', scores=scores, target=target, bands=bands, split=splits, games=games)
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for split, models in result['metrics'].items():
        print(split, json.dumps({name: rows['Overall'] for name, rows in models.items()}), flush=True)


if __name__ == '__main__':
    main()
