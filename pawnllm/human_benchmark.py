"""Probability-based human-move evaluation on verified, game-disjoint splits.

Calibration parameters are fitted on validation only. All legal moves get probability
mass. Test results include game-cluster bootstrap intervals and paired differences.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pawnllm.chess_data import move_index
from pawnllm.chess_train import loader, load_model, forward_batch
import chess

BANDS = ['600-999', '1000-1399', '1400-1799', '1800-2199', '2200-2600']


def key(row):
    return row['split'], row['game_id'], row['ply']


def probabilities(scores, temperature=1., uniform_mix=0.):
    """Stable softmax over finite (legal) scores, with optional uniform legal tail."""
    legal = np.isfinite(scores)
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    return (1 - uniform_mix) * p + uniform_mix * legal / legal.sum(axis=1, keepdims=True)


def losses(p, target):
    return -np.log(np.maximum(p[np.arange(len(p)), target], 1e-300))


def fit_calibration(scores, target, bands, temperatures):
    """Fit five independently calibrated rating bands; callers pass validation only."""
    fitted = {}
    for band in range(5):
        mask = bands == band
        if not mask.any():
            raise ValueError(f'No calibration positions in {BANDS[band]}')
        best = (float('inf'), 1., 0.)
        for temperature in temperatures:
            p = probabilities(scores[mask], float(temperature))
            legal = np.isfinite(scores[mask])
            uniform = legal / legal.sum(axis=1, keepdims=True)
            for mix in sorted(set([0., .01, .03] + [i/20 for i in range(1, 20)])):
                loss = float(losses((1-mix)*p + mix*uniform, target[mask]).mean())
                if loss < best[0]:
                    best = loss, float(temperature), mix
        fitted[str(band)] = dict(temperature=best[1], uniform_mix=best[2],
                                validation_nll=best[0], positions=int(mask.sum()))
    return fitted


def apply_calibration(scores, bands, fitted):
    p = np.zeros_like(scores, dtype=np.float64)
    for band in range(5):
        mask = bands == band
        if mask.any():
            params = fitted[str(band)]
            p[mask] = probabilities(scores[mask], params['temperature'], params['uniform_mix'])
    return p


def summarize(p, target, bands, mask, ranking_hits=None):
    p, target, bands = p[mask], target[mask], bands[mask]
    if not len(target):
        return {'positions': 0}
    nll = losses(p, target)
    # Stable ties preserve the supplied move ordering (Stockfish's MultiPV order).
    order = np.argsort(-p, axis=1, kind='stable')
    correct = order[:, 0] == target
    top3 = (order[:, :3] == target[:, None]).any(axis=1)
    if ranking_hits is not None:
        # Keep torch's original action-space topk/argmax tie-breaking. Reordering
        # tied BF16 scores in Stockfish move order would bias model accuracy.
        correct, top3 = ranking_hits[mask].T
    confidence = p.max(axis=1)
    reliability = []
    for i in range(10):
        select = np.minimum((confidence*10).astype(int), 9) == i
        if select.any():
            reliability.append(dict(lower=i/10, upper=(i+1)/10, positions=int(select.sum()),
                                    confidence=float(confidence[select].mean()), accuracy=float(correct[select].mean())))
    return dict(positions=len(target), nll=float(nll.mean()),
                macro_nll=float(np.mean([nll[bands == b].mean() for b in np.unique(bands)])),
                perplexity=float(np.exp(nll.mean())), top1=float(correct.mean()), top3=float(top3.mean()),
                policy_brier=float((np.square(p).sum(axis=1) - 2*p[np.arange(len(p)), target] + 1).mean()),
                top_label_ece=sum(r['positions']*abs(r['confidence']-r['accuracy']) for r in reliability)/len(p),
                reliability_bins=reliability)


def bootstrap_macro(values, bands, games, repeats=2000):
    """Resample whole games, preserving positions and paired model comparisons."""
    names, inverse = np.unique(games, return_inverse=True)
    sums, counts = np.zeros((len(names), 5)), np.zeros((len(names), 5))
    np.add.at(sums, (inverse, bands), values)
    np.add.at(counts, (inverse, bands), 1)
    rng = np.random.default_rng(20260915)
    estimates = []
    for _ in range(repeats):
        draw = rng.integers(len(names), size=len(names))
        denominator = counts[draw].sum(axis=0)
        if np.all(denominator > 0):
            estimates.append(float(np.mean(sums[draw].sum(axis=0)/denominator)))
    return dict(low=float(np.quantile(estimates, .025)), high=float(np.quantile(estimates, .975)),
                replicates=len(estimates), games=len(names))


@torch.inference_mode()
def extract_model(checkpoint, data, records, sf_rows, fixed=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, state = load_model(checkpoint, device)
    model.eval()
    scores = np.full((len(records), 256), -np.inf, dtype=np.float64)
    hits = np.zeros((len(records), 2), dtype=bool)
    index = {key(row): i for i, row in enumerate(records)}
    for split in ('val', 'test'):
        split_records = [r for r in records if r['split'] == split]
        cursor = 0
        for batch in loader(data, split, 128, 0):
            originals = batch['ratings'].clone()
            if fixed:
                batch['ratings'] = torch.full_like(batch['ratings'], 1500)
            policy, _, _, _, moved = forward_batch(model, batch, device)
            top1 = (policy.argmax(-1) == moved['policy']).cpu().numpy()
            top3 = (policy.topk(3, dim=-1).indices == moved['policy'][:, None]).any(-1).cpu().numpy()
            for j in range(len(policy)):
                row = split_records[cursor]
                turn = row['ply'] % 2 == 0
                if (int(originals[j, 0]) != row['rating'] or int(batch['ply'][j]) != row['ply']
                    or int(batch['policy'][j]) != move_index(chess.Move.from_uci(row['human_move']), turn)):
                    raise AssertionError('Stored sample and restored game do not match')
                indices = [move_index(chess.Move.from_uci(m), turn) for m in sf_rows[key(row)]['stockfish_moves']]
                legal = batch['legal'][j].tolist()
                if set(indices) != {x for x in legal if x >= 0}:
                    raise AssertionError('Stockfish did not score exactly all legal actions')
                scores[index[key(row)], :len(indices)] = policy[j, indices].float().cpu().numpy()
                hits[index[key(row)]] = [top1[j], top3[j]]
                cursor += 1
        if cursor != len(split_records):
            raise AssertionError('Incomplete model evaluation')
    return scores, hits, dict(checkpoint=str(checkpoint), step=state['step'], fixed_1500=fixed,
                        policy_calibration=state.get('policy_calibration'),
                        sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.positions).read_text().splitlines()]
    sf_list = [json.loads(line) for line in Path(args.stockfish).read_text().splitlines()]
    sf = {key(row): row for row in sf_list}
    if len(sf) != len(sf_list) or len({key(r) for r in records}) != len(records) or set(sf) != {key(r) for r in records}:
        raise ValueError('Position keys must match exactly, without duplicates')
    for record in records:
        engine_row = sf[key(record)]
        if any(engine_row[field] != record[field] for field in ('human_move', 'rating')):
            raise ValueError('Engine results disagree with the recorded human target or rating')
        if len(engine_row['centipawns']) != len(engine_row['stockfish_moves']):
            raise ValueError('Each engine move must have one score')
    val = np.array([r['split'] == 'val' for r in records])
    test = np.array([r['split'] == 'test' for r in records])
    games = np.array([r['game_id'] for r in records])
    if set(games[val]) & set(games[test]):
        raise ValueError('Game leakage between validation and test')
    bands = np.array([min(4, max(0, (r['rating']-600)//400)) for r in records])
    target = np.array([sf[key(r)]['stockfish_moves'].index(r['human_move']) for r in records])
    cp = np.full((len(records), 256), -np.inf)
    for i, record in enumerate(records):
        values = sf[key(record)]['centipawns']
        cp[i, :len(values)] = values
    cp_loss = np.maximum(0, cp.max(axis=1) - cp[np.arange(len(cp)), target])
    # The clipping is only for probability fitting; strata use unmodified scores.
    sf_scores = np.where(np.isfinite(cp), np.clip(cp, -2000, 2000), -np.inf)
    models, metadata, hits = {}, {}, {}
    for name, path, fixed in [('original', args.original, False), ('refined', args.refined, False),
                              ('refined_fixed1500', args.refined, True)]:
        models[name], hits[name], metadata[name] = extract_model(path, args.data, records, sf, fixed)
        print(json.dumps(dict(extracted=name, positions=len(records))), flush=True)
    models['stockfish'] = sf_scores
    np.savez_compressed(out / 'scores.npz', **models, **{k+'_hits': v for k, v in hits.items()}, target=target, bands=bands, val=val, test=test,
                        games=games, ply=np.array([r['ply'] for r in records]), cp_loss=cp_loss)
    calibration, distributions = {}, {}
    for name, scores in models.items():
        if name == 'refined_fixed1500':
            continue
        temperatures = np.geomspace(10, 5000, 81) if name == 'stockfish' else np.geomspace(.25, 4, 81)
        calibration[name] = fit_calibration(scores[val], target[val], bands[val], temperatures)
        distributions[name+'_calibrated'] = apply_calibration(scores, bands, calibration[name])
        if name != 'stockfish':
            distributions[name+'_raw'] = probabilities(scores)
    # Reuse the true-rating model's calibration to isolate changing the model inputs.
    distributions['refined_fixed1500_raw'] = probabilities(models['refined_fixed1500'])
    distributions['refined_fixed1500_calibrated'] = apply_calibration(models['refined_fixed1500'], bands, calibration['refined'])
    report = dict(metadata=metadata, calibration=calibration, metrics={}, paired_test_macro_nll_difference={},
                  protocol=dict(calibration_split='val only; also used for checkpoint selection',
                    checkpoint_selection='equal-weight Elo-band validation policy NLL',
                    score='natural-log human-move NLL, lower is better; macro gives each of five Elo bands equal weight',
                    stockfish_distribution='softmax(clip(centipawns, -2000, 2000)/T) mixed with uniform legal moves',
                    calibration='81 temperatures x 22 uniform mixtures (0 through .95), independently per Elo band for both systems',
                    ranking_ties='PawnLLM preserves torch action-space argmax/topk before calibration. SF probability ranking uses clipped scores, with MultiPV order for ties.',
                    uncertainty='2000 paired game-cluster bootstrap draws; fixed fitted models/calibrators',
                    quality_strata='approximate centipawn loss from finite 100k-node all-legal MultiPV; not authoritative blunder labels',
                    limitations='Same previously inspected pilot test set; exploratory, not fresh confirmatory evaluation. No clock data or temporal holdout.',
                    positions_sha256=hashlib.sha256(Path(args.positions).read_bytes()).hexdigest(),
                    stockfish_sha256=hashlib.sha256(Path(args.stockfish).read_bytes()).hexdigest()))
    ply = np.array([r['ply'] for r in records])
    for name, p in distributions.items():
        report['metrics'][name] = {}
        for split, mask in [('val', val), ('test', test)]:
            subsets = {'Overall': mask, **{label: mask & (bands == b) for b, label in enumerate(BANDS)},
                       'opening_ply_lt20': mask & (ply < 20), 'after_opening_ply_ge20': mask & (ply >= 20),
                       'cp_loss_lt50': mask & (cp_loss < 50),
                       'cp_loss_50_199': mask & (cp_loss >= 50) & (cp_loss < 200),
                       'cp_loss_ge200': mask & (cp_loss >= 200)}
            model_name = name.removesuffix('_raw').removesuffix('_calibrated')
            report['metrics'][name][split] = {label: summarize(p, target, bands, selected, hits.get(model_name)) for label, selected in subsets.items()}
        report['metrics'][name]['test']['Overall']['macro_nll_ci95'] = bootstrap_macro(losses(p[test], target[test]), bands[test], games[test])
    for a, b in [('refined_raw', 'original_raw'), ('refined_calibrated', 'original_calibrated'),
                 ('refined_calibrated', 'stockfish_calibrated'),
                 ('refined_raw', 'refined_fixed1500_raw')]:
        difference = losses(distributions[a][test], target[test]) - losses(distributions[b][test], target[test])
        report['paired_test_macro_nll_difference'][a+' minus '+b] = dict(
            estimate=float(np.mean([difference[bands[test] == band].mean() for band in range(5)])),
            ci95=bootstrap_macro(difference, bands[test], games[test]))
    (out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.export_checkpoint:
        export = Path(args.export_checkpoint)
        if export.resolve() in {Path(args.original).resolve(), Path(args.refined).resolve()}:
            raise ValueError('Export must not overwrite a source checkpoint')
        state = torch.load(args.refined, map_location='cpu', weights_only=True)
        state['policy_calibration'] = calibration['refined']
        state['calibration_protocol'] = report['protocol']
        export.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, export.with_suffix('.tmp'))
        export.with_suffix('.tmp').replace(export)
    print(json.dumps({name: values['test']['Overall'] for name, values in report['metrics'].items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/chess-human')
    parser.add_argument('--positions', default='runs/stockfish/positions.jsonl')
    parser.add_argument('--stockfish', default='runs/stockfish/all-legal-100000/results.jsonl')
    parser.add_argument('--original', default='runs/chess-human/best.pt')
    parser.add_argument('--refined', default='runs/chess-human-refined/best.pt')
    parser.add_argument('--output', default='runs/human-comparison')
    parser.add_argument('--export-checkpoint', help='Export refined weights with inference calibration, to a new path')
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args)


if __name__ == '__main__':
    main()
