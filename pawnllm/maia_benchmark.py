"""Evaluate official Maia-3 weights on the existing verified human test positions.

Clone CSSLab/maia3 and pass --source. No changes to the official model or tokenizer.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import chess
import numpy as np
import torch

from pawnllm.chess_data import move_index
from pawnllm.stockfish_benchmark import shard_rows
from pawnllm.human_benchmark import BANDS, probabilities, losses, bootstrap_macro, fit_calibration, apply_calibration


def aggregate(p, target, bands, mask):
    nll = losses(p[mask], target[mask])
    selected = bands[mask]
    macro = float(np.mean([nll[selected == b].mean() for b in np.unique(selected)]))
    return dict(positions=int(mask.sum()), nll=float(nll.mean()), macro_nll=macro,
                human_move_probability=float(np.exp(-macro)))


@torch.inference_mode()
def run(args):
    sys.path.insert(0, str(Path(args.source).resolve()))
    from maia3.uci import parse_args
    from maia3.models import MAIA3Model
    from maia3.model_registry import resolve_checkpoint_path
    from maia3.dataset import tokenize_board, get_historical_tokens, get_legal_moves_mask
    from maia3.utils import get_all_possible_moves, mirror_move

    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = parse_args(['--model', '23m', '--device', args.device, '--use-uci-history', '--no-use-amp'])
    cfg.checkpoint_path = resolve_checkpoint_path(cfg.model_spec, cache_dir=args.cache)
    state = torch.load(cfg.checkpoint_path, map_location='cpu', weights_only=True)
    weights = state['model_state_dict'] if 'model_state_dict' in state else state
    weights = {k.replace('smolgen', 'gab'): v for k, v in weights.items()}
    model = MAIA3Model(cfg)
    model.load_state_dict(weights, strict=True)
    model.to(args.device).eval()
    vocabulary = {move: i for i, move in enumerate(get_all_possible_moves())}
    records = [json.loads(line) for line in Path(args.positions).read_text().splitlines()]
    sample_iters = {split: iter(shard_rows(Path(args.data), split)) for split in ('val', 'test')}
    n = len(records)
    scores = np.full((n, 256), -np.inf)
    targets = np.empty(n, dtype=int)
    ratings = np.empty((n, 2), dtype=np.float32)
    start = time.monotonic()
    # Bounded batches; full games are replayed, with exactly the last eight snapshots.
    for begin in range(0, n, args.batch_size):
        tensors, masks, indices = [], [], []
        for i in range(begin, min(n, begin+args.batch_size)):
            row = records[i]
            sample = next(sample_iters[row['split']])
            board = chess.Board()
            history = deque(maxlen=cfg.history)
            if len(row['moves']) < cfg.history:
                history.append(tokenize_board(board))
            for ply, uci in enumerate(row['moves'], 1):
                board.push_uci(uci)
                if ply > len(row['moves'])-cfg.history:
                    history.append(tokenize_board(board))
            if board.fen() != row['fen']:
                raise AssertionError('Reconstructed Maia position differs')
            human = chess.Move.from_uci(row['human_move'])
            if (int(sample['ply']) != row['ply'] or int(sample['ratings'][0]) != row['rating']
                    or int(sample['policy']) != move_index(human, board.turn)):
                raise AssertionError('Stored ratings/target do not align with reconstructed history')
            ratings[i] = sample['ratings']
            legal = list(board.legal_moves)
            encoded = [vocabulary[m.uci() if board.turn else mirror_move(m.uci())] for m in legal]
            mask = get_legal_moves_mask(board, vocabulary)
            if mask.sum().item() != len(legal) or set(encoded) != set(mask.nonzero().flatten().tolist()):
                raise AssertionError('Official vocabulary omitted a legal move')
            targets[i] = legal.index(human)
            tensors.append(get_historical_tokens(history, cfg, 0., 0., 0., 0.))
            masks.append(mask)
            indices.append(encoded)
        batch = torch.stack(tensors).to(args.device)
        elos = torch.tensor(ratings[begin:begin+len(tensors)], dtype=torch.long, device=args.device)
        logits, _, _ = model(batch, elos[:, 0], elos[:, 1])
        logits = logits.float().masked_fill(~torch.stack(masks).to(args.device), -torch.inf)
        for j, encoded in enumerate(indices):
            scores[begin+j, :len(encoded)] = logits[j, encoded].cpu().numpy()
        if begin % 1024 == 0:
            print(json.dumps(dict(completed=begin+len(tensors), seconds=time.monotonic()-start)), flush=True)
    for iterator in sample_iters.values():
        if next(iterator, None) is not None:
            raise AssertionError('Incomplete official-model evaluation')
    bands = np.minimum(4, np.maximum(0, ((ratings[:, 0]-600)//400).astype(int)))
    val = np.array([r['split'] == 'val' for r in records])
    test = np.array([r['split'] == 'test' for r in records])
    games = np.array([r['game_id'] for r in records])
    ply = np.array([r['ply'] for r in records])
    fitted = fit_calibration(scores[val], targets[val], bands[val], np.geomspace(.25, 4, 81))
    distributions = {'maia3_23m_official': probabilities(scores),
                     'maia3_23m_calibrated': apply_calibration(scores, bands, fitted)}
    result = dict(metadata=dict(
        repository='https://github.com/CSSLab/maia3', model='https://huggingface.co/UofTCSSLab/Maia3-23M',
        git_commit=subprocess.check_output(['git', '-C', args.source, 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint_path=str(Path(cfg.checkpoint_path).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(cfg.checkpoint_path).read_bytes()).hexdigest(),
        positions_sha256=hashlib.sha256(Path(args.positions).read_bytes()).hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()), device=args.device, precision='float32, no autocast',
        history='Official tokenizer, chronological last 8 positions; earliest-position padding; real reconstructed history',
        ratings='Actual active/opponent ratings without clipping to pilot bands',
        clocks='Official released preset does not include time info; unused ponder feature set to zero as official UCI',
        training='Released weights, no local fine-tuning. Official column is uncalibrated T=1 over all legal moves.',
        caveat='Possible overlap with Maia training games/period; these are only verified held-out from PawnLLM.',
        seconds=time.monotonic()-start), calibration=fitted, metrics={})
    for name, p in distributions.items():
        result['metrics'][name] = {}
        for split, mask in [('val', val), ('test', test)]:
            subsets = {'Overall': mask, **{label: mask & (bands == i) for i, label in enumerate(BANDS)},
                       'opening_ply_lt20': mask & (ply < 20), 'after_opening_ply_ge20': mask & (ply >= 20)}
            result['metrics'][name][split] = {label: aggregate(p, targets, bands, selected) for label, selected in subsets.items()}
        ci = bootstrap_macro(losses(p[test], targets[test]), bands[test], games[test])
        result['metrics'][name]['test']['Overall']['probability_ci95'] = dict(low=float(np.exp(-ci['high'])), high=float(np.exp(-ci['low'])))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'scores.npz', scores=scores, target=targets, ratings=ratings,
                        bands=bands, val=val, test=test, games=games, ply=ply)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/maia3-official')
    parser.add_argument('--cache', default='runs/maia3-cache')
    parser.add_argument('--data', default='data/chess-human')
    parser.add_argument('--positions', default='runs/stockfish/positions.jsonl')
    parser.add_argument('--output', default='runs/maia3-comparison')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    run(args)


if __name__ == '__main__':
    main()
