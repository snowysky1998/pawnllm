"""Reconstruct exact held-out games, then score Stockfish human-move agreement."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import threading
import time

import chess
import chess.engine
import numpy as np

from pawnllm.chess_data import prepare, move_index

KEYS = ('pieces', 'rules', 'legal', 'policy', 'value', 'ratings', 'ply')


def shard_rows(root, split):
    for path in sorted((root / split).glob('*.npz')):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in KEYS}
        for i in range(len(arrays['value'])):
            yield {key: array[i] for key, array in arrays.items()}


def reconstruct(data, output):
    manifest = json.loads((data / 'manifest.json').read_text())
    if not manifest.get('rating_conditioned'):
        raise ValueError('Expected rating-conditioned data with stored ply indices')
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'positions.jsonl'
    if target.exists():
        raise ValueError(f'Already reconstructed: {target}')
    expected = {split: iter(shard_rows(data, split)) for split in ('val', 'test')}
    counts = Counter()
    args = argparse.Namespace(**manifest['args'])
    args.output = str(output / 'replay')
    with target.with_suffix('.tmp').open('w') as out:
        def verify_game(columns, samples, split):
            if split == 'train':
                return
            for sample in samples:
                row = next(expected[split])
                for key, value in zip(KEYS, sample):
                    if not np.array_equal(row[key], value):
                        raise AssertionError(f'Replay differs at {split}/{counts[split]}: {key}')
                counts[split] += 1
            selected = {int(s[-1]): s for s in samples}
            board = chess.Board()
            prefix = []
            for ply, san in enumerate(columns[6].split()):
                move = board.parse_san(san)
                if ply in selected:
                    sample = selected[ply]
                    assert move_index(move, board.turn) == sample[3]
                    record = dict(split=split, game_id=columns[0], ply=ply, moves=list(prefix),
                                  human_move=move.uci(), rating=int(sample[5][0]), fen=board.fen())
                    out.write(json.dumps(record) + '\n')
                prefix.append(move.uci())
                board.push(move)
        with (output / 'replay.log').open('w') as log, redirect_stdout(log):
            prepare(args, game_callback=verify_game)
        for split in expected:
            if next(expected[split], None) is not None or counts[split] != manifest['positions'][split]:
                raise AssertionError('Incomplete reconstruction')
    target.with_suffix('.tmp').replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / 'positions-verification.json').write_text(json.dumps(dict(
        counts=dict(counts), sha256=digest, data=str(data.resolve()),
        check='All seven stored arrays matched exactly for every validation/test sample; full histories restored'
    ), indent=2) + '\n')
    print(json.dumps(dict(positions=str(target), verified=dict(counts))), flush=True)


def benchmark(positions, engine_path, output, nodes, workers, all_legal=False):
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'results.jsonl'
    if result_path.exists():
        raise ValueError(f'Results already exist: {result_path}; choose another output')
    settings = dict(engine_path=str(engine_path.resolve()), nodes=nodes, workers=workers,
                    threads_per_engine=1, hash_mib_per_engine=64, multipv='all legal' if all_legal else 3,
                    skill_level=20, uci_limit_strength=False, tablebases=False,
                    fresh_hash_each_position=True,
                    positions_sha256=hashlib.sha256(positions.read_bytes()).hexdigest(),
                    binary_sha256=hashlib.sha256(engine_path.read_bytes()).hexdigest())
    local = threading.local()
    engines, lock = [], threading.Lock()
    def score(record):
        if not hasattr(local, 'engine'):
            engine = chess.engine.SimpleEngine.popen_uci(str(engine_path.resolve()))
            engine.configure({'Threads': 1, 'Hash': 64, 'Skill Level': 20, 'UCI_LimitStrength': False})
            with lock:
                engines.append(engine)
                settings['engine_id'] = engine.id
            local.engine = engine
        board = chess.Board()
        for move in record['moves']:
            board.push_uci(move)
        if board.fen() != record['fen']:
            raise AssertionError('Restored board differs')
        human = chess.Move.from_uci(record['human_move'])
        if human not in board.legal_moves:
            raise AssertionError('Human target is illegal')
        requested = board.legal_moves.count() if all_legal else min(3, board.legal_moves.count())
        info = local.engine.analyse(board, chess.engine.Limit(nodes=nodes), multipv=requested, game=object(),
                                    info=chess.engine.INFO_ALL)
        info.sort(key=lambda row: row.get('multipv', 1))
        moves = [row['pv'][0].uci() for row in info if row.get('pv')]
        if len(moves) != requested or len(set(moves)) != len(moves):
            raise RuntimeError('Incomplete or duplicate MultiPV result')
        result = dict(split=record['split'], game_id=record['game_id'], ply=record['ply'],
                    rating=record['rating'], human_move=human.uci(), stockfish_moves=moves,
                    top1=human.uci() == moves[0], top3=human.uci() in moves[:3],
                    depths=[row.get('depth') for row in info], nodes=max(row.get('nodes', 0) for row in info))
        if all_legal:
            result['centipawns'] = [row['score'].pov(board.turn).score(mate_score=100000) for row in info]
        return result
    counts = {}
    start = time.monotonic()
    try:
        with positions.open() as source, result_path.open('w') as out, ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(score, map(json.loads, source)), 1):
                out.write(json.dumps(result) + '\n')
                band = min(4, max(0, (result['rating']-600)//400))
                label = f'{600+400*band}-{999+400*band if band<4 else 2600}'
                for key in (label, 'Overall'):
                    row = counts.setdefault(result['split'], {}).setdefault(key, dict(n=0, top1=0, top3=0))
                    row['n'] += 1
                    row['top1'] += result['top1']
                    row['top3'] += result['top3']
                if i % 500 == 0:
                    out.flush()
                    print(json.dumps(dict(completed=i, elapsed_seconds=time.monotonic()-start)), flush=True)
    finally:
        for engine in engines:
            engine.quit()
    metrics = {split: {band: dict(positions=r['n'], top1=r['top1']/r['n'], top3=r['top3']/r['n'])
                       for band, r in bands.items()} for split, bands in counts.items()}
    report = dict(settings=settings, elapsed_seconds=time.monotonic()-start, metrics=metrics)
    (output / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--data', type=Path, default=Path('data/chess-human'))
    prep.add_argument('--output', type=Path, default=Path('runs/stockfish'))
    run = sub.add_parser('run')
    run.add_argument('--positions', type=Path, default=Path('runs/stockfish/positions.jsonl'))
    run.add_argument('--engine', type=Path, required=True)
    run.add_argument('--output', type=Path, default=Path('runs/stockfish/nodes-10000'))
    run.add_argument('--nodes', type=int, default=10000)
    run.add_argument('--workers', type=int, default=4)
    run.add_argument('--all-legal', action='store_true', help='Score every legal move for probability calibration')
    args = p.parse_args()
    if args.command == 'prepare':
        reconstruct(args.data, args.output)
    else:
        if args.nodes < 1 or not 1 <= args.workers <= 16:
            p.error('nodes must be positive; workers must be 1–16')
        benchmark(args.positions, args.engine, args.output, args.nodes, args.workers, args.all_legal)


if __name__ == '__main__':
    main()
