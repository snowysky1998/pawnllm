"""Bounded-memory archive preparation and compact position shard loading."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import random

import chess
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
import zstandard

from pawnllm.chess_model import FORMAT_VERSION

PROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def canonical_square(square, turn):
    return square if turn == chess.WHITE else square ^ 56


def move_index(move, turn):
    source, dest = (canonical_square(s, turn) for s in (move.from_square, move.to_square))
    if move.promotion:
        return 4096 + ((source % 8) * 8 + dest % 8) * 4 + PROMOTIONS.index(move.promotion)
    return source * 64 + dest


def index_move(index, turn):
    if index >= 4096:
        pair, promo = divmod(index - 4096, 4)
        source_file, dest_file = divmod(pair, 8)
        return chess.Move(canonical_square(48 + source_file, turn),
                          canonical_square(56 + dest_file, turn), PROMOTIONS[promo])
    source, dest = divmod(index, 64)
    return chess.Move(canonical_square(source, turn), canonical_square(dest, turn))


def encode_board(board, history=8):
    turn = board.turn
    cursor = board.copy(stack=history - 1)
    pieces = np.zeros((history, 64), dtype=np.uint8)
    for t in range(history):
        for square, piece in cursor.piece_map().items():
            pieces[t, canonical_square(square, turn)] = piece.piece_type + (6 if piece.color != turn else 0)
        if cursor.move_stack:
            cursor.pop()
    repetition = sum(board.is_repetition(n) for n in (2, 3, 4, 5)) / 4
    rules = np.array([
        board.has_kingside_castling_rights(turn), board.has_queenside_castling_rights(turn),
        board.has_kingside_castling_rights(not turn), board.has_queenside_castling_rights(not turn),
        min(board.halfmove_clock, 150) / 150, repetition,
        canonical_square(board.ep_square, turn) if board.ep_square is not None else -1,
    ], dtype=np.float32)
    return pieces, rules


def outcome_class(result, turn):
    if result == '1/2-1/2':
        return 1
    return 2 if (result == '1-0') == turn else 0


def game_split(game_id):
    bucket = int.from_bytes(hashlib.blake2b(game_id.encode(), digest_size=8).digest(), 'big') % 100
    return 'test' if bucket < 2 else 'val' if bucket < 4 else 'train'


def speed_class(control):
    if control == '-':
        return 'correspondence'
    base, increment = map(int, control.split('+'))
    if base < 0 or increment < 0:
        raise ValueError('Negative time control')
    seconds = base + 40 * increment
    return 'bullet' if seconds < 180 else 'blitz' if seconds < 480 else 'rapid' if seconds < 1500 else 'classical'


def game_samples(columns, history, per_game, min_ply, rng, rating_conditioned=False):
    board = chess.Board()
    sans = columns[6].split()
    if len(sans) > 1024:
        raise ValueError('Game exceeds 1024 ply bound')
    chosen = set(rng.sample(range(min_ply, len(sans)), min(per_game, max(0, len(sans) - min_ply))))
    samples = []
    for ply, san in enumerate(sans):
        move = board.parse_san(san)
        if not move or not board.is_legal(move):
            raise ValueError('Illegal or null move')
        if ply in chosen:
            pieces, rules = encode_board(board, history)
            legal_indices = [move_index(m, board.turn) for m in board.legal_moves]
            if len(legal_indices) > 256:
                raise ValueError('Legal move bound exceeded')
            legal = np.full(256, -1, dtype=np.int16)
            legal[:len(legal_indices)] = legal_indices
            sample = (pieces, rules, legal, move_index(move, board.turn), outcome_class(columns[5], board.turn))
            if rating_conditioned:
                ratings = [int(columns[1]), int(columns[2])]
                if not board.turn:
                    ratings.reverse()
                sample += (np.array(ratings, dtype=np.float32), np.int16(ply))
            samples.append(sample)
        board.push(move)
    return samples


class ShardWriter:
    def __init__(self, root, shard_size):
        self.root, self.shard_size = root, shard_size
        self.buffers = {split: [] for split in ('train', 'val', 'test')}
        self.counts, self.shards = Counter(), Counter()
        for split in self.buffers:
            (root / split).mkdir(parents=True)

    def add(self, split, samples):
        for sample in samples:
            self.buffers[split].append(sample)
            self.counts[split] += 1
            if len(self.buffers[split]) >= self.shard_size:
                self.flush(split)

    def flush(self, split):
        buffer = self.buffers[split]
        if not buffer:
            return
        path = self.root / split / f'{self.shards[split]:06d}.npz'
        arrays = dict(zip(('pieces', 'rules', 'legal', 'policy', 'value', 'ratings', 'ply'), map(np.stack, zip(*buffer))))
        arrays['policy'] = arrays['policy'].astype(np.int16)
        arrays['value'] = arrays['value'].astype(np.uint8)
        with path.with_suffix('.tmp').open('wb') as out:
            np.savez_compressed(out, **arrays)
        path.with_suffix('.tmp').replace(path)
        buffer.clear()
        self.shards[split] += 1


def prepare(args, game_callback=None):
    paths = sorted(Path(args.source).glob('*.tsv.zst'))
    if not paths:
        raise ValueError(f'No TSV archives in {args.source}')
    root = Path(args.output)
    if root.exists():
        raise ValueError(f'Output already exists: {root}; use a new directory')
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    writer = ShardWriter(root, args.shard_size)
    stats, accepted_speed, accepted_rating = Counter(), Counter(), Counter()
    conditioned = getattr(args, 'rating_conditioned', False)
    # Equal quotas bound speed imbalance without retaining games in RAM. Prefix
    # sampling is a pilot, not a uniform sample of each entire monthly archive.
    speeds = args.speeds.split(',')
    if len(set(speeds)) != len(speeds) or not set(speeds) <= {'blitz', 'rapid', 'classical'}:
        raise ValueError('speeds must be distinct blitz,rapid,classical classes')
    quotas = {s: args.games // len(speeds) + (i < args.games % len(speeds)) for i, s in enumerate(speeds)}
    rating_quotas = {i: args.games // 5 + (i < args.games % 5) for i in range(5)}
    if conditioned and speeds != ['blitz']:
        raise ValueError('Rating-conditioned pilot uses --speeds blitz to keep one rating pool')
    with ExitStack() as stack:
        streams = []
        for path in paths:
            raw = stack.enter_context(path.open('rb'))
            reader = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(raw))
            streams.append(stack.enter_context(io.TextIOWrapper(reader)))
        while streams and stats['accepted_games'] < args.games and stats['scanned_games'] < args.max_scan:
            for stream in streams[:]:
                if stats['scanned_games'] >= args.max_scan:
                    break
                line = stream.readline()
                if not line:
                    streams.remove(stream)
                    stream.close()
                    continue
                stats['scanned_games'] += 1
                cols = line.rstrip('\r\n').split('\t')
                try:
                    if len(cols) != 8:
                        raise ValueError('Expected eight columns')
                    speed = speed_class(cols[3])
                    ratings = [int(cols[1]), int(cols[2])]
                    rating_bin = min(4, max(0, (sum(ratings) // 2 - 600) // 400))
                    if (cols[5] not in ('1-0', '0-1', '1/2-1/2') or cols[4] != 'Normal'
                            or min(int(cols[1]), int(cols[2])) < args.min_elo
                            or speed not in quotas or accepted_speed[speed] >= quotas[speed]):
                        stats['filtered_games'] += 1
                        continue
                    if conditioned and (min(ratings) < 600 or max(ratings) > 2600 or accepted_rating[rating_bin] >= rating_quotas[rating_bin]):
                        stats['filtered_games'] += 1
                        continue
                    samples = game_samples(cols, args.history, args.positions_per_game, args.min_ply, rng, conditioned)
                    if not samples:
                        stats['short_games'] += 1
                        continue
                except ValueError:
                    stats['invalid_games'] += 1
                    continue
                split = game_split(cols[0])
                if game_callback is not None:
                    game_callback(cols, samples, split)
                writer.add(split, samples)
                stats['accepted_games'] += 1
                accepted_speed[speed] += 1
                if conditioned:
                    accepted_rating[rating_bin] += 1
                stats['games_with_evals'] += bool(cols[7])
                if stats['accepted_games'] % 1000 == 0:
                    print(json.dumps(dict(stats)), flush=True)
                if stats['accepted_games'] >= args.games or stats['scanned_games'] >= args.max_scan:
                    break
    for split in writer.buffers:
        writer.flush(split)
    manifest = dict(format_version=FORMAT_VERSION, args=vars(args), stats=dict(stats),
                    speed_counts=dict(accepted_speed), positions=dict(writer.counts),
                    source_files=[str(p.resolve()) for p in paths], split='blake2b(game_id) mod 100: test <2, val <4',
                    value_target='game result, side-to-move, loss/draw/win')
    if conditioned:
        manifest['rating_conditioned'] = True
        manifest['rating_range'] = [600, 2600]
        manifest['rating_game_counts'] = {f'{600 + 400*i}-{999 + 400*i if i < 4 else 2600}': accepted_rating[i] for i in range(5)}
        manifest['rating_balance'] = 'equal game quotas by mean player rating; five 400-point bins'
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


class PositionDataset(IterableDataset):
    def __init__(self, root, split, seed=0, shuffle=False):
        self.paths = sorted((Path(root) / split).glob('*.npz'))
        if not self.paths:
            raise ValueError(f'No {split} shards in {root}')
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0

    def __iter__(self):
        worker = get_worker_info()
        rng = np.random.default_rng(self.seed + self.epoch)
        paths = list(self.paths)
        if self.shuffle:
            rng.shuffle(paths)
        if worker:
            paths = paths[worker.id::worker.num_workers]
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            order = np.arange(len(arrays['value']))
            if self.shuffle:
                rng.shuffle(order)
            for i in order:
                yield {key: torch.from_numpy(np.asarray(value[i]).copy()) for key, value in arrays.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='/mnt/archive/lichess/strength-filtered')
    p.add_argument('--output', default='data/chess-pilot')
    p.add_argument('--games', type=int, default=10000)
    p.add_argument('--max-scan', type=int, default=1000000)
    p.add_argument('--positions-per-game', type=int, default=8)
    p.add_argument('--min-ply', type=int, default=20)
    p.add_argument('--history', type=int, default=8)
    p.add_argument('--min-elo', type=int, default=2000)
    p.add_argument('--speeds', default='blitz,rapid,classical')
    p.add_argument('--shard-size', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rating-conditioned', action='store_true', help='Store both ratings; balance 600–2600 blitz games by mean rating')
    args = p.parse_args()
    if min(args.games, args.max_scan, args.positions_per_game, args.history, args.shard_size) < 1 or args.min_ply < 0:
        p.error('Counts must be positive and min-ply nonnegative')
    prepare(args)


if __name__ == '__main__':
    main()
