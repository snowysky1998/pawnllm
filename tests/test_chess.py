import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import chess
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import zstandard

from pawnllm.chess_data import (encode_board, game_samples, game_split, index_move,
                                move_index, outcome_class, PositionDataset, prepare)
from pawnllm.chess_model import ACTION_SIZE, ChessConfig, ChessModel, masked_policy
from pawnllm.chess_train import load_model, save_checkpoint


class ChessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_special_moves_roundtrip(self):
        fens = [chess.STARTING_FEN,
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
                '1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1']
        for fen in fens:
            board = chess.Board(fen)
            for b in (board, board.mirror()):
                indices = []
                for move in b.legal_moves:
                    index = move_index(move, b.turn)
                    self.assertEqual(index_move(index, b.turn), move)
                    self.assertTrue(0 <= index < ACTION_SIZE)
                    indices.append(index)
                self.assertEqual(len(indices), len(set(indices)))
        board = chess.Board(fens[-1])
        self.assertEqual(len([m for m in board.legal_moves if m.promotion]), 8)

    def test_mirrored_board_has_same_encoding(self):
        board = chess.Board('r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 17 12')
        a, ar = encode_board(board)
        b, br = encode_board(board.mirror())
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(ar, br)
        self.assertEqual(ar[-1], chess.D6)
        self.assertEqual(outcome_class('1-0', chess.WHITE), 2)
        self.assertEqual(outcome_class('1-0', chess.BLACK), 0)
        self.assertEqual(outcome_class('1/2-1/2', chess.BLACK), 1)

    def test_history_and_repetition(self):
        board = chess.Board()
        initial, _ = encode_board(board)
        np.testing.assert_array_equal(initial[0], initial[-1])
        for san in ('Nf3', 'Nf6', 'Ng1', 'Ng8') * 2:
            board.push_san(san)
        pieces, rules = encode_board(board)
        self.assertEqual(rules[5], 0.5)  # third occurrence
        self.assertFalse(np.array_equal(pieces[0], pieces[1]))
        self.assertEqual(len(board.move_stack), 8)

    def test_samples_target_pre_move_position(self):
        cols = ['abc', '2100', '2100', '300+0', 'Normal', '1-0', 'e4 e5 Nf3 Nc6', '']
        samples = game_samples(cols, 8, 8, 0, random.Random(1))
        self.assertEqual(len(samples), 4)
        for i, (_, _, legal, policy, value) in enumerate(samples):
            self.assertIn(policy, legal)
            self.assertEqual(value, 2 if i % 2 == 0 else 0)
        self.assertEqual(samples[0][3], move_index(chess.Move.from_uci('e2e4'), chess.WHITE))
        self.assertEqual(samples[1][0][0, chess.E7 ^ 56], chess.PAWN)
        with self.assertRaises(ValueError):
            game_samples(cols[:6] + ['e4 e5 INVALID', ''], 8, 8, 0, random.Random(1))

    def test_model_can_learn_and_mask(self):
        torch.manual_seed(1)
        model = ChessModel(ChessConfig(dim=32, layers=2, heads=4, gab_size=8))
        pieces, rules = encode_board(chess.Board())
        pieces = torch.from_numpy(np.stack([pieces] * 2))
        rules = torch.from_numpy(np.stack([rules] * 2))
        moves = [move_index(m, chess.WHITE) for m in chess.Board().legal_moves]
        legal = torch.tensor([moves + [-1], moves + [-1]])
        target = torch.tensor([move_index(chess.Move.from_uci('e2e4'), chess.WHITE)] * 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        losses = []
        for _ in range(120):
            policy, value = model(pieces, rules)
            masked = masked_policy(policy, legal)
            loss = F.cross_entropy(masked, target) + F.cross_entropy(value, torch.tensor([2, 2]))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(torch.isfinite(model.templates.grad).all())
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * 0.5)
        self.assertEqual(masked.argmax(-1).tolist(), target.tolist())
        self.assertTrue(torch.isneginf(masked[:, 0]).all())

    def test_prepare_and_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            rows = ['\t'.join([f'game{i}', '2100', '2200', '300+0', 'Normal', '1-0', 'e4 e5 Nf3 Nc6', '']) for i in range(150)]
            (source / '2026-01.tsv.zst').write_bytes(zstandard.ZstdCompressor().compress(('\n'.join(rows) + '\n').encode()))
            args = argparse.Namespace(source=str(source), output=str(root / 'prepared'), games=150,
                                      max_scan=200, seed=42, shard_size=32, speeds='blitz', min_elo=2000,
                                      history=8, positions_per_game=4, min_ply=0)
            with redirect_stdout(io.StringIO()):
                prepare(args)
            manifest = json.loads((root / 'prepared/manifest.json').read_text())
            self.assertEqual(manifest['stats']['accepted_games'], 150)
            for split in ('train', 'val', 'test'):
                expected = sum(game_split(f'game{i}') == split for i in range(150)) * 4
                data = list(PositionDataset(root / 'prepared', split))
                self.assertEqual(len(data), expected)
                self.assertTrue(all(int(row['policy']) in row['legal'] for row in data))
                loaded = sum(len(batch['value']) for batch in DataLoader(PositionDataset(root / 'prepared', split), batch_size=17, num_workers=2))
                self.assertEqual(loaded, expected)
            with self.assertRaises(ValueError):
                prepare(args)

    def test_scan_limit_on_filtered_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source'
            source.mkdir()
            row = 'id\t1000\t1000\t300+0\tNormal\t1-0\te4 e5\t\n'
            for month in ('01', '02'):
                (source / f'2026-{month}.tsv.zst').write_bytes(zstandard.ZstdCompressor().compress((row * 10).encode()))
            args = argparse.Namespace(source=str(source), output=str(root / 'prepared'), games=10,
                                      max_scan=3, seed=42, shard_size=32, speeds='blitz', min_elo=2000,
                                      history=8, positions_per_game=4, min_ply=0)
            with redirect_stdout(io.StringIO()):
                prepare(args)
            manifest = json.loads((root / 'prepared/manifest.json').read_text())
            self.assertEqual(manifest['stats']['scanned_games'], 3)

    def test_checkpoint_roundtrip(self):
        model = ChessModel(ChessConfig(dim=32, heads=4, layers=1, gab_size=8))
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'last.pt'
            save_checkpoint(path, model, optimizer, 3, 1.0, argparse.Namespace(seed=1), {})
            restored, state = load_model(path, torch.device('cpu'))
            self.assertEqual(state['step'], 3)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b)


if __name__ == '__main__':
    unittest.main()
