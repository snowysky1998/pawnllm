from types import SimpleNamespace
import unittest

import chess
import torch

from pawnllm.chess_analysis import analyze_board, predict_boards
from pawnllm.chess_model import ACTION_SIZE, ChessConfig, ChessModel


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(history=1, variant='pilot')

    def forward(self, pieces, rules):
        return torch.zeros(len(pieces), ACTION_SIZE), torch.tensor([[0.1, 0.2, 0.7]]).log().expand(len(pieces), -1)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = FixedModel()
        self.device = torch.device('cpu')

    def test_value_perspective_and_policy_normalization(self):
        white = chess.Board()
        black = chess.Board()
        black.push_uci('e2e4')
        a, b = predict_boards(self.model, [white, black], self.device)
        for got, expected in zip(a['wdl'], [0.7, 0.2, 0.1]):
            self.assertAlmostEqual(got, expected, places=6)
        for got, expected in zip(b['wdl'], [0.1, 0.2, 0.7]):
            self.assertAlmostEqual(got, expected, places=6)
        self.assertAlmostEqual(sum(row['probability'] for row in a['policy']), 1.0, places=6)
        self.assertEqual(len(a['policy']), 20)

    def test_rollouts_are_legal_bounded_and_preserve_board(self):
        board = chess.Board()
        for depth, expected in ((1, 0.2), (2, 0.8), (3, 0.2)):
            report = analyze_board(self.model, board, self.device, depth)
            self.assertEqual(len(report['candidates']), 8)
            for row in report['candidates']:
                cursor = board.copy()
                self.assertEqual(len(row['line']), depth)
                self.assertAlmostEqual(row['white_score'], expected, places=6)
                for step in row['line']:
                    move = cursor.parse_uci(step['uci'])
                    self.assertEqual(step['san'], cursor.san(move))
                    cursor.push(move)
            self.assertEqual(board.fen(), chess.STARTING_FEN)
        for depth in (0, 4, True, '3'):
            with self.assertRaises(ValueError):
                analyze_board(self.model, board, self.device, depth)

    def test_terminal_results_are_exact(self):
        board = chess.Board()
        for san in ('f3', 'e5', 'g4', 'Qh4#'):
            board.push_san(san)
        report = analyze_board(self.model, board, self.device)
        self.assertEqual(report['wdl'], [0, 0, 1])
        self.assertTrue(report['terminal'])
        self.assertEqual(report['candidates'], [])
        draw = chess.Board('8/8/8/8/8/8/4K3/7k w - - 0 1')
        self.assertEqual(predict_boards(None, [draw], self.device)[0]['wdl'], [0, 1, 0])

    def test_enhanced_model_gradients_and_state(self):
        config = ChessConfig(dim=32, layers=2, heads=4, gab_size=8, variant='enhanced')
        model = ChessModel(config)
        pieces = torch.randint(0, 13, (2, 8, 64), dtype=torch.uint8)
        rules = torch.zeros(2, 7)
        policy, value = model(pieces, rules)
        (policy.square().mean() + value.square().mean()).backward()
        for block in model.blocks:
            self.assertTrue(torch.isfinite(block.relative.grad).all())
            self.assertGreater(block.relative.grad.abs().sum(), 0)
        restored = ChessModel(config)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(pieces, rules)[0], policy)


if __name__ == '__main__':
    unittest.main()
