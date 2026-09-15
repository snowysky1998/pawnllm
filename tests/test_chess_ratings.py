import random
from types import SimpleNamespace
import unittest

import chess
import numpy as np
import torch

from pawnllm.chess_data import game_samples
from pawnllm.chess_model import ChessConfig, ChessModel, ACTION_SIZE
from pawnllm.chess_analysis import predict_boards


class RatingTests(unittest.TestCase):
    def test_data_stores_side_to_move_rating(self):
        cols = ['id', '900', '2100', '300+0', 'Normal', '1-0', 'e4 e5 Nf3', '']
        samples = game_samples(cols, 8, 8, 0, random.Random(1), True)
        np.testing.assert_array_equal(samples[0][5], [900, 2100])
        np.testing.assert_array_equal(samples[1][5], [2100, 900])
        self.assertEqual(samples[2][6], 2)

    def test_analysis_swaps_white_black_ratings(self):
        class CaptureModel(torch.nn.Module):
            config = SimpleNamespace(history=1, rating_conditioned=True)
            def forward(self, pieces, rules, ratings):
                self.seen = ratings.tolist()
                return torch.zeros(len(pieces), ACTION_SIZE), torch.zeros(len(pieces), 3)
        model = CaptureModel()
        white, black = chess.Board(), chess.Board()
        black.push_uci('e2e4')
        predict_boards(model, [white, black], torch.device('cpu'), 800, 2400)
        self.assertEqual(model.seen, [[800, 2400], [2400, 800]])

    def test_conditioning_is_required_and_trainable(self):
        torch.set_num_threads(2)
        model = ChessModel(ChessConfig(dim=32, layers=1, heads=4, gab_size=8, rating_conditioned=True))
        pieces = torch.randint(0, 13, (2, 8, 64), dtype=torch.uint8)
        rules = torch.zeros(2, 7)
        with self.assertRaises(ValueError):
            model(pieces, rules)
        policy, value = model(pieces, rules, torch.tensor([[800, 800], [2400, 2400]]))
        (policy.square().mean() + value.square().mean()).backward()
        self.assertGreater(model.rating_embedding[0].weight.grad.abs().sum(), 0)
        restored = ChessModel(model.config)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(pieces, rules, torch.tensor([[800, 800], [2400, 2400]]))[0], policy)


if __name__ == '__main__':
    unittest.main()
