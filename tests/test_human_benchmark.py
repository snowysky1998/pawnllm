import unittest

import numpy as np
import torch

from pawnllm.chess_analysis import policy_probabilities
from pawnllm.maia_benchmark import aggregate
from pawnllm.chess_train import selection_score
from pawnllm.human_benchmark import probabilities, losses, fit_calibration, apply_calibration, summarize, bootstrap_macro


class HumanMetricTests(unittest.TestCase):
    def test_probability_report_uses_equal_band_geometric_mean(self):
        p = np.array([[.5,.5],[.5,.5],[.125,.875]])
        target = np.zeros(3,dtype=int)
        bands = np.array([0,0,1])
        result = aggregate(p,target,bands,np.ones(3,dtype=bool))
        self.assertAlmostEqual(result['human_move_probability'],.25)
        self.assertEqual(aggregate(p,target,bands,np.zeros(3,dtype=bool)),{'positions':0})

    def test_served_probabilities_match_benchmark_calibration(self):
        scores = np.array([[2., 1., -3.]])
        calibration = {str(i): dict(temperature=1.+i, uniform_mix=.05*i) for i in range(5)}
        for rating, band in [(600, 0), (999, 0), (1000, 1), (1500, 2), (2000, 3), (2600, 4)]:
            actual = policy_probabilities(torch.tensor(scores[0]), rating, calibration).numpy()
            expected = apply_calibration(scores, np.array([band]), calibration)[0]
            np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_legal_distribution_and_log_loss(self):
        scores = np.array([[np.log(.4), np.log(.6), -np.inf], [0., -np.inf, -np.inf]])
        p = probabilities(scores)
        np.testing.assert_allclose(p.sum(1), 1)
        self.assertEqual(p[0, 2], 0.)
        np.testing.assert_allclose(losses(p, np.array([0, 0])), [-np.log(.4), 0.])
        mixed = probabilities(scores, 2., .1)
        self.assertEqual(mixed[0, 2], 0.)
        self.assertEqual(mixed[1, 0], 1.)

    def test_macro_weights_bands_equally(self):
        p = np.array([[.9, .1], [.9, .1], [.1, .9]])
        summary = summarize(p, np.array([0, 0, 0]), np.array([0, 0, 1]), np.ones(3, bool))
        self.assertAlmostEqual(summary['macro_nll'], (-np.log(.9)-np.log(.1))/2)
        self.assertNotAlmostEqual(summary['macro_nll'], summary['nll'])
        self.assertEqual(summary['top3'], 1.)
        self.assertAlmostEqual(summary['top_label_ece'], .9-2/3)
        self.assertAlmostEqual(summary['policy_brier'], (.02+.02+1.62)/3)
        self.assertEqual(selection_score({'macro_policy_loss': 2., 'policy_loss': 9.}, 'macro-policy', .5), 2.)

    def test_calibration_uses_supplied_labels_and_preserves_ranking(self):
        scores = np.tile([[4., 0., -np.inf]], (20, 1))
        target = np.tile([0, 0, 1, 1], 5)
        bands = np.repeat(np.arange(5), 4)
        fitted = fit_calibration(scores, target, bands, [1., 4.])
        p = apply_calibration(scores, bands, fitted)
        self.assertLess(losses(p, target).mean(), losses(probabilities(scores), target).mean())
        np.testing.assert_array_equal(p.argmax(1), scores.argmax(1))
        np.testing.assert_array_equal(p[:, 2], 0.)

    def test_cluster_bootstrap_preserves_constant_paired_difference(self):
        bands = np.tile(np.arange(5), 20)
        games = np.repeat(np.arange(20).astype(str), 5)
        interval = bootstrap_macro(np.full(100, -.2), bands, games, repeats=100)
        self.assertEqual(interval['games'], 20)
        self.assertAlmostEqual(interval['low'], -.2)
        self.assertAlmostEqual(interval['high'], -.2)

    def test_explicit_model_ties_are_not_broken_in_engine_order(self):
        p = np.full((1, 5), .2)
        args = (p, np.array([0]), np.array([0]), np.array([True]))
        self.assertEqual(summarize(*args)['top3'], 1.)
        preserved = summarize(*args, ranking_hits=np.array([[False, False]]))
        self.assertEqual(preserved['top3'], 0.)
        self.assertEqual(preserved['top1'], 0.)
        self.assertAlmostEqual(preserved['top_label_ece'], .2)


if __name__ == '__main__':
    unittest.main()
