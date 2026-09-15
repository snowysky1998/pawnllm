import threading
import unittest


from pawnllm.chess_web import play_request


class WebGameTests(unittest.TestCase):
    def request(self, moves, color='white'):
        return play_request(dict(moves=moves, color=color), None, None, threading.Lock())

    def test_white_starts_without_inference(self):
        state = self.request([])
        self.assertEqual(state['moves'], [])
        self.assertEqual(len(state['legal']), 20)
        self.assertFalse(state['over'])

    def test_both_sides_remain_manual(self):
        history = ['e2e4']
        state = self.request(history)
        self.assertEqual(state['moves'], ['e2e4'])
        self.assertEqual(state['sans'], ['e4'])
        self.assertEqual(state['status'], 'Black to move')
        self.assertEqual(history, ['e2e4'])
        self.assertEqual(self.request([], 'black')['moves'], [])
        self.assertEqual(self.request(['e2e4', 'e7e5'])['status'], 'White to move')

    def test_mate_and_no_continuation(self):
        moves = ['f2f3', 'e7e5', 'g2g4', 'd8h4']
        state = self.request(moves)
        self.assertTrue(state['over'])
        self.assertEqual(state['legal'], [])
        self.assertEqual(state['status'], 'Black wins · checkmate')
        with self.assertRaises(ValueError):
            self.request(moves + ['a2a3'])

    def test_bad_requests(self):
        for moves, color in [(['e2e5'], 'white'), (['0000'], 'white'), ([], 'green'),
                             ('e2e4', 'white'), ([None], 'white'), (['e2e4'] * 1025, 'white')]:
            with self.subTest(moves=str(moves)[:30], color=color), self.assertRaises(ValueError):
                self.request(moves, color)


if __name__ == '__main__':
    unittest.main()
