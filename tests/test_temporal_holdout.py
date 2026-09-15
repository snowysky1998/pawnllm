from datetime import datetime
import io
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import chess
import chess.pgn
import zstandard

from pawnllm.chess_data import prepare
from pawnllm.temporal_holdout import complete_games, eligible_header, select_position, range_bytes


class TemporalHoldoutTests(unittest.TestCase):
    def test_full_archive_response_is_rejected_before_reading_body(self):
        response=MagicMock();response.status_code=200;response.headers={}
        response.__enter__.return_value=response
        with patch('pawnllm.temporal_holdout.requests.get',return_value=response):
            with self.assertRaisesRegex(ValueError,'exact byte range'):
                range_bytes('https://example.test/archive',100,199,1000,'etag')
        response.iter_content.assert_not_called()

    def headers(self):
        return dict(Event='Rated Blitz game',Site='https://lichess.org/abcdefgh',UTCDate='2026.08.10',UTCTime='12:00:00',
                    WhiteElo='1500',BlackElo='1800',TimeControl='180+2',Termination='Time forfeit',Result='1-0')

    def game(self):
        game=chess.pgn.Game();game.headers['TimeControl']='180+2'
        cursor=game
        for uci in ('g1f3','g8f6','b1c3','b8c6','e2e3','e7e6','f1e2','f8e7','e1g1','e8g8',
                    'd2d3','d7d6','c1d2','c8d7','a2a3','a7a6','h2h3','h7h6','b2b3','b7b6',
                    'c3a2','c6a7','a2c3','a7c6','c3a2','c6a7'):
            cursor=cursor.add_variation(chess.Move.from_uci(uci));cursor.set_clock(90.)
        return game

    def test_month_and_prior_game_exclusion(self):
        headers=self.headers()
        self.assertIsNotNone(eligible_header(headers,'2026-08',set()))
        self.assertIsNone(eligible_header(headers,'2026-07',set()))
        self.assertIsNone(eligible_header(headers,'2026-08',{'abcdefgh'}))
        headers['WhiteTitle']='BOT'
        self.assertIsNone(eligible_header(headers,'2026-08',set()))

    def test_frame_boundaries_drop_incomplete_games(self):
        frame=b'partial earlier game\n[Event "one"]\n\n1. e4 1-0\n\n[Event "two"]\n\n1. d4'
        games=list(complete_games(frame))
        self.assertEqual(len(games),1)
        self.assertTrue(games[0].startswith(b'[Event "one"]'))

    def test_clock_crossing_excluded_even_if_clock_later_recovers(self):
        game=self.game();nodes=list(game.mainline())
        nodes[22].set_clock(29.)
        nodes[24].set_clock(40.)
        for seed in range(10):
            chosen=select_position(game,chess.WHITE,seed)
            self.assertEqual(chosen[0],20)
            self.assertEqual(chosen[1:],(90.,90.,90.))

    def test_missing_clock_stops_eligibility_and_opening_is_excluded(self):
        game=self.game();nodes=list(game.mainline())
        nodes[18].comment=''
        self.assertIsNone(select_position(game,chess.WHITE,42))
        game=self.game()
        for color in [chess.WHITE,chess.BLACK]:
            chosen=select_position(game,color,42)
            self.assertGreaterEqual(chosen[0],20)
            self.assertEqual(chosen[0]%2,0 if color else 1)

    def test_training_preparation_respects_reserved_ids_and_date_cutoff(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            row=lambda game_id: f'{game_id}\t1500\t1500\t180+2\tNormal\t1-0\te4 e5 Nf3\t\n'
            for month,text in [('2026-06',row('reserved')+row('allowed1')),('2026-07',row('later001'))]:
                (source/f'{month}.tsv.zst').write_bytes(zstandard.ZstdCompressor().compress(text.encode()))
            excluded=root/'exclude.json';excluded.write_text(json.dumps([dict(game_id='reserved')]))
            args=argparse.Namespace(source=str(source),output=str(root/'output'),games=5,max_scan=10,
                positions_per_game=1,min_ply=0,history=1,min_elo=0,speeds='blitz',shard_size=10,seed=42,
                rating_conditioned=False,before_month='2026-07',exclude_games=str(excluded))
            seen=[]
            with redirect_stdout(io.StringIO()):
                prepare(args,lambda cols,samples,split: seen.append(cols[0]))
            self.assertEqual(seen,['allowed1'])
            manifest=json.loads((root/'output'/'manifest.json').read_text())
            self.assertEqual(manifest['stats']['excluded_holdout_games'],1)


if __name__=='__main__':
    unittest.main()
