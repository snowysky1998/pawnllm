"""Audit frozen later-date holdouts without running any model predictions."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import chess
import chess.pgn
import numpy as np
import zstandard

from pawnllm.chess_data import encode_board, move_index, outcome_class
from pawnllm.stockfish_benchmark import shard_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='/mnt/archive/lichess/temporal-holdout-2026')
    args=parser.parse_args();root=Path(args.data)
    manifest=json.loads((root/'manifest.json').read_text())
    for relative,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,relative
    publication=manifest['maia_publication']
    stamp=next(c['date'] for c in publication['commits'] if c['commit']==publication['revision'])
    cutoff=datetime.fromisoformat(stamp)
    rows=[json.loads(line) for line in (root/'positions.jsonl').read_text().splitlines()]
    reserved={r['game_id'] for r in rows}
    assert len(rows)==len(reserved)
    excluded={r['game_id'] for r in json.loads((root/'excluded-prior-games.json').read_text())}
    assert not reserved & excluded
    lookup={r['game_id']:r for r in rows}
    samples={split:iter(shard_rows(root,split)) for split in ['val','test']}
    counts=Counter();bands={'val':Counter(),'test':Counter()};dates={'val':[],'test':[]};seen=set()
    with (root/'selected-games.pgn.zst').open('rb') as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
            text=io.TextIOWrapper(reader)
            while game:=chess.pgn.read_game(text):
                assert not game.errors
                game_id=game.headers['Site'].rsplit('/',1)[-1]
                assert game_id not in seen;seen.add(game_id)
                r=lookup[game_id];split=r['split'];sample=next(samples[split])
                when=datetime.strptime(game.headers['UTCDate']+' '+game.headers['UTCTime'],'%Y.%m.%d %H:%M:%S').replace(tzinfo=cutoff.tzinfo)
                assert when>cutoff and when.isoformat()==r['utc']
                assert when.strftime('%Y-%m')==manifest['args']['val_month' if split=='val' else 'test_month']
                board=game.board();clocks={True:float(game.headers['TimeControl'].split('+')[0]),False:float(game.headers['TimeControl'].split('+')[0])}
                nodes=list(game.mainline());target=nodes[r['ply']]
                for node in nodes[:r['ply']]:
                    assert node.clock() is not None and node.clock()>=30
                    clocks[board.turn]=node.clock();board.push(node.move)
                assert r['ply']>=20 and min(clocks.values())>=30 and target.clock()>=30
                assert r['moves']==[m.uci() for m in board.move_stack]
                assert r['fen']==board.fen() and r['human_move']==target.move.uci()
                assert r['active_clock_before']==clocks[board.turn] and r['opponent_clock_before']==clocks[not board.turn]
                assert r['active_clock_after']==target.clock()
                assert not board.is_game_over(claim_draw=True)
                ratings=[int(game.headers['WhiteElo']),int(game.headers['BlackElo'])]
                if not board.turn:ratings.reverse()
                assert ratings==[r['rating'],r['opponent_rating']]
                np.testing.assert_array_equal(sample['ratings'],ratings)
                pieces,rules=encode_board(board,8)
                np.testing.assert_array_equal(sample['pieces'],pieces);np.testing.assert_array_equal(sample['rules'],rules)
                assert int(sample['ply'])==r['ply'] and int(sample['policy'])==move_index(target.move,board.turn)
                assert int(sample['value'])==outcome_class(game.headers['Result'],board.turn)
                assert set(sample['legal'][sample['legal']>=0])=={move_index(m,board.turn) for m in board.legal_moves}
                counts[split]+=1;bands[split][min(4,(r['rating']-600)//400)]+=1;dates[split].append(r['utc'])
    assert seen==reserved
    for split,it in samples.items():
        assert next(it,None) is None
        assert counts[split]==manifest['positions'][split]
        expected=manifest['args']['val_per_band' if split=='val' else 'test_per_band']
        assert bands[split]==Counter({b:expected for b in range(5)})
    assert max(dates['val'])<min(dates['test'])
    report=dict(verified=True,positions=dict(counts),games=dict(counts),prior_game_overlap=0,cross_split_overlap=0,
        publication_cutoff=stamp,maia_revision=publication['revision'],
        dates={k:dict(first=min(v),last=max(v),distinct_days=len({d[:10] for d in v})) for k,v in dates.items()},
        checks='Every file checksum, PGN legality, date, clocks, history, seven shard fields, exact Elo quota, and unique game ID; no model inference',
        manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
