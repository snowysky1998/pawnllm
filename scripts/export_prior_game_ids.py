"""Replay the original pilot and verify all fields before exporting its game IDs.

Run once in a fresh runs/temporal-holdout directory before building the new holdout.
"""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contextlib import redirect_stdout
import numpy as np
from pawnllm.chess_data import prepare
from pawnllm.stockfish_benchmark import shard_rows,KEYS
Path('runs/temporal-holdout').mkdir(parents=True,exist_ok=True)
root=Path('data/chess-human')
manifest=json.loads((root/'manifest.json').read_text())
args=argparse.Namespace(**manifest['args']);args.output='runs/temporal-holdout/prior-replay'
expected={split:iter(shard_rows(root,split)) for split in ['train','val','test']}
ids=[]
def callback(cols,samples,split):
 for sample in samples:
  row=next(expected[split])
  for key,value in zip(KEYS,sample):
   if not np.array_equal(row[key],value):raise AssertionError('Prior-data replay differs')
 ids.append(dict(game_id=cols[0],split=split))
with open('runs/temporal-holdout/prior-replay.log','w') as log,redirect_stdout(log):prepare(args,callback)
for rows in expected.values():assert next(rows,None) is None
assert len(ids)==len({x['game_id'] for x in ids})==25000
Path('runs/temporal-holdout/prior-games.json').write_text(json.dumps(ids)+'\n')
print('Reconstructed and verified all 25000 prior games',flush=True)
