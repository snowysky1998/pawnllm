"""Score existing, frozen systems on the verified temporal holdout; never fit parameters."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import torch

from pawnllm.human_benchmark import BANDS, key, extract_model, apply_calibration, probabilities, losses, bootstrap_macro
from pawnllm.maia_benchmark import aggregate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_inputs(args):
    root=Path(args.data)
    verification=json.loads((root/'verification.json').read_text())
    if not verification['verified'] or sha(root/'manifest.json')!=verification['manifest_sha256']:
        raise ValueError('Dataset manifest does not match its completed audit')
    manifest=json.loads((root/'manifest.json').read_text())
    if sha(root/'positions.jsonl')!=manifest['files_sha256']['positions.jsonl']:
        raise ValueError('Position file changed after verification')
    for relative,digest in manifest['files_sha256'].items():
        if relative.endswith('.npz') and sha(root/relative)!=digest:
            raise ValueError('Data shard changed after verification')
    records=[json.loads(line) for line in (root/'positions.jsonl').read_text().splitlines()]
    legal={key(r):dict(stockfish_moves=[m.uci() for m in chess.Board(r['fen']).legal_moves]) for r in records}
    if len(legal)!=len(records):
        raise ValueError('Duplicate position keys')
    target=np.array([legal[key(r)]['stockfish_moves'].index(r['human_move']) for r in records])
    bands=np.array([min(4,(r['rating']-600)//400) for r in records])
    return records,legal,target,bands,verification


def extract(args):
    records,legal,target,bands,verification=data_inputs(args)
    scores,hits,metadata=extract_model(args.checkpoint,args.data,records,legal)
    if not metadata['policy_calibration']:
        raise ValueError('Expected the current calibrated inference checkpoint')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'pawn.npz',scores=scores,target=target,bands=bands)
    metadata['positions_sha256']=sha(Path(args.data)/'positions.jsonl')
    metadata['manifest_sha256']=verification['manifest_sha256']
    (output/'pawn.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(extracted=len(records),checkpoint_step=metadata['step'])),flush=True)


def report(args):
    root=Path(args.output)
    records,legal,target,bands,verification=data_inputs(args)
    pawn=np.load(root/'pawn.npz');maia=np.load(root/'maia'/'scores.npz')
    pawn_meta=json.loads((root/'pawn.json').read_text())
    maia_report=json.loads((root/'maia'/'summary.json').read_text())
    sf_report=json.loads((root/'stockfish'/'summary.json').read_text())
    position_hash=sha(Path(args.data)/'positions.jsonl')
    for recorded in [pawn_meta['positions_sha256'],maia_report['metadata']['positions_sha256'],sf_report['settings']['positions_sha256']]:
        if recorded!=position_hash:
            raise ValueError('Systems were evaluated on different position files')
    np.testing.assert_array_equal(target,pawn['target']);np.testing.assert_array_equal(target,maia['target'])
    np.testing.assert_array_equal(bands,pawn['bands']);np.testing.assert_array_equal(bands,maia['bands'])
    np.testing.assert_array_equal(np.isfinite(pawn['scores']),np.isfinite(maia['scores']))
    sf_rows=[json.loads(line) for line in (root/'stockfish'/'results.jsonl').read_text().splitlines()]
    sf={key(r):r for r in sf_rows}
    if len(sf_rows)!=len(sf) or set(sf)!=set(legal):
        raise ValueError('Stockfish coverage differs')
    scores=np.full_like(pawn['scores'],-np.inf)
    for i,row in enumerate(records):
        result=sf[key(row)]
        if result['human_move']!=row['human_move'] or result['rating']!=row['rating']:
            raise ValueError('Stockfish targets/ratings differ')
        if len(result['stockfish_moves'])!=len(result['centipawns']):
            raise ValueError('Incomplete Stockfish move scores')
        mapping=dict(zip(result['stockfish_moves'],result['centipawns']))
        moves=legal[key(row)]['stockfish_moves']
        if len(mapping)!=len(result['stockfish_moves']) or set(mapping)!=set(moves):
            raise ValueError('Stockfish did not score exactly every legal move')
        scores[i,:len(moves)]=np.clip([mapping[m] for m in moves],-2000,2000)
    old=json.loads(Path(args.calibration).read_text())
    if old['calibration']['refined']!=pawn_meta['policy_calibration']:
        raise ValueError('Served PawnLLM calibration differs from the previous comparison')
    distributions=dict(
        pawnllm=apply_calibration(pawn['scores'],bands,pawn_meta['policy_calibration']),
        maia3=probabilities(maia['scores']),
        stockfish=apply_calibration(scores,bands,old['calibration']['stockfish']))
    legal_mask=np.isfinite(pawn['scores'])
    distributions['random']=legal_mask/legal_mask.sum(axis=1,keepdims=True)
    split_array=np.array([r['split'] for r in records]);games=np.array([r['game_id'] for r in records])
    np.testing.assert_array_equal(games,maia['games'])
    result=dict(evaluated_at=datetime.now(timezone.utc).isoformat(),metrics={},
        protocol=dict(score='Geometric mean human-move probability = exp(-mean NLL); equal weight per Elo band in Overall',
            calibration='Frozen from the old pilot validation set; no fitting on July or August; official Maia temperature 1',
            training='Current PawnLLM remains trained on 24,018 games / 192,144 positions; no scaling run yet',
            test_status='First current-system evaluation requested by user; test results are now inspected and cannot be called untouched in future experiments',
            limitations='Different training budgets and implementations. Same later-game examples; recurring positions/players may overlap. Chunk sampling is not uniform over all monthly games.'),
        provenance=dict(dataset_verification=verification,pawnllm=pawn_meta,maia=maia_report['metadata'],
            stockfish=sf_report['settings'],frozen_calibration_file=args.calibration,frozen_calibration_sha256=sha(args.calibration)))
    for split in ['val','test']:
        select=split_array==split
        result['metrics'][split]={}
        for name,p in distributions.items():
            subsets={'Overall':select,**{label:select&(bands==i) for i,label in enumerate(BANDS)}}
            result['metrics'][split][name]={label:aggregate(p,target,bands,mask) for label,mask in subsets.items()}
            ci=bootstrap_macro(losses(p[select],target[select]),bands[select],games[select])
            result['metrics'][split][name]['Overall']['probability_ci95']=dict(low=float(np.exp(-ci['high'])),high=float(np.exp(-ci['low'])))
    np.savez_compressed(root/'per-position-losses.npz',**{name:losses(p,target) for name,p in distributions.items()},
                        bands=bands,split=split_array,games=games)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for split,metrics in result['metrics'].items():
        print(split,json.dumps({name:rows['Overall'] for name,rows in metrics.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['extract','report'])
    parser.add_argument('--data',default='/mnt/archive/lichess/temporal-holdout-2026')
    parser.add_argument('--checkpoint',default='runs/chess-human-calibrated/best.pt')
    parser.add_argument('--calibration',default='runs/human-comparison/summary.json')
    parser.add_argument('--output',default='runs/temporal-scores')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.command=='extract':extract(args)
    else:report(args)


if __name__=='__main__':main()
