"""Cross-check saved benchmark probabilities against official position-wise UCI code."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import chess
sys.path.insert(0, str(Path('runs/maia3-official').resolve()))
from maia3.uci import parse_args, Maia3UCIEngine

records = [json.loads(line) for line in Path('runs/stockfish/positions.jsonl').read_text().splitlines()]
r = json.loads(Path('runs/maia3-comparison/summary.json').read_text())
scores = np.load('runs/maia3-comparison/scores.npz')
chosen = {0, 1}
for predicate in [lambda r: r['ply'] < 8, lambda r: len(r['human_move']) == 5]:
    match = next((i for i, r in enumerate(records) if predicate(r)), None)
    if match is not None:
        chosen.add(match)
cfg = parse_args(['--model','23m','--checkpoint-path',r['metadata']['checkpoint_path'],
                  '--use-uci-history','--no-use-amp','--device','cuda','--multipv','256'])
torch.set_num_threads(4)
engine = Maia3UCIEngine(cfg)
engine.ensure_model_loaded()
for i in sorted(chosen):
    row = records[i]
    engine.cmd_position('position startpos moves '+' '.join(row['moves']))
    engine.self_elo, engine.oppo_elo = map(int, scores['ratings'][i])
    _, moves = engine.score_moves()
    official = {x['move'].uci(): x['policy'] for x in moves}
    legal = list(engine.board.legal_moves)
    logits = scores['scores'][i, :len(legal)]
    p = np.exp(logits-logits.max()); p /= p.sum()
    assert len(official) == min(20, len(legal))
    saved = {m.uci(): float(v) for m, v in zip(legal, p)}
    np.testing.assert_allclose([saved[m] for m in official], list(official.values()), rtol=2e-4, atol=2e-6)
    print('Matched official UCI distribution:', row['game_id'], 'ply', row['ply'], 'human', row['human_move'])
print('Official adapter cross-check passed.')
