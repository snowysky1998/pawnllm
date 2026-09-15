"""Verify streamed training shards and their game-level temporal exclusions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawnllm.chess_data import game_split
from pawnllm.chess_model import ACTION_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    root = Path(args.data)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['args']['train_only']
    assert manifest['stats']['accepted_games'] == manifest['args']['games']
    assert all(Path(p).name[:7] < manifest['args']['before_month'] for p in manifest['source_files'])
    exclusion = manifest['excluded_games']
    excluded_bytes = Path(exclusion['file']).read_bytes()
    assert hashlib.sha256(excluded_bytes).hexdigest() == exclusion['sha256']
    excluded = {r['game_id'] if isinstance(r, dict) else r for r in json.loads(excluded_bytes)}
    seen = set()
    expected_positions = 0
    for line in (root / 'games.jsonl').open():
        row = json.loads(line)
        gid = row['game_id']
        assert gid not in seen and gid not in excluded
        assert game_split(gid) == row['split'] == 'train'
        seen.add(gid)
        expected_positions += row['positions']
    assert len(seen) == manifest['stats']['accepted_games']
    assert not list((root / 'val').glob('*.npz')) and not list((root / 'test').glob('*.npz'))
    count = 0
    bands = Counter()
    checksums = {}
    for path in sorted((root / 'train').glob('*.npz')):
        checksums[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == {'pieces', 'rules', 'legal', 'policy', 'value', 'ratings', 'ply'}
            n = len(data['policy'])
            assert all(len(data[k]) == n for k in data.files)
            assert data['pieces'].shape == (n, manifest['args']['history'], 64)
            assert ((data['pieces'] >= 0) & (data['pieces'] <= 12)).all()
            assert np.isfinite(data['rules']).all()
            assert ((data['policy'][:, None] == data['legal']).any(axis=1)).all()
            assert ((data['legal'] >= -1) & (data['legal'] < ACTION_SIZE)).all()
            assert ((data['value'] >= 0) & (data['value'] <= 2)).all()
            assert ((data['ratings'] >= 600) & (data['ratings'] <= 2600)).all()
            assert (data['ply'] >= manifest['args']['min_ply']).all()
            bands.update(np.minimum(4, (data['ratings'][:, 0].astype(int) - 600) // 400).tolist())
            count += n
    assert count == expected_positions == manifest['positions']['train']
    report = dict(verified=True, games=len(seen), positions=count, holdout_game_overlap=0,
                  active_rating_positions=dict(bands), files_sha256=checksums,
                  manifest_sha256=hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(),
                  games_sha256=hashlib.sha256((root / 'games.jsonl').read_bytes()).hexdigest(),
                  checks='All shards decompressed, shapes/ranges/targets checked; unique training-hash IDs, reserved-game exclusion, monthly cutoff. Does not replay source SAN.')
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__':
    main()
