"""Measure BF16 score ties and policy-head precision on validation only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pawnllm.chess_train import evaluate, forward_batch, loader, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model(args.checkpoint, device)
    model.eval()
    result = dict(checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  data=args.data, split='val', calibration='none', metrics={})
    batch = next(iter(loader(args.data, 'val', 128, 0)))
    for fixed in (False, True):
        model.config.fp32_policy = fixed
        with torch.inference_mode():
            policy, _, _, _, moved = forward_batch(model, batch, device)
        unique = [len(p[l[l >= 0].long()].unique()) / int((l >= 0).sum()) for p, l in zip(policy, moved['legal'])]
        metrics = evaluate(model, loader(args.data, 'val', 128, 2), device, 0)
        metrics['human_move_probability'] = math.exp(-metrics.get('macro_policy_loss', metrics['policy_loss']))
        metrics['first_batch_median_unique_legal_fraction'] = float(np.median(unique))
        result['metrics']['fp32_head' if fixed else 'legacy_bf16'] = metrics
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
