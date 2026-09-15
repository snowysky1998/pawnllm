"""Train/evaluate the chess encoder on compact shards within a fixed step budget."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from pawnllm.chess_data import PositionDataset
from pawnllm.chess_model import ChessConfig, ChessModel, FORMAT_VERSION, masked_policy


def host_pss_gib():
    """Linux RAM attributed to trainer + loader workers (shared pages counted proportionally)."""
    total = 0
    for pid in [os.getpid()] + [p.pid for p in multiprocessing.active_children()]:
        try:
            lines = Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()
            total += int(next(line.split()[1] for line in lines if line.startswith('Pss:')))
        except (OSError, StopIteration):
            return None
    return total / 2**20


def loader(root, split, batch_size, workers, seed=0, shuffle=False):
    dataset = PositionDataset(root, split, seed, shuffle)
    options = dict(batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available())
    if workers:
        options['prefetch_factor'] = 2
    return DataLoader(dataset, **options)


def amp_context(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def forward_batch(model, batch, device):
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with amp_context(device):
        policy, value = model(batch['pieces'], batch['rules'], batch.get('ratings'))
    policy = masked_policy(policy, batch['legal'])
    policy_loss = F.cross_entropy(policy, batch['policy'].long())
    value_loss = F.cross_entropy(value.float(), batch['value'].long())
    return policy, value.float(), policy_loss, value_loss, batch


@torch.inference_mode()
def evaluate(model, data, device, max_batches):
    model.eval()
    n = top1 = top3 = value_correct = 0
    pl = vl = brier = 0.0
    bands = {}
    for i, batch in enumerate(data):
        if max_batches and i >= max_batches:
            break
        policy, value, policy_loss, value_loss, batch = forward_batch(model, batch, device)
        size = len(value)
        n += size
        pl += policy_loss.item() * size
        vl += value_loss.item() * size
        top1 += (policy.argmax(-1) == batch['policy']).sum().item()
        top3 += (policy.topk(3, dim=-1).indices == batch['policy'][:, None]).any(-1).sum().item()
        value_correct += (value.argmax(-1) == batch['value']).sum().item()
        target = F.one_hot(batch['value'].long(), 3)
        brier += ((value.softmax(-1) - target) ** 2).sum().item()
        if 'ratings' in batch:
            for low in range(600, 2600, 400):
                high = low + 399 if low < 2200 else 2600
                mask = (batch['ratings'][:, 0] >= low) & (batch['ratings'][:, 0] <= high)
                count = int(mask.sum())
                if count:
                    stats = bands.setdefault(f'{low}-{high}', dict(positions=0, correct=0, top3=0, policy_loss=0., value_correct=0))
                    stats['positions'] += count
                    stats['correct'] += int(((policy.argmax(-1) == batch['policy']) & mask).sum())
                    stats['top3'] += int(((policy.topk(3, dim=-1).indices == batch['policy'][:, None]).any(-1) & mask).sum())
                    stats['policy_loss'] += F.cross_entropy(policy[mask], batch['policy'][mask].long(), reduction='sum').item()
                    stats['value_correct'] += int(((value.argmax(-1) == batch['value']) & mask).sum())
    if not n:
        raise ValueError('Evaluation set is empty')
    metrics = dict(positions=n, policy_loss=pl/n, value_loss=vl/n, top1=top1/n,
                   top3=top3/n, value_accuracy=value_correct/n, value_brier=brier/n)
    if bands:
        metrics['by_rating'] = {band: dict(positions=s['positions'], top1=s['correct']/s['positions'],
                                         top3=s['top3']/s['positions'], policy_loss=s['policy_loss']/s['positions'],
                                         value_accuracy=s['value_correct']/s['positions']) for band, s in bands.items()}
        metrics['macro_policy_loss'] = float(np.mean([s['policy_loss'] for s in metrics['by_rating'].values()]))
    return metrics


def selection_score(metrics, metric, value_weight):
    if metric == 'macro-policy':
        if 'macro_policy_loss' not in metrics:
            raise ValueError('macro-policy selection requires rating-conditioned validation data')
        return metrics['macro_policy_loss']
    if metric == 'policy':
        return metrics['policy_loss']
    return metrics['policy_loss'] + value_weight * metrics['value_loss']


def save_checkpoint(path, model, optimizer, step, best, args, manifest):
    state = dict(format_version=FORMAT_VERSION, config=asdict(model.config), model=model.state_dict(),
                 optimizer=optimizer.state_dict(), step=step, best=best, args=vars(args), manifest=manifest,
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 python_rng=random.getstate())
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def load_model(path, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['format_version'] != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format')
    model = ChessModel(ChessConfig(**state['config'])).to(device)
    model.load_state_dict(state['model'])
    # Optional inference-only calibration; training/evaluate retain raw logits.
    model.policy_calibration = state.get('policy_calibration')
    return model, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/chess-pilot')
    p.add_argument('--validation-data', help='Separate validation dataset; never reads its test split')
    p.add_argument('--init-from', help='Initialize weights on new data without restoring optimizer or step')
    p.add_argument('--fp32-policy', action='store_true')
    p.add_argument('--schedule', choices=('constant', 'cosine'), default='constant')
    p.add_argument('--output', default='runs/chess-pilot')
    p.add_argument('--steps', type=int, default=1000, help='Total optimizer steps, including resumed steps')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--value-weight', type=float, default=0.5)
    p.add_argument('--selection-metric', choices=('auto', 'macro-policy', 'policy', 'joint'), default='auto',
                   help='auto uses equal-weight Elo-band policy loss for human models, joint loss otherwise')
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--eval-batches', type=int, default=0, help='0 evaluates every validation position')
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume')
    p.add_argument('--model', choices=('pilot', 'enhanced', 'human'), default='pilot', help='Architecture preset for a new run; resume uses checkpoint config')
    p.add_argument('--evaluate', choices=('val', 'test'), help='Evaluate --resume without training')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.accumulation, args.eval_every, args.log_every) < 1 or min(args.workers, args.warmup, args.eval_batches) < 0:
        p.error('Invalid count or interval')
    if args.lr <= 0 or args.value_weight < 0:
        p.error('lr must be positive and value-weight nonnegative')
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive')
    if args.evaluate and not args.resume:
        p.error('--evaluate requires --resume')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This trainer requires BF16 support on CUDA')
        torch.set_float32_matmul_precision('high')
        torch.cuda.reset_peak_memory_stats()
    manifest = json.loads((Path(args.data) / 'manifest.json').read_text())
    if manifest['format_version'] != FORMAT_VERSION:
        raise ValueError('Unsupported data format')
    if args.resume or args.init_from:
        model, state = load_model(args.resume or args.init_from, device)
        if model.config.history != manifest['args']['history']:
            raise ValueError('History differs between checkpoint and data')
    else:
        config = ChessConfig(history=manifest['args']['history'])
        if args.model in ('enhanced', 'human'):
            config = ChessConfig(history=config.history, dim=512, layers=8, heads=16, gab_size=64, variant='enhanced')
            config.rating_conditioned = args.model == 'human'
        model = ChessModel(config).to(device)
        state = None
    if args.fp32_policy:
        model.config.fp32_policy = True
    if model.config.rating_conditioned != manifest.get('rating_conditioned', False):
        raise ValueError('Model and dataset rating conditioning must match')
    if args.selection_metric == 'auto':
        args.selection_metric = 'macro-policy' if model.config.rating_conditioned else 'joint'
    if args.evaluate:
        metrics = evaluate(model, loader(args.data, args.evaluate, args.batch_size, args.workers), device, args.eval_batches)
        print(json.dumps(dict(split=args.evaluate, **metrics)))
        return
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Output is not empty; use --resume or a new directory')
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step, best = 0, float('inf')
    validation_root = args.validation_data or args.data
    validation_manifest = json.loads((Path(validation_root) / 'manifest.json').read_text())
    if (validation_manifest['format_version'] != FORMAT_VERSION
            or validation_manifest['args']['history'] != model.config.history
            or validation_manifest.get('rating_conditioned', False) != model.config.rating_conditioned):
        raise ValueError('Validation format, history and rating conditioning must match model')
    args.validation_manifest = validation_manifest
    if state and args.resume:
        if state['args'].get('validation_manifest', state['manifest']) != validation_manifest:
            raise ValueError('Resume requires the same validation manifest; use --init-from for a new experiment')
        if state['manifest'] != manifest:
            raise ValueError('Resume requires the same data manifest')
        optimizer.load_state_dict(state['optimizer'])
        step, best = state['step'], state['best']
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if device.type == 'cuda' and state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    train = loader(args.data, 'train', args.batch_size, args.workers, args.seed + step, True)
    val = loader(validation_root, 'val', args.batch_size, args.workers)
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()), device=str(device), config=asdict(model.config))), flush=True)
    with (output / 'metrics.jsonl').open('a') as log:
        def record(event):
            text = json.dumps(event)
            print(text, flush=True)
            log.write(text + '\n')
            log.flush()
        initial_metrics = evaluate(model, val, device, args.eval_batches)
        record(dict(event='validation', step=step, **initial_metrics))
        # Recompute the baseline when resuming: selection criteria may have changed.
        best = selection_score(initial_metrics, args.selection_metric, args.value_weight)
        previous_best = output / 'best.pt'
        keep_previous = False
        if previous_best.exists():
            incumbent, incumbent_state = load_model(previous_best, device)
            if incumbent_state['manifest'] != manifest:
                raise ValueError('Output best checkpoint belongs to another dataset')
            incumbent_score = selection_score(evaluate(incumbent, val, device, args.eval_batches),
                                              args.selection_metric, args.value_weight)
            keep_previous = incumbent_score <= best
            if keep_previous:
                best = incumbent_score
            del incumbent, incumbent_state
        if not keep_previous:
            save_checkpoint(previous_best, model, optimizer, step, best, args, manifest)
        iterator = iter(train)
        start = time.monotonic()
        seen = 0
        while step < args.steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = np.zeros(2)
            lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
            if args.schedule == 'cosine' and step >= args.warmup:
                progress = min(1., (step - args.warmup) / max(1, args.steps - args.warmup - 1))
                lr *= 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group['lr'] = lr
            for _ in range(args.accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    train.dataset.epoch += 1
                    iterator = iter(train)
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        raise ValueError('Training set is empty') from None
                _, _, pl, vl, moved = forward_batch(model, batch, device)
                loss = (pl + args.value_weight * vl) / args.accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                losses += [pl.item(), vl.item()]
                seen += len(moved['value'])
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                elapsed = time.monotonic() - start
                record(dict(event='train', step=step, policy_loss=losses[0]/args.accumulation,
                            value_loss=losses[1]/args.accumulation, lr=lr, positions_per_second=seen/elapsed,
                            elapsed_seconds=elapsed, peak_parent_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                            sampled_host_pss_gib=host_pss_gib(),
                            peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == 'cuda' else 0,
                            peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30 if device.type == 'cuda' else 0))
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, val, device, args.eval_batches)
                record(dict(event='validation', step=step, **metrics))
                score = selection_score(metrics, args.selection_metric, args.value_weight)
                if score < best:
                    best = score
                    save_checkpoint(output / 'best.pt', model, optimizer, step, best, args, manifest)
                save_checkpoint(output / 'last.pt', model, optimizer, step, best, args, manifest)


if __name__ == '__main__':
    main()
