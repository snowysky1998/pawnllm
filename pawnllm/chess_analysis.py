"""Batched network scores and bounded greedy-policy continuations, not tree search."""
import chess
import numpy as np
import torch

from pawnllm.chess_data import encode_board, move_index
from pawnllm.chess_train import amp_context


def terminal_wdl(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return [0.0, 1.0, 0.0] if outcome.winner is None else [1.0, 0.0, 0.0] if outcome.winner else [0.0, 0.0, 1.0]


def policy_probabilities(legal_logits, rating, calibration=None):
    """Apply validation-fitted calibration across every legal move, before top-k."""
    if calibration is None:
        return legal_logits.float().softmax(-1)
    band = str(min(4, max(0, (int(rating) - 600) // 400)))
    params = calibration[band]
    temperature, mix = params['temperature'], params['uniform_mix']
    if not np.isfinite(temperature) or temperature <= 0 or not 0 <= mix <= 1:
        raise ValueError('Invalid policy calibration')
    p = (legal_logits.float() / temperature).softmax(-1)
    return (1 - mix) * p + mix / len(p)


@torch.inference_mode()
def predict_boards(model, boards, device, white_elo=1500, black_elo=1500):
    results = [None] * len(boards)
    active = []
    for i, board in enumerate(boards):
        exact = terminal_wdl(board)
        if exact is not None:
            results[i] = dict(wdl=exact, policy=[], terminal=True)
        else:
            active.append(i)
    if active:
        encoded = [encode_board(boards[i], model.config.history) for i in active]
        pieces = torch.from_numpy(np.stack([x[0] for x in encoded])).to(device)
        rules = torch.from_numpy(np.stack([x[1] for x in encoded])).to(device)
        with amp_context(device):
            if getattr(model.config, 'rating_conditioned', False):
                ratings = torch.tensor([[white_elo, black_elo] if boards[i].turn else [black_elo, white_elo]
                                        for i in active], dtype=torch.float32, device=device)
                logits, value = model(pieces, rules, ratings)
            else:
                logits, value = model(pieces, rules)
        values = value.float().softmax(-1).cpu().tolist()
        for j, i in enumerate(active):
            board = boards[i]
            legal = list(board.legal_moves)
            indices = [move_index(move, board.turn) for move in legal]
            probabilities = policy_probabilities(logits[j, indices], white_elo if board.turn else black_elo,
                                                 getattr(model, 'policy_calibration', None)).cpu().tolist()
            policy = sorted([dict(uci=move.uci(), san=board.san(move), probability=p)
                             for move, p in zip(legal, probabilities)], key=lambda item: -item['probability'])
            loss, draw, win = values[j]
            results[i] = dict(wdl=[win, draw, loss] if board.turn else [loss, draw, win], policy=policy, terminal=False)
    return results


def analyze_board(model, board, device, depth=1, candidates=8, white_elo=1500, black_elo=1500):
    if type(depth) is not int or not 1 <= depth <= 3:
        raise ValueError('Analysis depth must be 1, 2, or 3 half-moves')
    root = predict_boards(model, [board], device, white_elo, black_elo)[0]
    rows = [dict(item, line=[dict(uci=item['uci'], san=item['san'])]) for item in root['policy'][:candidates]]
    branches = []
    for row in rows:
        child = board.copy()
        child.push_uci(row['uci'])
        branches.append(child)
    for ply in range(depth):
        predictions = predict_boards(model, branches, device, white_elo, black_elo)
        for row, child, prediction in zip(rows, branches, predictions):
            row['leaf_wdl'] = prediction['wdl']
            row['white_score'] = prediction['wdl'][0] + 0.5 * prediction['wdl'][1]
            row['terminal'] = prediction['terminal']
            if ply < depth - 1 and prediction['policy']:
                best = prediction['policy'][0]
                row['line'].append(dict(uci=best['uci'], san=best['san']))
                child.push_uci(best['uci'])
    return dict(wdl=root['wdl'], terminal=root['terminal'], candidates=rows, depth=depth,
                turn='white' if board.turn else 'black', legal_count=len(root['policy']),
                shown_probability=sum(row['probability'] for row in rows),
                model='human' if getattr(model.config, 'rating_conditioned', False) else model.config.variant,
                rating_conditioned=getattr(model.config, 'rating_conditioned', False),
                policy_calibrated=getattr(model, 'policy_calibration', None) is not None,
                white_elo=white_elo, black_elo=black_elo,
                parameters=sum(p.numel() for p in model.parameters()))
