"""Terminal chess, legal policy inference, and matches against random or UCI opponents."""
import argparse
import json
from pathlib import Path
import random

import chess
import chess.engine
import chess.pgn
import torch

from pawnllm.chess_data import encode_board, move_index
from pawnllm.chess_train import amp_context, load_model


@torch.inference_mode()
def choose_move(model, board, device):
    moves = list(board.legal_moves)
    if not moves or board.is_game_over(claim_draw=True):
        return None, None
    pieces, rules = encode_board(board, model.config.history)
    with amp_context(device):
        policy, value = model(torch.from_numpy(pieces[None]).to(device), torch.from_numpy(rules[None]).to(device))
    indices = [move_index(m, board.turn) for m in moves]
    move = moves[policy[0, indices].argmax().item()]
    return move, value[0].float().softmax(-1).cpu().tolist()


def play_human(model, board, device, color):
    human = color == 'white'
    print(f'You play {color}. Enter SAN (e4, Nf3, O-O) or UCI (e2e4).')
    print('Commands: moves (legal moves), resign, quit. Draws are claimed automatically.')
    while True:
        ranks = range(7, -1, -1) if human else range(8)
        files = range(8) if human else range(7, -1, -1)
        print()
        for rank in ranks:
            row = [board.piece_at(chess.square(file, rank)) for file in files]
            print(f'{rank + 1}  ' + ' '.join(piece.symbol() if piece else '.' for piece in row))
        print('   ' + ' '.join(chess.FILE_NAMES[file] for file in files))
        result = board.outcome(claim_draw=True)
        if result:
            print(f'Game over: {result.result()} ({result.termination.name.lower().replace("_", " ")}).')
            return
        if board.turn != human:
            move, _ = choose_move(model, board, device)
            print(f'Engine: {board.san(move)} ({move.uci()})')
            board.push(move)
            continue
        while True:
            try:
                command = input('Your move: ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\nGame ended.')
                return
            if command.lower() in ('quit', 'q', 'exit'):
                print('Game ended.')
                return
            if command.lower() == 'resign':
                print('You resigned. Engine wins.')
                return
            if command.lower() == 'moves':
                print(' '.join(board.san(move) for move in board.legal_moves))
                continue
            try:
                move = board.parse_san(command)
                if not move or not board.is_legal(move):
                    raise ValueError('Illegal move')
            except ValueError:
                print('Invalid or illegal move. Try e4, g1f3, or type moves.')
                continue
            board.push(move)
            break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--fen', default=chess.STARTING_FEN)
    p.add_argument('--human', choices=('white', 'black'), help='Play against the model in your terminal')
    p.add_argument('--games', type=int, default=0, help='Even number: paired colors, shared randomized openings')
    p.add_argument('--engine', help='UCI executable; omitted uses uniformly random legal moves')
    p.add_argument('--nodes', type=int, default=1000)
    p.add_argument('--max-plies', type=int, default=300)
    p.add_argument('--pgn', default='runs/matches.pgn')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.human and (args.games or args.engine):
        p.error('--human cannot be combined with --games or --engine')
    if args.games < 0 or args.games % 2 or args.nodes < 1 or args.max_plies < 1:
        p.error('games must be nonnegative and even; nodes/max-plies must be positive')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, device)
    model.eval()
    board = chess.Board(args.fen)
    if not board.is_valid():
        p.error('Invalid board')
    if args.human:
        play_human(model, board, device, args.human)
        return
    if not args.games:
        move, wdl = choose_move(model, board, device)
        print(json.dumps(dict(move=move.uci() if move else None, wdl_loss_draw_win=wdl)))
        return
    engine = chess.engine.SimpleEngine.popen_uci(args.engine) if args.engine else None
    rng = random.Random(args.seed)
    scores = dict(wins=0, draws=0, losses=0, truncated=0)
    Path(args.pgn).parent.mkdir(parents=True, exist_ok=True)
    try:
        if engine:
            engine.configure({'Threads': 1, 'Hash': 64})
        with Path(args.pgn).open('a') as out:
            for number in range(args.games):
                if number % 2 == 0:
                    opening = chess.Board(args.fen)
                    for _ in range(8):
                        if opening.is_game_over(claim_draw=True):
                            break
                        opening.push(rng.choice(list(opening.legal_moves)))
                board = opening.copy()
                model_color = chess.WHITE if number % 2 == 0 else chess.BLACK
                for _ in range(args.max_plies):
                    if board.is_game_over(claim_draw=True):
                        break
                    if board.turn == model_color:
                        move, _ = choose_move(model, board, device)
                    elif engine:
                        move = engine.play(board, chess.engine.Limit(nodes=args.nodes), game=number).move
                    else:
                        move = rng.choice(list(board.legal_moves))
                    board.push(move)
                result = board.outcome(claim_draw=True)
                if result is None:
                    scores['truncated'] += 1
                elif result.winner is None:
                    scores['draws'] += 1
                else:
                    scores['wins' if result.winner == model_color else 'losses'] += 1
                game = chess.pgn.Game.from_board(board)
                game.headers['White'] = 'PawnLLM' if model_color else (args.engine or 'Random')
                game.headers['Black'] = (args.engine or 'Random') if model_color else 'PawnLLM'
                game.headers['Result'] = result.result() if result else '*'
                game.headers['Termination'] = 'normal' if result else 'ply limit'
                print(game, file=out, end='\n\n')
                out.flush()
                print(json.dumps(dict(games=number + 1, **scores)), flush=True)
    finally:
        if engine:
            engine.quit()


if __name__ == '__main__':
    main()
