"""Local browser interface: uv run python -m pawnllm.chess_web."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit

import chess
import chess.svg
import torch

from pawnllm.chess_analysis import analyze_board
from pawnllm.chess_train import load_model

WEB_ROOT = Path(__file__).with_name('web')


def replay_request(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected a game object')
    history, color = payload.get('moves', []), payload.get('color', 'white')
    if color not in ('white', 'black'):
        raise ValueError('Choose white or black')
    if not isinstance(history, list) or len(history) > 1024:
        raise ValueError('Invalid move history')
    board = chess.Board()
    sans = []
    for uci in history:
        if not isinstance(uci, str) or len(uci) > 5:
            raise ValueError('Invalid move')
        if board.is_game_over(claim_draw=True):
            raise ValueError('The game has already ended')
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f'Illegal move: {uci}')
        sans.append(board.san(move))
        board.push(move)
    return board, sans, color


def play_request(payload, model, device, lock):
    # Both endpoints analyze exactly the submitted moves. Neither side is automated.
    board, sans, color = replay_request(payload)
    return position_state(board, sans, color)


def position_state(board, sans, color):
    outcome = board.outcome(claim_draw=True)
    if outcome:
        reason = outcome.termination.name.lower().replace('_', ' ')
        status = f'Draw · {reason}' if outcome.winner is None else f'{"White wins" if outcome.winner else "Black wins"} · {reason}'
    else:
        status = ('White' if board.turn else 'Black') + ' to move' + (' · check' if board.is_check() else '')
    return dict(moves=[m.uci() for m in board.move_stack], sans=sans, color=color,
                pieces={chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
                legal=[] if outcome else [m.uci() for m in board.legal_moves],
                status=status, over=outcome is not None,
                check=chess.square_name(board.king(board.turn)) if board.is_check() else None)


def select_analysis_model(model, opening_model, board):
    return opening_model if opening_model is not None and board.ply() < 20 else model


def handler_for(model, device, opening_model=None):
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, mime='application/json'):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store' if mime == 'application/json' else 'no-cache')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/', '/app.js', '/style.css'):
                filename, mime = {'/': ('index.html', 'text/html; charset=utf-8'),
                                  '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                                  '/style.css': ('style.css', 'text/css; charset=utf-8')}[path]
                self.send(200, (WEB_ROOT / filename).read_bytes(), mime)
            elif path.startswith('/piece/') and len(path) == len('/piece/P.svg') and path[-4:] == '.svg' and path[7] in 'pnbrqkPNBRQK':
                svg = chess.svg.piece(chess.Piece.from_symbol(path[7]), size=80)
                self.send(200, svg.encode(), 'image/svg+xml')
            else:
                self.send(404, b'{"error":"Not found"}')

        def do_POST(self):
            if self.path not in ('/api/game', '/api/analyze'):
                self.send(404, b'{"error":"Not found"}')
                return
            # Requests are JSON from this local page; no cross-origin API access.
            origin = self.headers.get('Origin')
            if origin and origin != f'http://{self.headers.get("Host")}':
                self.send(403, b'{"error":"Origin rejected"}')
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError('Expected a game object')
                depth = payload.get('depth', 1)
                if type(depth) is not int or not 1 <= depth <= 3:
                    raise ValueError('Depth must be 1, 2, or 3 half-moves')
                white_elo, black_elo = payload.get('white_elo', 1500), payload.get('black_elo', 1500)
                if any(type(elo) is not int or not 600 <= elo <= 2600 for elo in (white_elo, black_elo)):
                    raise ValueError('Player ratings must be integers from 600 to 2600')
                if self.path == '/api/game':
                    state = play_request(payload, model, device, inference_lock)
                    board, _, _ = replay_request(state)
                else:
                    board, sans, color = replay_request(payload)
                    state = position_state(board, sans, color)
                with inference_lock:
                    active_model = select_analysis_model(model, opening_model, board)
                    state['analysis'] = analyze_board(active_model, board, device, depth, white_elo=white_elo, black_elo=black_elo)
                    state['analysis']['opening_model_used'] = active_model is opening_model
            except (ValueError, TypeError) as exc:
                self.send(400, json.dumps(dict(error=str(exc))).encode())
                return
            except Exception:
                self.log_error('Engine request failed')
                self.send(500, b'{"error":"Engine request failed. See server terminal."}')
                raise
            self.send(200, json.dumps(state).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='runs/chess-human-250k/best.pt')
    parser.add_argument('--opening-checkpoint', help='Model for positions before move 11; defaults to the previous model with the 250k checkpoint')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, device)
    if not model.config.rating_conditioned:
        parser.error('The selectable-rating analysis UI requires a --model human checkpoint')
    model.eval()
    opening_path = args.opening_checkpoint
    if opening_path is None and Path(args.checkpoint).resolve() == Path('runs/chess-human-250k/best.pt').resolve():
        opening_path = 'runs/chess-human-calibrated/best.pt'
    opening_model = None
    if opening_path:
        opening_model, _ = load_model(opening_path, device)
        if not opening_model.config.rating_conditioned:
            parser.error('The opening checkpoint must also be rating conditioned')
        opening_model.eval()
    server = ThreadingHTTPServer((args.host, args.port), handler_for(model, device, opening_model))
    print(f'Play PawnLLM at http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
