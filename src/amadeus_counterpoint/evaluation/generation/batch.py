"""Efficient batched advancement of many independent variable-length games
while preserving the semantics of single-game generation.
"""

import chess
import torch

from amadeus_counterpoint.chess import create_board, check_end
from amadeus_counterpoint.encoding import (
    encode_history,
    legal_move_mask,
    policy_index_to_move,
)
from amadeus_counterpoint.evaluation.generation.single import MAX_PLIES


class _GameState:
    """Mutable per-game state carried across batched inference steps."""

    def __init__(self, white_elo, black_elo, seed):
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.board = create_board()
        self.history = [self.board.copy(stack=False)]
        self.moves = []
        self.generator = torch.Generator().manual_seed(seed)
        self.result = None


def play_games(model, white_elos, black_elos, seeds):
    """Generate many complete self-play games, batching model inference.

    `white_elos`, `black_elos`, and `seeds` are parallel sequences, one entry
    per game. Each game keeps its own board, history, move list, and seeded
    `torch.Generator`, so a game's sampled moves depend only on its own seed
    -- never on its slot in the batch, execution order, or when other games
    finish. On every iteration, every still-unfinished game's history is
    encoded and stacked into a single batch for one `model(...)` call; the
    legal-move mask, softmax, and seeded sample are then applied per game,
    exactly as in `single.play_game`, so a game generated here matches the
    same game generated alone (same model, Elos, seed) up to floating-point
    differences between batched and single-example inference.

    Returns a list of game records, one per input game, in input order, each
    shaped like `single.play_game`'s return value (`white_elo`, `black_elo`,
    `result`, `moves`).
    """
    model.eval()

    games = [
        _GameState(white_elo, black_elo, seed)
        for white_elo, black_elo, seed in zip(white_elos, black_elos, seeds)
    ]

    while True:
        # 1 & 2. finalize any game that has just reached normal termination
        # or the 500-ply cap; already-finished games are left untouched.
        for game in games:
            if game.result is not None:
                continue

            outcome = check_end(game.board)
            if outcome is not None:
                game.result = outcome.result()
            elif len(game.moves) >= MAX_PLIES:
                # See single.play_game: no adjudicated result is defined yet
                # for ply-cap truncation; left as an open placeholder.
                game.result = ""

        active = [game for game in games if game.result is None]
        if not active:
            break

        # 3. who is actually moving in each active game?
        player_elos = []
        opponent_elos = []
        for game in active:
            if game.board.turn == chess.WHITE:
                player_elos.append(game.white_elo)
                opponent_elos.append(game.black_elo)
            else:
                player_elos.append(game.black_elo)
                opponent_elos.append(game.white_elo)

        # 4. encode each active game's history and stack into one batch
        x = torch.stack([encode_history(game.history) for game in active])

        # 5. Elo -> tensors
        player_elo_t = torch.tensor(player_elos, dtype=torch.long)
        opponent_elo_t = torch.tensor(opponent_elos, dtype=torch.long)

        # 6. one batched inference call for every active game
        with torch.no_grad():
            policy_logits, _ = model(x, player_elo_t, opponent_elo_t)

        # 7-13. per game: mask, softmax(T=1), sample with its own generator,
        # decode, push, and record -- identical to single.play_game.
        for game, logits in zip(active, policy_logits):
            legal_mask = legal_move_mask(game.board)
            logits = logits.masked_fill(
                ~legal_mask, torch.finfo(logits.dtype).min
            )

            probs = torch.softmax(logits, dim=-1)
            index = torch.multinomial(
                probs, num_samples=1, generator=game.generator
            ).item()
            move = policy_index_to_move(index, game.board)

            game.board.push(move)
            game.moves.append(move.uci())
            game.history.append(game.board.copy(stack=False))

    return [
        {
            "white_elo": game.white_elo,
            "black_elo": game.black_elo,
            "result": game.result,
            "moves": game.moves,
        }
        for game in games
    ]
