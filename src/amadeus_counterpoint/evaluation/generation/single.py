"""Correctness-first generation of one complete neural chess game using
existing chess/model/encoding primitives.
"""

import chess
import torch

from amadeus_counterpoint.chess import check_end, create_board
from amadeus_counterpoint.encoding import (
    encode_history,
    legal_move_mask,
    policy_index_to_move,
)

MAX_PLIES = 500


def play_game(model, white_elo, black_elo, seed):
    """Generate one complete self-play game and return it as a game record.

    `model` is run in eval mode under `torch.no_grad()`; sampling is driven
    by a `torch.Generator` seeded with `seed`, so the same inputs reproduce
    the same game. Returns `white_elo`, `black_elo`, `result`, `censored`,
    and `moves`; capped games have `result=None` and `censored=True` rather
    than an adjudicated result.
    """
    model.eval()

    board = create_board()
    history = [board.copy(stack=False)]
    moves = []

    generator = torch.Generator().manual_seed(seed)

    result = None
    censored = False

    while True:
        # 1. normal chess termination?
        outcome = check_end(board)
        if outcome is not None:
            result = outcome.result()
            break

        # 2. 500 ply?
        if len(moves) >= MAX_PLIES:
            # A ply-cap truncation is not an adjudicated draw.
            censored = True
            break

        # 3. who is actually moving?
        if board.turn == chess.WHITE:
            player_elo = white_elo
            opponent_elo = black_elo
        else:
            player_elo = black_elo
            opponent_elo = white_elo

        # 4. encode history
        x = encode_history(history).unsqueeze(0)

        # 5. Elo -> tensors
        player_elo_t = torch.tensor([player_elo], dtype=torch.float32)
        opponent_elo_t = torch.tensor([opponent_elo], dtype=torch.float32)

        # 6. inference
        with torch.no_grad():
            policy_logits, _ = model(x, player_elo_t, opponent_elo_t)

        # 7. legal mask
        legal_mask = legal_move_mask(board)

        # 8. mask logits
        policy_logits = policy_logits.squeeze(0).masked_fill(
            ~legal_mask, torch.finfo(policy_logits.dtype).min
        )

        # 9. softmax(T=1)
        probs = torch.softmax(policy_logits, dim=-1)

        # 10. seeded sample
        index = torch.multinomial(probs, num_samples=1, generator=generator).item()

        # 11. index -> move
        move = policy_index_to_move(index, board)

        # 12. push
        board.push(move)

        # 13. record move + board snapshot
        moves.append(move.uci())
        history.append(board.copy(stack=False))

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "censored": censored,
        "moves": moves,
    }
