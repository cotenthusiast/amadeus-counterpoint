"""Method-2 (style-residual) personalized game generation.

Kept separate from the ordinary (non-personalized) path in `single.py`/
`batch.py`, which are unmodified and untouched by this module, and separate
from Method 1's generation. Reuses `score_candidates` (Stage 3) with
`target_index=None` rather than duplicating the candidate-selection /
plane-construction / CNN / residual pipeline.
"""

import chess
import torch

from amadeus_counterpoint.chess import check_end, create_board
from amadeus_counterpoint.encoding import encode_history, legal_move_mask, policy_index_to_move
from amadeus_counterpoint.evaluation.generation.single import MAX_PLIES
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.models.style_scoring import score_candidates


def play_game_method2(
    base: Chessformer,
    cnn: MoveStyleCNN,
    table: PlayerStyleTable,
    residual: StyleResidual,
    player_id: int,
    player_color: chess.Color,
    player_elo: float,
    opponent_elo: float,
    k: int,
    seed: int,
) -> dict:
    """Generate one self-play game with a Method-2 personalized player.

    `player_color` is played by candidate-restricted style-residual scoring
    (`score_candidates` with `target_index=None`, i.e. no target-append: the
    candidate set is exactly the raw top-`k` legal moves, width `k`, and
    sampling is a softmax over only that candidate set -- moves outside it
    receive zero probability). The other color is played by `base` directly
    under ordinary full-legal-policy generation, mirroring `single.play_game`
    exactly, with no style machinery at all.

    Returns a dict shaped like `single.play_game`'s (`white_elo`,
    `black_elo`, `result`, `censored`, `moves`).
    """
    base.eval()
    cnn.eval()
    table.eval()
    residual.eval()

    board = create_board()
    history = [board.copy(stack=False)]
    moves = []

    generator = torch.Generator().manual_seed(seed)

    result = None
    censored = False

    player_id_t = torch.tensor([player_id])

    while True:
        outcome = check_end(board)
        if outcome is not None:
            result = outcome.result()
            break

        if len(moves) >= MAX_PLIES:
            censored = True
            break

        x = encode_history(history).unsqueeze(0)
        mask = legal_move_mask(board).unsqueeze(0)

        if board.turn == player_color:
            mover_elo = torch.tensor([player_elo], dtype=torch.float32)
            opp_elo = torch.tensor([opponent_elo], dtype=torch.float32)

            with torch.no_grad():
                output = score_candidates(
                    base, cnn, table, residual,
                    x, mover_elo, opp_elo, player_id_t, mask,
                    k, target_index=None,
                )

            probs = torch.softmax(output.candidate_scores.squeeze(0), dim=-1)
            local_index = torch.multinomial(probs, num_samples=1, generator=generator).item()
            index = output.candidate_indices.squeeze(0)[local_index].item()
        else:
            mover_elo = torch.tensor([opponent_elo], dtype=torch.float32)
            opp_elo = torch.tensor([player_elo], dtype=torch.float32)

            with torch.no_grad():
                policy_logits, _ = base(x, mover_elo, opp_elo)
            policy_logits = policy_logits.squeeze(0).masked_fill(
                ~mask.squeeze(0), torch.finfo(policy_logits.dtype).min
            )

            probs = torch.softmax(policy_logits, dim=-1)
            index = torch.multinomial(probs, num_samples=1, generator=generator).item()

        move = policy_index_to_move(index, board)
        board.push(move)

        moves.append(move.uci())
        history.append(board.copy(stack=False))

    if player_color == chess.WHITE:
        white_elo, black_elo = player_elo, opponent_elo
    else:
        white_elo, black_elo = opponent_elo, player_elo

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "censored": censored,
        "moves": moves,
    }


def play_game_method2_ab(
    base: Chessformer,
    cnn: MoveStyleCNN,
    table: PlayerStyleTable,
    residual: StyleResidual,
    player_id_a: int,
    player_id_b: int,
    a_color: chess.Color,
    elo_a: float,
    elo_b: float,
    k: int,
    seed: int,
) -> dict:
    """Generate one self-play game where BOTH sides are Method-2 personalized.

    `a_color` is the color player A has this game (`chess.WHITE` for the
    A_WHITE orientation, `chess.BLACK` for B_WHITE); player B always has the
    other color. On each ply, candidate-restricted style-residual scoring
    (`score_candidates`, `target_index=None`) is run with the mover's own
    `player_id`, selecting that player's row out of the shared
    `PlayerStyleTable`; `cnn`, `residual`, and the frozen `base` are the same
    object for both sides -- exactly Method 2's jointly-trained style stack,
    just evaluated for whichever player is on move.

    Returns a dict shaped like `single.play_game`'s (`white_elo`,
    `black_elo`, `result`, `censored`, `moves`).
    """
    base.eval()
    cnn.eval()
    table.eval()
    residual.eval()

    board = create_board()
    history = [board.copy(stack=False)]
    moves = []

    generator = torch.Generator().manual_seed(seed)

    result = None
    censored = False

    player_id_a_t = torch.tensor([player_id_a])
    player_id_b_t = torch.tensor([player_id_b])

    while True:
        outcome = check_end(board)
        if outcome is not None:
            result = outcome.result()
            break

        if len(moves) >= MAX_PLIES:
            censored = True
            break

        x = encode_history(history).unsqueeze(0)
        mask = legal_move_mask(board).unsqueeze(0)

        if board.turn == a_color:
            mover_id_t, mover_elo, opp_elo = player_id_a_t, elo_a, elo_b
        else:
            mover_id_t, mover_elo, opp_elo = player_id_b_t, elo_b, elo_a

        mover_elo_t = torch.tensor([mover_elo], dtype=torch.float32)
        opp_elo_t = torch.tensor([opp_elo], dtype=torch.float32)

        with torch.no_grad():
            output = score_candidates(
                base, cnn, table, residual,
                x, mover_elo_t, opp_elo_t, mover_id_t, mask,
                k, target_index=None,
            )

        probs = torch.softmax(output.candidate_scores.squeeze(0), dim=-1)
        local_index = torch.multinomial(probs, num_samples=1, generator=generator).item()
        index = output.candidate_indices.squeeze(0)[local_index].item()

        move = policy_index_to_move(index, board)
        board.push(move)

        moves.append(move.uci())
        history.append(board.copy(stack=False))

    if a_color == chess.WHITE:
        white_elo, black_elo = elo_a, elo_b
    else:
        white_elo, black_elo = elo_b, elo_a

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "censored": censored,
        "moves": moves,
    }
