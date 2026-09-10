"""Method-1 (player-embedding) personalized game generation.

Kept separate from the ordinary (non-personalized) path in `single.py`/
`batch.py`, which are unmodified and untouched by this module. No top-K
restriction, no style CNN/residual/candidate selector -- full legal-policy
softmax throughout, mirroring `single.play_game`'s mask/softmax/sample/decode
mechanics exactly.
"""

import chess
import torch

from amadeus_counterpoint.chess import check_end, create_board
from amadeus_counterpoint.encoding import encode_history, legal_move_mask, policy_index_to_move
from amadeus_counterpoint.evaluation.generation.single import MAX_PLIES
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer


def play_game_method1(
    wrapper: PersonalizedChessformer,
    wrapper_color: chess.Color,
    opponent_elo: float,
    seed: int,
) -> dict:
    """Generate one self-play game with a Method-1 personalized player.

    `wrapper_color` is played by `wrapper` (its trainable `z_player`
    replaces only the mover embedding on those plies, via
    `Chessformer.forward`'s `player_emb_override`; opponent conditioning is
    always the ordinary `interpolate_elo(opponent_elo)`, per
    `PersonalizedChessformer.forward`). The other color is played by
    `wrapper.base` directly under ordinary Elo interpolation for both
    sides -- the personalized player is represented to their opponent by
    `wrapper.nominal_elo`, not by `z_player` (single-sided personalization,
    matching Method 1's frozen design).

    Returns a dict shaped like `single.play_game`'s (`white_elo`,
    `black_elo`, `result`, `censored`, `moves`); `white_elo`/`black_elo`
    report `wrapper.nominal_elo` for `wrapper_color` and `opponent_elo` for
    the other side.
    """
    wrapper.eval()

    board = create_board()
    history = [board.copy(stack=False)]
    moves = []

    generator = torch.Generator().manual_seed(seed)

    result = None
    censored = False

    while True:
        outcome = check_end(board)
        if outcome is not None:
            result = outcome.result()
            break

        if len(moves) >= MAX_PLIES:
            censored = True
            break

        x = encode_history(history).unsqueeze(0)

        if board.turn == wrapper_color:
            mover_elo = torch.tensor([wrapper.nominal_elo], dtype=torch.float32)
            opp_elo = torch.tensor([opponent_elo], dtype=torch.float32)
            with torch.no_grad():
                policy_logits, _ = wrapper(x, mover_elo, opp_elo)
        else:
            mover_elo = torch.tensor([opponent_elo], dtype=torch.float32)
            opp_elo = torch.tensor([wrapper.nominal_elo], dtype=torch.float32)
            with torch.no_grad():
                policy_logits, _ = wrapper.base(x, mover_elo, opp_elo)

        mask = legal_move_mask(board)
        policy_logits = policy_logits.squeeze(0).masked_fill(
            ~mask, torch.finfo(policy_logits.dtype).min
        )

        probs = torch.softmax(policy_logits, dim=-1)
        index = torch.multinomial(probs, num_samples=1, generator=generator).item()

        move = policy_index_to_move(index, board)
        board.push(move)

        moves.append(move.uci())
        history.append(board.copy(stack=False))

    if wrapper_color == chess.WHITE:
        white_elo, black_elo = wrapper.nominal_elo, opponent_elo
    else:
        white_elo, black_elo = opponent_elo, wrapper.nominal_elo

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "censored": censored,
        "moves": moves,
    }


def play_game_method1_ab(
    wrapper_a: PersonalizedChessformer,
    wrapper_b: PersonalizedChessformer,
    a_color: chess.Color,
    seed: int,
) -> dict:
    """Generate one self-play game where BOTH sides are Method-1 personalized.

    `a_color` is the color player A has this game (`chess.WHITE` for the
    A_WHITE orientation, `chess.BLACK` for B_WHITE); player B always has the
    other color. On each ply, the mover's own wrapper is called, so its own
    `z_player` replaces its own mover embedding; the *other* wrapper's
    `nominal_elo` is passed as `opponent_elo`, so opponent conditioning is
    still the ordinary `interpolate_elo` of the opponent's fixed
    representative Elo -- exactly as in single-sided `play_game_method1`,
    just with a personalized opponent instead of a generic one. `wrapper_a`
    and `wrapper_b` are expected to share the same frozen base model.

    Returns a dict shaped like `single.play_game`'s (`white_elo`,
    `black_elo`, `result`, `censored`, `moves`); `white_elo`/`black_elo`
    report whichever wrapper's `nominal_elo` has that color.
    """
    wrapper_a.eval()
    wrapper_b.eval()

    board = create_board()
    history = [board.copy(stack=False)]
    moves = []

    generator = torch.Generator().manual_seed(seed)

    result = None
    censored = False

    while True:
        outcome = check_end(board)
        if outcome is not None:
            result = outcome.result()
            break

        if len(moves) >= MAX_PLIES:
            censored = True
            break

        x = encode_history(history).unsqueeze(0)

        if board.turn == a_color:
            mover, opponent = wrapper_a, wrapper_b
        else:
            mover, opponent = wrapper_b, wrapper_a

        mover_elo = torch.tensor([mover.nominal_elo], dtype=torch.float32)
        opp_elo = torch.tensor([opponent.nominal_elo], dtype=torch.float32)

        with torch.no_grad():
            policy_logits, _ = mover(x, mover_elo, opp_elo)

        mask = legal_move_mask(board)
        policy_logits = policy_logits.squeeze(0).masked_fill(
            ~mask, torch.finfo(policy_logits.dtype).min
        )

        probs = torch.softmax(policy_logits, dim=-1)
        index = torch.multinomial(probs, num_samples=1, generator=generator).item()

        move = policy_index_to_move(index, board)
        board.push(move)

        moves.append(move.uci())
        history.append(board.copy(stack=False))

    if a_color == chess.WHITE:
        white_elo, black_elo = wrapper_a.nominal_elo, wrapper_b.nominal_elo
    else:
        white_elo, black_elo = wrapper_b.nominal_elo, wrapper_a.nominal_elo

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "censored": censored,
        "moves": moves,
    }
