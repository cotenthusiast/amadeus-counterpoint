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
    device = next(wrapper.parameters()).device

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

        x = encode_history(history).unsqueeze(0).to(device)

        if board.turn == wrapper_color:
            mover_elo = torch.tensor([wrapper.nominal_elo], dtype=torch.float32, device=device)
            opp_elo = torch.tensor([opponent_elo], dtype=torch.float32, device=device)
            with torch.no_grad():
                policy_logits, _ = wrapper(x, mover_elo, opp_elo)
        else:
            mover_elo = torch.tensor([opponent_elo], dtype=torch.float32, device=device)
            opp_elo = torch.tensor([wrapper.nominal_elo], dtype=torch.float32, device=device)
            with torch.no_grad():
                policy_logits, _ = wrapper.base(x, mover_elo, opp_elo)

        mask = legal_move_mask(board).to(device)
        policy_logits = policy_logits.squeeze(0).masked_fill(
            ~mask, torch.finfo(policy_logits.dtype).min
        )

        # `generator` is a plain (CPU) torch.Generator -- sampling must
        # happen on CPU to match it (a CUDA generator would change which
        # game a given seed produces, not just fix a device mismatch).
        probs = torch.softmax(policy_logits, dim=-1)
        index = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item()

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
    device = next(wrapper_a.parameters()).device

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

        x = encode_history(history).unsqueeze(0).to(device)

        if board.turn == a_color:
            mover, opponent = wrapper_a, wrapper_b
        else:
            mover, opponent = wrapper_b, wrapper_a

        mover_elo = torch.tensor([mover.nominal_elo], dtype=torch.float32, device=device)
        opp_elo = torch.tensor([opponent.nominal_elo], dtype=torch.float32, device=device)

        with torch.no_grad():
            policy_logits, _ = mover(x, mover_elo, opp_elo)

        mask = legal_move_mask(board).to(device)
        policy_logits = policy_logits.squeeze(0).masked_fill(
            ~mask, torch.finfo(policy_logits.dtype).min
        )

        # `generator` is a plain (CPU) torch.Generator -- sampling must
        # happen on CPU to match it (a CUDA generator would change which
        # game a given seed produces, not just fix a device mismatch).
        probs = torch.softmax(policy_logits, dim=-1)
        index = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item()

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


# ---------------------------------------------------------------------------
# Batched (many-games-in-parallel) variants, mirroring
# evaluation.generation.batch.play_games' active-games architecture.
#
# Key simplification, true by construction: one call here always covers one
# (dyad, condition, orientation) CELL -- every game in the batch shares the
# same wrapper_color/opponent identity/elos, and all games start at ply 0
# and advance exactly one ply per active game per outer-loop iteration.
# Therefore every game still in `active` at a given iteration has had the
# identical number of plies pushed so far, so `active[0].board.turn` alone
# tells you whose turn it is for the WHOLE active batch -- there is no need
# to route different games in one batch through different representations
# per ply. This is what makes one batched model call per ply correct here,
# not an approximation.
# ---------------------------------------------------------------------------


class _Method1GameState:
    """Mutable per-game state for play_games_method1 (single-sided)."""

    def __init__(self, opponent_elo, seed):
        self.opponent_elo = opponent_elo
        self.board = create_board()
        self.history = [self.board.copy(stack=False)]
        self.moves = []
        self.generator = torch.Generator().manual_seed(seed)
        self.result = None
        self.censored = False


def play_games_method1(
    wrapper: PersonalizedChessformer,
    wrapper_color: chess.Color,
    opponent_elos,
    seeds,
) -> list[dict]:
    """Batched version of play_game_method1: many games, all sharing the
    same wrapper/wrapper_color (one AG-or-GB cell's worth). Each game keeps
    its own board, history, move list, and seeded torch.Generator, so a
    game's sampled moves depend only on its own seed -- never its slot in
    the batch, execution order, or when other games finish. Finished games
    drop out of the active batch; still-active games are advanced with one
    batched model call per ply. Returns a list of game dicts, one per input
    game, in input order, shaped exactly like play_game_method1's.
    """
    if len(opponent_elos) != len(seeds):
        raise ValueError("opponent_elos and seeds must have equal lengths")

    wrapper.eval()
    device = next(wrapper.parameters()).device

    games = [_Method1GameState(elo, seed) for elo, seed in zip(opponent_elos, seeds)]

    while True:
        for game in games:
            if game.result is not None or game.censored:
                continue
            outcome = check_end(game.board)
            if outcome is not None:
                game.result = outcome.result()
            elif len(game.moves) >= MAX_PLIES:
                game.censored = True

        active = [g for g in games if g.result is None and not g.censored]
        if not active:
            break

        is_wrapper_turn = active[0].board.turn == wrapper_color

        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        opp_elo_t = torch.tensor([g.opponent_elo for g in active], dtype=torch.float32, device=device)

        if is_wrapper_turn:
            mover_elo_t = torch.full(
                (len(active),), wrapper.nominal_elo, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                policy_logits, _ = wrapper(x, mover_elo_t, opp_elo_t)
        else:
            mover_elo_t = opp_elo_t
            opp_elo_t = torch.full(
                (len(active),), wrapper.nominal_elo, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                policy_logits, _ = wrapper.base(x, mover_elo_t, opp_elo_t)

        for game, logits in zip(active, policy_logits):
            legal_mask = legal_move_mask(game.board).to(device)
            logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)

            # game.generator is a plain (CPU) torch.Generator -- sampling
            # must happen on CPU to match it.
            probs = torch.softmax(logits, dim=-1)
            index = torch.multinomial(probs.cpu(), num_samples=1, generator=game.generator).item()

            move = policy_index_to_move(index, game.board)
            game.board.push(move)
            game.moves.append(move.uci())
            game.history.append(game.board.copy(stack=False))

    results = []
    for game in games:
        if wrapper_color == chess.WHITE:
            white_elo, black_elo = wrapper.nominal_elo, game.opponent_elo
        else:
            white_elo, black_elo = game.opponent_elo, wrapper.nominal_elo
        results.append({
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": game.result,
            "censored": game.censored,
            "moves": game.moves,
        })
    return results


class _Method1ABGameState:
    """Mutable per-game state for play_games_method1_ab (both-sided)."""

    def __init__(self, seed):
        self.board = create_board()
        self.history = [self.board.copy(stack=False)]
        self.moves = []
        self.generator = torch.Generator().manual_seed(seed)
        self.result = None
        self.censored = False


def play_games_method1_ab(
    wrapper_a: PersonalizedChessformer,
    wrapper_b: PersonalizedChessformer,
    a_color: chess.Color,
    seeds,
) -> list[dict]:
    """Batched version of play_game_method1_ab: many games, all sharing the
    same wrapper_a/wrapper_b/a_color (one AB cell's worth). Same active-batch
    architecture as play_games_method1. `wrapper_a` and `wrapper_b` are
    expected to share the same frozen base model.
    """
    wrapper_a.eval()
    wrapper_b.eval()
    device = next(wrapper_a.parameters()).device

    games = [_Method1ABGameState(seed) for seed in seeds]

    while True:
        for game in games:
            if game.result is not None or game.censored:
                continue
            outcome = check_end(game.board)
            if outcome is not None:
                game.result = outcome.result()
            elif len(game.moves) >= MAX_PLIES:
                game.censored = True

        active = [g for g in games if g.result is None and not g.censored]
        if not active:
            break

        if active[0].board.turn == a_color:
            mover, opponent = wrapper_a, wrapper_b
        else:
            mover, opponent = wrapper_b, wrapper_a

        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        mover_elo_t = torch.full((len(active),), mover.nominal_elo, dtype=torch.float32, device=device)
        opp_elo_t = torch.full((len(active),), opponent.nominal_elo, dtype=torch.float32, device=device)

        with torch.no_grad():
            policy_logits, _ = mover(x, mover_elo_t, opp_elo_t)

        for game, logits in zip(active, policy_logits):
            legal_mask = legal_move_mask(game.board).to(device)
            logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)

            probs = torch.softmax(logits, dim=-1)
            index = torch.multinomial(probs.cpu(), num_samples=1, generator=game.generator).item()

            move = policy_index_to_move(index, game.board)
            game.board.push(move)
            game.moves.append(move.uci())
            game.history.append(game.board.copy(stack=False))

    results = []
    for game in games:
        if a_color == chess.WHITE:
            white_elo, black_elo = wrapper_a.nominal_elo, wrapper_b.nominal_elo
        else:
            white_elo, black_elo = wrapper_b.nominal_elo, wrapper_a.nominal_elo
        results.append({
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": game.result,
            "censored": game.censored,
            "moves": game.moves,
        })
    return results
