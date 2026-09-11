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


def _infer_device(module) -> torch.device:
    """Best-effort device detection from a module's own parameters, so
    per-move tensors can be moved to wherever the model actually lives
    instead of assuming CPU or hardcoding CUDA. Falls back to CPU for
    parameter-less test doubles (e.g. FakeBase), which are only ever used
    in CPU-only tests."""
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


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
    device = _infer_device(base)

    board = create_board()
    history = [board.copy(stack=False)]
    moves = []

    generator = torch.Generator().manual_seed(seed)

    result = None
    censored = False

    player_id_t = torch.tensor([player_id], device=device)

    while True:
        outcome = check_end(board)
        if outcome is not None:
            result = outcome.result()
            break

        if len(moves) >= MAX_PLIES:
            censored = True
            break

        x = encode_history(history).unsqueeze(0).to(device)
        mask = legal_move_mask(board).unsqueeze(0).to(device)

        if board.turn == player_color:
            mover_elo = torch.tensor([player_elo], dtype=torch.float32, device=device)
            opp_elo = torch.tensor([opponent_elo], dtype=torch.float32, device=device)

            with torch.no_grad():
                output = score_candidates(
                    base, cnn, table, residual,
                    x, mover_elo, opp_elo, player_id_t, mask,
                    k, target_index=None,
                )

            # `generator` is a plain (CPU) torch.Generator -- sampling must
            # happen on CPU to match it (a CUDA generator would change which
            # game a given seed produces, not just fix a device mismatch).
            probs = torch.softmax(output.candidate_scores.squeeze(0), dim=-1)
            local_index = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item()
            index = output.candidate_indices.squeeze(0)[local_index].item()
        else:
            mover_elo = torch.tensor([opponent_elo], dtype=torch.float32, device=device)
            opp_elo = torch.tensor([player_elo], dtype=torch.float32, device=device)

            with torch.no_grad():
                policy_logits, _ = base(x, mover_elo, opp_elo)
            policy_logits = policy_logits.squeeze(0).masked_fill(
                ~mask.squeeze(0), torch.finfo(policy_logits.dtype).min
            )

            probs = torch.softmax(policy_logits, dim=-1)
            index = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item()

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
    device = _infer_device(base)

    board = create_board()
    history = [board.copy(stack=False)]
    moves = []

    generator = torch.Generator().manual_seed(seed)

    result = None
    censored = False

    player_id_a_t = torch.tensor([player_id_a], device=device)
    player_id_b_t = torch.tensor([player_id_b], device=device)

    while True:
        outcome = check_end(board)
        if outcome is not None:
            result = outcome.result()
            break

        if len(moves) >= MAX_PLIES:
            censored = True
            break

        x = encode_history(history).unsqueeze(0).to(device)
        mask = legal_move_mask(board).unsqueeze(0).to(device)

        if board.turn == a_color:
            mover_id_t, mover_elo, opp_elo = player_id_a_t, elo_a, elo_b
        else:
            mover_id_t, mover_elo, opp_elo = player_id_b_t, elo_b, elo_a

        mover_elo_t = torch.tensor([mover_elo], dtype=torch.float32, device=device)
        opp_elo_t = torch.tensor([opp_elo], dtype=torch.float32, device=device)

        with torch.no_grad():
            output = score_candidates(
                base, cnn, table, residual,
                x, mover_elo_t, opp_elo_t, mover_id_t, mask,
                k, target_index=None,
            )

        # `generator` is a plain (CPU) torch.Generator -- sampling must
        # happen on CPU to match it (a CUDA generator would change which
        # game a given seed produces, not just fix a device mismatch).
        probs = torch.softmax(output.candidate_scores.squeeze(0), dim=-1)
        local_index = torch.multinomial(probs.cpu(), num_samples=1, generator=generator).item()
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


# ---------------------------------------------------------------------------
# Batched (many-games-in-parallel) variants, mirroring
# evaluation.generation.batch.play_games' active-games architecture. See
# method1_personalized.py's batched section for why one batched model call
# per ply is exact (not an approximation) when one call always covers a
# single (dyad, condition, orientation) cell: every game in the batch
# shares the same identity/elos, so all still-active games are always on
# the same side-to-move at any given iteration.
# ---------------------------------------------------------------------------


class _Method2GameState:
    """Mutable per-game state for play_games_method2 (single-sided)."""

    def __init__(self, opponent_elo, seed):
        self.opponent_elo = opponent_elo
        self.board = create_board()
        self.history = [self.board.copy(stack=False)]
        self.moves = []
        self.generator = torch.Generator().manual_seed(seed)
        self.result = None
        self.censored = False


def play_games_method2(
    base: Chessformer,
    cnn: MoveStyleCNN,
    table: PlayerStyleTable,
    residual: StyleResidual,
    player_id: int,
    player_color: chess.Color,
    player_elo: float,
    opponent_elos,
    k: int,
    seeds,
) -> list[dict]:
    """Batched version of play_game_method2: many games, all sharing the
    same player_id/player_color/player_elo (one AG-or-GB cell's worth).
    Same active-batch architecture as method1_personalized.play_games_method1.
    Returns a list of game dicts, one per input game, in input order, shaped
    exactly like play_game_method2's.
    """
    if len(opponent_elos) != len(seeds):
        raise ValueError("opponent_elos and seeds must have equal lengths")

    base.eval()
    cnn.eval()
    table.eval()
    residual.eval()
    device = _infer_device(base)

    games = [_Method2GameState(elo, seed) for elo, seed in zip(opponent_elos, seeds)]

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

        is_player_turn = active[0].board.turn == player_color

        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        mask = torch.stack([legal_move_mask(g.board) for g in active]).to(device)
        opp_elo_t = torch.tensor([g.opponent_elo for g in active], dtype=torch.float32, device=device)

        if is_player_turn:
            mover_elo_t = torch.full((len(active),), player_elo, dtype=torch.float32, device=device)
            player_id_t = torch.full((len(active),), player_id, dtype=torch.long, device=device)

            with torch.no_grad():
                output = score_candidates(
                    base, cnn, table, residual, x, mover_elo_t, opp_elo_t, player_id_t, mask,
                    k, target_index=None,
                )

            for game, scores, indices in zip(active, output.candidate_scores, output.candidate_indices):
                probs = torch.softmax(scores, dim=-1)
                local_index = torch.multinomial(probs.cpu(), num_samples=1, generator=game.generator).item()
                index = indices[local_index].item()
                move = policy_index_to_move(index, game.board)
                game.board.push(move)
                game.moves.append(move.uci())
                game.history.append(game.board.copy(stack=False))
        else:
            mover_elo_t = opp_elo_t
            opp_elo_t = torch.full((len(active),), player_elo, dtype=torch.float32, device=device)

            with torch.no_grad():
                policy_logits, _ = base(x, mover_elo_t, opp_elo_t)
            policy_logits = policy_logits.masked_fill(~mask, torch.finfo(policy_logits.dtype).min)

            for game, logits in zip(active, policy_logits):
                probs = torch.softmax(logits, dim=-1)
                index = torch.multinomial(probs.cpu(), num_samples=1, generator=game.generator).item()
                move = policy_index_to_move(index, game.board)
                game.board.push(move)
                game.moves.append(move.uci())
                game.history.append(game.board.copy(stack=False))

    results = []
    for game in games:
        if player_color == chess.WHITE:
            white_elo, black_elo = player_elo, game.opponent_elo
        else:
            white_elo, black_elo = game.opponent_elo, player_elo
        results.append({
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": game.result,
            "censored": game.censored,
            "moves": game.moves,
        })
    return results


class _Method2ABGameState:
    """Mutable per-game state for play_games_method2_ab (both-sided)."""

    def __init__(self, seed):
        self.board = create_board()
        self.history = [self.board.copy(stack=False)]
        self.moves = []
        self.generator = torch.Generator().manual_seed(seed)
        self.result = None
        self.censored = False


def play_games_method2_ab(
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
    seeds,
) -> list[dict]:
    """Batched version of play_game_method2_ab: many games, all sharing the
    same player_id_a/player_id_b/a_color/elo_a/elo_b (one AB cell's worth).
    """
    base.eval()
    cnn.eval()
    table.eval()
    residual.eval()
    device = _infer_device(base)

    games = [_Method2ABGameState(seed) for seed in seeds]

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
            mover_id, mover_elo, opp_elo = player_id_a, elo_a, elo_b
        else:
            mover_id, mover_elo, opp_elo = player_id_b, elo_b, elo_a

        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        mask = torch.stack([legal_move_mask(g.board) for g in active]).to(device)
        mover_elo_t = torch.full((len(active),), mover_elo, dtype=torch.float32, device=device)
        opp_elo_t = torch.full((len(active),), opp_elo, dtype=torch.float32, device=device)
        mover_id_t = torch.full((len(active),), mover_id, dtype=torch.long, device=device)

        with torch.no_grad():
            output = score_candidates(
                base, cnn, table, residual, x, mover_elo_t, opp_elo_t, mover_id_t, mask,
                k, target_index=None,
            )

        for game, scores, indices in zip(active, output.candidate_scores, output.candidate_indices):
            probs = torch.softmax(scores, dim=-1)
            local_index = torch.multinomial(probs.cpu(), num_samples=1, generator=game.generator).item()
            index = indices[local_index].item()
            move = policy_index_to_move(index, game.board)
            game.board.push(move)
            game.moves.append(move.uci())
            game.history.append(game.board.copy(stack=False))

    results = []
    for game in games:
        if a_color == chess.WHITE:
            white_elo, black_elo = elo_a, elo_b
        else:
            white_elo, black_elo = elo_b, elo_a
        results.append({
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": game.result,
            "censored": game.censored,
            "moves": game.moves,
        })
    return results
