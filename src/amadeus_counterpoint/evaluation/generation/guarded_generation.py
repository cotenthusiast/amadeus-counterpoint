"""Frozen strength-aware guardrail wired into the batched production
generation path (see rollout_quality_exploratory_2026-09-12/
final_sampler_freeze/FINAL_SAMPLER_REPORT.md for the frozen specification:
K=5, Stockfish 19 depth 8, lambda=2.0).

This module does NOT reimplement the guardrail's math -- it imports and
calls `strength_guardrail.strength_reweight` and
`strength_guardrail.candidate_cheap_losses` (the same functions the
diagnostic already validated with 11 tests, commit ca94af9) so there is
exactly one source of truth for score formula, mover-POV orientation, mate
handling, and the nonzero-probability numerical floor.

Kept separate from the existing unguarded `single.py` / `batch.py` /
`method1_personalized.py` / `method2_personalized.py`, which are NOT
modified by this module and remain the exact unguarded path used whenever
`experiment.py`'s cell-batched functions are called without a guardrail
(the existing default, `guardrail=None`) -- this module is additive, never
a replacement.

Architecture, one paragraph: every guarded batched entry point below
builds a small per-condition closure that (a) runs ONE batched model
forward pass over all still-active games this ply (exactly like the
existing unguarded functions already do, for GPU throughput) and (b)
returns each active game's own top-K candidate indices + p_behavior. That
per-condition closure is the ONLY piece that differs across
generic/Method-1/Method-2 and single- vs both-sided personalization --
state management, termination, batched Stockfish evaluation, per-game
reweight+sample, and move application are written EXACTLY ONCE in
`_run_batched_guarded_loop`, so the five guarded entry points cannot
silently drift apart from each other the way five independent copies of
the whole game loop could.

Engine orchestration: `StockfishEnginePool` is a persistent pool of
Stockfish engine WORKER PROCESSES (one engine per worker, opened once,
reused for the pool's entire lifetime -- never restarted per move or per
candidate). It uses Python's `spawn` start method explicitly, never the
default `fork`: an earlier attempt at a different (throwaway diagnostic)
parallel design used a fork-based Pool of workers that ALSO loaded a
PyTorch model and hit a real, silent hang (fork() can leave a child
holding a mutex some other thread of the parent held at fork time, with
that thread no longer existing post-fork -- a well-documented PyTorch/
multiprocessing interaction). These Stockfish workers never import torch
at all, and `spawn` additionally guarantees they start as fresh
interpreters with no inherited state from the CUDA/PyTorch generation
process regardless. `evaluate_many` uses `pool.starmap`, which is
blocking and ORDER-PRESERVING: results come back in the same order as the
input tasks no matter which worker finishes first, which is what keeps
each game's seeded `torch.Generator` draw independent of worker
completion order/timing.

Scope note: this pass implements the simpler, synchronous per-ply design
(compute this ply's candidates for the whole active batch -> one blocking
batched Stockfish call -> sample -> next ply) rather than an asynchronous
pipeline that overlaps one cohort's Stockfish evaluation with another
cohort's GPU inference. See FINAL_SAMPLER_REPORT.md's throughput section
for the measured numbers this choice implies and whether that overlap is
worth building next.
"""

import multiprocessing as mp

import chess
import torch

from amadeus_counterpoint.chess import check_end, create_board
from amadeus_counterpoint.encoding import encode_history, legal_move_mask, policy_index_to_move
from amadeus_counterpoint.evaluation.generation.single import MAX_PLIES
from amadeus_counterpoint.evaluation.generation.strength_guardrail import (
    StrengthGuardrailConfig,
    candidate_cheap_losses,
    strength_reweight,
    top_k_candidates,
)
from amadeus_counterpoint.models.style_scoring import score_candidates

_SPAWN_CTX = mp.get_context("spawn")

_ENGINE = None  # per-worker-process global; set once by _pool_worker_init


def _pool_worker_init(stockfish_path, gcc_lib_path, threads, hash_mb):
    import os

    import chess.engine as _ce

    global _ENGINE
    env = os.environ.copy()
    if gcc_lib_path:
        env["LD_LIBRARY_PATH"] = gcc_lib_path + ":" + env.get("LD_LIBRARY_PATH", "")
    _ENGINE = _ce.SimpleEngine.popen_uci(stockfish_path, env=env)
    _ENGINE.configure({"Threads": threads, "Hash": hash_mb})


def _pool_worker_eval(task):
    fen, candidate_ucis, depth = task
    board = chess.Board(fen)
    moves = [chess.Move.from_uci(u) for u in candidate_ucis]
    return candidate_cheap_losses(_ENGINE, board, moves, depth)


def _pool_worker_shutdown(_ignored):
    global _ENGINE
    if _ENGINE is not None:
        _ENGINE.quit()
        _ENGINE = None


class StockfishEnginePool:
    """See module docstring. `n_workers` persistent engine processes,
    `spawn`-started, reused for the pool's whole lifetime. A worker crash
    (engine died, UCI error, etc.) raises out of `evaluate_many` --
    deliberately no try/except here that would fall back to unguarded
    sampling; a broken engine must stop generation, not silently degrade
    the frozen scientific guarantee that every retained candidate is
    actually Stockfish-evaluated."""

    def __init__(self, n_workers, stockfish_path, gcc_lib_path=None, threads=1, hash_mb=128):
        self.n_workers = n_workers
        self._pool = _SPAWN_CTX.Pool(
            n_workers,
            initializer=_pool_worker_init,
            initargs=(stockfish_path, gcc_lib_path, threads, hash_mb),
        )
        self._closed = False

    def evaluate_many(self, tasks: list[tuple[str, list[str]]], depth: int) -> list[list[float]]:
        """`tasks`: list of (fen, candidate_ucis). Returns cheap-loss lists
        in the SAME order as `tasks` (order-preserving `starmap`; see
        module docstring for why this matters for reproducibility)."""
        full_tasks = [(fen, ucis, depth) for fen, ucis in tasks]
        return self._pool.starmap(_pool_worker_eval, full_tasks)

    def close(self):
        if self._closed:
            return
        # Let each worker quit its own engine cleanly before the pool tears
        # the processes down, rather than relying on process-exit cleanup.
        self._pool.map(_pool_worker_shutdown, [None] * self.n_workers)
        self._pool.close()
        self._pool.join()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class _GuardedGameState:
    """Mutable per-game state, shared shape for every guarded batched
    entry point below (mirrors `batch._GameState` / method1's /
    method2's per-condition state classes, unified into one)."""

    def __init__(self, seed, white_elo=None, black_elo=None, opponent_elo=None):
        self.board = create_board()
        self.history = [self.board.copy(stack=False)]
        self.moves = []
        self.generator = torch.Generator().manual_seed(seed)
        self.result = None
        self.censored = False
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.opponent_elo = opponent_elo  # single-sided AG/GB entry points only


def _active_games(games):
    for game in games:
        if game.result is not None or game.censored:
            continue
        outcome = check_end(game.board)
        if outcome is not None:
            game.result = outcome.result()
        elif len(game.moves) >= MAX_PLIES:
            game.censored = True
    return [g for g in games if g.result is None and not g.censored]


def _run_batched_guarded_loop(games, get_mover_candidates, engine_pool, config: StrengthGuardrailConfig):
    """The ONE shared driver for every guarded batched entry point below.

    `get_mover_candidates(active_games)` -> list of (candidate_indices:
    list[int], p_behavior: list[float]), one per active game, in the SAME
    order as `active_games`. This closure is the only thing that differs
    per condition/method; everything else here is written exactly once.
    """
    while True:
        active = _active_games(games)
        if not active:
            break

        candidates = get_mover_candidates(active)

        tasks = []
        for game, (indices, _p_behavior) in zip(active, candidates):
            ucis = [policy_index_to_move(idx, game.board).uci() for idx in indices]
            tasks.append((game.board.fen(), ucis))
        cheap_losses_per_game = engine_pool.evaluate_many(tasks, config.cheap_depth)

        for game, (indices, p_behavior), cheap_losses in zip(active, candidates, cheap_losses_per_game):
            q = strength_reweight(p_behavior, cheap_losses, config.lam)
            local_idx = int(torch.multinomial(q, num_samples=1, generator=game.generator).item())
            move = policy_index_to_move(indices[local_idx], game.board)
            game.board.push(move)
            game.moves.append(move.uci())
            game.history.append(game.board.copy(stack=False))

    return games


def _finish_games(games):
    return [
        {
            "white_elo": game.white_elo, "black_elo": game.black_elo,
            "result": game.result, "censored": game.censored, "moves": game.moves,
        }
        for game in games
    ]


def _generic_forward_candidates(base, device, k):
    """Shared candidate-extraction closure for any "generic policy" mover
    -- used both by the pure-GG entry point and by the ungoverned side of
    single-sided Method-1/Method-2 personalization below."""

    def compute(active, player_elos, opponent_elos):
        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        player_elo_t = torch.tensor(player_elos, dtype=torch.float32, device=device)
        opponent_elo_t = torch.tensor(opponent_elos, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = base(x, player_elo_t, opponent_elo_t)
        out = []
        for game, lg in zip(active, logits):
            mask = legal_move_mask(game.board).to(device)
            probs = torch.softmax(lg.masked_fill(~mask, torch.finfo(lg.dtype).min), dim=-1)
            indices, _, p_behavior = top_k_candidates(probs, game.board, k)
            out.append((indices, p_behavior))
        return out

    return compute


def play_games_guarded(base, white_elos, black_elos, seeds, config: StrengthGuardrailConfig, engine_pool):
    """Guarded mirror of `batch.play_games` (generic-vs-generic): the SAME
    top-K + strength-reweighting mechanism is applied to the generic
    policy on BOTH sides, matching the frozen "identical infrastructure
    across GG/AG/GB/AB" requirement."""
    lengths = (len(white_elos), len(black_elos), len(seeds))
    if len(set(lengths)) != 1:
        raise ValueError("white_elos, black_elos, and seeds must have equal lengths")

    base.eval()
    device = next(base.parameters()).device
    games = [_GuardedGameState(seed, w, b) for w, b, seed in zip(white_elos, black_elos, seeds)]
    forward = _generic_forward_candidates(base, device, config.k)

    def get_mover_candidates(active):
        player_elos, opponent_elos = [], []
        for game in active:
            if game.board.turn == chess.WHITE:
                player_elos.append(game.white_elo)
                opponent_elos.append(game.black_elo)
            else:
                player_elos.append(game.black_elo)
                opponent_elos.append(game.white_elo)
        return forward(active, player_elos, opponent_elos)

    _run_batched_guarded_loop(games, get_mover_candidates, engine_pool, config)
    return _finish_games(games)


def play_games_method1_guarded(wrapper, wrapper_color, opponent_elos, seeds,
                                config: StrengthGuardrailConfig, engine_pool):
    """Guarded mirror of `method1_personalized.play_games_method1`
    (single-sided AG/GB): `wrapper_color` uses the player's own
    `PersonalizedChessformer`; the opposite color uses the frozen base
    (generic) -- BOTH sides guarded identically, per the frozen spec.

    Relies on the same invariant the existing unguarded batched functions
    already document and rely on: every game in one (dyad, condition,
    orientation) cell shares the same identity/elo assignment, so all
    still-active games are on the same side-to-move at any given
    iteration -- `active[0].board.turn` is checked once per ply, not once
    per game.
    """
    if len(opponent_elos) != len(seeds):
        raise ValueError("opponent_elos and seeds must have equal lengths")

    wrapper.eval()
    device = next(wrapper.parameters()).device
    games = [_GuardedGameState(seed, opponent_elo=elo) for elo, seed in zip(opponent_elos, seeds)]
    k = config.k

    def get_mover_candidates(active):
        is_wrapper_turn = active[0].board.turn == wrapper_color
        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        opp_elo_t = torch.tensor([g.opponent_elo for g in active], dtype=torch.float32, device=device)

        if is_wrapper_turn:
            mover_elo_t = torch.full((len(active),), wrapper.nominal_elo, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = wrapper(x, mover_elo_t, opp_elo_t)
        else:
            mover_elo_t = opp_elo_t
            opp_elo_for_base = torch.full((len(active),), wrapper.nominal_elo, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, _ = wrapper.base(x, mover_elo_t, opp_elo_for_base)

        out = []
        for game, lg in zip(active, logits):
            mask = legal_move_mask(game.board).to(device)
            probs = torch.softmax(lg.masked_fill(~mask, torch.finfo(lg.dtype).min), dim=-1)
            indices, _, p_behavior = top_k_candidates(probs, game.board, k)
            out.append((indices, p_behavior))
        return out

    _run_batched_guarded_loop(games, get_mover_candidates, engine_pool, config)

    return [
        {
            "white_elo": g.opponent_elo if wrapper_color == chess.BLACK else wrapper.nominal_elo,
            "black_elo": g.opponent_elo if wrapper_color == chess.WHITE else wrapper.nominal_elo,
            "result": g.result, "censored": g.censored, "moves": g.moves,
        }
        for g in games
    ]


def play_games_method1_ab_guarded(wrapper_a, wrapper_b, a_color, seeds,
                                   config: StrengthGuardrailConfig, engine_pool):
    """Guarded mirror of `method1_personalized.play_games_method1_ab`
    (both sides personalized, Method 1)."""
    wrapper_a.eval()
    wrapper_b.eval()
    device = next(wrapper_a.parameters()).device
    games = [_GuardedGameState(seed) for seed in seeds]
    k = config.k

    def get_mover_candidates(active):
        is_a_turn = active[0].board.turn == a_color
        mover_wrapper, opp_wrapper = (wrapper_a, wrapper_b) if is_a_turn else (wrapper_b, wrapper_a)

        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        mover_elo_t = torch.full((len(active),), mover_wrapper.nominal_elo, dtype=torch.float32, device=device)
        opp_elo_t = torch.full((len(active),), opp_wrapper.nominal_elo, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, _ = mover_wrapper(x, mover_elo_t, opp_elo_t)

        out = []
        for game, lg in zip(active, logits):
            mask = legal_move_mask(game.board).to(device)
            probs = torch.softmax(lg.masked_fill(~mask, torch.finfo(lg.dtype).min), dim=-1)
            indices, _, p_behavior = top_k_candidates(probs, game.board, k)
            out.append((indices, p_behavior))
        return out

    _run_batched_guarded_loop(games, get_mover_candidates, engine_pool, config)

    results = []
    for g in games:
        white_elo, black_elo = (
            (wrapper_a.nominal_elo, wrapper_b.nominal_elo) if a_color == chess.WHITE
            else (wrapper_b.nominal_elo, wrapper_a.nominal_elo)
        )
        results.append({"white_elo": white_elo, "black_elo": black_elo,
                         "result": g.result, "censored": g.censored, "moves": g.moves})
    return results


def _method2_candidates(base, cnn, table, residual, x, mover_elo_t, opp_elo_t, player_id_t, mask, k):
    """Shared Method-2 candidate extraction: raw base top-K then style
    rerank (`score_candidates`), filtering the invalid padding slots that
    occur when fewer than K legal moves exist -- an earlier diagnostic
    hit exactly this bug (pushing an invalid placeholder move onto a real
    board) by forgetting this filter; fixed there, and built in here from
    the start. Returns, per row in the batch: (candidate_indices,
    p_behavior) where p_behavior is softmax(candidate_scores) over ONLY
    the valid candidates."""
    with torch.no_grad():
        output = score_candidates(base, cnn, table, residual, x, mover_elo_t, opp_elo_t, player_id_t, mask, k, target_index=None)
    out = []
    for b in range(output.candidate_indices.shape[0]):
        valid = output.candidate_valid[b]
        indices = output.candidate_indices[b][valid].tolist()
        scores = output.candidate_scores[b][valid]
        probs = torch.softmax(scores, dim=-1).tolist()
        out.append((indices, probs))
    return out


def play_games_method2_guarded(base, cnn, table, residual, player_id, player_color, player_elo,
                                opponent_elos, k, seeds, config: StrengthGuardrailConfig, engine_pool):
    """Guarded mirror of `method2_personalized.play_games_method2`
    (single-sided AG/GB): the personalized side uses Method 2's existing
    candidate construction + style rerank as p_behavior (unchanged); the
    opposite (generic) side is ALSO guarded, via the same top-K mechanism
    as the pure-GG path, per the frozen "identical infrastructure"
    requirement."""
    if len(opponent_elos) != len(seeds):
        raise ValueError("opponent_elos and seeds must have equal lengths")

    base.eval()
    cnn.eval()
    table.eval()
    residual.eval()
    device = next(base.parameters()).device
    games = [_GuardedGameState(seed, opponent_elo=elo) for elo, seed in zip(opponent_elos, seeds)]
    generic_forward = _generic_forward_candidates(base, device, config.k)

    def get_mover_candidates(active):
        is_player_turn = active[0].board.turn == player_color
        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        mask = torch.stack([legal_move_mask(g.board) for g in active]).to(device)
        opp_elo_t = torch.tensor([g.opponent_elo for g in active], dtype=torch.float32, device=device)

        if is_player_turn:
            mover_elo_t = torch.full((len(active),), player_elo, dtype=torch.float32, device=device)
            player_id_t = torch.full((len(active),), player_id, dtype=torch.long, device=device)
            return _method2_candidates(base, cnn, table, residual, x, mover_elo_t, opp_elo_t, player_id_t, mask, k)
        else:
            player_elos_list = [player_elo] * len(active)
            opp_elos_list = [g.opponent_elo for g in active]
            return generic_forward(active, opp_elos_list, player_elos_list)

    _run_batched_guarded_loop(games, get_mover_candidates, engine_pool, config)

    return [
        {
            "white_elo": g.opponent_elo if player_color == chess.BLACK else player_elo,
            "black_elo": g.opponent_elo if player_color == chess.WHITE else player_elo,
            "result": g.result, "censored": g.censored, "moves": g.moves,
        }
        for g in games
    ]


def play_games_method2_ab_guarded(base, cnn, table, residual, player_id_a, player_id_b, a_color,
                                   elo_a, elo_b, k, seeds, config: StrengthGuardrailConfig, engine_pool):
    """Guarded mirror of `method2_personalized.play_games_method2_ab`
    (both sides personalized, Method 2)."""
    base.eval()
    cnn.eval()
    table.eval()
    residual.eval()
    device = next(base.parameters()).device
    games = [_GuardedGameState(seed) for seed in seeds]

    def get_mover_candidates(active):
        is_a_turn = active[0].board.turn == a_color
        mover_id, mover_elo, opp_elo = (player_id_a, elo_a, elo_b) if is_a_turn else (player_id_b, elo_b, elo_a)

        x = torch.stack([encode_history(g.history) for g in active]).to(device)
        mask = torch.stack([legal_move_mask(g.board) for g in active]).to(device)
        mover_elo_t = torch.full((len(active),), mover_elo, dtype=torch.float32, device=device)
        opp_elo_t = torch.full((len(active),), opp_elo, dtype=torch.float32, device=device)
        mover_id_t = torch.full((len(active),), mover_id, dtype=torch.long, device=device)
        return _method2_candidates(base, cnn, table, residual, x, mover_elo_t, opp_elo_t, mover_id_t, mask, k)

    _run_batched_guarded_loop(games, get_mover_candidates, engine_pool, config)

    white_elo, black_elo = (elo_a, elo_b) if a_color == chess.WHITE else (elo_b, elo_a)
    return [
        {"white_elo": white_elo, "black_elo": black_elo,
         "result": g.result, "censored": g.censored, "moves": g.moves}
        for g in games
    ]
