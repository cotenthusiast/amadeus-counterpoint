"""Shared strength-aware soft-reweighting move-selection layer.

Frozen result of the exploratory rollout-quality investigation
(rollout_quality_exploratory_2026-09-12/): base/Method-1/Method-2 policies
already place a strong move in their own top-K almost everywhere on
non-sealed real GM positions -- even where their own argmax/unguarded
sampling collapses -- so a cheap post-hoc reweighting step recovers most
of the quality gap without touching any model weight.

Formula (units preserved exactly from the diagnostics that chose them --
see rollout_quality_exploratory_2026-09-12/final_sampler_freeze/
FINAL_SAMPLER_REPORT.md for the full derivation and lambda sweep):

    score_i = log(p_behavior_i) - lambda * (cheap_loss_i / 100.0)
    q_i = softmax(score_i) over the top-K candidate set

`cheap_loss_i` is a depth-8 Stockfish centipawn loss (mover POV) relative
to the best of the K candidates at that same cheap depth. Dividing by 100
means lambda is "nats of log-odds cost per 100cp (one pawn) of cheap-
search quality deficit relative to the best candidate in the set."

This mechanism is used IDENTICALLY for the generic policy, Method 1's
personalized policy, and Method 2's already-style-reranked candidate
distribution -- only `p_behavior` differs per caller; the reweighting math
is one function.

IMPORTANT, stated explicitly rather than glossed over: restricting to the
top-K candidates BEFORE reweighting is itself a (usually small) behavior
change relative to the old full-legal-vocabulary sampler, independent of
lambda. `lambda=0.0` reproduces `p_behavior` renormalized over the
retained top-K set, NOT the full original per-move probability vector --
moves ranked outside the top-K receive exactly zero probability at every
lambda, including zero. See the frozen report for measured top-K coverage
(top-5 cumulative probability is typically ~0.95).

Stockfish search (candidate cheap-loss evaluation) is inherently
per-position -- chess search does not batch across different boards.
`strength_reweight` (the pure reweighting math) DOES support batched
input (`[..., K]` tensors), so many already-evaluated positions' final
distributions can be computed in one vectorized call; batched and
unbatched calls are numerically identical by construction (plain
broadcasting torch ops, no shape-dependent branching).
"""

from dataclasses import dataclass

import chess
import chess.engine
import torch

from amadeus_counterpoint.encoding import policy_index_to_move

DEFAULT_K = 5
DEFAULT_CHEAP_DEPTH = 8
MATE_SCORE = 100000


@dataclass(frozen=True)
class StrengthGuardrailConfig:
    """Frozen sampler configuration. Every field here is a scientific
    protocol decision (see FINAL_SAMPLER_REPORT.md) -- do not change a
    field's value without re-running the freeze pass; construction alone
    never validates against sealed data."""

    lam: float
    k: int = DEFAULT_K
    cheap_depth: int = DEFAULT_CHEAP_DEPTH
    threads: int = 1
    hash_mb: int = 128


def strength_reweight(p_behavior: torch.Tensor, cheap_losses: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute q_i = softmax(log(p_behavior_i) - lam * cheap_losses_i / 100)
    over the last dimension. `p_behavior` and `cheap_losses` may be `[K]`
    (one candidate set) or `[..., K]` (a batch of candidate sets, e.g.
    `[B, K]`) -- the same formula, broadcast identically either way, so a
    batched call and a loop of unbatched calls always agree exactly
    (float64 recommended for that exact-agreement test; float32 agrees up
    to ordinary floating-point tolerance).

    `lam=0.0` returns `p_behavior` renormalized (not necessarily already
    normalized on input) -- it does NOT restore probability mass to any
    candidate outside the passed-in set. Candidate-set construction (which
    moves are "in") happens before this function is called.
    """
    p_behavior = torch.as_tensor(p_behavior)
    cheap_losses = torch.as_tensor(cheap_losses, dtype=p_behavior.dtype)
    log_p = torch.log(p_behavior.clamp_min(1e-12))
    scores = log_p - lam * (cheap_losses / 100.0)
    q = torch.softmax(scores, dim=-1)
    # A sufficiently large (lambda * loss) can underflow softmax to an exact
    # floating-point 0.0 for the worst candidate even though the formula
    # never assigns exact zero mathematically -- "bad moves keep nonzero
    # probability" is a scientific requirement (elite humans can blunder),
    # so floor and renormalize rather than let floating point silently
    # violate it. The floor is far below any realistic candidate's true
    # probability at the lambda values this project actually uses.
    q = q.clamp_min(1e-12)
    return q / q.sum(dim=-1, keepdim=True)


def _raw_white_pov_eval(engine: "chess.engine.SimpleEngine", board: chess.Board, depth: int) -> int:
    """Depth-`depth` Stockfish evaluation of `board`, always from White's
    POV (mate handled explicitly since a terminal position cannot be
    handed to the engine)."""
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    if board.is_game_over():
        return 0
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].white().score(mate_score=MATE_SCORE)


def cheap_eval_mover_pov(engine: "chess.engine.SimpleEngine", board: chess.Board, depth: int) -> int:
    """Depth-`depth` Stockfish evaluation of `board`, from the perspective
    of whichever side is to move IN THIS board right now. Only meaningful
    for evaluating a position before a candidate move is chosen from it --
    NOT for scoring a position reached AFTER a candidate move, where the
    side to move has flipped to the opponent (see `candidate_cheap_losses`,
    which fixes the mover's sign before any candidate is pushed, instead of
    re-deriving it from the post-move board)."""
    return (1 if board.turn == chess.WHITE else -1) * _raw_white_pov_eval(engine, board, depth)


def top_k_candidates(probs_full: torch.Tensor, board: chess.Board, k: int):
    """Top-`k` (or fewer, if fewer legal moves exist) candidates from a
    full legal-masked policy distribution over the board's legal moves.
    Returns (policy_indices: list[int], moves: list[chess.Move],
    p_behavior: list[float])."""
    n_legal = board.legal_moves.count()
    k_eff = min(k, n_legal)
    topk = torch.topk(probs_full, k=k_eff)
    indices = topk.indices.tolist()
    moves = [policy_index_to_move(idx, board) for idx in indices]
    p_behavior = topk.values.tolist()
    return indices, moves, p_behavior


def candidate_cheap_losses(engine: "chess.engine.SimpleEngine", board: chess.Board,
                            moves: list[chess.Move], depth: int) -> list[float]:
    """Depth-`depth` cheap-search loss of each of `moves` (must all be
    legal in `board`), relative to the best of THIS candidate set at that
    same depth -- not necessarily the true best legal move on the board.
    Mutates `board` internally via push/pop but restores it exactly.

    The mover's sign is fixed from `board.turn` BEFORE any candidate is
    pushed and reused for every candidate: after a move is pushed, the
    side to move flips to the opponent, so re-deriving the sign from the
    post-move board (as `cheap_eval_mover_pov` does for a pre-move
    position) would silently score every candidate from the OPPONENT's
    perspective instead of the actual mover's -- caught by
    test_candidate_cheap_losses_relative_to_best_in_set_and_restores_board.
    """
    mover_sign = 1 if board.turn == chess.WHITE else -1
    evals = []
    for move in moves:
        board.push(move)
        evals.append(mover_sign * _raw_white_pov_eval(engine, board, depth))
        board.pop()
    best = max(evals)
    return [max(0.0, best - e) for e in evals]


def sample_guarded_move(
    probs_full: torch.Tensor,
    board: chess.Board,
    engine: "chess.engine.SimpleEngine",
    generator: torch.Generator,
    config: StrengthGuardrailConfig,
) -> chess.Move:
    """One end-to-end guarded selection: top-K -> depth-`config.cheap_depth`
    Stockfish loss -> soft reweight -> seeded stochastic sample. Always
    samples (never argmax) and always leaves every retained candidate with
    nonzero probability for finite `config.lam`."""
    _, moves, p_behavior = top_k_candidates(probs_full, board, config.k)
    cheap_losses = candidate_cheap_losses(engine, board, moves, config.cheap_depth)
    q = strength_reweight(torch.tensor(p_behavior), torch.tensor(cheap_losses), config.lam)
    local_idx = int(torch.multinomial(q, num_samples=1, generator=generator).item())
    return moves[local_idx]
