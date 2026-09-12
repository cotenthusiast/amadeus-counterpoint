"""Tests for the shared strength-aware soft-reweighting selection layer
(evaluation.generation.strength_guardrail), frozen by the rollout-quality
exploratory investigation. No model or real Stockfish process is needed:
`strength_reweight` is pure tensor math, and a small FakeEngine stands in
for `chess.engine.SimpleEngine` (same `.analyse(board, Limit(depth=..))
-> {"score": PovScore}` interface) for the board/orientation-facing
helpers.
"""

import chess
import chess.engine
import pytest
import torch

from amadeus_counterpoint.evaluation.generation.strength_guardrail import (
    StrengthGuardrailConfig,
    candidate_cheap_losses,
    cheap_eval_mover_pov,
    sample_guarded_move,
    strength_reweight,
    top_k_candidates,
)


class FakeEngine:
    """Maps a board's FEN to a fixed centipawn (White-POV) score, mimicking
    chess.engine.SimpleEngine.analyse's return shape exactly."""

    def __init__(self, scores_by_fen):
        self.scores_by_fen = scores_by_fen

    def analyse(self, board, limit):
        key = board.board_fen() + (" w" if board.turn == chess.WHITE else " b")
        cp = self.scores_by_fen[key]
        return {"score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE)}


# ---------------------------------------------------------------------------
# strength_reweight: pure math properties
# ---------------------------------------------------------------------------

def test_lambda_zero_reproduces_renormalized_p_behavior():
    p = torch.tensor([0.5, 0.3, 0.2])
    losses = torch.tensor([0.0, 40.0, 500.0])
    q = strength_reweight(p, losses, lam=0.0)
    assert torch.allclose(q, p / p.sum(), atol=1e-10)


def test_q_normalizes_to_one():
    p = torch.tensor([0.6, 0.25, 0.1, 0.03, 0.02])
    losses = torch.tensor([0.0, 15.0, 80.0, 220.0, 900.0])
    for lam in (0.0, 1.0, 5.0, 20.0):
        q = strength_reweight(p, losses, lam=lam)
        assert torch.isclose(q.sum(), torch.tensor(1.0), atol=1e-9)


def test_all_candidates_remain_nonzero_for_finite_lambda():
    p = torch.tensor([0.7, 0.29, 0.01])
    losses = torch.tensor([0.0, 5.0, 5000.0])  # one catastrophic candidate
    q = strength_reweight(p, losses, lam=20.0)
    assert torch.all(q > 0.0)


def test_stronger_candidate_gets_intended_multiplicative_advantage():
    # two candidates with EQUAL p_behavior but different cheap loss: the
    # weaker one's odds must fall by exactly exp(-lam * delta_loss / 100)
    p = torch.tensor([0.5, 0.5])
    losses = torch.tensor([0.0, 100.0])  # exactly one pawn worse
    lam = 3.0
    q = strength_reweight(p, losses, lam=lam)
    ratio = (q[0] / q[1]).item()
    expected_ratio = torch.exp(torch.tensor(lam * 100.0 / 100.0)).item()  # e^{lam}
    assert ratio == pytest.approx(expected_ratio, rel=1e-5)


def test_batched_and_unbatched_agree_exactly():
    p = torch.tensor([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]], dtype=torch.float64)
    losses = torch.tensor([[0.0, 20.0, 150.0], [0.0, 5.0, 60.0]], dtype=torch.float64)
    lam = 4.0
    batched = strength_reweight(p, losses, lam=lam)
    unbatched = torch.stack([strength_reweight(p[i], losses[i], lam=lam) for i in range(p.shape[0])])
    assert torch.allclose(batched, unbatched, atol=1e-12)


def test_generic_and_personalized_share_the_same_mechanism():
    # The function has no notion of "which agent" p_behavior came from --
    # calling it with a Method-1-shaped or Method-2-shaped p_behavior/loss
    # pair uses literally the same code path as a generic one.
    generic_p = torch.tensor([0.55, 0.3, 0.15])
    method1_p = torch.tensor([0.4, 0.35, 0.25])
    losses = torch.tensor([0.0, 30.0, 300.0])
    q_generic = strength_reweight(generic_p, losses, lam=5.0)
    q_method1 = strength_reweight(method1_p, losses, lam=5.0)
    # different inputs -> generally different outputs, but both still valid distributions
    assert torch.isclose(q_generic.sum(), torch.tensor(1.0), atol=1e-9)
    assert torch.isclose(q_method1.sum(), torch.tensor(1.0), atol=1e-9)


# ---------------------------------------------------------------------------
# board/orientation-facing helpers
# ---------------------------------------------------------------------------

def test_cheap_eval_mover_pov_orientation_white_and_black():
    board_white_to_move = chess.Board()  # White to move, "White is +37cp"
    engine = FakeEngine({board_white_to_move.board_fen() + " w": 37})
    assert cheap_eval_mover_pov(engine, board_white_to_move, depth=8) == 37

    board_black_to_move = chess.Board()
    board_black_to_move.push_san("e4")  # now Black to move; same White-POV +37cp
    engine2 = FakeEngine({board_black_to_move.board_fen() + " b": 37})
    # mover (Black) POV must be the NEGATION of the White-POV score
    assert cheap_eval_mover_pov(engine2, board_black_to_move, depth=8) == -37


def test_candidate_cheap_losses_relative_to_best_in_set_and_restores_board():
    board = chess.Board()
    e4 = chess.Move.from_uci("e2e4")
    d4 = chess.Move.from_uci("d2d4")
    board_after_e4 = board.copy()
    board_after_e4.push(e4)
    board_after_d4 = board.copy()
    board_after_d4.push(d4)

    engine = FakeEngine({
        board_after_e4.board_fen() + " b": 50,   # White-POV eval after 1.e4
        board_after_d4.board_fen() + " b": 10,   # White-POV eval after 1.d4 (worse for White)
    })
    fen_before = board.fen()
    losses = candidate_cheap_losses(engine, board, [e4, d4], depth=8)
    assert board.fen() == fen_before  # push/pop must be perfectly balanced
    assert losses[0] == pytest.approx(0.0)   # e4 is the best of the two -> zero loss
    assert losses[1] == pytest.approx(40.0)  # d4 is 40cp worse


def test_top_k_candidates_caps_at_number_of_legal_moves():
    board = chess.Board(fen="8/8/8/8/8/8/8/K1k5 w - - 0 1")  # only 3 legal moves for White king
    n_legal = board.legal_moves.count()
    assert n_legal < 5
    probs = torch.zeros(4352)
    from amadeus_counterpoint.encoding import move_to_policy_index
    for i, move in enumerate(board.legal_moves):
        probs[move_to_policy_index(move, board)] = float(i + 1)
    probs = probs / probs.sum()
    indices, moves, p_behavior = top_k_candidates(probs, board, k=5)
    assert len(moves) == n_legal  # capped, not padded


# ---------------------------------------------------------------------------
# end-to-end sampling: determinism and nonzero-support
# ---------------------------------------------------------------------------

def _uniform_probs_over_legal(board):
    from amadeus_counterpoint.encoding import move_to_policy_index
    probs = torch.zeros(4352)
    legal = list(board.legal_moves)
    for move in legal:
        probs[move_to_policy_index(move, board)] = 1.0 / len(legal)
    return probs


def test_sample_guarded_move_is_deterministic_under_fixed_seed():
    board = chess.Board()
    probs = _uniform_probs_over_legal(board)
    scores = {chess.Board().board_fen() + " w": 0}
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        scores[b2.board_fen() + " b"] = hash(move.uci()) % 200 - 100  # arbitrary but fixed per move
    engine = FakeEngine(scores)
    config = StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8)

    picks = []
    for _ in range(3):
        generator = torch.Generator().manual_seed(12345)
        move = sample_guarded_move(probs, board.copy(), engine, generator, config)
        picks.append(move.uci())
    assert len(set(picks)) == 1  # same seed -> same pick, every time


def test_sample_guarded_move_only_ever_returns_a_retained_candidate():
    board = chess.Board()
    probs = _uniform_probs_over_legal(board)
    scores = {}
    for move in board.legal_moves:
        b2 = board.copy()
        b2.push(move)
        scores[b2.board_fen() + " b"] = hash(move.uci()) % 200 - 100
    engine = FakeEngine(scores)
    config = StrengthGuardrailConfig(lam=5.0, k=5, cheap_depth=8)

    _, expected_moves, _ = top_k_candidates(probs, board, config.k)
    expected_ucis = {m.uci() for m in expected_moves}

    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        move = sample_guarded_move(probs, board.copy(), engine, generator, config)
        assert move.uci() in expected_ucis
