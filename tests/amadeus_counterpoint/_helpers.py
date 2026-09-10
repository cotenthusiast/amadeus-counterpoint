"""Shared test-only helpers for the Stage 1-5 personalization tests.

Plain module, imported explicitly (`from _helpers import ...`) -- not a
conftest.py, so nothing here is pytest-fixture magic.
"""

import chess
import torch

# Tiny Chessformer config shared by every personalization test that needs a
# real (but small/fast) model.
CONFIG = dict(
    d_model=32,
    num_heads=4,
    num_layers=2,
    dropout=0.0,
    d1=4,
    d2=8,
    d3=4,
    d_ff=64,
    head_hid_dim=16,
    input_dim=96,
    elo_dim=8,
)


def replay_and_check_legal(moves):
    """Replay UCI moves from the start position, asserting each is legal."""
    board = chess.Board()
    for uci in moves:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        board.push(move)
    return board


class FakeBase:
    """Deterministic stand-in for Chessformer: always returns the same fixed
    policy logits, so tests can control candidate ranking exactly instead of
    depending on a real, randomly-initialized model's output.

    `eval()` and `requires_grad_()` are harmless no-ops, present because some
    callers (StyleTrainer, the generation functions) call them on `base`;
    tests that don't need them simply never call them.
    """

    def __init__(self, policy_logits: torch.Tensor):
        self.policy_logits = policy_logits

    def __call__(self, x, player_elo, opponent_elo):
        return self.policy_logits, torch.zeros(x.shape[0], 3)

    def eval(self):
        return self

    def requires_grad_(self, flag):
        return self
