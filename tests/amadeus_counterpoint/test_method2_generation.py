import chess
import torch

from _helpers import CONFIG, FakeBase, replay_and_check_legal
from amadeus_counterpoint.chess import create_board
from amadeus_counterpoint.encoding import move_to_policy_index
from amadeus_counterpoint.evaluation.generation import method2_personalized
from amadeus_counterpoint.evaluation.generation.method2_personalized import (
    play_game_method2,
    play_game_method2_ab,
)
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN

STYLE_DIM = 16


class _CapturingPlayerStyleTable(PlayerStyleTable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def forward(self, player_id):
        self.calls.append(player_id.clone())
        return super().forward(player_id)


def build_real_style_stack(num_players: int = 2, style_dim: int = STYLE_DIM):
    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=style_dim)
    table = PlayerStyleTable(num_players=num_players, style_dim=style_dim)
    residual = StyleResidual(style_dim=style_dim, init_s=0.17)
    return base, cnn, table, residual


def _one_ply_outcome():
    return chess.Outcome(chess.Termination.CHECKMATE, chess.WHITE)


def _stop_after_one_ply(monkeypatch):
    calls = {"n": 0}

    def check_end_after_one_ply(board):
        calls["n"] += 1
        return None if calls["n"] == 1 else _one_ply_outcome()

    monkeypatch.setattr(method2_personalized, "check_end", check_end_after_one_ply)


def _ranked_start_position_logits():
    """Every opening legal move gets a distinct, strictly increasing logit,
    so ranking (and therefore top-K candidate selection) is fully known."""
    board = create_board()
    legal_moves = list(board.legal_moves)
    policy_logits = torch.full((1, 4352), float("-inf"))
    for rank, move in enumerate(legal_moves):
        policy_logits[0, move_to_policy_index(move, board)] = float(rank)
    return policy_logits, legal_moves


# --- deterministic, single-ply behavior (fake base) -------------------------


def test_candidate_width_is_exactly_k_not_k_plus_1(monkeypatch):
    _stop_after_one_ply(monkeypatch)
    policy_logits, legal_moves = _ranked_start_position_logits()

    base = FakeBase(policy_logits)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=1, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.17)

    game = play_game_method2(
        base, cnn, table, residual,
        player_id=0, player_color=chess.WHITE,
        player_elo=1500.0, opponent_elo=1500.0,
        k=5, seed=1,
    )

    assert len(game["moves"]) == 1
    played = chess.Move.from_uci(game["moves"][0])
    # Ranks are strictly increasing with list position -> the top-5 are the
    # LAST 5 entries of legal_moves; only those 5 can ever be sampled.
    assert played in legal_moves[-5:]


def test_local_to_global_index_mapping_is_correct_with_k_equals_one(monkeypatch):
    # With exactly one candidate, softmax gives it probability 1 regardless
    # of the style residual, so the played move is fully deterministic:
    # exactly the single highest-ranked legal move by the frozen base logit.
    _stop_after_one_ply(monkeypatch)
    policy_logits, legal_moves = _ranked_start_position_logits()

    base = FakeBase(policy_logits)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=1, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.17)

    game = play_game_method2(
        base, cnn, table, residual,
        player_id=0, player_color=chess.WHITE,
        player_elo=1500.0, opponent_elo=1500.0,
        k=1, seed=123,
    )

    expected_move = legal_moves[-1]  # rank 19, the single highest logit
    assert game["moves"][0] == expected_move.uci()


def test_player_id_is_forwarded_to_player_style_table(monkeypatch):
    _stop_after_one_ply(monkeypatch)
    policy_logits, _ = _ranked_start_position_logits()

    base = FakeBase(policy_logits)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = _CapturingPlayerStyleTable(num_players=3, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.17)

    play_game_method2(
        base, cnn, table, residual,
        player_id=2, player_color=chess.WHITE,
        player_elo=1500.0, opponent_elo=1500.0,
        k=5, seed=1,
    )

    assert table.calls[0].item() == 2


# --- end-to-end, real model --------------------------------------------------


def test_generation_produces_only_legal_moves():
    base, cnn, table, residual = build_real_style_stack()

    game = play_game_method2(
        base, cnn, table, residual,
        player_id=0, player_color=chess.WHITE,
        player_elo=1500.0, opponent_elo=1600.0,
        k=5, seed=1,
    )

    assert len(game["moves"]) > 0
    replay_and_check_legal(game["moves"])


def test_seeded_generation_is_reproducible():
    base, cnn, table, residual = build_real_style_stack()

    game_a = play_game_method2(
        base, cnn, table, residual,
        player_id=0, player_color=chess.WHITE,
        player_elo=1500.0, opponent_elo=1600.0,
        k=5, seed=9,
    )
    game_b = play_game_method2(
        base, cnn, table, residual,
        player_id=0, player_color=chess.WHITE,
        player_elo=1500.0, opponent_elo=1600.0,
        k=5, seed=9,
    )

    assert game_a == game_b


def test_k_is_configurable():
    base, cnn, table, residual = build_real_style_stack()

    for k in (3, 10):
        game = play_game_method2(
            base, cnn, table, residual,
            player_id=0, player_color=chess.WHITE,
            player_elo=1500.0, opponent_elo=1600.0,
            k=k, seed=9,
        )
        assert len(game["moves"]) > 0
        replay_and_check_legal(game["moves"])


# --- AB (both sides personalized) -------------------------------------------


def test_ab_generation_produces_only_legal_moves():
    base, cnn, table, residual = build_real_style_stack(num_players=2)

    game = play_game_method2_ab(
        base, cnn, table, residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.WHITE,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=1,
    )

    assert len(game["moves"]) > 0
    replay_and_check_legal(game["moves"])


def test_ab_hits_the_existing_ply_cap(monkeypatch):
    monkeypatch.setattr(method2_personalized, "MAX_PLIES", 0)
    base, cnn, table, residual = build_real_style_stack(num_players=2)

    game = play_game_method2_ab(
        base, cnn, table, residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.WHITE,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=1,
    )

    assert game["result"] is None
    assert game["censored"] is True
    assert game["moves"] == []


def test_ab_selects_personalization_by_whose_turn_it_is():
    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = _CapturingPlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.17)

    play_game_method2_ab(
        base, cnn, table, residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.WHITE,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=1,
    )

    assert len(table.calls) >= 2  # at least two plies were played
    assert table.calls[0].item() == 0  # White's move == A == player_id 0
    assert table.calls[1].item() == 1  # Black's move == B == player_id 1


def test_ab_color_assignment_swaps_without_changing_player_identity():
    base, cnn, table, residual = build_real_style_stack(num_players=2)

    a_white = play_game_method2_ab(
        base, cnn, table, residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.WHITE,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=5,
    )
    b_white = play_game_method2_ab(
        base, cnn, table, residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.BLACK,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=5,
    )

    assert a_white["white_elo"] == 1500.0 and a_white["black_elo"] == 1600.0
    assert b_white["white_elo"] == 1600.0 and b_white["black_elo"] == 1500.0


def test_ab_seeded_generation_is_reproducible():
    base, cnn, table, residual = build_real_style_stack(num_players=2)

    game_a = play_game_method2_ab(
        base, cnn, table, residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.WHITE,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=9,
    )
    game_b = play_game_method2_ab(
        base, cnn, table, residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.WHITE,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=9,
    )

    assert game_a == game_b
