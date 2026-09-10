import chess
import torch

from _helpers import CONFIG, replay_and_check_legal
from amadeus_counterpoint.evaluation.generation import method1_personalized, single
from amadeus_counterpoint.evaluation.generation.method1_personalized import (
    play_game_method1,
    play_game_method1_ab,
)
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer


class _CapturingChessformer(Chessformer):
    """Records every forward() call's arguments, so generation-time
    conditioning can be inspected precisely rather than inferred."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def forward(self, x, player_elo, opponent_elo, player_emb_override=None):
        self.calls.append(
            {
                "player_elo": player_elo.clone(),
                "opponent_elo": opponent_elo.clone(),
                "override": None if player_emb_override is None else player_emb_override.clone(),
            }
        )
        return super().forward(x, player_elo, opponent_elo, player_emb_override=player_emb_override)


def build_wrapper(nominal_elo: float = 1600.0) -> PersonalizedChessformer:
    base = Chessformer(**CONFIG)
    return PersonalizedChessformer(base, nominal_elo=nominal_elo, identity="test-player")


def test_generation_produces_only_legal_moves():
    wrapper = build_wrapper()
    game = play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=1)

    assert len(game["moves"]) > 0
    replay_and_check_legal(game["moves"])


def test_seeded_generation_is_reproducible():
    wrapper = build_wrapper()

    game_a = play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=7)
    game_b = play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=7)

    assert game_a == game_b


def test_different_seeds_can_diverge():
    wrapper = build_wrapper()

    game_a = play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=1)
    game_b = play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=2)

    assert game_a["moves"] != game_b["moves"]


def test_no_topk_restriction_full_legal_policy_is_reachable():
    # With enough distinct seeds, more than one distinct first move must
    # appear -- proving sampling isn't collapsed onto some small restricted
    # candidate subset (there is no candidate selector in Method 1 at all).
    wrapper = build_wrapper()

    seen_first_moves = {
        play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=seed)["moves"][0]
        for seed in range(20)
    }

    assert len(seen_first_moves) > 1


def test_wrapper_override_used_only_on_its_own_plies_opponent_stays_ordinary():
    base = _CapturingChessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="test-player")

    play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=3)

    assert len(base.calls) >= 2  # at least two plies were played

    # ply 0: White == wrapper_color -> override present, ordinary opponent Elo.
    first_call = base.calls[0]
    assert first_call["override"] is not None
    assert torch.allclose(first_call["override"], wrapper.z_player.unsqueeze(0))
    assert first_call["opponent_elo"].item() == 1800.0

    # ply 1: Black == the opponent -> no override, ordinary Elo on both sides.
    second_call = base.calls[1]
    assert second_call["override"] is None
    assert second_call["player_elo"].item() == 1800.0
    assert second_call["opponent_elo"].item() == 1600.0


def test_identity_z_player_reproduces_ordinary_generation_exactly():
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="test-player")
    # No training happened -- z_player is still exactly interpolate_elo(1600).

    personalized = play_game_method1(wrapper, chess.WHITE, opponent_elo=1800.0, seed=42)
    ordinary = single.play_game(base, white_elo=1600.0, black_elo=1800.0, seed=42)

    assert personalized == ordinary


# --- AB (both sides personalized) -------------------------------------------


def build_ab_wrappers(base=None, elo_a: float = 1600.0, elo_b: float = 1900.0):
    base = base if base is not None else Chessformer(**CONFIG)
    wrapper_a = PersonalizedChessformer(base, nominal_elo=elo_a, identity="A")
    wrapper_b = PersonalizedChessformer(base, nominal_elo=elo_b, identity="B")
    return wrapper_a, wrapper_b


def test_ab_generation_produces_only_legal_moves():
    wrapper_a, wrapper_b = build_ab_wrappers()

    game = play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=1)

    assert len(game["moves"]) > 0
    replay_and_check_legal(game["moves"])


def test_ab_hits_the_existing_ply_cap(monkeypatch):
    monkeypatch.setattr(method1_personalized, "MAX_PLIES", 0)
    wrapper_a, wrapper_b = build_ab_wrappers()

    game = play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=1)

    assert game["result"] is None
    assert game["censored"] is True
    assert game["moves"] == []


def test_ab_selects_personalization_by_whose_turn_it_is():
    base = _CapturingChessformer(**CONFIG)
    wrapper_a, wrapper_b = build_ab_wrappers(base=base, elo_a=1600.0, elo_b=1900.0)

    play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=3)

    assert len(base.calls) >= 2  # at least two plies were played

    # ply 0: White == A -> A's override, opponent Elo is B's nominal Elo.
    first_call = base.calls[0]
    assert torch.allclose(first_call["override"], wrapper_a.z_player.unsqueeze(0))
    assert first_call["opponent_elo"].item() == 1900.0

    # ply 1: Black == B -> B's override, opponent Elo is A's nominal Elo.
    second_call = base.calls[1]
    assert torch.allclose(second_call["override"], wrapper_b.z_player.unsqueeze(0))
    assert second_call["opponent_elo"].item() == 1600.0


def test_ab_color_assignment_swaps_without_changing_player_identity():
    wrapper_a, wrapper_b = build_ab_wrappers(elo_a=1600.0, elo_b=1900.0)

    a_white = play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=5)
    b_white = play_game_method1_ab(wrapper_a, wrapper_b, chess.BLACK, seed=5)

    assert a_white["white_elo"] == 1600.0 and a_white["black_elo"] == 1900.0
    assert b_white["white_elo"] == 1900.0 and b_white["black_elo"] == 1600.0


def test_ab_seeded_generation_is_reproducible():
    wrapper_a, wrapper_b = build_ab_wrappers()

    game_a = play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=7)
    game_b = play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=7)

    assert game_a == game_b


def test_ab_identity_z_players_reproduce_ordinary_generation_exactly():
    base = Chessformer(**CONFIG)
    # No training happened -- both z_players are still exactly
    # interpolate_elo(their own nominal_elo), so the AB path must reduce
    # exactly to ordinary two-sided generation with those same Elos.
    wrapper_a, wrapper_b = build_ab_wrappers(base=base, elo_a=1600.0, elo_b=1800.0)

    personalized = play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=42)
    ordinary = single.play_game(base, white_elo=1600.0, black_elo=1800.0, seed=42)

    assert personalized == ordinary
