import chess

from _helpers import CONFIG
from amadeus_counterpoint.evaluation import artifacts
from amadeus_counterpoint.evaluation.generation import method1_personalized, method2_personalized, single
from amadeus_counterpoint.evaluation.generation.experiment import (
    generate_method1_cell,
    generate_method1_experiment,
    generate_method2_cell,
    generate_method2_experiment,
)
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN

STYLE_DIM = 16


def build_method1_fixture():
    base = Chessformer(**CONFIG)
    wrapper0 = PersonalizedChessformer(base, nominal_elo=1600.0, identity="p0")
    wrapper1 = PersonalizedChessformer(base, nominal_elo=1800.0, identity="p1")
    wrappers = {0: wrapper0, 1: wrapper1}
    representative_elos = {0: 1600.0, 1: 1800.0}
    return wrappers, base, representative_elos


def build_method2_fixture():
    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM)
    representative_elos = {0: 1600.0, 1: 1800.0}
    return base, cnn, table, residual, representative_elos


# --- Method 1 ----------------------------------------------------------


def test_method1_generates_all_conditions_orientations_and_respects_game_count():
    wrappers, base, elos = build_method1_fixture()

    games = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=2, root_seed=123, checkpoint_identity="ckpt-m1",
    )

    # 1 dyad * 2 orientations * 4 conditions * 2 games each.
    assert len(games) == 16
    assert {g["condition"] for g in games} == {"GG", "AG", "GB", "AB"}
    assert {g["orientation"] for g in games} == {"A_WHITE", "B_WHITE"}


def test_method1_dyad_identity_is_fixed_across_orientations():
    wrappers, base, elos = build_method1_fixture()

    games = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=1, root_seed=1, checkpoint_identity="ckpt-m1",
    )

    assert {g["dyad"] for g in games} == {"0__1"}


def test_method1_personalization_matches_condition_regardless_of_color():
    wrappers, base, elos = build_method1_fixture()

    games = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=1, root_seed=7, checkpoint_identity="ckpt-m1",
    )
    by_condition_orientation = {(g["condition"], g["orientation"]): g for g in games}

    for orientation in ("A_WHITE", "B_WHITE"):
        gg = by_condition_orientation[("GG", orientation)]
        assert gg["white_representation_identity"] == "global"
        assert gg["black_representation_identity"] == "global"

        ag = by_condition_orientation[("AG", orientation)]
        assert {ag["white_representation_identity"], ag["black_representation_identity"]} == {"p0", "global"}

        gb = by_condition_orientation[("GB", orientation)]
        assert {gb["white_representation_identity"], gb["black_representation_identity"]} == {"p1", "global"}

        ab = by_condition_orientation[("AB", orientation)]
        assert {ab["white_representation_identity"], ab["black_representation_identity"]} == {"p0", "p1"}


def test_method1_seeds_are_all_distinct():
    wrappers, base, elos = build_method1_fixture()

    games = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=2, root_seed=42, checkpoint_identity="ckpt-m1",
    )

    seeds = [g["seed"] for g in games]
    assert len(set(seeds)) == len(seeds)


def test_method1_same_inputs_reproduce_the_same_experiment():
    # Reuse the SAME wrappers/base for both calls -- a freshly-constructed
    # Chessformer has randomly-initialized weights, so two independently
    # built models would legitimately generate different games regardless
    # of the orchestrator's own seed determinism.
    wrappers, base, elos = build_method1_fixture()

    games_a = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=1, root_seed=99, checkpoint_identity="ckpt-m1",
    )
    games_b = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=1, root_seed=99, checkpoint_identity="ckpt-m1",
    )

    assert games_a == games_b


def test_method1_game_record_has_correct_provenance_fields():
    wrappers, base, elos = build_method1_fixture()

    games = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=1, root_seed=5, checkpoint_identity="ckpt-m1",
    )
    record = next(g for g in games if g["condition"] == "AB" and g["orientation"] == "A_WHITE")

    assert record["dyad"] == "0__1"
    assert record["game_index"] == 0
    assert record["checkpoint_identity"] == "ckpt-m1"
    assert isinstance(record["seed"], int)
    assert record["white_elo"] == 1600.0
    assert record["black_elo"] == 1800.0


def test_method1_ply_cap_is_recorded_as_censored_not_a_draw(monkeypatch):
    monkeypatch.setattr(single, "MAX_PLIES", 0)
    monkeypatch.setattr(method1_personalized, "MAX_PLIES", 0)
    wrappers, base, elos = build_method1_fixture()

    games = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=1, root_seed=1, checkpoint_identity="ckpt-m1",
    )

    assert len(games) == 8  # 1 dyad * 2 orientations * 4 conditions * 1 game
    for game in games:
        assert game["censored"] is True
        assert game["result"] is None
        assert game["moves"] == []


def test_method1_records_are_accepted_by_the_existing_artifact_writer(tmp_path):
    # write_games requires game_index to be unique WITHIN one call -- that's
    # one (dyad, condition, orientation) cell's worth of games, matching how
    # a production run would shard its writes (one shard per cell), not an
    # entire multi-cell experiment at once.
    wrappers, base, elos = build_method1_fixture()

    games = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=3, root_seed=1, checkpoint_identity="ckpt-m1",
    )
    one_cell = [g for g in games if g["condition"] == "AB" and g["orientation"] == "A_WHITE"]
    assert len(one_cell) == 3

    path = tmp_path / "method1_ab_a_white.parquet"
    metadata = {
        "protocol_version": "1", "code_commit": "abc123",
        "root_seed": "1", "temperature": "1.0",
    }
    artifacts.write_games(path, one_cell, metadata)

    read_back = artifacts.read_games(path)
    assert read_back.num_rows == len(one_cell)


# --- Method 2 ------------------------------------------------------------


def test_method2_generates_all_conditions_orientations_and_respects_game_count():
    base, cnn, table, residual, elos = build_method2_fixture()

    games = generate_method2_experiment(
        base, cnn, table, residual, [0, 1], elos, k=5,
        n_games_per_orientation=2, root_seed=123, checkpoint_identity="ckpt-m2",
    )

    assert len(games) == 16
    assert {g["condition"] for g in games} == {"GG", "AG", "GB", "AB"}
    assert {g["orientation"] for g in games} == {"A_WHITE", "B_WHITE"}


def test_method2_personalization_matches_condition_regardless_of_color():
    base, cnn, table, residual, elos = build_method2_fixture()

    games = generate_method2_experiment(
        base, cnn, table, residual, [0, 1], elos, k=5,
        n_games_per_orientation=1, root_seed=7, checkpoint_identity="ckpt-m2",
    )
    by_condition_orientation = {(g["condition"], g["orientation"]): g for g in games}

    for orientation in ("A_WHITE", "B_WHITE"):
        gg = by_condition_orientation[("GG", orientation)]
        assert gg["white_representation_identity"] == "global"
        assert gg["black_representation_identity"] == "global"

        ag = by_condition_orientation[("AG", orientation)]
        assert {ag["white_representation_identity"], ag["black_representation_identity"]} == {"0", "global"}

        gb = by_condition_orientation[("GB", orientation)]
        assert {gb["white_representation_identity"], gb["black_representation_identity"]} == {"1", "global"}

        ab = by_condition_orientation[("AB", orientation)]
        assert {ab["white_representation_identity"], ab["black_representation_identity"]} == {"0", "1"}


def test_method2_same_inputs_reproduce_the_same_experiment():
    # Reuse the SAME base/cnn/table/residual for both calls -- see the
    # matching comment on the Method-1 version of this test.
    base, cnn, table, residual, elos = build_method2_fixture()

    games_a = generate_method2_experiment(
        base, cnn, table, residual, [0, 1], elos, k=5,
        n_games_per_orientation=1, root_seed=99, checkpoint_identity="ckpt-m2",
    )
    games_b = generate_method2_experiment(
        base, cnn, table, residual, [0, 1], elos, k=5,
        n_games_per_orientation=1, root_seed=99, checkpoint_identity="ckpt-m2",
    )

    assert games_a == games_b


def test_method2_ply_cap_is_recorded_as_censored_not_a_draw(monkeypatch):
    monkeypatch.setattr(single, "MAX_PLIES", 0)
    monkeypatch.setattr(method2_personalized, "MAX_PLIES", 0)
    base, cnn, table, residual, elos = build_method2_fixture()

    games = generate_method2_experiment(
        base, cnn, table, residual, [0, 1], elos, k=5,
        n_games_per_orientation=1, root_seed=1, checkpoint_identity="ckpt-m2",
    )

    assert len(games) == 8
    for game in games:
        assert game["censored"] is True
        assert game["result"] is None
        assert game["moves"] == []


def test_method1_and_method2_seed_the_same_dyad_condition_orientation_differently():
    # Same dyad/condition/orientation/game_index, different method -> the
    # `method` tag folded into the seed's dyad string must still separate them.
    wrappers, base1, elos = build_method1_fixture()
    base2, cnn, table, residual, _ = build_method2_fixture()

    m1_games = generate_method1_experiment(
        wrappers, base1, [0, 1], elos,
        n_games_per_orientation=1, root_seed=1, checkpoint_identity="ckpt-m1",
    )
    m2_games = generate_method2_experiment(
        base2, cnn, table, residual, [0, 1], elos, k=5,
        n_games_per_orientation=1, root_seed=1, checkpoint_identity="ckpt-m2",
    )

    m1_ab_seed = next(g["seed"] for g in m1_games if g["condition"] == "AB" and g["orientation"] == "A_WHITE")
    m2_ab_seed = next(g["seed"] for g in m2_games if g["condition"] == "AB" and g["orientation"] == "A_WHITE")
    assert m1_ab_seed != m2_ab_seed


# --- standalone per-cell functions (used directly by production scripts) --


def test_generate_method1_cell_matches_the_corresponding_slice_of_the_full_experiment():
    wrappers, base, elos = build_method1_fixture()

    full = generate_method1_experiment(
        wrappers, base, [0, 1], elos,
        n_games_per_orientation=1, root_seed=11, checkpoint_identity="ckpt-m1",
    )
    one_cell = generate_method1_cell(
        wrappers[0], wrappers[1], base, chess.WHITE, "AB",
        elos[0], elos[1], "0__1", n_games=1, root_seed=11, checkpoint_identity="ckpt-m1",
    )

    expected = [g for g in full if g["condition"] == "AB" and g["orientation"] == "A_WHITE"]
    assert one_cell == expected


def test_generate_method2_cell_matches_the_corresponding_slice_of_the_full_experiment():
    base, cnn, table, residual, elos = build_method2_fixture()

    full = generate_method2_experiment(
        base, cnn, table, residual, [0, 1], elos, k=5,
        n_games_per_orientation=1, root_seed=11, checkpoint_identity="ckpt-m2",
    )
    one_cell = generate_method2_cell(
        base, cnn, table, residual, 0, 1, chess.WHITE, "AB",
        elos[0], elos[1], k=5, dyad="0__1", n_games=1, root_seed=11, checkpoint_identity="ckpt-m2",
    )

    expected = [g for g in full if g["condition"] == "AB" and g["orientation"] == "A_WHITE"]
    assert one_cell == expected
