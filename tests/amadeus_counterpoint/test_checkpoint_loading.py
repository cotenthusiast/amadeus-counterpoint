import chess
import pytest
import torch

from _helpers import CONFIG, replay_and_check_legal
from amadeus_counterpoint.evaluation.generation.checkpoint_loading import (
    load_method1_wrapper_for_generation,
    load_method2_style_stack_for_generation,
)
from amadeus_counterpoint.evaluation.generation.method1_personalized import play_game_method1_ab
from amadeus_counterpoint.evaluation.generation.method2_personalized import play_game_method2_ab
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.checkpoints import save_method1_checkpoint, save_method2_checkpoint

STYLE_DIM = 16


# --- Method 1 -----------------------------------------------------------


def test_load_method1_wrapper_matches_saved_z_player(tmp_path):
    base = Chessformer(**CONFIG)
    trained = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")
    with torch.no_grad():
        trained.z_player.add_(1.0)  # simulate a trained (non-initial) z_player

    path = tmp_path / "method1.pt"
    save_method1_checkpoint(path, trained)

    wrapper = load_method1_wrapper_for_generation(
        base, path, nominal_elo=1600.0, identity="alpha"
    )

    assert torch.equal(wrapper.z_player, trained.z_player)
    assert wrapper.base is base  # same object, never reloaded/reconstructed
    assert wrapper.training is False  # eval mode


def test_loaded_method1_wrapper_works_in_ab_generation(tmp_path):
    base = Chessformer(**CONFIG)
    trained_a = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")
    trained_b = PersonalizedChessformer(base, nominal_elo=1900.0, identity="beta")

    path_a = tmp_path / "a.pt"
    path_b = tmp_path / "b.pt"
    save_method1_checkpoint(path_a, trained_a)
    save_method1_checkpoint(path_b, trained_b)

    wrapper_a = load_method1_wrapper_for_generation(base, path_a, nominal_elo=1600.0, identity="alpha")
    wrapper_b = load_method1_wrapper_for_generation(base, path_b, nominal_elo=1900.0, identity="beta")

    game = play_game_method1_ab(wrapper_a, wrapper_b, chess.WHITE, seed=1)

    assert len(game["moves"]) > 0
    replay_and_check_legal(game["moves"])


def test_load_method1_wrapper_rejects_identity_mismatch(tmp_path):
    base = Chessformer(**CONFIG)
    trained = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")
    path = tmp_path / "method1.pt"
    save_method1_checkpoint(path, trained)

    with pytest.raises(ValueError, match="identity"):
        load_method1_wrapper_for_generation(base, path, nominal_elo=1600.0, identity="wrong-name")


def test_load_method1_wrapper_rejects_nominal_elo_mismatch(tmp_path):
    base = Chessformer(**CONFIG)
    trained = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")
    path = tmp_path / "method1.pt"
    save_method1_checkpoint(path, trained)

    with pytest.raises(ValueError, match="nominal_elo"):
        load_method1_wrapper_for_generation(base, path, nominal_elo=1700.0, identity="alpha")


def test_load_method1_wrapper_does_not_reconstruct_the_base(tmp_path):
    base = Chessformer(**CONFIG)
    original_state = {k: v.clone() for k, v in base.state_dict().items()}
    trained = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")
    path = tmp_path / "method1.pt"
    save_method1_checkpoint(path, trained)

    wrapper = load_method1_wrapper_for_generation(base, path, nominal_elo=1600.0, identity="alpha")

    assert wrapper.base is base
    for key, value in base.state_dict().items():
        assert torch.equal(value, original_state[key])  # base weights untouched


# --- Method 2 -----------------------------------------------------------


def test_load_method2_style_stack_matches_saved_state(tmp_path):
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.42)

    path = tmp_path / "method2.pt"
    save_method2_checkpoint(path, cnn, table, residual, style_dim=STYLE_DIM)

    base = Chessformer(**CONFIG)
    loaded_cnn, loaded_table, loaded_residual, metadata = load_method2_style_stack_for_generation(
        base, path, num_players=2, style_dim=STYLE_DIM
    )

    for key, value in cnn.state_dict().items():
        assert torch.equal(loaded_cnn.state_dict()[key], value)
    for key, value in table.state_dict().items():
        assert torch.equal(loaded_table.state_dict()[key], value)
    assert torch.equal(loaded_residual.s, residual.s)
    assert loaded_cnn.training is False
    assert loaded_table.training is False
    assert loaded_residual.training is False


def test_loaded_method2_style_stack_works_in_ab_generation(tmp_path):
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM)
    path = tmp_path / "method2.pt"
    save_method2_checkpoint(path, cnn, table, residual, style_dim=STYLE_DIM)

    base = Chessformer(**CONFIG)
    loaded_cnn, loaded_table, loaded_residual, _ = load_method2_style_stack_for_generation(
        base, path, num_players=2, style_dim=STYLE_DIM
    )

    game = play_game_method2_ab(
        base, loaded_cnn, loaded_table, loaded_residual,
        player_id_a=0, player_id_b=1,
        a_color=chess.WHITE,
        elo_a=1500.0, elo_b=1600.0,
        k=5, seed=1,
    )

    assert len(game["moves"]) > 0
    replay_and_check_legal(game["moves"])


def test_load_method2_style_stack_restores_player_id_map(tmp_path):
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM)
    path = tmp_path / "method2.pt"
    player_id_map = {"alpha": 0, "beta": 1}
    save_method2_checkpoint(path, cnn, table, residual, player_id_map=player_id_map)

    base = Chessformer(**CONFIG)
    _, _, _, metadata = load_method2_style_stack_for_generation(
        base, path, num_players=2, style_dim=STYLE_DIM
    )

    assert metadata["player_id_map"] == player_id_map


def test_load_method2_style_stack_rejects_style_dim_mismatch(tmp_path):
    # A style_dim mismatch changes tensor shapes throughout the CNN/table, so
    # torch's own load_state_dict raises first with a shape-mismatch
    # RuntimeError; the explicit style_dim check below is a second line of
    # defense for the (data-integrity, not shape) case where the checkpoint's
    # own recorded style_dim disagrees with the caller's -- either way, this
    # must never silently succeed.
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM)
    path = tmp_path / "method2.pt"
    save_method2_checkpoint(path, cnn, table, residual, style_dim=STYLE_DIM)

    base = Chessformer(**CONFIG)
    with pytest.raises((ValueError, RuntimeError), match="style_dim|size mismatch"):
        load_method2_style_stack_for_generation(base, path, num_players=2, style_dim=STYLE_DIM + 1)


def test_load_method2_style_stack_rejects_mismatched_style_dim_metadata(tmp_path):
    # Here the tensors themselves match the caller's style_dim (so
    # load_state_dict succeeds), but the checkpoint's own recorded style_dim
    # metadata disagrees -- only the explicit check below catches this.
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM)
    path = tmp_path / "method2.pt"
    save_method2_checkpoint(path, cnn, table, residual, style_dim=STYLE_DIM + 1)  # wrong on purpose

    base = Chessformer(**CONFIG)
    with pytest.raises(ValueError, match="style_dim"):
        load_method2_style_stack_for_generation(base, path, num_players=2, style_dim=STYLE_DIM)


def test_load_method2_style_stack_does_not_reconstruct_the_base(tmp_path):
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=2, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM)
    path = tmp_path / "method2.pt"
    save_method2_checkpoint(path, cnn, table, residual, style_dim=STYLE_DIM)

    base = Chessformer(**CONFIG)
    original_state = {k: v.clone() for k, v in base.state_dict().items()}

    load_method2_style_stack_for_generation(base, path, num_players=2, style_dim=STYLE_DIM)

    for key, value in base.state_dict().items():
        assert torch.equal(value, original_state[key])
