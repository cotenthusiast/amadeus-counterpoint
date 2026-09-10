import pytest
import torch

from _helpers import CONFIG
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.checkpoints import (
    load_method1_checkpoint,
    load_method2_checkpoint,
    save_method1_checkpoint,
    save_method2_checkpoint,
    verify_method1_checkpoint_identity,
)
from amadeus_counterpoint.training.style_trainer import StyleTrainer

STYLE_DIM = 16


def _assert_optimizer_states_equal(optimizer_a, optimizer_b):
    """optimizer.state_dict() contains tensors, so a plain == comparison
    raises on multi-element tensors; compare each field explicitly instead."""
    state_a = optimizer_a.state_dict()["state"]
    state_b = optimizer_b.state_dict()["state"]
    assert state_a.keys() == state_b.keys()

    for param_key in state_a:
        fields_a = state_a[param_key]
        fields_b = state_b[param_key]
        assert fields_a.keys() == fields_b.keys()

        for field_name in fields_a:
            value_a = fields_a[field_name]
            value_b = fields_b[field_name]
            if torch.is_tensor(value_a):
                assert torch.equal(value_a, value_b)
            else:
                assert value_a == value_b


# --- Method 1 -----------------------------------------------------------


def test_method1_checkpoint_round_trip(tmp_path):
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")
    optimizer = torch.optim.AdamW([wrapper.z_player], lr=0.01)

    # Take one training step so z_player and the optimizer both hold real state.
    wrapper.z_player.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    path = tmp_path / "method1.pt"
    save_method1_checkpoint(
        path, wrapper, optimizer=optimizer, epoch=3, step=150, best_val_loss=1.23,
        epochs_without_improvement=2,
    )

    fresh_base = Chessformer(**CONFIG)
    fresh_wrapper = PersonalizedChessformer(fresh_base, nominal_elo=1600.0, identity="alpha")
    fresh_optimizer = torch.optim.AdamW([fresh_wrapper.z_player], lr=0.01)

    metadata = load_method1_checkpoint(path, fresh_wrapper, optimizer=fresh_optimizer)

    assert torch.equal(fresh_wrapper.z_player, wrapper.z_player)
    _assert_optimizer_states_equal(fresh_optimizer, optimizer)
    assert metadata["identity"] == "alpha"
    assert metadata["nominal_elo"] == 1600.0
    assert metadata["epoch"] == 3
    assert metadata["step"] == 150
    assert metadata["best_val_loss"] == 1.23
    assert metadata["epochs_without_improvement"] == 2


def test_method1_checkpoint_without_epochs_without_improvement_defaults_to_zero(tmp_path):
    # A legacy checkpoint saved before this field existed must resume as if
    # early-stopping patience had not yet been used, matching a fresh
    # trainer's own starting value -- not silently omitted.
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")

    path = tmp_path / "method1.pt"
    save_method1_checkpoint(path, wrapper, epoch=1, best_val_loss=0.5)
    checkpoint = torch.load(path)
    del checkpoint["epochs_without_improvement"]
    torch.save(checkpoint, path)

    fresh_base = Chessformer(**CONFIG)
    fresh_wrapper = PersonalizedChessformer(fresh_base, nominal_elo=1600.0, identity="alpha")
    metadata = load_method1_checkpoint(path, fresh_wrapper)

    assert metadata["epochs_without_improvement"] == 0


def test_method1_checkpoint_without_optimizer_or_metadata(tmp_path):
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")

    path = tmp_path / "method1.pt"
    save_method1_checkpoint(path, wrapper)

    fresh_base = Chessformer(**CONFIG)
    fresh_wrapper = PersonalizedChessformer(fresh_base, nominal_elo=1600.0, identity="alpha")

    metadata = load_method1_checkpoint(path, fresh_wrapper)

    assert torch.equal(fresh_wrapper.z_player, wrapper.z_player)
    assert metadata["epoch"] is None
    assert metadata["step"] is None
    assert metadata["best_val_loss"] is None


def test_method1_checkpoint_contains_only_personalization_state(tmp_path):
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="alpha")

    path = tmp_path / "method1.pt"
    save_method1_checkpoint(path, wrapper)

    checkpoint = torch.load(path)

    assert list(checkpoint.keys()) == [
        "z_player", "identity", "nominal_elo", "optimizer", "epoch", "step", "best_val_loss",
        "epochs_without_improvement",
    ]
    assert isinstance(checkpoint["z_player"], torch.Tensor)
    assert checkpoint["z_player"].shape == (CONFIG["elo_dim"],)


# --- verify_method1_checkpoint_identity ----------------------------------


def test_verify_method1_checkpoint_identity_accepts_matching_metadata():
    metadata = {"identity": "alpha", "nominal_elo": 1600.0}
    verify_method1_checkpoint_identity(metadata, "alpha", 1600.0)  # must not raise


def test_verify_method1_checkpoint_identity_rejects_identity_mismatch():
    metadata = {"identity": "alpha", "nominal_elo": 1600.0}
    with pytest.raises(ValueError, match="identity"):
        verify_method1_checkpoint_identity(metadata, "wrong-name", 1600.0)


def test_verify_method1_checkpoint_identity_rejects_nominal_elo_mismatch():
    metadata = {"identity": "alpha", "nominal_elo": 1600.0}
    with pytest.raises(ValueError, match="nominal_elo"):
        verify_method1_checkpoint_identity(metadata, "alpha", 1700.0)


# --- Method 2 -----------------------------------------------------------


def _build_style_modules(num_players=3, style_dim=STYLE_DIM):
    cnn = MoveStyleCNN(style_dim=style_dim)
    table = PlayerStyleTable(num_players=num_players, style_dim=style_dim)
    residual = StyleResidual(style_dim=style_dim, init_s=0.17)
    return cnn, table, residual


def test_method2_checkpoint_round_trip(tmp_path):
    cnn, table, residual = _build_style_modules()
    optimizer = torch.optim.AdamW(
        list(cnn.parameters()) + list(table.parameters()) + list(residual.parameters()),
        lr=0.01,
    )

    # Take one training step so every module and the optimizer hold real state.
    planes = torch.randn(4, 18, 8, 8)
    move_features = cnn(planes).unsqueeze(1)
    z_u = table(torch.tensor([0, 1, 2, 0]))
    scores = residual(move_features, z_u)
    scores.sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    path = tmp_path / "method2.pt"
    save_method2_checkpoint(
        path, cnn, table, residual, optimizer=optimizer,
        epoch=2, step=75, best_val_loss=0.9, epochs_without_improvement=1,
        k=5, style_dim=STYLE_DIM, player_id_map={"alice": 0, "bob": 1, "carol": 2},
    )

    fresh_cnn, fresh_table, fresh_residual = _build_style_modules()
    fresh_optimizer = torch.optim.AdamW(
        list(fresh_cnn.parameters()) + list(fresh_table.parameters()) + list(fresh_residual.parameters()),
        lr=0.01,
    )

    metadata = load_method2_checkpoint(
        path, fresh_cnn, fresh_table, fresh_residual, optimizer=fresh_optimizer,
    )

    for name, value in cnn.state_dict().items():
        assert torch.equal(value, fresh_cnn.state_dict()[name])
    assert torch.equal(table.embeddings.weight, fresh_table.embeddings.weight)
    assert torch.equal(residual.s, fresh_residual.s)
    _assert_optimizer_states_equal(fresh_optimizer, optimizer)

    assert metadata["epoch"] == 2
    assert metadata["step"] == 75
    assert metadata["best_val_loss"] == 0.9
    assert metadata["epochs_without_improvement"] == 1
    assert metadata["k"] == 5
    assert metadata["style_dim"] == STYLE_DIM
    assert metadata["player_id_map"] == {"alice": 0, "bob": 1, "carol": 2}


def test_method2_checkpoint_without_optimizer_or_metadata(tmp_path):
    cnn, table, residual = _build_style_modules(num_players=2)

    path = tmp_path / "method2.pt"
    save_method2_checkpoint(path, cnn, table, residual)

    fresh_cnn, fresh_table, fresh_residual = _build_style_modules(num_players=2)

    metadata = load_method2_checkpoint(path, fresh_cnn, fresh_table, fresh_residual)

    assert torch.equal(residual.s, fresh_residual.s)
    assert metadata["k"] is None
    assert metadata["player_id_map"] is None


def test_method2_checkpoint_contains_only_style_module_state(tmp_path):
    cnn, table, residual = _build_style_modules()

    path = tmp_path / "method2.pt"
    save_method2_checkpoint(path, cnn, table, residual)

    checkpoint = torch.load(path)

    assert list(checkpoint.keys()) == [
        "cnn", "table", "residual", "optimizer",
        "epoch", "step", "best_val_loss", "epochs_without_improvement",
        "k", "style_dim", "player_id_map",
    ]
    assert set(checkpoint["cnn"].keys()) == set(cnn.state_dict().keys())
    assert set(checkpoint["table"].keys()) == set(table.state_dict().keys())
    assert set(checkpoint["residual"].keys()) == set(residual.state_dict().keys())


def test_method2_checkpoint_saves_style_trainer_progress_state(tmp_path):
    base = Chessformer(**CONFIG)
    cnn, table, residual = _build_style_modules(num_players=2)
    trainer = StyleTrainer(base, cnn, table, residual, k=5)

    trainer.current_epoch = 4
    trainer.best_val_loss = 0.5
    trainer.epochs_without_improvement = 2

    path = tmp_path / "method2.pt"
    save_method2_checkpoint(
        path, trainer.cnn, trainer.table, trainer.residual, optimizer=trainer.optimizer,
        epoch=trainer.current_epoch,
        best_val_loss=trainer.best_val_loss,
        epochs_without_improvement=trainer.epochs_without_improvement,
        k=trainer.k,
    )

    checkpoint = torch.load(path)
    assert checkpoint["epoch"] == 4
    assert checkpoint["best_val_loss"] == 0.5
    assert checkpoint["epochs_without_improvement"] == 2
    assert checkpoint["k"] == 5
