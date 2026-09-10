import torch

from _helpers import CONFIG
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.training.method1_trainer import Method1Trainer


def _tiny_batch(n: int = 4) -> dict:
    return {
        "x": torch.zeros(n, 64, 96),
        "player_id": torch.zeros(n, dtype=torch.long),
        "player_elo": torch.full((n,), 1600.0),
        "opponent_elo": torch.full((n,), 1800.0),
        "policy_target": torch.zeros(n, dtype=torch.long),
        "legal_mask": torch.ones(n, 4352, dtype=torch.bool),
    }


def test_train_epoch_updates_z_player_and_leaves_base_untouched():
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="p0")
    original_base_state = {k: v.clone() for k, v in base.state_dict().items()}
    original_z_player = wrapper.z_player.detach().clone()

    trainer = Method1Trainer(wrapper, lr=0.1)
    train_loss = trainer.train_epoch([_tiny_batch(), _tiny_batch()])

    assert isinstance(train_loss, float)
    assert not torch.equal(wrapper.z_player, original_z_player)
    for key, value in base.state_dict().items():
        assert torch.equal(value, original_base_state[key])
    assert trainer.current_epoch == 1


def test_only_z_player_is_in_the_optimizer():
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="p0")
    trainer = Method1Trainer(wrapper, lr=0.1)

    optimized_params = trainer.optimizer.param_groups[0]["params"]
    assert len(optimized_params) == 1
    assert optimized_params[0] is wrapper.z_player


def test_validate_tracks_best_val_loss_and_early_stopping_counter():
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="p0")
    trainer = Method1Trainer(wrapper, lr=0.1)

    losses = iter([1.0, 2.0, 0.5])
    trainer._policy_loss = lambda batch: torch.tensor(next(losses))
    val_loader = [_tiny_batch()]

    first = trainer.validate(val_loader)
    assert first == 1.0
    assert trainer.best_val_loss == 1.0
    assert trainer.epochs_without_improvement == 0

    second = trainer.validate(val_loader)
    assert second == 2.0
    assert trainer.best_val_loss == 1.0  # unchanged -- second was worse
    assert trainer.epochs_without_improvement == 1

    third = trainer.validate(val_loader)
    assert third == 0.5
    assert trainer.best_val_loss == 0.5  # improved
    assert trainer.epochs_without_improvement == 0


# --- device handling -------------------------------------------------------


def test_batch_forward_succeeds_when_base_is_on_a_non_cpu_device():
    """Regression test for a real bug found during Kelvin2 GPU smoke testing:
    train_epoch/validate never moved DataLoader batches (always plain CPU
    tensors) onto the model's device, so every real CUDA run crashed the
    moment a batch was used. `meta` is a real, distinct device requiring no
    GPU: it reproduces the same "expected all tensors on the same device"
    failure as CPU-vs-CUDA did, so this fails on the old code and passes on
    the fix.
    """
    base = Chessformer(**CONFIG).to("meta")
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="p0")
    trainer = Method1Trainer(wrapper, lr=0.1)

    batch = _tiny_batch()  # plain CPU tensors, exactly what a real DataLoader yields
    assert batch["x"].device.type == "cpu"

    moved = trainer._to_device(batch)
    loss = trainer._policy_loss(moved)

    assert loss.device.type == "meta"


def test_train_epoch_and_validate_move_batches_before_use():
    """Wiring check: train_epoch/validate must actually call the device
    transfer, not just have it available -- guards against someone removing
    the call while leaving `_to_device` itself correct."""
    base = Chessformer(**CONFIG)
    wrapper = PersonalizedChessformer(base, nominal_elo=1600.0, identity="p0")
    trainer = Method1Trainer(wrapper, lr=0.1)

    calls = []
    original = trainer._to_device
    trainer._to_device = lambda batch: (calls.append(1), original(batch))[1]

    trainer.train_epoch([_tiny_batch(), _tiny_batch()])
    trainer.validate([_tiny_batch()])

    assert len(calls) == 3
