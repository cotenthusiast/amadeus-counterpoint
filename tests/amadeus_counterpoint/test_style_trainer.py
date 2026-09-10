import torch

from _helpers import CONFIG, FakeBase
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.style_trainer import StyleTrainer

NUM_PLAYERS = 3
STYLE_DIM = 16


def build_trainer() -> StyleTrainer:
    base = Chessformer(**CONFIG)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=NUM_PLAYERS, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.17)
    return StyleTrainer(base, cnn, table, residual, k=5, lr=1e-2, weight_decay=0.0)


def synthetic_batch(B: int, num_players: int = NUM_PLAYERS) -> dict:
    x = torch.randn(B, 64, CONFIG["input_dim"])
    player_elo = torch.randint(1000, 3000, (B,)).float()
    opponent_elo = torch.randint(1000, 3000, (B,)).float()
    player_id = torch.randint(0, num_players, (B,))

    legal_mask = torch.zeros(B, 4352, dtype=torch.bool)
    policy_target = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        legal_indices = torch.randperm(4352)[:8]
        legal_mask[b, legal_indices] = True
        policy_target[b] = legal_indices[0]

    return {
        "x": x,
        "player_id": player_id,
        "player_elo": player_elo,
        "opponent_elo": opponent_elo,
        "policy_target": policy_target,
        "legal_mask": legal_mask,
    }


# --- freezing ---------------------------------------------------------------


def test_base_is_frozen_on_construction():
    trainer = build_trainer()
    assert all(not p.requires_grad for p in trainer.base.parameters())


def test_optimizer_contains_exactly_the_style_parameters():
    trainer = build_trainer()

    optimizer_params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    base_params = {id(p) for p in trainer.base.parameters()}
    style_params = (
        {id(p) for p in trainer.cnn.parameters()}
        | {id(p) for p in trainer.table.parameters()}
        | {id(p) for p in trainer.residual.parameters()}
    )

    assert optimizer_params.isdisjoint(base_params)
    assert optimizer_params == style_params


def test_base_receives_no_gradient_and_is_unchanged_after_a_step():
    trainer = build_trainer()
    base_before = [p.detach().clone() for p in trainer.base.parameters()]

    trainer.train_epoch([synthetic_batch(4)])

    assert all(p.grad is None for p in trainer.base.parameters())
    for before, after in zip(base_before, trainer.base.parameters()):
        assert torch.equal(before, after)


def test_style_parameters_receive_gradients_and_change_after_a_step():
    trainer = build_trainer()
    cnn_before = [p.detach().clone() for p in trainer.cnn.parameters()]
    table_before = trainer.table.embeddings.weight.detach().clone()
    s_before = trainer.residual.s.detach().clone()

    trainer.train_epoch([synthetic_batch(4)])

    assert any(
        not torch.equal(before, after)
        for before, after in zip(cnn_before, trainer.cnn.parameters())
    )
    assert not torch.equal(table_before, trainer.table.embeddings.weight.detach())
    assert not torch.equal(s_before, trainer.residual.s.detach())


# --- train/validate mechanics ------------------------------------------------


def test_train_epoch_runs_end_to_end_and_returns_finite_mean_loss():
    trainer = build_trainer()
    batches = [synthetic_batch(4) for _ in range(3)]

    mean_loss = trainer.train_epoch(batches)

    assert torch.isfinite(torch.tensor(mean_loss))
    assert trainer.current_epoch == 1


def test_validate_runs_end_to_end_with_no_grad():
    trainer = build_trainer()
    batches = [synthetic_batch(4) for _ in range(2)]

    report = trainer.validate(batches)

    assert torch.isfinite(torch.tensor(report.mean_ce))
    assert 0.0 <= report.coverage <= 1.0
    assert report.num_examples == 8
    assert all(p.grad is None for p in trainer.cnn.parameters())
    assert trainer.residual.s.grad is None


def test_validate_reports_per_player_metrics():
    trainer = build_trainer()
    batch = synthetic_batch(6)

    report = trainer.validate([batch])

    observed_ids = set(batch["player_id"].tolist())
    assert set(report.per_player.keys()) == observed_ids
    for player_id in observed_ids:
        metrics = report.per_player[player_id]
        expected_count = int((batch["player_id"] == player_id).sum().item())
        assert metrics.num_examples == expected_count
        assert 0.0 <= metrics.coverage <= 1.0
        assert metrics.player_id == player_id


def test_best_val_loss_and_early_stopping_counter_update():
    trainer = build_trainer()
    batch = synthetic_batch(4)

    report_1 = trainer.validate([batch])
    assert trainer.best_val_loss == report_1.mean_ce
    assert trainer.epochs_without_improvement == 0

    # Same style parameters, same batch, no training step in between -> the
    # identical loss is not an improvement, so the counter must increment.
    report_2 = trainer.validate([batch])
    assert report_2.mean_ce == report_1.mean_ce
    assert trainer.best_val_loss == report_1.mean_ce
    assert trainer.epochs_without_improvement == 1


def test_no_checkpoint_serialization_exists_in_stage_3():
    trainer = build_trainer()
    assert not hasattr(trainer, "save_checkpoint")
    assert not hasattr(trainer, "load_checkpoint")


# --- difficult candidate case, through the trainer --------------------------


def test_validate_target_outside_topk_gives_finite_loss_and_false_coverage():
    policy_logits = torch.zeros(1, 4352)
    legal = [10, 20, 30, 40, 50]
    for idx, val in zip(legal, [9.0, 7.0, 5.0, 3.0, 1.0]):
        policy_logits[0, idx] = val
    legal_mask = torch.zeros(1, 4352, dtype=torch.bool)
    legal_mask[0, legal] = True

    base = FakeBase(policy_logits)
    cnn = MoveStyleCNN(style_dim=STYLE_DIM)
    table = PlayerStyleTable(num_players=1, style_dim=STYLE_DIM)
    residual = StyleResidual(style_dim=STYLE_DIM, init_s=0.17)
    trainer = StyleTrainer(base, cnn, table, residual, k=3)

    batch = {
        "x": torch.randn(1, 64, CONFIG["input_dim"]),
        "player_id": torch.tensor([0]),
        "player_elo": torch.tensor([1500.0]),
        "opponent_elo": torch.tensor([1500.0]),
        "policy_target": torch.tensor([50]),  # rank 5th, outside top-3
        "legal_mask": legal_mask,
    }

    report = trainer.validate([batch])

    assert torch.isfinite(torch.tensor(report.mean_ce))
    assert report.coverage == 0.0
