import torch

from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.trainer import (
    load_checkpoint,
    save_checkpoint,
    train,
)

TINY_CONFIG = {
    "d_model": 32,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.0,
    "d1": 4,
    "d2": 8,
    "d3": 4,
    "d_ff": 64,
    "head_hid_dim": 16,
    "input_dim": 96,
    "elo_dim": 8,
}

DEVICE = torch.device("cpu")


def _build_model():
    torch.manual_seed(0)
    return Chessformer(**TINY_CONFIG)


def _build_optimizer_and_scheduler(model, lr=1e-2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    return optimizer, scheduler


def _make_batch(batch_size=2):
    return {
        "x": torch.randn(batch_size, 64, TINY_CONFIG["input_dim"]),
        "player_elo": torch.randint(0, 5000, (batch_size,)),
        "opponent_elo": torch.randint(0, 5000, (batch_size,)),
        "policy_target": torch.randint(0, 4352, (batch_size,)),
        "value_target": torch.randint(0, 3, (batch_size,)),
        # All-legal mask: trainer-level tests exercise loop mechanics, not
        # masking correctness (that is loss.py's responsibility).
        "legal_mask": torch.ones(batch_size, 4352, dtype=torch.bool),
    }


def _make_dataloader(num_batches, batch_size=2):
    torch.manual_seed(1)
    return [_make_batch(batch_size) for _ in range(num_batches)]


# --- basic training loop -----------------------------------------------


def test_train_updates_model_parameters():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)
    before = [p.clone() for p in model.parameters()]

    dataloader = _make_dataloader(num_batches=8)
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=2, accumulation_steps=4, log_every=1000,
    )

    assert global_step == 2
    after = list(model.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_accumulation_steps_causes_one_update_per_n_microbatches():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    step_calls = []
    original_step = optimizer.step
    optimizer.step = lambda *a, **kw: step_calls.append(1) or original_step(*a, **kw)

    scheduler_calls = []
    original_sched_step = scheduler.step
    scheduler.step = lambda *a, **kw: scheduler_calls.append(1) or original_sched_step(*a, **kw)

    dataloader = _make_dataloader(num_batches=12)
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=3, accumulation_steps=4, log_every=1000,
    )

    assert global_step == 3
    assert len(step_calls) == 3
    assert len(scheduler_calls) == 3


def test_clip_grad_norm_executes_once_per_optimizer_update(monkeypatch):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    clip_calls = []
    original_clip = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *a, **kw: clip_calls.append(1) or original_clip(*a, **kw),
    )

    dataloader = _make_dataloader(num_batches=8)
    train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=2, accumulation_steps=4, log_every=1000,
    )

    assert len(clip_calls) == 2


def test_requested_global_step_count_stops_training_early():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    # 20 microbatches would allow 5 optimizer updates at accumulation_steps=4,
    # but num_steps caps training at 2 updates.
    consumed = []
    dataloader = _make_dataloader(num_batches=20)

    def counting_dataloader():
        for batch in dataloader:
            consumed.append(1)
            yield batch

    global_step = train(
        model, counting_dataloader(), optimizer, scheduler, DEVICE,
        num_steps=2, accumulation_steps=4, log_every=1000,
    )

    assert global_step == 2
    assert len(consumed) == 8  # exactly 2 * accumulation_steps microbatches


def test_dataloader_exhaustion_before_num_steps_does_not_crash():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    # Only enough microbatches for 1 optimizer update, but num_steps asks for 5.
    dataloader = _make_dataloader(num_batches=4)
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=5, accumulation_steps=4, log_every=1000,
    )

    assert global_step == 1


def test_use_amp_on_cpu_is_a_safe_no_op():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    dataloader = _make_dataloader(num_batches=8)
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=2, accumulation_steps=4, log_every=1000, use_amp=True,
    )

    assert global_step == 2


# --- checkpointing -----------------------------------------------------


def test_checkpoint_created_at_expected_global_steps(tmp_path):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    dataloader = _make_dataloader(num_batches=16)
    train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=4, accumulation_steps=4, log_every=1000,
        checkpoint_dir=tmp_path, checkpoint_every=2,
    )

    assert (tmp_path / "step_00000002.pt").exists()
    assert (tmp_path / "step_00000004.pt").exists()
    assert not (tmp_path / "step_00000001.pt").exists()
    assert not (tmp_path / "step_00000003.pt").exists()


def test_checkpoint_contains_model_optimizer_scheduler_and_step(tmp_path):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    dataloader = _make_dataloader(num_batches=4)
    train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=1, accumulation_steps=4, log_every=1000,
        checkpoint_dir=tmp_path, checkpoint_every=1,
    )

    checkpoint = torch.load(tmp_path / "step_00000001.pt", map_location=DEVICE)
    assert set(checkpoint) == {"model", "optimizer", "scheduler", "global_step"}
    assert checkpoint["global_step"] == 1
    assert set(checkpoint["model"]) == set(model.state_dict())


def test_save_checkpoint_creates_missing_directories(tmp_path):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    nested = tmp_path / "a" / "b" / "step_00000001.pt"
    save_checkpoint(nested, model, optimizer, scheduler, global_step=1)

    assert nested.exists()


def test_resume_from_checkpoint_restores_state_exactly(tmp_path):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    dataloader = _make_dataloader(num_batches=8)
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=2, accumulation_steps=4, log_every=1000,
        checkpoint_dir=tmp_path, checkpoint_every=2,
    )
    assert global_step == 2

    # Fresh model/optimizer/scheduler with different random init.
    torch.manual_seed(999)
    resumed_model = Chessformer(**TINY_CONFIG)
    resumed_optimizer, resumed_scheduler = _build_optimizer_and_scheduler(resumed_model)

    restored_step = load_checkpoint(
        tmp_path / "step_00000002.pt",
        resumed_model, resumed_optimizer, resumed_scheduler, DEVICE,
    )

    assert restored_step == 2
    for trained_p, resumed_p in zip(model.parameters(), resumed_model.parameters()):
        assert torch.equal(trained_p, resumed_p)
    assert scheduler.get_last_lr() == resumed_scheduler.get_last_lr()

    # Training can continue seamlessly from the restored step.
    more_batches = _make_dataloader(num_batches=8)
    final_step = train(
        resumed_model, more_batches, resumed_optimizer, resumed_scheduler, DEVICE,
        num_steps=4, accumulation_steps=4, log_every=1000, start_step=restored_step,
    )
    assert final_step == 4
