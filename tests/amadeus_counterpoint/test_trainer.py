import pytest
import torch

from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.loss import chessformer_loss
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


def test_gradient_accumulation_split_is_equivalent_to_one_big_microbatch():
    """Splitting one fixed effective batch into more/fewer microbatches
    (with accumulation_steps adjusted to match) must produce the same
    parameter update either way.

    This is the numerical guarantee behind safely trading physical batch
    size for gradient-accumulation steps at a fixed effective batch: since
    chessformer_loss uses mean reduction and trainer.py divides by
    accumulation_steps before backward, the mean of N equal-size
    microbatch means equals the true effective-batch mean exactly (up to
    floating-point summation order), for any equal split of the same data.
    """
    torch.manual_seed(2)
    big_batch = _make_batch(batch_size=8)

    def run(accumulation_steps, microbatch_size):
        model = _build_model()
        optimizer, scheduler = _build_optimizer_and_scheduler(model)
        microbatches = [
            {k: v[i : i + microbatch_size] for k, v in big_batch.items()}
            for i in range(0, 8, microbatch_size)
        ]
        train(
            model, microbatches, optimizer, scheduler, DEVICE,
            num_steps=1, accumulation_steps=accumulation_steps, log_every=1000,
        )
        return list(model.parameters())

    params_single = run(accumulation_steps=1, microbatch_size=8)
    params_split_2 = run(accumulation_steps=2, microbatch_size=4)
    params_split_4 = run(accumulation_steps=4, microbatch_size=2)

    for p_single, p2, p4 in zip(params_single, params_split_2, params_split_4):
        assert torch.allclose(p_single, p2, atol=1e-5, rtol=1e-4)
        assert torch.allclose(p_single, p4, atol=1e-5, rtol=1e-4)


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


# --- multi-pass training: corpus exhaustion starts a fresh pass ------------
#
# Training is fixed-step (num_steps optimizer updates), not fixed-epoch.
# When a repeatedly-iterable dataloader (a real DataLoader over
# ChessDataset, or a plain list, as here) runs out mid-run, train() does
# not stop early -- it re-iterates the SAME dataloader for a fresh
# "corpus pass" and keeps going until num_steps is reached. For
# ChessDataset specifically, re-iterating reshuffles shard order and
# re-samples positions/history-masking from scratch (see
# amadeus_counterpoint.data.dataset.ChessDataset.__iter__), so a pass
# boundary is never a cached replay of the previous pass.


def test_dataloader_exhaustion_before_num_steps_continues_into_a_fresh_pass():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    # Only 1 optimizer update's worth of microbatches per pass; num_steps=5
    # requires 5 passes over this same 4-batch list.
    dataloader = _make_dataloader(num_batches=4)
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=5, accumulation_steps=4, log_every=1000,
    )

    assert global_step == 5


def test_global_step_and_scheduler_do_not_reset_across_pass_boundaries():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    dataloader = _make_dataloader(num_batches=4)  # 1 optimizer update/pass
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=5, accumulation_steps=4, log_every=1000,
    )

    # Monotonic 1, 2, 3, 4, 5 across 5 passes -- a reset at any pass
    # boundary would leave global_step, and the scheduler's own internal
    # step counter, short of 5.
    assert global_step == 5
    assert scheduler.last_epoch == 5


def test_optimizer_scheduler_scaler_state_persist_across_pass_boundary():
    """Crossing a pass boundary (re-iterating the dataloader) must be
    numerically indistinguishable from one continuous pass over the same
    batches concatenated -- proving optimizer/scheduler/scaler state is
    never reset or reinitialized at a pass boundary.
    """
    batches = _make_dataloader(num_batches=4, batch_size=2)

    model_a = _build_model()
    optimizer_a, scheduler_a = _build_optimizer_and_scheduler(model_a)
    scaler_a = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=64.0, growth_interval=1)
    step_a = train(
        model_a, batches, optimizer_a, scheduler_a, DEVICE,
        num_steps=2, accumulation_steps=4, log_every=1000,
        use_amp=True, scaler=scaler_a,
    )  # 2 passes over the same 4-batch list

    model_b = _build_model()
    optimizer_b, scheduler_b = _build_optimizer_and_scheduler(model_b)
    scaler_b = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=64.0, growth_interval=1)
    step_b = train(
        model_b, batches + batches, optimizer_b, scheduler_b, DEVICE,
        num_steps=2, accumulation_steps=4, log_every=1000,
        use_amp=True, scaler=scaler_b,
    )  # one continuous pass over the duplicated batches

    assert step_a == step_b == 2
    assert scaler_a.get_scale() == scaler_b.get_scale()
    assert scheduler_a.last_epoch == scheduler_b.last_epoch
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, atol=1e-6, rtol=1e-5)


def test_model_keeps_improving_across_many_pass_boundaries():
    """A single-microbatch 'corpus' forces one pass boundary per optimizer
    update; loss should still trend down, confirming learning is not
    disrupted at pass boundaries."""
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model, lr=5e-2)

    fixed_batch = _make_batch(batch_size=4)
    dataloader = [fixed_batch]  # 1 microbatch/pass at accumulation_steps=1

    def policy_loss_of():
        model.eval()
        with torch.no_grad():
            policy_logits, value_logits = model(
                fixed_batch["x"], fixed_batch["player_elo"], fixed_batch["opponent_elo"]
            )
            _, policy_loss, _ = chessformer_loss(
                policy_logits, value_logits,
                fixed_batch["policy_target"], fixed_batch["value_target"], fixed_batch["legal_mask"],
            )
        model.train()
        return policy_loss.item()

    initial_loss = policy_loss_of()
    # 50 optimizer updates from a 1-batch "corpus" => 50 pass boundaries.
    # The batch holds 4 distinct random positions/targets (not one
    # repeated example), so this converges gradually rather than
    # overfitting sharply -- the assertion only needs to show real,
    # sustained improvement across many pass boundaries, not convergence.
    train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=50, accumulation_steps=1, log_every=1000,
    )
    final_loss = policy_loss_of()

    assert final_loss < initial_loss * 0.9


def test_empty_dataloader_raises_clearly_instead_of_looping_forever():
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    with pytest.raises(RuntimeError, match="zero microbatches"):
        train(
            model, [], optimizer, scheduler, DEVICE,
            num_steps=1, accumulation_steps=1, log_every=1000,
        )


def test_pass_that_goes_empty_mid_run_raises_clearly_instead_of_looping_forever():
    """A one-shot (self-exhausting) iterable simulates a corpus with
    genuinely no more data for a second pass -- e.g. a bug that leaves a
    dataloader non-restartable. This must fail clearly, not hang.
    """
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    def one_shot_dataloader():
        yield from _make_dataloader(num_batches=4)  # exactly 1 optimizer update

    with pytest.raises(RuntimeError, match="zero microbatches"):
        train(
            model, one_shot_dataloader(), optimizer, scheduler, DEVICE,
            num_steps=2, accumulation_steps=4, log_every=1000,
        )


def test_smaller_final_microbatch_still_completes_an_optimizer_update():
    """train() has no concept of an intended physical batch size -- it
    treats whatever batch it receives as one microbatch. This is exactly
    why scripts/train.py's DataLoader must use drop_last=True at
    accumulation_steps=1: without it, an undersized trailing batch at a
    corpus-pass boundary would silently complete an optimizer update at a
    smaller-than-intended effective batch. This test pins that underlying
    mechanism so the drop_last requirement doesn't silently bit-rot.
    """
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    full_batch = _make_batch(batch_size=4)
    undersized_batch = _make_batch(batch_size=1)

    step_calls = []
    original_step = optimizer.step
    optimizer.step = lambda *a, **kw: step_calls.append(1) or original_step(*a, **kw)

    train(
        model, [full_batch, undersized_batch], optimizer, scheduler, DEVICE,
        num_steps=2, accumulation_steps=1, log_every=1000,
    )

    # Both the full-size and undersized batches completed their own
    # optimizer update -- train() does not protect against this itself.
    assert len(step_calls) == 2


def test_checkpoint_resume_works_across_a_multipass_run(tmp_path):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    dataloader = _make_dataloader(num_batches=4)  # 1 optimizer update/pass
    global_step = train(
        model, dataloader, optimizer, scheduler, DEVICE,
        num_steps=3, accumulation_steps=4, log_every=1000,
        checkpoint_dir=tmp_path, checkpoint_every=1,
    )
    assert global_step == 3
    assert (tmp_path / "step_00000003.pt").exists()

    resumed_model = _build_model()
    resumed_optimizer, resumed_scheduler = _build_optimizer_and_scheduler(resumed_model)
    restored_step = load_checkpoint(
        tmp_path / "step_00000003.pt", resumed_model, resumed_optimizer, resumed_scheduler, DEVICE,
    )
    assert restored_step == 3

    # Resume for 2 more updates -- itself spanning 2 more pass boundaries.
    final_step = train(
        resumed_model, dataloader, resumed_optimizer, resumed_scheduler, DEVICE,
        num_steps=5, accumulation_steps=4, log_every=1000, start_step=restored_step,
    )
    assert final_step == 5


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
    assert set(checkpoint) == {"model", "optimizer", "scheduler", "global_step", "scaler"}
    assert checkpoint["global_step"] == 1
    assert set(checkpoint["model"]) == set(model.state_dict())


def test_save_checkpoint_creates_missing_directories(tmp_path):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    nested = tmp_path / "a" / "b" / "step_00000001.pt"
    save_checkpoint(nested, model, optimizer, scheduler, global_step=1)

    assert nested.exists()


def test_checkpoint_round_trips_amp_scaler_state(tmp_path):
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=123.0)
    # Simulate a scaler that has grown/backed off partway through training,
    # rather than sitting at its freshly-constructed init_scale.
    scaler.load_state_dict(
        {"scale": 512.0, "growth_factor": 2.0, "backoff_factor": 0.5,
         "growth_interval": 2000, "_growth_tracker": 7}
    )

    save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, global_step=1, scaler=scaler)

    resumed_scaler = torch.amp.GradScaler(device="cpu", enabled=True, init_scale=999.0)
    load_checkpoint(
        tmp_path / "ckpt.pt", model, optimizer, scheduler, DEVICE, scaler=resumed_scaler,
    )

    assert resumed_scaler.state_dict() == scaler.state_dict()


def test_load_checkpoint_without_scaler_arg_ignores_missing_scaler_state(tmp_path):
    # save_checkpoint(scaler=None) (the default) must not fail load_checkpoint
    # for callers who don't care about AMP state.
    model = _build_model()
    optimizer, scheduler = _build_optimizer_and_scheduler(model)

    save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, global_step=1)

    restored_step = load_checkpoint(tmp_path / "ckpt.pt", model, optimizer, scheduler, DEVICE)

    assert restored_step == 1


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
