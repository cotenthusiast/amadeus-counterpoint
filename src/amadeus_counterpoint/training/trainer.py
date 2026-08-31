"""Training loop for Chessformer."""

from pathlib import Path

import torch

from amadeus_counterpoint.training.loss import chessformer_loss

# GradScaler configuration verified from the Chessformer paper's Table 4
# training recipe (use_amp: true).
AMP_INIT_SCALE = 256
AMP_GROWTH_FACTOR = 1.5
AMP_GROWTH_INTERVAL = 2000
AMP_BACKOFF_FACTOR = 0.5


def save_checkpoint(path, model, optimizer, scheduler, global_step: int) -> None:
    """Save model/optimizer/scheduler state and the optimizer step count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device) -> int:
    """Load model/optimizer/scheduler state and return the saved global step."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["global_step"]


def train(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    num_steps: int,
    accumulation_steps: int = 4,
    max_grad_norm: float = 3.5,
    value_coefficient: float = 0.1,
    log_every: int = 100,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 1000,
    start_step: int = 0,
    use_amp: bool = False,
) -> int:
    """Train Chessformer for a fixed number of optimizer updates.

    Checkpoints (model/optimizer/scheduler/global_step) are written every
    `checkpoint_every` OPTIMIZER steps to `checkpoint_dir`, if given. Returns
    the final global step, so training can be resumed by passing that value
    back in as `start_step` together with `load_checkpoint`.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # AMP is only meaningful on CUDA; the scaler and autocast below are
    # no-ops everywhere else, so the training loop needs no separate CPU path.
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=use_amp,
        init_scale=AMP_INIT_SCALE,
        growth_factor=AMP_GROWTH_FACTOR,
        backoff_factor=AMP_BACKOFF_FACTOR,
        growth_interval=AMP_GROWTH_INTERVAL,
    )

    global_step = start_step
    accumulation_counter = 0

    running_total_loss = 0.0
    running_policy_loss = 0.0
    running_value_loss = 0.0
    running_batches = 0

    for batch in dataloader:
        # Move the batch from CPU memory to the training device.
        x = batch["x"].to(device)
        player_elo = batch["player_elo"].to(device)
        opponent_elo = batch["opponent_elo"].to(device)
        policy_target = batch["policy_target"].to(device)
        value_target = batch["value_target"].to(device)
        legal_mask = batch["legal_mask"].to(device)

        # Forward pass and loss.
        with torch.autocast(device_type=device.type, enabled=use_amp):
            policy_logits, value_logits = model(
                x,
                player_elo,
                opponent_elo,
            )

            total_loss, policy_loss, value_loss = chessformer_loss(
                policy_logits,
                value_logits,
                policy_target,
                value_target,
                legal_mask,
                value_coefficient,
            )

        # Track the real, unscaled losses for logging.
        running_total_loss += total_loss.item()
        running_policy_loss += policy_loss.item()
        running_value_loss += value_loss.item()
        running_batches += 1

        # Divide because gradients from multiple microbatches are accumulated.
        loss = total_loss / accumulation_steps
        scaler.scale(loss).backward()

        accumulation_counter += 1

        if accumulation_counter == accumulation_steps:
            # Gradients must be unscaled before clipping, or the clip
            # threshold would be applied to AMP-scaled gradients.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            accumulation_counter = 0

            if global_step % log_every == 0:
                print(
                    f"step {global_step}/{num_steps} | "
                    f"loss {running_total_loss / running_batches:.4f} | "
                    f"policy {running_policy_loss / running_batches:.4f} | "
                    f"value {running_value_loss / running_batches:.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e}"
                )

                running_total_loss = 0.0
                running_policy_loss = 0.0
                running_value_loss = 0.0
                running_batches = 0

            if checkpoint_dir is not None and global_step % checkpoint_every == 0:
                save_checkpoint(
                    Path(checkpoint_dir) / f"step_{global_step:08d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                )

            if global_step == num_steps:
                break

    return global_step
