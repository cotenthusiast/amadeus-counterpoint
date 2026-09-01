"""Launch Chessformer training."""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.dataset import ChessDataset
from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.trainer import (
    AMP_BACKOFF_FACTOR,
    AMP_GROWTH_FACTOR,
    AMP_GROWTH_INTERVAL,
    AMP_INIT_SCALE,
    load_checkpoint,
    train,
)

# ---------------------------------------------------------------------------
# Model configuration: Chessformer 79M
# ---------------------------------------------------------------------------

D_MODEL = 1024
NUM_HEADS = 32
NUM_LAYERS = 8

D1 = 32
D2 = 128
D3 = 128

D_FF = 2048
ELO_DIM = 128
HEAD_HID_DIM = 1024

RAW_INPUT_DIM = 96  # Chessformer's input_dim: it appends 2*elo_dim internally

DROPOUT = 0.0


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

# Physical batch / accumulation verified on Kelvin2 (A100 + H100, real
# production pipeline, real sharded Parquet data): 512x1 beats 128x4 and
# 256x2 on both GPUs once shards outnumber DataLoader workers. Effective
# batch is unchanged at 512 -- see the numerical-equivalence regression
# test in tests/amadeus_counterpoint/test_trainer.py.
BATCH_SIZE = 512
ACCUMULATION_STEPS = 1

LEARNING_RATE = 5e-5
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-6

WARMUP_STEPS = 1_000
NUM_STEPS = 1_000_000
CYCLE_STEPS = 50_000  # Table 4 "cosine_cycles": length of each cosine restart

MAX_GRAD_NORM = 3.5
VALUE_COEFFICIENT = 0.1

USE_AMP = True
CHECKPOINT_EVERY = 1_000
RUN_NAME = "chessformer-79m"

SEED = 0  # project convention; not paper-specified (AdamW betas/eps and RNG
          # of comparable sort are already-documented unpublished details).

# ---------------------------------------------------------------------------
# Data loading: engineering knobs, not model/training hyperparameters.
# num_workers=4 is the throughput-study winner on both A100 and H100 at the
# 512x1 batch config; tune further only if a future study finds otherwise.
# ---------------------------------------------------------------------------

NUM_WORKERS = 4
SHUFFLE_BUFFER_SIZE = 1024

DEFAULT_SHARD_DIR = Path("data/processed/train")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints") / RUN_NAME


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_run_metadata(
    checkpoint_dir: Path, device: torch.device, shard_dir: Path,
    num_steps: int, checkpoint_every: int,
) -> None:
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "git_commit": _git_commit(),
        "seed": SEED,
        "shard_dir": str(shard_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "config": {
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "d1": D1,
            "d2": D2,
            "d3": D3,
            "d_ff": D_FF,
            "elo_dim": ELO_DIM,
            "head_hid_dim": HEAD_HID_DIM,
            "input_dim": RAW_INPUT_DIM,
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "effective_batch": BATCH_SIZE * ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "num_steps": num_steps,
            "cycle_steps": CYCLE_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "value_coefficient": VALUE_COEFFICIENT,
            "use_amp": USE_AMP,
            "checkpoint_every": checkpoint_every,
            "num_workers": NUM_WORKERS,
            "shuffle_buffer_size": SHUFFLE_BUFFER_SIZE,
        },
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_path = checkpoint_dir / f"run_metadata_{int(time.time())}.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Run metadata written to {out_path}")
    print(json.dumps(metadata, indent=2))


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--num-steps", type=int, default=NUM_STEPS,
        help="Override for smoke/gate runs; production launches use the default (1,000,000).",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
        help="Override for smoke/gate runs; production launches use the default (1,000).",
    )
    parser.add_argument(
        "--log-every", type=int, default=100,
        help="Override for smoke/gate runs to see progress on short runs.",
    )
    args = parser.parse_args()

    shard_dir = args.shard_dir
    checkpoint_dir = args.checkpoint_dir

    # Use the GPU when one is available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")
    torch.manual_seed(SEED)

    _write_run_metadata(checkpoint_dir, device, shard_dir, args.num_steps, args.checkpoint_every)

    # -----------------------------------------------------------------------
    # Dataset and DataLoader
    # -----------------------------------------------------------------------

    dataset = ChessDataset(
        shard_dir=shard_dir,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    # drop_last=True: with accumulation_steps=1, every microbatch IS an
    # optimizer update, so an undersized trailing batch at the end of a
    # corpus pass would otherwise silently shrink that one update's
    # effective batch below 512. Dropping it keeps every optimizer update
    # at exactly BATCH_SIZE * ACCUMULATION_STEPS examples.
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = Chessformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        d1=D1,
        d2=D2,
        d3=D3,
        d_ff=D_FF,
        head_hid_dim=HEAD_HID_DIM,
        input_dim=RAW_INPUT_DIM,
        elo_dim=ELO_DIM,
    )

    model = model.to(device)

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------------------------
    # Learning-rate schedule
    #
    # Paper Table 4 confirms warmup_steps=1000, cosine_cycles=50000, and
    # Sec. 7 confirms a "cyclic cosine annealed" schedule over the full
    # 1M-step run -- but the exact restart mechanics (whether warmup repeats,
    # what "refresh" resets) are not spelled out anywhere in the paper. This
    # is our best-effort reconstruction: a one-time linear warmup, followed
    # by repeated 50,000-step cosine cycles between the peak and minimum LR.
    # -----------------------------------------------------------------------

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=WARMUP_STEPS,
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=CYCLE_STEPS,
        eta_min=MIN_LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            warmup_scheduler,
            cosine_scheduler,
        ],
        milestones=[WARMUP_STEPS],
    )

    # -----------------------------------------------------------------------
    # AMP scaler -- constructed here (rather than inside train()) so its
    # state can be restored from a checkpoint before training resumes.
    # -----------------------------------------------------------------------

    use_amp = USE_AMP and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=use_amp,
        init_scale=AMP_INIT_SCALE,
        growth_factor=AMP_GROWTH_FACTOR,
        backoff_factor=AMP_BACKOFF_FACTOR,
        growth_interval=AMP_GROWTH_INTERVAL,
    )

    # -----------------------------------------------------------------------
    # Resume from the latest checkpoint in checkpoint_dir, if one exists.
    # This is what lets the run survive unplanned restarts (node failure,
    # preemption, or exceeding a single SLURM walltime allocation).
    # -----------------------------------------------------------------------

    start_step = 0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_checkpoint(checkpoint_dir)
    if latest is not None:
        start_step = load_checkpoint(latest, model, optimizer, scheduler, device, scaler=scaler)
        print(f"Resumed from {latest} at global_step={start_step}")
    else:
        print("No existing checkpoint found -- starting from step 0")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------

    final_step = train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_steps=args.num_steps,
        accumulation_steps=ACCUMULATION_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        value_coefficient=VALUE_COEFFICIENT,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        start_step=start_step,
        use_amp=USE_AMP,
        scaler=scaler,
    )

    print(f"Training stopped at global_step={final_step}")


if __name__ == "__main__":
    main()
