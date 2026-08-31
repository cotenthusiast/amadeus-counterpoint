"""Launch Chessformer training."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.dataset import ChessDataset
from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.trainer import train

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

BATCH_SIZE = 128
ACCUMULATION_STEPS = 4

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

# ---------------------------------------------------------------------------
# Data loading: engineering knobs, not model/training hyperparameters.
# Tune these for the machine running training; they don't affect fidelity.
# ---------------------------------------------------------------------------

NUM_WORKERS = 8
SHUFFLE_BUFFER_SIZE = 1024

SHARD_DIR = Path("data/processed/train")
CHECKPOINT_DIR = Path("checkpoints") / RUN_NAME


def main():
    # Use the GPU when one is available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Dataset and DataLoader
    # -----------------------------------------------------------------------

    dataset = ChessDataset(
        shard_dir=SHARD_DIR,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
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
    # Train
    # -----------------------------------------------------------------------

    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_steps=NUM_STEPS,
        accumulation_steps=ACCUMULATION_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        value_coefficient=VALUE_COEFFICIENT,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_every=CHECKPOINT_EVERY,
        use_amp=USE_AMP,
    )


if __name__ == "__main__":
    main()