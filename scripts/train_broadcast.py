"""Launch Broadcast domain-adaptation training for the finished population
Chessformer.

This is ordinary continued Chessformer training on a new domain (target-free
elite Broadcast games), not a new architecture and not personalization. It
reuses population training's model config, optimizer family, loss, AMP
behavior, checkpoint format, and DataLoader/shard mechanics verbatim (see
scripts/train.py); the only real differences are:

  - Weights-only initialization from the finished population checkpoint
    (--init-checkpoint) on a fresh run: model state_dict is loaded, but the
    optimizer/scheduler/global_step start fresh at step 0. This is
    deliberate -- resuming the population optimizer/scheduler as-is would
    restore them at global_step~1,000,000 with an already-decayed LR, which
    is population-training *continuation*, not domain adaptation. A true
    resume of an in-progress *adaptation* run (a checkpoint already present
    in --checkpoint-dir, written by this same script) still restores
    everything, exactly like scripts/train.py, because at that point the
    optimizer/scheduler state genuinely belongs to this adaptation run.
  - A lower peak/min LR (see ADAPTATION_* constants below), justified in the
    corpus/launch report, not here.
  - A smaller default --num-steps (this is a several-pass adaptation corpus,
    not a from-scratch 1,000,000-step run) and a wider default
    --checkpoint-every, since best-checkpoint selection for this stage is
    done via a post-hoc validation-loss sweep over saved checkpoints
    (scripts/eval_broadcast_val_loss.py) rather than a live in-loop
    validation hook -- trainer.train() has no such hook, and adding one was
    judged not worth modifying the exact module that produced the proven
    population checkpoint.

Target exclusion for the Broadcast shards this script consumes was NOT done
via configs/target_aliases.json (that is the Lichess-username scheme for the
population corpus). It was done upstream, in scripts/build_broadcast_shards.py,
against the amadeus-broadcast-data sibling workspace's
manifests/target_free_elite_adaptation.parquet -- an independently audited,
mechanically-tested (10/10 invariant tests) zero-target game set. See that
script's docstring and the launch report for the full audit trail.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import chess
import pyarrow
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
# Model configuration: Chessformer 79M -- IDENTICAL to population training.
# Must match exactly for the population checkpoint's state_dict to load.
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

# Effective batch preserved from population training (see scripts/train.py).
BATCH_SIZE = 512
ACCUMULATION_STEPS = 1

# Adaptation LR: population's own MIN LR (1e-5) as the new PEAK LR. This is
# not an arbitrary reduction -- it is a rate this exact model already trained
# stably at, repeatedly, at the floor of every one of its ~20 cosine cycles
# over the full 1,000,000-step run. It is 5x below population's peak (5e-5),
# consistent with common fine-tuning practice (peak LR reduced ~5-10x from
# pretraining). The peak:floor ratio (5:1) is kept identical to population's
# schedule, just uniformly scaled down, so the schedule SHAPE is unchanged.
LEARNING_RATE = 5e-5          # population's peak, for reference/comparison only
MIN_LEARNING_RATE = 1e-5      # population's floor, for reference/comparison only
ADAPTATION_LEARNING_RATE = 1e-5
ADAPTATION_MIN_LEARNING_RATE = 2e-6
WEIGHT_DECAY = 1e-6

WARMUP_STEPS = 1_000  # unchanged from population: 1000/200000 = 0.5% of budget
NUM_STEPS = 200_000   # ~3.9 corpus passes at 26.24M train positions -- see
                       # the launch report for the full steps-vs-passes table
CYCLE_STEPS = 50_000  # unchanged: 200,000 / 50,000 = exactly 4 complete
                       # cosine cycles, a clean fit against this budget

MAX_GRAD_NORM = 3.5
VALUE_COEFFICIENT = 0.1

USE_AMP = True
CHECKPOINT_EVERY = 10_000  # 20 checkpoints over the run, for the post-hoc
                            # validation-loss best-checkpoint sweep
RUN_NAME = "chessformer-79m-broadcast"

SEED = 0

# ---------------------------------------------------------------------------
# Data loading: engineering knobs, unchanged from the population-training
# throughput study (same model, same batch config, same shard format).
# ---------------------------------------------------------------------------

NUM_WORKERS = 4
SHUFFLE_BUFFER_SIZE = 1024

DEFAULT_SHARD_DIR = Path("data/processed/broadcast_train")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints") / RUN_NAME


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent
        ).decode()
    except Exception:  # noqa: BLE001
        return True
    return bool(output.strip())


def _gpu_uuid(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"]
        ).decode().strip()
        return output or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _shard_manifest_hash(shard_dir: Path) -> str:
    names = sorted(p.name for p in Path(shard_dir).glob("*.parquet"))
    joined = "\n".join(names)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _broadcast_exclusion_meta(report_path: str) -> dict:
    """Provenance for the target-free Broadcast corpus this run consumes --
    NOT configs/target_aliases.json (that's the population corpus's
    Lichess-username scheme; irrelevant here). See
    scripts/build_broadcast_shards.py for how this report was produced."""
    try:
        raw = Path(report_path).read_text(encoding="utf-8")
    except OSError:
        return {"report_path": report_path, "available": False}
    report = json.loads(raw)
    return {
        "report_path": report_path,
        "available": True,
        "source_manifest": (
            "/mnt/scratch2/users/40482774/repos/amadeus-broadcast-data/"
            "manifests/target_free_elite_adaptation.parquet"
        ),
        "raw_target_free_games": report.get("raw_target_free_games"),
        "retained_train": report.get("retained_train"),
        "retained_val": report.get("retained_val"),
        "positions_train": report.get("positions_train"),
        "positions_val": report.get("positions_val"),
    }


def _write_run_metadata(
    checkpoint_dir: Path, device: torch.device, shard_dir: Path,
    num_steps: int, checkpoint_every: int, start_step: int, args: argparse.Namespace,
    init_checkpoint_used: str | None,
) -> None:
    shard_count = len(list(Path(shard_dir).glob("*.parquet")))
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "gpu_uuid": _gpu_uuid(device),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_chess_version": chess.__version__,
        "pyarrow_version": pyarrow.__version__,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "cli_argv": sys.argv,
        "cli_args": {k: str(v) for k, v in vars(args).items()},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_running_under_slurm"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "not_running_under_slurm"),
        "seed": SEED,
        "shard_dir": str(shard_dir),
        "shard_count": shard_count,
        "shard_manifest_hash": _shard_manifest_hash(shard_dir),
        "broadcast_target_exclusion": _broadcast_exclusion_meta(str(args.build_report)),
        "checkpoint_dir": str(checkpoint_dir),
        "start_step": start_step,
        "init_checkpoint_used": init_checkpoint_used,
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
            "population_learning_rate": LEARNING_RATE,
            "population_min_learning_rate": MIN_LEARNING_RATE,
            "adaptation_learning_rate": ADAPTATION_LEARNING_RATE,
            "adaptation_min_learning_rate": ADAPTATION_MIN_LEARNING_RATE,
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
        "--init-checkpoint", type=Path, default=None,
        help="Finished population checkpoint to load MODEL WEIGHTS ONLY from, "
        "on a fresh run (no existing checkpoint yet in --checkpoint-dir). "
        "Ignored if --checkpoint-dir already has a step_*.pt (that's a resume "
        "of THIS adaptation run instead).",
    )
    parser.add_argument(
        "--build-report", type=Path,
        default=Path("/mnt/scratch2/users/40482774/lichess_data/broadcast_shard_build_report.json"),
        help="build_broadcast_shards.py's report, recorded into run metadata "
        "for provenance.",
    )
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    shard_dir = args.shard_dir
    checkpoint_dir = args.checkpoint_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # Dataset and DataLoader
    # -----------------------------------------------------------------------

    dataset = ChessDataset(
        shard_dir=shard_dir,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    )

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
        lr=ADAPTATION_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------------------------
    # Learning-rate schedule: same shape as population (linear warmup ->
    # cosine-annealing-with-restarts), peak/min uniformly scaled down 5x.
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
        eta_min=ADAPTATION_MIN_LEARNING_RATE,
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
    # AMP scaler
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
    # Initialization: resume THIS adaptation run if it has its own
    # checkpoint already; otherwise, if given, load population's model
    # WEIGHTS ONLY (fresh optimizer/scheduler/scaler/global_step=0).
    # -----------------------------------------------------------------------

    start_step = 0
    init_checkpoint_used = None
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = _latest_checkpoint(checkpoint_dir)
    if latest is not None:
        start_step = load_checkpoint(latest, model, optimizer, scheduler, device, scaler=scaler)
        print(f"Resumed adaptation run from {latest} at global_step={start_step}")
    elif args.init_checkpoint is not None:
        population_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(population_ckpt["model"])
        init_checkpoint_used = str(args.init_checkpoint)
        print(
            f"Initialized model weights from population checkpoint "
            f"{args.init_checkpoint} (population global_step="
            f"{population_ckpt['global_step']}); optimizer/scheduler/scaler "
            f"start fresh at global_step=0"
        )
    else:
        print("No existing adaptation checkpoint and no --init-checkpoint given "
              "-- starting from a freshly initialized model at step 0")

    _write_run_metadata(
        checkpoint_dir, device, shard_dir, args.num_steps, args.checkpoint_every,
        start_step, args, init_checkpoint_used,
    )

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
