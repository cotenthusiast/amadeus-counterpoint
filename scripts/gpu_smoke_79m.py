#!/usr/bin/env python3
"""GPU smoke test for the full 79M Chessformer configuration on real data.

Exercises forward/backward, AMP, gradient accumulation, the optimizer,
scheduler, legal masking, value loss, and checkpointing for a short run,
watching for NaN/inf loss, exploding gradients, AMP scaler instability,
and CUDA OOM. Also supports checkpoint/resume: run once to some step
count, then run again with --resume to verify model/optimizer/scheduler/
GradScaler/global_step are restored and training continues correctly.

Not a production training script -- the LR schedule here is a minimal
placeholder (linear warmup then constant) for mechanical smoke-testing
only; the paper's cyclic cosine restart schedule is a separate, already
documented, deliberately-unreproduced deviation (see repo docs).
"""

import argparse
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.dataset import ChessDataset
from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.trainer import load_checkpoint, train

FULL_79M_CONFIG = {
    "d_model": 1024,
    "num_heads": 32,
    "num_layers": 8,
    "dropout": 0.0,
    "d1": 32,
    "d2": 128,
    "d3": 128,
    "d_ff": 2048,
    "head_hid_dim": 1024,
    "input_dim": 96,
    "elo_dim": 128,
}

PEAK_LR = 5e-5
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-6
WARMUP_STEPS = 1000
MAX_GRAD_NORM = 3.5
VALUE_COEFFICIENT = 0.1


def warmup_then_constant(step: int) -> float:
    """Linear warmup to 1.0, then hold -- placeholder schedule for the smoke
    test only (see module docstring: the real cosine-restart schedule is a
    separate, documented reconstruction, out of scope here)."""
    if step < WARMUP_STEPS:
        floor = MIN_LR / PEAK_LR
        return floor + (1.0 - floor) * (step + 1) / WARMUP_STEPS
    return 1.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir")
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--num-more-steps", type=int, default=3,
                         help="optimizer updates to run in THIS invocation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    torch.manual_seed(0)
    model = Chessformer(**FULL_79M_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameter count: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_constant)
    scaler = torch.amp.GradScaler(
        device=device.type, enabled=(device.type == "cuda"),
        init_scale=256, growth_factor=1.5, backoff_factor=0.5, growth_interval=2000,
    )

    start_step = 0
    if args.resume:
        ckpts = sorted(pathlib.Path(args.checkpoint_dir).glob("step_*.pt"))
        if not ckpts:
            raise RuntimeError(f"--resume given but no checkpoint found in {args.checkpoint_dir}")
        ckpt_path = ckpts[-1]
        start_step = load_checkpoint(ckpt_path, model, optimizer, scheduler, device, scaler=scaler)
        print(f"resumed from {ckpt_path} at global_step={start_step}")
        print(f"restored scaler scale: {scaler.get_scale()}")
        print(f"restored LR: {optimizer.param_groups[0]['lr']:.3e}")

    dataset = ChessDataset(args.shard_dir, shuffle_buffer_size=256)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    target_step = start_step + args.num_more_steps
    t0 = time.perf_counter()
    global_step = train(
        model, dataloader, optimizer, scheduler, device,
        num_steps=target_step,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=MAX_GRAD_NORM,
        value_coefficient=VALUE_COEFFICIENT,
        log_every=1,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.num_more_steps,
        start_step=start_step,
        use_amp=(device.type == "cuda"),
        scaler=scaler,
    )
    elapsed = time.perf_counter() - t0

    print(f"finished at global_step={global_step} (target {target_step})")
    print(f"elapsed: {elapsed:.2f}s for {global_step - start_step} optimizer step(s)")
    if global_step > start_step:
        print(f"seconds/optimizer_step: {elapsed / (global_step - start_step):.3f}")
    print(f"final scaler scale: {scaler.get_scale()}")
    print(f"final LR: {optimizer.param_groups[0]['lr']:.3e}")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"peak VRAM allocated: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()
