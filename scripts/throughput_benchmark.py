#!/usr/bin/env python3
"""Production-throughput benchmark for the real 79M Chessformer recipe.

Uses the actual trainer.train() loop, the actual Chessformer model, and
real preprocessed Parquet data -- this script only varies engineering
knobs (physical batch / accumulation split, DataLoader settings). It
makes NO change to loss normalization, optimizer/scheduler update
frequency, gradient clipping, or AMP semantics; physical_batch *
accumulation_steps must equal --effective-batch for every run (enforced
below).

Prints one JSON line per run to stdout (and appends to --results-file if
given), so a driver script can sweep many configs and collect results.

Optional --profile wraps the run in torch.profiler and prints a top-N
CUDA-time-by-operator table -- adds overhead, so use a short, separate
run for this, not the run whose throughput number you intend to report.
"""

import argparse
import json
import pathlib
import time

import torch
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.dataset import ChessDataset
from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.trainer import train

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
    """Placeholder smoke/benchmark schedule -- see gpu_smoke_79m.py; the
    real cosine-restart schedule is a separate, already documented gap."""
    if step < WARMUP_STEPS:
        floor = MIN_LR / PEAK_LR
        return floor + (1.0 - floor) * (step + 1) / WARMUP_STEPS
    return 1.0


class TimedIterable:
    """Wraps a DataLoader to record wall time spent waiting for each next
    batch, without altering what train() receives. Pure external
    instrumentation -- train() sees the same batches in the same order.
    """

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.fetch_times: list[float] = []

    def __iter__(self):
        it = iter(self.dataloader)
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                return
            self.fetch_times.append(time.perf_counter() - t0)
            yield batch


def run(args) -> dict:
    if args.physical_batch * args.accumulation_steps != args.effective_batch:
        raise ValueError(
            f"physical_batch ({args.physical_batch}) * accumulation_steps "
            f"({args.accumulation_steps}) != effective_batch ({args.effective_batch})"
        )

    device = torch.device(args.device)

    dl_kwargs = {
        "batch_size": args.physical_batch,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    persistent_workers = args.num_workers > 0 and not args.no_persistent_workers
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = persistent_workers
        if args.prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = args.prefetch_factor

    dataset = ChessDataset(args.shard_dir, shuffle_buffer_size=args.shuffle_buffer_size)
    dataloader = DataLoader(dataset, **dl_kwargs)
    timed = TimedIterable(dataloader)

    torch.manual_seed(0)
    model = Chessformer(**FULL_79M_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_then_constant)
    scaler = torch.amp.GradScaler(
        device=device.type, enabled=(device.type == "cuda"),
        init_scale=256, growth_factor=1.5, backoff_factor=0.5, growth_interval=2000,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    profiler_ctx = None
    if args.profile:
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
        )
        profiler_ctx.__enter__()

    t0 = time.perf_counter()
    global_step = train(
        model, timed, optimizer, scheduler, device,
        num_steps=args.num_steps,
        accumulation_steps=args.accumulation_steps,
        max_grad_norm=MAX_GRAD_NORM,
        value_coefficient=VALUE_COEFFICIENT,
        log_every=max(1, args.num_steps // 5),
        use_amp=(device.type == "cuda"),
        scaler=scaler,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        print(profiler_ctx.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    n_microbatches = global_step * args.accumulation_steps
    result = {
        "physical_batch": args.physical_batch,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch": args.physical_batch * args.accumulation_steps,
        "num_workers": args.num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": args.pin_memory,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "device": args.device,
        "requested_steps": args.num_steps,
        "optimizer_steps": global_step,
        "elapsed_s": round(elapsed, 4),
        "sec_per_microbatch": round(elapsed / n_microbatches, 5) if n_microbatches else None,
        "sec_per_optimizer_step": round(elapsed / global_step, 5) if global_step else None,
        "optimizer_updates_per_sec": round(global_step / elapsed, 4) if elapsed else None,
        "examples_per_sec": round(n_microbatches * args.physical_batch / elapsed, 2) if elapsed else None,
        "avg_dataloader_fetch_wait_s": round(sum(timed.fetch_times) / len(timed.fetch_times), 5)
        if timed.fetch_times else None,
        "total_dataloader_fetch_wait_s": round(sum(timed.fetch_times), 4),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3)
        if device.type == "cuda" else None,
        "projected_1M_step_hours": round(1_000_000 * elapsed / global_step / 3600, 2)
        if global_step else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir")
    parser.add_argument("--physical-batch", type=int, default=128)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--effective-batch", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--shuffle-buffer-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--results-file", default=None)
    parser.add_argument("--tag", default=None, help="free-text label for this run")
    args = parser.parse_args()

    result = run(args)
    if args.tag:
        result["tag"] = args.tag

    print(json.dumps(result))
    if args.results_file:
        path = pathlib.Path(args.results_file)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
