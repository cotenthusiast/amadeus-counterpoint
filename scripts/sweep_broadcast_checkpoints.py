"""Sweep Broadcast-adaptation checkpoints over a held-out validation shard
set and report the checkpoint with the lowest validation policy loss.

The validation shard directory must already exclude all 8 target players --
this script performs no additional exclusion or filtering itself.

    python scripts/sweep_broadcast_checkpoints.py \\
        --checkpoint-dir checkpoints/broadcast_adaptation \\
        --val-shard-dir data/broadcast_val_shards \\
        --report-path checkpoints/broadcast_adaptation/val_sweep.json
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.dataset import ChessDataset
from amadeus_counterpoint.evaluation.broadcast_validation import evaluate_checkpoint, select_best_checkpoint
from amadeus_counterpoint.models import Chessformer

# Must match the architecture the Broadcast-adaptation checkpoints were
# trained with (scripts/train.py's "Chessformer 79M" configuration).
D_MODEL = 1024
NUM_HEADS = 32
NUM_LAYERS = 8
D1 = 32
D2 = 128
D3 = 128
D_FF = 2048
ELO_DIM = 128
HEAD_HID_DIM = 1024
RAW_INPUT_DIM = 96
DROPOUT = 0.0


def build_model() -> Chessformer:
    return Chessformer(
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
        d1=D1, d2=D2, d3=D3, d_ff=D_FF, head_hid_dim=HEAD_HID_DIM,
        input_dim=RAW_INPUT_DIM, elo_dim=ELO_DIM,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--val-shard-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--report-path", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoints = sorted(args.checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no step_*.pt checkpoints found in {args.checkpoint_dir}")

    results = {}
    for checkpoint_path in checkpoints:
        model = build_model().to(device)
        payload = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(payload["model"])

        dataset = ChessDataset(shard_dir=args.val_shard_dir, shuffle_buffer_size=1)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        metrics = evaluate_checkpoint(model, dataloader, device)
        results[checkpoint_path.name] = metrics
        print(
            f"{checkpoint_path.name}: policy_loss={metrics['mean_policy_loss']:.4f} "
            f"value_loss={metrics['mean_value_loss']:.4f} "
            f"value_acc={metrics['value_accuracy']:.4f} "
            f"move_match_acc={metrics['move_match_accuracy']:.4f} "
            f"n={metrics['num_examples']}"
        )

    best = select_best_checkpoint(results)
    print(f"\nBest checkpoint by validation policy loss: {best}")

    if args.report_path is not None:
        args.report_path.write_text(json.dumps({"results": results, "best": best}, indent=2))


if __name__ == "__main__":
    main()
