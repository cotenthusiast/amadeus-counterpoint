"""Outer training loop for Method 2's joint style-residual personalization.

One run trains ALL 8 target players jointly: one shared MoveStyleCNN,
PlayerStyleTable (one row per player), and StyleResidual, over a frozen
Broadcast-adapted base. `training.style_trainer.StyleTrainer` already
implements the per-epoch train/validate mechanics; this script is only the
missing outer loop (epochs, early stopping, checkpoint saving).

    python scripts/train_method2.py \\
        --records-path data/style_records.json \\
        --base-checkpoint checkpoints/broadcast_adaptation/step_00200000.pt \\
        --checkpoint-path checkpoints/method2/joint.pt \\
        --player-id-map '{"Magnus Carlsen": 0, "Wesley So": 1}'
"""

import argparse
import json
from pathlib import Path

import torch

from amadeus_counterpoint.data.broadcast_ingest import load_game_records
from amadeus_counterpoint.data.style_dataset import build_style_dataloaders, build_style_datasets
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.player_style import PlayerStyleTable, StyleResidual
from amadeus_counterpoint.models.style_cnn import MoveStyleCNN
from amadeus_counterpoint.training.checkpoints import load_method2_checkpoint, save_method2_checkpoint
from amadeus_counterpoint.training.style_trainer import StyleTrainer

# Must match the architecture the Broadcast-adapted base checkpoint was
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


def build_base(checkpoint_path, device) -> Chessformer:
    base = Chessformer(
        d_model=D_MODEL, num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
        d1=D1, d2=D2, d3=D3, d_ff=D_FF, head_hid_dim=HEAD_HID_DIM,
        input_dim=RAW_INPUT_DIM, elo_dim=ELO_DIM,
    )
    payload = torch.load(checkpoint_path, map_location=device)
    base.load_state_dict(payload["model"])
    return base.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-path", type=Path, required=True,
        help="JSON file of all 8 players' StyleGameRecords (data.broadcast_ingest.save_game_records)",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--player-id-map", type=str, required=True,
        help='JSON string mapping identity -> player_id, e.g. \'{"Magnus Carlsen": 0}\'',
    )
    parser.add_argument("--num-players", type=int, default=8)
    parser.add_argument("--style-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--balance-seed", type=int, default=0)
    args = parser.parse_args()

    player_id_map = json.loads(args.player_id_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records = load_game_records(args.records_path)
    train_dataset, val_dataset, split_report = build_style_datasets(
        records, train_fraction=args.train_fraction,
        split_seed=args.split_seed, balance_seed=args.balance_seed,
    )
    train_loader, val_loader = build_style_dataloaders(
        train_dataset, val_dataset, batch_size=args.batch_size
    )
    print(f"split report: {split_report['split']}")

    base = build_base(args.base_checkpoint, device)
    cnn = MoveStyleCNN(style_dim=args.style_dim).to(device)
    table = PlayerStyleTable(num_players=args.num_players, style_dim=args.style_dim).to(device)
    residual = StyleResidual(style_dim=args.style_dim).to(device)

    trainer = StyleTrainer(base, cnn, table, residual, k=args.k, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.checkpoint_path.exists():
        metadata = load_method2_checkpoint(
            args.checkpoint_path, cnn, table, residual, optimizer=trainer.optimizer
        )
        best_val_loss = metadata["best_val_loss"] or float("inf")
        trainer.best_val_loss = best_val_loss
        trainer.epochs_without_improvement = metadata["epochs_without_improvement"] or 0
        start_epoch = (metadata["epoch"] or 0) + 1
        print(f"resumed from {args.checkpoint_path} at epoch {metadata['epoch']}")

    for epoch in range(start_epoch, args.max_epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        report = trainer.validate(val_loader)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={report.mean_ce:.4f} "
            f"coverage={report.coverage:.4f} best={trainer.best_val_loss:.4f} "
            f"no_improve={trainer.epochs_without_improvement}"
        )

        if report.mean_ce <= best_val_loss:
            best_val_loss = report.mean_ce
            save_method2_checkpoint(
                args.checkpoint_path, cnn, table, residual, optimizer=trainer.optimizer,
                epoch=epoch, step=None, best_val_loss=best_val_loss,
                epochs_without_improvement=trainer.epochs_without_improvement,
                k=args.k, style_dim=args.style_dim, player_id_map=player_id_map,
            )
            print(f"  saved best checkpoint to {args.checkpoint_path}")

        if trainer.epochs_without_improvement >= args.patience:
            print(f"early stopping after {epoch} epochs (patience={args.patience})")
            break


if __name__ == "__main__":
    main()
