"""Train ONE Method-1 player's z_player against cohort-excluded Broadcast
personalization records.

CRITICAL: one PersonalizedChessformer = one z_player = one player. Run this
once per target player (8 independent runs) -- it never mixes player
identities; records are filtered to the requested --player-id BEFORE any
dataset construction.

    python scripts/train_method1_player.py \\
        --player-id 0 --identity "Magnus Carlsen" --nominal-elo 2853.0 \\
        --records-path data/style_records.json \\
        --base-checkpoint checkpoints/broadcast_adaptation/step_00200000.pt \\
        --checkpoint-path checkpoints/method1/player_0.pt
"""

import argparse
from pathlib import Path

import torch

from amadeus_counterpoint.data.broadcast_ingest import load_game_records
from amadeus_counterpoint.data.style_dataset import (
    build_style_dataloaders,
    build_style_datasets,
    filter_records_by_player,
)
from amadeus_counterpoint.models.chessformer import Chessformer
from amadeus_counterpoint.models.personalized_chessformer import PersonalizedChessformer
from amadeus_counterpoint.training.checkpoints import (
    load_method1_checkpoint,
    save_method1_checkpoint,
    verify_method1_checkpoint_identity,
)
from amadeus_counterpoint.training.method1_trainer import Method1Trainer

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
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--identity", type=str, required=True)
    parser.add_argument("--nominal-elo", type=float, required=True)
    parser.add_argument(
        "--records-path", type=Path, required=True,
        help="JSON file of StyleGameRecords (data.broadcast_ingest.save_game_records)",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--balance-seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_records = load_game_records(args.records_path)
    player_records = filter_records_by_player(all_records, args.player_id)
    if not player_records:
        raise SystemExit(f"no records found for player_id={args.player_id} in {args.records_path}")

    train_dataset, val_dataset, split_report = build_style_datasets(
        player_records, train_fraction=args.train_fraction,
        split_seed=args.split_seed, balance_seed=args.balance_seed,
    )
    train_loader, val_loader = build_style_dataloaders(
        train_dataset, val_dataset, batch_size=args.batch_size
    )
    print(f"player {args.player_id} split: {split_report['split'][args.player_id]}")

    base = build_base(args.base_checkpoint, device)
    wrapper = PersonalizedChessformer(
        base, nominal_elo=args.nominal_elo, identity=args.identity
    ).to(device)

    trainer = Method1Trainer(wrapper, lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_loss = float("inf")
    if args.checkpoint_path.exists():
        metadata = load_method1_checkpoint(args.checkpoint_path, wrapper, optimizer=trainer.optimizer)
        verify_method1_checkpoint_identity(metadata, args.identity, args.nominal_elo)
        best_val_loss = metadata["best_val_loss"] or float("inf")
        trainer.best_val_loss = best_val_loss
        trainer.epochs_without_improvement = metadata["epochs_without_improvement"]
        start_epoch = (metadata["epoch"] or 0) + 1
        print(f"resumed from {args.checkpoint_path} at epoch {metadata['epoch']}")

    for epoch in range(start_epoch, args.max_epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"best={trainer.best_val_loss:.4f} no_improve={trainer.epochs_without_improvement}"
        )

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_method1_checkpoint(
                args.checkpoint_path, wrapper, optimizer=trainer.optimizer,
                epoch=epoch, step=None, best_val_loss=best_val_loss,
                epochs_without_improvement=trainer.epochs_without_improvement,
            )
            print(f"  saved best checkpoint to {args.checkpoint_path}")

        if trainer.epochs_without_improvement >= args.patience:
            print(f"early stopping after {epoch} epochs (patience={args.patience})")
            break


if __name__ == "__main__":
    main()
