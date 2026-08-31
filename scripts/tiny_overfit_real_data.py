#!/usr/bin/env python3
"""Learnability sanity check: overfit a small (non-79M) Chessformer on a
fixed batch of real examples produced by the production preprocessing
pipeline (PGN -> preprocess_pgn -> Parquet -> iter_game_examples).

Confirms the real chain -- preprocessing, encoding, targets, legal mask,
model, loss, backward, optimizer -- is wired correctly end to end, before
committing GPU time to the full 79M smoke.
"""

import argparse
import pathlib
import sys

import torch

from amadeus_counterpoint.data.dataset import iter_game_examples, iter_shard_records
from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.loss import chessformer_loss
from amadeus_counterpoint.training.trainer import train

TINY_CONFIG = {
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.0,
    "d1": 8,
    "d2": 16,
    "d3": 8,
    "d_ff": 128,
    "head_hid_dim": 32,
    "input_dim": 96,
    "elo_dim": 16,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dir")
    parser.add_argument("--num-examples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    records = []
    for path in sorted(pathlib.Path(args.shard_dir).glob("*.parquet")):
        records.extend(iter_shard_records(path))
    if not records:
        print(f"no records found in {args.shard_dir}", file=sys.stderr)
        sys.exit(1)

    examples = []
    for record in records:
        # history_mask_prob=0: a clean, deterministic fixed batch to overfit.
        examples.extend(iter_game_examples(record, history_mask_prob=0.0))
        if len(examples) >= args.num_examples:
            break
    examples = examples[: args.num_examples]
    print(f"real examples in fixed overfit batch: {len(examples)} (from {len(records)} retained games)")

    batch = {
        "x": torch.stack([e["x"] for e in examples]).to(device),
        "player_elo": torch.tensor([e["player_elo"] for e in examples], dtype=torch.long).to(device),
        "opponent_elo": torch.tensor([e["opponent_elo"] for e in examples], dtype=torch.long).to(device),
        "policy_target": torch.tensor([e["policy_target"] for e in examples], dtype=torch.long).to(device),
        "value_target": torch.tensor([e["value_target"] for e in examples], dtype=torch.long).to(device),
        "legal_mask": torch.stack([e["legal_mask"] for e in examples]).to(device),
    }

    torch.manual_seed(0)
    model = Chessformer(**TINY_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def policy_loss_of():
        model.eval()
        with torch.no_grad():
            policy_logits, value_logits = model(batch["x"], batch["player_elo"], batch["opponent_elo"])
            _, policy_loss, _ = chessformer_loss(
                policy_logits, value_logits,
                batch["policy_target"], batch["value_target"], batch["legal_mask"],
            )
        model.train()
        return policy_loss.item()

    initial_loss = policy_loss_of()
    train(
        model, [batch] * args.steps, optimizer, scheduler, device,
        num_steps=args.steps, accumulation_steps=1, log_every=max(1, args.steps // 5),
    )
    final_loss = policy_loss_of()

    ratio = final_loss / initial_loss if initial_loss else float("nan")
    print(f"device={device}")
    print(f"initial policy loss: {initial_loss:.4f}")
    print(f"final policy loss:   {final_loss:.4f}")
    print(f"ratio (final/initial): {ratio:.4f}")
    print(f"overfit verdict: {'PASS' if ratio < 0.1 else 'FAIL'} (threshold: ratio < 0.1)")


if __name__ == "__main__":
    main()
