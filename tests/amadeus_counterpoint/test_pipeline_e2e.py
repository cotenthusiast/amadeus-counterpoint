"""End-to-end tests: PGN -> preprocess -> Parquet -> dataset -> model -> loss -> optimizer.

Uses a small hand-written PGN corpus and a tiny (non-79M) Chessformer config,
per the audit's requirement not to instantiate the full model in routine tests.
"""

import torch
from torch.utils.data import DataLoader

from amadeus_counterpoint.data.dataset import ChessDataset, iter_game_examples
from amadeus_counterpoint.data.preprocess import preprocess_pgn
from amadeus_counterpoint.models import Chessformer
from amadeus_counterpoint.training.loss import chessformer_loss
from amadeus_counterpoint.training.trainer import train

TINY_CONFIG = {
    "d_model": 32,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.0,
    "d1": 4,
    "d2": 8,
    "d3": 4,
    "d_ff": 64,
    "head_hid_dim": 16,
    "input_dim": 96,
    "elo_dim": 8,
}

SAMPLE_PGN = """\
[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1600"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0

[Event "Test"]
[White "C"]
[Black "D"]
[Result "0-1"]
[WhiteElo "1800"]
[BlackElo "2000"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 0-1

[Event "Test"]
[White "E"]
[Black "F"]
[Result "1/2-1/2"]
[WhiteElo "2200"]
[BlackElo "2100"]

1. c4 c5 2. Nf3 Nf6 3. g3 b6 1/2-1/2

"""


def test_pgn_to_optimizer_step(tmp_path):
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(SAMPLE_PGN, encoding="utf-8")

    shard_dir = tmp_path / "shards"
    preprocess_pgn(pgn_path, shard_dir, shard_size=10)
    assert list(shard_dir.glob("*.parquet"))

    dataset = ChessDataset(shard_dir, shuffle_buffer_size=8)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0)

    torch.manual_seed(0)
    model = Chessformer(**TINY_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = next(iter(dataloader))
    policy_logits, value_logits = model(
        batch["x"], batch["player_elo"], batch["opponent_elo"]
    )
    total_loss, _, _ = chessformer_loss(
        policy_logits,
        value_logits,
        batch["policy_target"],
        batch["value_target"],
        batch["legal_mask"],
    )

    assert torch.isfinite(total_loss)

    before = [p.clone() for p in model.parameters()]
    total_loss.backward()
    optimizer.step()
    after = list(model.parameters())

    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_full_dataloader_epoch_runs_through_training_loop(tmp_path):
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(SAMPLE_PGN, encoding="utf-8")

    shard_dir = tmp_path / "shards"
    preprocess_pgn(pgn_path, shard_dir, shard_size=10)

    dataset = ChessDataset(shard_dir, shuffle_buffer_size=8)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    torch.manual_seed(0)
    model = Chessformer(**TINY_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    global_step = train(
        model, dataloader, optimizer, scheduler, torch.device("cpu"),
        num_steps=100, accumulation_steps=1, log_every=1000,
    )

    # The tiny dataset exhausts long before 100 optimizer updates.
    assert 0 < global_step < 100


def test_tiny_overfit_reduces_policy_loss_substantially():
    """A model should be able to memorize a single repeated example.

    This is a training-flow sanity check (correct targets/gradients wired up
    end to end), not a claim about real model quality.
    """
    record = {
        "white_elo": 1500,
        "black_elo": 1500,
        "result": "1-0",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
    }
    example = next(iter(iter_game_examples(record)))

    batch_size = 8
    batch = {
        "x": example["x"].unsqueeze(0).repeat(batch_size, 1, 1),
        "player_elo": torch.full((batch_size,), example["player_elo"]),
        "opponent_elo": torch.full((batch_size,), example["opponent_elo"]),
        "policy_target": torch.full((batch_size,), example["policy_target"], dtype=torch.long),
        "value_target": torch.full((batch_size,), example["value_target"], dtype=torch.long),
        "legal_mask": example["legal_mask"].unsqueeze(0).repeat(batch_size, 1),
    }

    torch.manual_seed(0)
    model = Chessformer(**TINY_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def policy_loss_of(model):
        model.eval()
        with torch.no_grad():
            policy_logits, value_logits = model(
                batch["x"], batch["player_elo"], batch["opponent_elo"]
            )
            _, policy_loss, _ = chessformer_loss(
                policy_logits, value_logits,
                batch["policy_target"], batch["value_target"], batch["legal_mask"],
            )
        model.train()
        return policy_loss.item()

    initial_loss = policy_loss_of(model)

    repeated_batches = [batch] * 200
    train(
        model, repeated_batches, optimizer, scheduler, torch.device("cpu"),
        num_steps=200, accumulation_steps=1, log_every=1000,
    )

    final_loss = policy_loss_of(model)

    assert final_loss < initial_loss * 0.1
