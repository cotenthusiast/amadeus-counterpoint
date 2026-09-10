"""Turn a list of `StyleGameRecord`s into train/validation `StyleDataset`s
that yield exactly `StyleTrainer`'s existing batch contract:

    x               [64, 96]
    player_id       int
    player_elo      float  -- the target player's own recorded Elo
    opponent_elo    float  -- the opponent's recorded Elo
    policy_target   int    -- global 0..4351 policy index of the human move
    legal_mask      [4352] bool

Pipeline: split games by player (before balancing) -> list each split's
target-mover decisions -> label each decision's game phase (data/phase.py)
-> balance each player's decisions across phases, per split, independently
-> wrap the balanced decisions in a Dataset that encodes lazily.
"""

import random

import chess
import torch
from torch.utils.data import DataLoader, Dataset

from amadeus_counterpoint.data.phase import divide_game, phase_at_ply
from amadeus_counterpoint.encoding import encode_history, legal_move_mask, move_to_policy_index

PHASES = ("opening", "middlegame", "endgame")


def filter_records_by_player(records: list[dict], player_id: int) -> list[dict]:
    """Keep only one player's records.

    Method 1 trains exactly one player's z_player per run (one
    PersonalizedChessformer = one z_player = one player) and must never mix
    identities -- callers filter with this BEFORE any dataset construction.
    """
    return [record for record in records if record["player_id"] == player_id]


def split_games_by_player(records: list[dict], train_fraction: float = 0.8, seed: int = 0):
    """Split each player's eligible games independently into train/validation.

    Rounding: for each player, `n_train = round(n_games * train_fraction)`.
    If the player has at least 2 games, `n_train` is then clamped to
    `[1, n_games - 1]` so both splits are non-empty whenever that's
    possible at all. A player with exactly 1 eligible game has that game
    placed entirely in train, leaving validation empty for that player --
    reported explicitly, not silently patched.

    One `random.Random(seed)` is shared across all players, processed in
    sorted player_id order. The guarantee this provides is: the same input
    `records` sequence with the same `seed` always produces the same split.
    It does NOT guarantee the split is independent of `records`' input
    order -- `random.shuffle` on the same seed can still land differently
    if a player's games arrive in a different relative order beforehand.
    In practice this is not a problem: the real pipeline
    (`broadcast_ingest.iter_target_game_records`) streams PGN files in a
    fixed, caller-supplied order, so `records`' order -- and therefore the
    split -- is itself already stable run to run.

    Returns (train_records, val_records, report) where report maps
    player_id to {"total_eligible_games", "train_games", "val_games"}.
    """
    records_by_player: dict[int, list[dict]] = {}
    for record in records:
        records_by_player.setdefault(record["player_id"], []).append(record)

    rng = random.Random(seed)

    train_records = []
    val_records = []
    report = {}

    for player_id in sorted(records_by_player):
        player_games = list(records_by_player[player_id])
        rng.shuffle(player_games)

        n_games = len(player_games)
        n_train = round(n_games * train_fraction)
        if n_games >= 2:
            n_train = max(1, min(n_train, n_games - 1))

        train_records.extend(player_games[:n_train])
        val_records.extend(player_games[n_train:])

        report[player_id] = {
            "total_eligible_games": n_games,
            "train_games": n_train,
            "val_games": n_games - n_train,
        }

    return train_records, val_records, report


def list_decisions(records: list[dict]) -> list[tuple[int, int]]:
    """List every target-mover decision across `records` as (game_index, ply).

    White-mover games contribute even plies (0, 2, 4, ...); black-mover
    games contribute odd plies (1, 3, 5, ...); both up to the game's last
    move. No board replay is needed for this step.
    """
    decisions = []
    for game_index, record in enumerate(records):
        start_ply = 0 if record["mover_color"] == "white" else 1
        for ply in range(start_ply, len(record["moves"]), 2):
            decisions.append((game_index, ply))
    return decisions


def _boards_before_each_ply(record: dict) -> list[chess.Board]:
    """One board per ply of the game, each the position BEFORE that ply is
    played -- the frame `data.phase.divide_game` expects."""
    board = chess.Board()
    boards = [board.copy(stack=False)]
    for move_uci in record["moves"]:
        board.push(chess.Move.from_uci(move_uci))
        boards.append(board.copy(stack=False))
    return boards[:-1]  # drop the position after the final move


def label_phases(records: list[dict], decisions: list[tuple[int, int]]) -> list[str]:
    """Label each (game_index, ply) decision with its game phase, in the
    same order as `decisions`. Each distinct game is replayed and divided
    only once, even if it contributes many decisions."""
    division_by_game = {}
    labels = []
    for game_index, ply in decisions:
        if game_index not in division_by_game:
            boards = _boards_before_each_ply(records[game_index])
            division_by_game[game_index] = divide_game(boards)
        labels.append(phase_at_ply(division_by_game[game_index], ply))
    return labels


def balance_by_phase(
    decisions: list[tuple[int, int]],
    phases: list[str],
    records: list[dict],
    seed: int = 0,
    max_decisions_per_player: int | None = None,
):
    """Downsample each player's decisions so all three phases contribute
    equally, independently per player.

    For each player: n = the smallest of that player's three phase counts
    (0 if any phase is entirely absent -- see below). Each phase bucket is
    downsampled to exactly n examples via `random.sample` (no replacement,
    so no duplicates and no oversampling), deterministic under `seed`.

    If `max_decisions_per_player` is given, n is additionally capped at
    `max_decisions_per_player // 3`, keeping the result phase-balanced.
    Default is no cap.

    A phase with zero available examples forces n=0 for that player,
    meaning ALL of that player's phases are downsampled to zero -- this is
    reported explicitly in the per-player "available_counts", not hidden.

    Returns (balanced_decisions, report) where report maps player_id to
    {"available_counts": {phase: count}, "balanced_count_per_phase": n,
    "total_balanced": n * 3}.
    """
    rng = random.Random(seed)

    buckets: dict[tuple[int, str], list[int]] = {}
    for decision_index, ((game_index, _ply), phase) in enumerate(zip(decisions, phases)):
        player_id = records[game_index]["player_id"]
        buckets.setdefault((player_id, phase), []).append(decision_index)

    player_ids = sorted({records[game_index]["player_id"] for game_index, _ply in decisions})

    balanced_indices = []
    report = {}

    for player_id in player_ids:
        available_counts = {phase: len(buckets.get((player_id, phase), [])) for phase in PHASES}
        n = min(available_counts.values())
        if max_decisions_per_player is not None:
            n = min(n, max_decisions_per_player // 3)

        for phase in PHASES:
            available = buckets.get((player_id, phase), [])
            chosen = rng.sample(available, n) if n > 0 else []
            balanced_indices.extend(chosen)

        report[player_id] = {
            "available_counts": available_counts,
            "balanced_count_per_phase": n,
            "total_balanced": n * len(PHASES),
        }

    balanced_indices.sort()
    balanced_decisions = [decisions[i] for i in balanced_indices]
    return balanced_decisions, report


class StyleDataset(Dataset):
    """Yields exactly StyleTrainer's batch fields, encoded lazily per item."""

    def __init__(self, records: list[dict], decisions: list[tuple[int, int]]):
        self.records = records
        self.decisions = decisions

    def __len__(self) -> int:
        return len(self.decisions)

    def __getitem__(self, index: int) -> dict:
        game_index, ply = self.decisions[index]
        record = self.records[game_index]

        board = chess.Board()
        history = [board.copy(stack=False)]
        for move_uci in record["moves"][:ply]:
            board.push(chess.Move.from_uci(move_uci))
            history.append(board.copy(stack=False))

        x = encode_history(history)
        legal_mask = legal_move_mask(board)

        move = chess.Move.from_uci(record["moves"][ply])
        policy_target = move_to_policy_index(move, board)

        if record["mover_color"] == "white":
            player_elo = record["white_elo"]
            opponent_elo = record["black_elo"]
        else:
            player_elo = record["black_elo"]
            opponent_elo = record["white_elo"]

        return {
            "x": x,
            "player_id": record["player_id"],
            # Explicit float32 tensors: DataLoader's default collation turns
            # a plain Python float into a float64 batch (a documented
            # PyTorch quirk), which would then dtype-mismatch against the
            # model's float32 tensors.
            "player_elo": torch.tensor(player_elo, dtype=torch.float32),
            "opponent_elo": torch.tensor(opponent_elo, dtype=torch.float32),
            "policy_target": policy_target,
            "legal_mask": legal_mask,
        }


def build_style_datasets(
    records: list[dict],
    train_fraction: float = 0.8,
    split_seed: int = 0,
    balance_seed: int = 0,
    max_decisions_per_player: int | None = None,
):
    """Full Stage-6 pipeline: split by game, then balance by phase,
    independently per split. Returns (train_dataset, val_dataset, report).
    """
    train_records, val_records, split_report = split_games_by_player(
        records, train_fraction=train_fraction, seed=split_seed
    )

    train_decisions_raw = list_decisions(train_records)
    val_decisions_raw = list_decisions(val_records)

    for player_id, stats in split_report.items():
        stats["train_decisions_before_balancing"] = sum(
            1 for game_index, _ply in train_decisions_raw
            if train_records[game_index]["player_id"] == player_id
        )
        stats["val_decisions_before_balancing"] = sum(
            1 for game_index, _ply in val_decisions_raw
            if val_records[game_index]["player_id"] == player_id
        )

    train_phases = label_phases(train_records, train_decisions_raw)
    val_phases = label_phases(val_records, val_decisions_raw)

    train_decisions, train_balance_report = balance_by_phase(
        train_decisions_raw, train_phases, train_records,
        seed=balance_seed, max_decisions_per_player=max_decisions_per_player,
    )
    val_decisions, val_balance_report = balance_by_phase(
        val_decisions_raw, val_phases, val_records,
        seed=balance_seed, max_decisions_per_player=max_decisions_per_player,
    )

    report = {
        "split": split_report,
        "train_balance": train_balance_report,
        "val_balance": val_balance_report,
    }

    train_dataset = StyleDataset(train_records, train_decisions)
    val_dataset = StyleDataset(val_records, val_decisions)

    return train_dataset, val_dataset, report


def build_style_dataloaders(train_dataset: Dataset, val_dataset: Dataset, batch_size: int = 32):
    """Thin DataLoader wrapper. Default collation already produces exactly
    StyleTrainer's batch shapes (x/legal_mask as tensors, the rest as
    LongTensor/FloatTensor) -- no custom collate_fn is needed."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
