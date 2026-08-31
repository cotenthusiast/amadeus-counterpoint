"""Streaming dataset utilities for preprocessed chess training data.

This module reads compact game records from Parquet shards, replays each game,
and yields one supervised training example per sampled human move (see
`iter_game_examples` for the sampling policy).
"""

import random
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import chess
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from amadeus_counterpoint.data.preprocess import GameRecord
from amadeus_counterpoint.encoding import (
    HISTORY_LENGTH,
    encode_history,
    legal_move_mask,
    move_to_policy_index,
)


class TrainingExample(TypedDict):
    """One supervised position used to train Chessformer."""

    x: torch.Tensor
    player_elo: int
    opponent_elo: int
    policy_target: int
    value_target: int
    legal_mask: torch.Tensor


def result_to_value_target(result: str, turn: chess.Color) -> int:
    """Convert a game result to loss/draw/win from the active player's view.

    Args:
        result: PGN game result: ``1-0``, ``0-1``, or ``1/2-1/2``.
        turn: Side to move in the current position.

    Returns:
        0 for loss, 1 for draw, and 2 for win.
    """
    if result == "1/2-1/2":
        return 1

    if turn == chess.WHITE:
        return 2 if result == "1-0" else 0

    return 2 if result == "0-1" else 0


DEFAULT_MAX_POSITIONS_PER_GAME = 32

# Chessformer Appendix E: with probability 5%, training examples have a
# uniformly random amount of history masked out, keeping low/no-history
# positions in-distribution. The 5% trigger probability and "uniformly
# random amount of history" are published; the exact original RNG sampling
# implementation is not, so `_mask_history` below is a documented,
# defensible reconstruction rather than a source-exact reproduction.
DEFAULT_HISTORY_MASK_PROB = 0.05


def _mask_history(
    history: list[chess.Board], rng: random.Random
) -> list[chess.Board]:
    """Randomly drop older boards, always keeping the current board.

    Uniformly chooses how many of the previous boards (0..available) to
    retain, keeping the most recent ones. The current board (`history[-1]`)
    is never dropped.
    """
    available_previous = len(history) - 1
    keep_previous = rng.randint(0, available_previous)
    return history[-(keep_previous + 1):]


def iter_game_examples(
    record: GameRecord,
    max_positions_per_game: int = DEFAULT_MAX_POSITIONS_PER_GAME,
    history_mask_prob: float = DEFAULT_HISTORY_MASK_PROB,
    rng: random.Random | None = None,
) -> Iterator[TrainingExample]:
    """Replay one game and yield an example for a sample of its eligible moves.

    Matches the Maia-3 recipe: up to `max_positions_per_game` positions are
    sampled uniformly at random without replacement from the game's eligible
    plies (see `eligible_ply_count` in preprocess.py for the <30s
    time-pressure cutoff); every eligible position is used if there are
    fewer than that. Every move up to the eligible cutoff is still replayed
    in order regardless of sampling, so board history stays correct for the
    positions that are yielded; moves past the cutoff are never replayed.

    Following Chessformer Appendix E, with probability `history_mask_prob`
    each yielded example has its history randomly shortened before encoding
    (see `_mask_history`); this training-time augmentation is applied here,
    not in `encode_history`, which stays deterministic for eval/inference use.

    Each example contains the encoded board history, active-player and opponent
    Elo ratings, policy target, value target, and legal-move mask.
    """
    if rng is None:
        rng = random.Random()

    white_elo = record["white_elo"]
    black_elo = record["black_elo"]
    result = record["result"]
    moves = record["moves"]

    # Records without this field (e.g. hand-built fixtures) treat every ply
    # as eligible, matching the pre-time-pressure-filter behavior.
    num_eligible = min(record.get("eligible_ply_count", len(moves)), len(moves))

    if num_eligible <= max_positions_per_game:
        sampled_plies = set(range(num_eligible))
    else:
        sampled_plies = set(rng.sample(range(num_eligible), max_positions_per_game))

    board = chess.Board()
    history = [board.copy(stack=False)]

    for ply, move_uci in enumerate(moves):
        if ply >= num_eligible:
            break

        move = chess.Move.from_uci(move_uci)

        # All targets describe the position before the human move is played.
        if board.turn == chess.WHITE:
            player_elo = white_elo
            opponent_elo = black_elo
        else:
            player_elo = black_elo
            opponent_elo = white_elo

        if ply in sampled_plies:
            example_history = history
            if rng.random() < history_mask_prob:
                example_history = _mask_history(history, rng)

            yield {
                "x": encode_history(example_history),
                "player_elo": player_elo,
                "opponent_elo": opponent_elo,
                "policy_target": move_to_policy_index(move, board),
                "value_target": result_to_value_target(result, board.turn),
                "legal_mask": legal_move_mask(board),
            }

        # Replay the recorded move and advance the rolling board history.
        board.push(move)
        history.append(board.copy(stack=False))

        if len(history) > HISTORY_LENGTH:
            history.pop(0)


def iter_shard_records(
    path: str | Path,
    batch_size: int = 1024,
) -> Iterator[GameRecord]:
    """Yield game records from a Parquet shard using bounded memory.

    Args:
        path: Path to a preprocessed Parquet shard.
        batch_size: Number of game records read from Parquet at a time.
    """
    parquet_file = pq.ParquetFile(path)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def shuffle_buffer(
    examples: Iterator[TrainingExample],
    buffer_size: int,
    rng: random.Random,
) -> Iterator[TrainingExample]:
    """Approximately shuffle a stream while using bounded memory.

    The buffer is first filled with examples. Each subsequent example replaces
    a randomly selected buffered example, which is yielded to the caller.
    Remaining examples are shuffled and emitted when the source is exhausted.
    """
    if buffer_size <= 1:
        yield from examples
        return

    buffer: list[TrainingExample] = []

    for example in examples:
        if len(buffer) < buffer_size:
            buffer.append(example)
            continue

        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = example

    rng.shuffle(buffer)
    yield from buffer


class ChessDataset(IterableDataset):
    """Stream Chessformer training examples from Parquet shards.

    DataLoader workers receive disjoint subsets of the available shards.
    Examples are then approximately shuffled using a bounded streaming buffer.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        shuffle_buffer_size: int = 1024,
    ):
        super().__init__()

        if shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative")

        shard_dir = Path(shard_dir)
        self.shards = sorted(shard_dir.glob("*.parquet"))
        self.shuffle_buffer_size = shuffle_buffer_size

        if not self.shards:
            raise ValueError(f"No Parquet shards found in {shard_dir}")

    def __iter__(self) -> Iterator[TrainingExample]:
        """Yield examples assigned to the current DataLoader worker."""
        worker_info = get_worker_info()

        if worker_info is None:
            # num_workers=0: the current process handles every shard.
            worker_shards = list(self.shards)
        else:
            # Give each worker a disjoint subset so games are not duplicated.
            worker_shards = list(
                self.shards[
                    worker_info.id :: worker_info.num_workers
                ]
            )

        # DataLoader gives each worker its own PyTorch seed.
        rng = random.Random(torch.initial_seed())
        rng.shuffle(worker_shards)

        def examples() -> Iterator[TrainingExample]:
            for shard_path in worker_shards:
                for record in iter_shard_records(shard_path):
                    yield from iter_game_examples(record, rng=rng)

        yield from shuffle_buffer(
            examples(),
            self.shuffle_buffer_size,
            rng,
        )