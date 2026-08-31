"""Offline preprocessing utilities for chess PGN data.

This module streams games from PGN files, validates the metadata required for
training, converts games into compact records, and writes those records into
compressed Parquet shards.
"""

import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

# Elo-bin balancing, verified verbatim from the Chessformer paper (Sec. 7):
# 22 bins -- one for mean Elo < 600, twenty 100-point bins spanning
# [600, 2600), and one for mean Elo >= 2600.
NUM_ELO_BINS = 22
ELO_BIN_WIDTH = 100
ELO_BIN_LOW = 600
ELO_BIN_HIGH = 2600


class GameRecord(TypedDict):
    """Compact representation of one validated chess game."""

    white_elo: int
    black_elo: int
    result: str
    moves: list[str]


def iter_pgn_games(path: str | Path) -> Iterator[chess.pgn.Game]:
    """Yield games from a PGN file one at a time.

    Games are parsed lazily so that large PGN files do not need to be loaded
    entirely into memory.

    Args:
        path: Path to the input PGN file.

    Yields:
        Parsed chess games in file order.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)

            # read_game() returns None once the file is exhausted.
            if game is None:
                break

            yield game


def game_to_record(game: chess.pgn.Game) -> GameRecord | None:
    """Convert one parsed game into a compact training record.

    Games without valid Elo ratings, a completed result, or any moves are
    discarded.

    Args:
        game: Parsed python-chess PGN game.

    Returns:
        A compact game record, or None if the game is unsuitable for training.
    """
    headers = game.headers

    # dataset.py replays games from chess.Board(), the standard starting
    # position, so games recorded from a custom position must be discarded.
    if "FEN" in headers or headers.get("SetUp") == "1":
        return None

    try:
        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
        result = headers["Result"]
    except (KeyError, ValueError):
        return None

    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return None

    # Store moves as reusable UCI strings rather than a one-use move iterator.
    moves = [move.uci() for move in game.mainline_moves()]

    if not moves:
        return None

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "moves": moves,
    }


def iter_records(path: str | Path) -> Iterator[GameRecord]:
    """Yield only valid compact game records from a PGN file.

    Args:
        path: Path to the input PGN file.

    Yields:
        Valid game records suitable for writing to training shards.
    """
    for game in iter_pgn_games(path):
        record = game_to_record(game)

        if record is None:
            continue

        yield record


def elo_bin(mean_elo: float) -> int:
    """Map a mean Elo rating to its 0..21 balancing bin index."""
    if mean_elo < ELO_BIN_LOW:
        return 0
    if mean_elo >= ELO_BIN_HIGH:
        return NUM_ELO_BINS - 1
    return 1 + int((mean_elo - ELO_BIN_LOW) // ELO_BIN_WIDTH)


def balance_by_elo(
    records: Iterable[GameRecord],
    chunk_size: int = 20_000,
    games_per_bin: int = 10,
) -> Iterator[GameRecord]:
    """Resample games so every Elo bin is represented, per the Chessformer recipe.

    Games are processed in sequential chunks of `chunk_size`. Within each
    chunk, games are visited in order and kept only while their bin (by mean
    of the two players' Elo) holds fewer than `games_per_bin` kept games so
    far; once every bin has reached the cap, the rest of the chunk is
    skipped. Bin counts reset at the start of each new chunk.

    Args:
        records: Valid game records, in their original sequential order.
        chunk_size: Number of raw games considered per balancing chunk.
        games_per_bin: Maximum games kept per Elo bin within one chunk.

    Yields:
        The retained subset of `records`, in their original order.
    """
    records = iter(records)

    while True:
        chunk = list(itertools.islice(records, chunk_size))
        if not chunk:
            return

        bin_counts = [0] * NUM_ELO_BINS
        filled_bins = 0

        for record in chunk:
            mean_elo = (record["white_elo"] + record["black_elo"]) / 2
            b = elo_bin(mean_elo)

            if bin_counts[b] >= games_per_bin:
                continue

            bin_counts[b] += 1
            yield record

            if bin_counts[b] == games_per_bin:
                filled_bins += 1
                if filled_bins == NUM_ELO_BINS:
                    break


def write_shard(records: list[GameRecord], path: str | Path) -> None:
    """Write a collection of game records to one compressed Parquet shard.

    Empty collections are ignored rather than producing useless empty shards.

    Args:
        records: Valid game records to write.
        path: Destination Parquet file.
    """
    if not records:
        return

    table = pa.Table.from_pylist(records)
    pq.write_table(table, path, compression="zstd")


def preprocess_pgn(
    input_path: str | Path,
    output_dir: str | Path,
    shard_size: int = 10_000, # TODO hyperparameter
    elo_chunk_size: int = 20_000,
    games_per_elo_bin: int = 10,
) -> None:
    """Preprocess a PGN file into compressed Parquet shards.

    Games are streamed from disk, validated, resampled for Elo-bin balance
    (see `balance_by_elo`), buffered up to ``shard_size``, and then written
    to sequentially numbered shard files.

    Args:
        input_path: Path to the source PGN file.
        output_dir: Directory in which Parquet shards will be created.
        shard_size: Maximum number of games stored in each shard.
        elo_chunk_size: Games per Elo-balancing chunk (see `balance_by_elo`).
        games_per_elo_bin: Games kept per Elo bin within one chunk.

    Raises:
        ValueError: If shard_size is not positive.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than 0")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[GameRecord] = []
    shard_index = 0

    balanced_records = balance_by_elo(
        iter_records(input_path),
        chunk_size=elo_chunk_size,
        games_per_bin=games_per_elo_bin,
    )

    for record in balanced_records:
        buffer.append(record)

        if len(buffer) >= shard_size:
            write_shard(
                buffer,
                output_dir / f"shard_{shard_index:05d}.parquet",
            )
            buffer = []
            shard_index += 1

    # Write the final partial shard, if there is one.
    if buffer:
        write_shard(
            buffer,
            output_dir / f"shard_{shard_index:05d}.parquet",
        )