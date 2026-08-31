"""Offline preprocessing utilities for chess PGN data.

This module streams games from PGN files, validates the metadata required for
training, converts games into compact records, and writes those records into
compressed Parquet shards.
"""

import itertools
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq

from amadeus_counterpoint.data.target_exclusion import (
    TargetExclusionRequiredError,
    is_target_game,
)

# Elo-bin balancing, verified verbatim from the Chessformer paper (Sec. 7):
# 22 bins -- one for mean Elo < 600, twenty 100-point bins spanning
# [600, 2600), and one for mean Elo >= 2600.
NUM_ELO_BINS = 22
ELO_BIN_WIDTH = 100
ELO_BIN_LOW = 600
ELO_BIN_HIGH = 2600

# Time-pressure cutoff, verified from the Chessformer paper (Sec. 4.1): "we
# retain the first 10 moves but discard moves made under time pressure in
# the same way [as the eval set]" -- i.e. remove positions that occur at or
# after the first time the mover has fewer than 30 seconds left on their
# clock. "Fewer than" is strict: exactly 30.0s does not trigger the cutoff.
TIME_PRESSURE_THRESHOLD_SECONDS = 30.0


class GameRecord(TypedDict):
    """Compact representation of one validated chess game."""

    white_elo: int
    black_elo: int
    result: str
    moves: list[str]
    eligible_ply_count: int


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


def eligible_ply_count(clocks_after_move: list[float | None]) -> int:
    """Return how many leading plies survive the <30s time-pressure cutoff.

    `clocks_after_move[i]` is the seconds left on the mover's clock right
    after playing ply `i` (as reported by `[%clk ...]`), or None if no clock
    annotation is present. Ply `i`'s own PRE-move clock is therefore the same
    player's previous reading, `clocks_after_move[i - 2]` (two plies back,
    since colors alternate); the first two plies of a game precede any
    same-color reading and are always eligible.

    Once a pre-move clock reading is found below the threshold, that ply and
    every later ply in the game are excluded (matches the paper's "discard
    moves made under time pressure" cutoff, not a scattered per-move filter).
    Missing clock data is treated as no evidence of time pressure and never
    triggers the cutoff -- the paper does not specify missing-clock handling,
    so this is our deliberate, conservative choice.

    Args:
        clocks_after_move: Post-move clock readings in seconds, one per ply.

    Returns:
        The number of leading plies (0..eligible_ply_count - 1) eligible for
        sampling as training positions.
    """
    for i in range(len(clocks_after_move)):
        pre_move_clock = clocks_after_move[i - 2] if i >= 2 else None
        if pre_move_clock is not None and pre_move_clock < TIME_PRESSURE_THRESHOLD_SECONDS:
            return i

    return len(clocks_after_move)


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

    # dataset.py replays games with a plain chess.Board() under standard
    # rules. python-chess auto-detects the PGN "Variant" header and parses
    # non-standard variants (Crazyhouse, Atomic, King of the Hill, ...) with
    # their own move semantics -- e.g. Crazyhouse drop moves -- which are not
    # valid standard-chess UCI and would corrupt (or crash) replay. Most
    # variants still start from the normal board, so the FEN/SetUp check
    # below does not catch them; only Chess960/"From Position" games do.
    if headers.get("Variant", "Standard").strip().lower() != "standard":
        return None

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
    nodes = list(game.mainline())
    moves = [node.move.uci() for node in nodes]

    if not moves:
        return None

    # node.clock() is the mover's remaining time AFTER their move, so it is
    # the pre-move clock reading for that same player's NEXT move (two plies
    # later, since colors alternate).
    clocks_after_move = [node.clock() for node in nodes]

    return {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "moves": moves,
        "eligible_ply_count": eligible_ply_count(clocks_after_move),
    }


def iter_records(
    path: str | Path,
    target_aliases: frozenset[str] | None = None,
    stats: dict[str, int] | None = None,
) -> Iterator[GameRecord]:
    """Yield only valid, non-target-excluded compact game records from a PGN file.

    Args:
        path: Path to the input PGN file.
        target_aliases: Normalized target-cohort Lichess usernames (see
            `target_exclusion.load_target_aliases`). A game is rejected if
            either player matches. `None` or empty applies no filtering.
        stats: If given, `stats["target_excluded_games"]` is created (at 0)
            and incremented whenever `target_aliases` is non-empty, so
            callers can distinguish "exclusion applied, zero matches" from
            "exclusion not applied at all" (the key is simply absent).

    Yields:
        Valid, non-target-excluded game records suitable for writing to
        training shards.
    """
    if target_aliases and stats is not None:
        stats.setdefault("target_excluded_games", 0)

    for game in iter_pgn_games(path):
        if target_aliases and is_target_game(
            game.headers.get("White", ""), game.headers.get("Black", ""), target_aliases
        ):
            if stats is not None:
                stats["target_excluded_games"] += 1
            continue

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
    target_aliases: frozenset[str] | None = None,
    allow_missing_target_exclusion: bool = False,
    stats: dict[str, int] | None = None,
) -> None:
    """Preprocess a PGN file into compressed Parquet shards.

    Games are streamed from disk, validated, target-excluded, resampled for
    Elo-bin balance (see `balance_by_elo`), buffered up to ``shard_size``,
    and then written to sequentially numbered shard files.

    Args:
        input_path: Path to the source PGN file.
        output_dir: Directory in which Parquet shards will be created.
        shard_size: Maximum number of games stored in each shard.
        elo_chunk_size: Games per Elo-balancing chunk (see `balance_by_elo`).
        games_per_elo_bin: Games kept per Elo bin within one chunk.
        target_aliases: Normalized target-cohort Lichess usernames (see
            `target_exclusion.load_target_aliases`). Required unless
            `allow_missing_target_exclusion` is set.
        allow_missing_target_exclusion: Deliberate opt-in to run WITHOUT
            target-cohort exclusion (e.g. `target_aliases` not yet
            populated). Emits a warning; the output is PILOT-ONLY and NOT
            VALID FOR FINAL POPULATION TRAINING.
        stats: If given, `stats["target_exclusion_applied"]` records whether
            exclusion ran, and (only when it did) `stats["target_excluded_games"]`
            counts the games it rejected.

    Raises:
        ValueError: If shard_size is not positive.
        TargetExclusionRequiredError: If `target_aliases` is empty/missing
            and `allow_missing_target_exclusion` was not explicitly set.
    """
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than 0")

    if not target_aliases:
        if not allow_missing_target_exclusion:
            raise TargetExclusionRequiredError(
                "No target-cohort aliases are configured (see "
                "configs/target_aliases.json). Population training must "
                "exclude the target cohort's Lichess accounts. Pass "
                "allow_missing_target_exclusion=True only for a "
                "non-production pilot run -- the resulting data is "
                "PILOT-ONLY and NOT VALID FOR FINAL POPULATION TRAINING."
            )
        warnings.warn(
            "PILOT-ONLY: target-cohort exclusion is not configured, so no "
            "target-cohort games are being filtered. This preprocessing "
            "run and its output are NOT VALID FOR FINAL POPULATION "
            "TRAINING.",
            stacklevel=2,
        )
        if stats is not None:
            stats["target_exclusion_applied"] = False
    elif stats is not None:
        stats["target_exclusion_applied"] = True

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer: list[GameRecord] = []
    shard_index = 0

    balanced_records = balance_by_elo(
        iter_records(input_path, target_aliases=target_aliases, stats=stats),
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