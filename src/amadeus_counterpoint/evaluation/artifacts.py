"""Persistence for immutable raw generated-game Parquet shards.

``write_games(path, games, metadata)`` writes one new zstd-compressed shard;
``read_games(path)`` returns its PyArrow table.  ``metadata`` is a plain
mapping (for example protocol version, code commit, root seed, temperature)
stored as Parquet schema metadata.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

GAME_SCHEMA = pa.schema([
    pa.field("game_index", pa.uint64(), nullable=False),
    pa.field("seed", pa.uint64(), nullable=False),
    pa.field("dyad", pa.string(), nullable=False),
    pa.field("condition", pa.string(), nullable=False),
    pa.field("orientation", pa.string(), nullable=False),
    pa.field("result", pa.string(), nullable=True),
    pa.field("censored", pa.bool_(), nullable=False),
    pa.field("moves", pa.list_(pa.string()), nullable=False),
    pa.field("white_elo", pa.float64(), nullable=False),
    pa.field("black_elo", pa.float64(), nullable=False),
    pa.field("checkpoint_identity", pa.string(), nullable=False),
    pa.field("white_representation_identity", pa.string(), nullable=False),
    pa.field("black_representation_identity", pa.string(), nullable=False),
])

_CONDITIONS = frozenset({"GG", "AG", "GB", "AB"})
_ORIENTATIONS = frozenset({"A_WHITE", "B_WHITE"})
_REQUIRED_FIELDS = frozenset(GAME_SCHEMA.names)
_REQUIRED_METADATA = frozenset({
    "protocol_version",
    "code_commit",
    "root_seed",
    "temperature",
})


def _validate_games(games: Sequence[Mapping[str, object]]) -> None:
    game_indices = set()

    for game in games:
        if not isinstance(game, Mapping) or set(game) != _REQUIRED_FIELDS:
            raise ValueError(
                "each game must contain exactly the fields in GAME_SCHEMA"
            )

        if game["condition"] not in _CONDITIONS:
            raise ValueError(f"unknown condition: {game['condition']!r}")
        if game["orientation"] not in _ORIENTATIONS:
            raise ValueError(f"unknown orientation: {game['orientation']!r}")

        censored = game["censored"]
        if not isinstance(censored, bool):
            raise TypeError("censored must be a bool")
        if censored != (game["result"] is None):
            raise ValueError(
                "censored games must have result=None and normal games "
                "must have a result"
            )

        game_index = game["game_index"]
        if game_index in game_indices:
            raise ValueError("game_index values must be unique within a shard")
        game_indices.add(game_index)


def _schema_with_metadata(metadata: Mapping[str, object]) -> pa.Schema:
    missing = _REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(
            "metadata is missing required keys: " + ", ".join(sorted(missing))
        )

    encoded_metadata = {
        str(key).encode(): str(value).encode()
        for key, value in metadata.items()
    }
    return GAME_SCHEMA.with_metadata(encoded_metadata)


def write_games(
    path: str | Path,
    games: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> None:
    """Write a new immutable zstd-compressed raw-game Parquet shard.

    Raises ``FileExistsError`` if ``path`` already exists, rather than
    replacing a previously generated shard.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)

    games = list(games)
    _validate_games(games)
    table = pa.Table.from_pylist(games, schema=_schema_with_metadata(metadata))
    pq.write_table(table, destination, compression="zstd")


def read_games(path: str | Path) -> pa.Table:
    """Read one immutable raw-game shard, including its schema metadata."""
    return pq.read_table(path)
