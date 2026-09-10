"""One-shard-per-cell production artifact writing, on top of the existing
immutable-shard writer (`evaluation.artifacts`). A "cell" is one
(method, dyad, condition, orientation) combination -- exactly the unit
`artifacts.write_games` already requires unique `game_index` within.

Resumability is plain file-existence: a cell whose shard already exists is
considered complete and is never regenerated or overwritten. No database,
no job queue.
"""

from pathlib import Path

from amadeus_counterpoint.evaluation import artifacts


def cell_artifact_path(output_root, method: str, dyad: str, condition: str, orientation: str) -> Path:
    """The deterministic file path for one (method, dyad, condition, orientation) cell."""
    return Path(output_root) / method / dyad / f"{condition}_{orientation}.parquet"


def cell_is_complete(output_root, method: str, dyad: str, condition: str, orientation: str) -> bool:
    """True if that cell's shard has already been written."""
    return cell_artifact_path(output_root, method, dyad, condition, orientation).exists()


def write_cell(
    output_root,
    method: str,
    dyad: str,
    condition: str,
    orientation: str,
    games: list[dict],
    metadata: dict,
) -> Path:
    """Write one cell's games as one immutable shard, unless it already exists.

    `games` must all share this exact (dyad, condition, orientation) --
    not checked here beyond what `artifacts.write_games` already validates
    (condition/orientation values, unique `game_index`). Returns the path
    written to, or the existing path if the cell was already complete (in
    which case `games` is never touched -- callers should check
    `cell_is_complete` first to avoid generating games only to discard them).
    """
    path = cell_artifact_path(output_root, method, dyad, condition, orientation)

    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.write_games(path, games, metadata)
    return path
