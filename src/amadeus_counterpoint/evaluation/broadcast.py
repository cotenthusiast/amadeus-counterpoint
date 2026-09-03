"""Read-only resolution of sealed Broadcast manifests against the canonical
normalized Broadcast corpus.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def load_manifest(path: str | Path) -> pa.Table:
    """Read a prepared Broadcast manifest.

    Manifests (e.g. `sealed_dyads/<A>__<B>.parquet`) are references keyed by
    `game_id`, not full copies of the canonical games -- some deliberately
    omit behavioral columns such as `result` and `moves_uci`.

    Args:
        path: Path to a manifest Parquet file.

    Returns:
        The manifest as a table, in its original row order.

    Raises:
        ValueError: If the manifest has no `game_id` column.
    """
    manifest = pq.read_table(path)

    if "game_id" not in manifest.column_names:
        raise ValueError(f"Manifest {path} has no 'game_id' column")

    return manifest


def resolve_manifest(manifest_path: str | Path, broadcast_root: str | Path) -> pa.Table:
    """Resolve a manifest's `game_id`s to their full canonical Broadcast rows.

    The canonical corpus is read as a `month=YYYY-MM`-partitioned Parquet
    dataset rooted at `broadcast_root` (e.g. `normalized/broadcast/`); only
    the rows the manifest actually references are pulled in.

    Args:
        manifest_path: Path to the manifest Parquet file.
        broadcast_root: Root directory of the canonical normalized Broadcast
            dataset.

    Returns:
        The canonical Broadcast rows for the manifest's `game_id`s, in the
        manifest's original row order.

    Raises:
        ValueError: If the manifest or canonical dataset has no `game_id`
            column, if the canonical dataset has duplicate `game_id`s among
            the rows the manifest references, or if any manifest `game_id`
            is missing from the canonical dataset.
    """
    manifest = load_manifest(manifest_path)
    manifest_ids = manifest.column("game_id").to_pylist()

    canonical = ds.dataset(broadcast_root, format="parquet", partitioning="hive")

    if "game_id" not in canonical.schema.names:
        raise ValueError(
            f"Canonical Broadcast dataset at {broadcast_root} has no 'game_id' column"
        )

    matched = canonical.to_table(filter=ds.field("game_id").isin(manifest_ids))
    matched_ids = matched.column("game_id").to_pylist()

    if len(matched_ids) != len(set(matched_ids)):
        raise ValueError(
            f"Canonical Broadcast dataset at {broadcast_root} has duplicate "
            "game_id values among the games referenced by "
            f"{manifest_path}"
        )

    row_by_id = {game_id: row for row, game_id in enumerate(matched_ids)}

    missing = [game_id for game_id in manifest_ids if game_id not in row_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} game_id(s) in {manifest_path} are missing from "
            f"the canonical Broadcast dataset at {broadcast_root}, e.g. "
            f"{missing[:5]}"
        )

    row_indices = pa.array([row_by_id[game_id] for game_id in manifest_ids], type=pa.int64())
    return matched.take(row_indices)
