import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from amadeus_counterpoint.evaluation.broadcast import load_manifest


def test_load_manifest_rejects_duplicate_game_ids(tmp_path):
    path = tmp_path / "manifest.parquet"
    pq.write_table(pa.table({"game_id": ["game-1", "game-1"]}), path)

    with pytest.raises(ValueError, match="duplicate.*game_id"):
        load_manifest(path)
