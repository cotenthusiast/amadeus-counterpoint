from amadeus_counterpoint.evaluation import artifacts
from amadeus_counterpoint.evaluation.generation.production import cell_artifact_path, cell_is_complete, write_cell

_METADATA = {"protocol_version": "1", "code_commit": "abc123", "root_seed": "1", "temperature": "1.0"}


def _game(game_index=0):
    return {
        "game_index": game_index, "seed": 1, "dyad": "0__1", "condition": "AB",
        "orientation": "A_WHITE", "result": "1-0", "censored": False,
        "moves": ["e2e4"], "white_elo": 1600.0, "black_elo": 1800.0,
        "checkpoint_identity": "ckpt", "white_representation_identity": "p0",
        "black_representation_identity": "p1",
    }


def test_cell_artifact_path_is_deterministic_and_distinct_per_cell(tmp_path):
    path_a = cell_artifact_path(tmp_path, "method1", "0__1", "AB", "A_WHITE")
    path_b = cell_artifact_path(tmp_path, "method1", "0__1", "AB", "A_WHITE")
    assert path_a == path_b

    assert path_a != cell_artifact_path(tmp_path, "method1", "0__1", "AB", "B_WHITE")
    assert path_a != cell_artifact_path(tmp_path, "method1", "0__1", "GG", "A_WHITE")
    assert path_a != cell_artifact_path(tmp_path, "method1", "0__2", "AB", "A_WHITE")
    assert path_a != cell_artifact_path(tmp_path, "method2", "0__1", "AB", "A_WHITE")


def test_write_cell_creates_a_readable_shard_and_marks_the_cell_complete(tmp_path):
    assert not cell_is_complete(tmp_path, "method1", "0__1", "AB", "A_WHITE")

    path = write_cell(tmp_path, "method1", "0__1", "AB", "A_WHITE", [_game()], _METADATA)

    assert cell_is_complete(tmp_path, "method1", "0__1", "AB", "A_WHITE")
    assert artifacts.read_games(path).num_rows == 1


def test_write_cell_skips_an_already_complete_cell_without_overwriting(tmp_path):
    write_cell(tmp_path, "method1", "0__1", "AB", "A_WHITE", [_game()], _METADATA)

    # Different games passed the second time -- must be ignored, not
    # merged or used to overwrite the existing shard.
    path = write_cell(
        tmp_path, "method1", "0__1", "AB", "A_WHITE", [_game(0), _game(1)], _METADATA
    )

    assert artifacts.read_games(path).num_rows == 1
