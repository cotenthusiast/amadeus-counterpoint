import csv

from amadeus_counterpoint.evaluation.aggregate import (
    aggregate_equal_dyad_weight,
    build_results_table,
    compute_condition_metrics,
)
from amadeus_counterpoint.evaluation.metrics.openings import load_opening_index

_COMMIT = "4b8622759e7ae6f93f011cc6c83a3823401ab45e"


def _write_openings(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("eco", "name", "pgn", "uci", "epd"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _game(result, orientation, moves=("e2e4",)):
    return {"result": result, "censored": False, "orientation": orientation, "moves": list(moves)}


def test_compute_condition_metrics_wires_wdl_and_opening_with_orientation_mean(tmp_path):
    path = tmp_path / "all.tsv"
    _write_openings(path, [])
    opening_index = load_opening_index(path, _COMMIT)

    synthetic = {
        "A_WHITE": [_game("1-0", "A_WHITE")],
        "B_WHITE": [_game("1-0", "B_WHITE")],
    }
    real = {
        "A_WHITE": [_game("1-0", "A_WHITE")],  # matches -> distance 0
        "B_WHITE": [_game("0-1", "B_WHITE")],  # mismatches -> distance 1
    }

    metrics = compute_condition_metrics(synthetic, real, opening_index)

    assert metrics["wdl"]["A_WHITE"]["distance"] == 0.0
    assert metrics["wdl"]["B_WHITE"]["distance"] == 1.0
    assert metrics["wdl"]["mean"] == 0.5  # equal 50/50 orientation combination
    assert "opening" in metrics


def test_aggregate_equal_dyad_weight_ignores_real_sample_size(tmp_path):
    path = tmp_path / "all.tsv"
    _write_openings(path, [])
    opening_index = load_opening_index(path, _COMMIT)

    def cell(result):
        return {"A_WHITE": [_game(result, "A_WHITE")], "B_WHITE": [_game(result, "B_WHITE")]}

    synthetic_games = {
        "0__1": {c: cell("1-0") for c in ("GG", "AG", "GB", "AB")},
        "0__2": {c: cell("1-0") for c in ("GG", "AG", "GB", "AB")},
    }
    real_games_by_dyad = {
        # Dyad 0__1: 1 real game, matches synthetic exactly -> distance 0.
        "0__1": cell("1-0"),
        # Dyad 0__2: 50 real games, all mismatching synthetic -> distance 1.
        # A high-game-count dyad must NOT dominate the equal-weight average.
        "0__2": {
            "A_WHITE": [_game("0-1", "A_WHITE")] * 50,
            "B_WHITE": [_game("0-1", "B_WHITE")] * 50,
        },
    }

    results_table = build_results_table(synthetic_games, real_games_by_dyad, opening_index)

    for condition in ("GG", "AG", "GB", "AB"):
        assert aggregate_equal_dyad_weight(results_table, condition, "wdl") == 0.5
