import csv

import chess
import pytest

from amadeus_counterpoint.evaluation.metrics.openings import (
    UNKNOWN_OPENING_FAMILY,
    classify_opening_family,
    load_opening_index,
    opening_distribution,
    opening_orientation_distance,
)
from amadeus_counterpoint.evaluation.metrics.wdl import (
    A_WIN,
    B_WIN,
    DRAW,
    outcome_for_a_b,
    wdl_orientation_distance,
    wdl_summary,
)


def _epd(moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board.epd()


def _write_openings(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("eco", "name", "pgn", "uci", "epd"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("orientation", "result", "expected"),
    [
        ("A_WHITE", "1-0", A_WIN),
        ("A_WHITE", "0-1", B_WIN),
        ("B_WHITE", "1-0", B_WIN),
        ("B_WHITE", "0-1", A_WIN),
        ("B_WHITE", "1/2-1/2", DRAW),
    ],
)
def test_wdl_outcome_maps_pgn_results_from_a_and_b_perspective(orientation, result, expected):
    assert outcome_for_a_b(result, orientation) == expected


def test_wdl_summary_excludes_censored_games_but_reports_them():
    summary = wdl_summary([
        {"result": "1-0", "censored": False, "orientation": "A_WHITE"},
        {"result": None, "censored": True, "orientation": "A_WHITE"},
    ])

    assert summary == {
        "distribution": {A_WIN: 1.0, DRAW: 0.0, B_WIN: 0.0},
        "completed_count": 1,
        "censored_count": 1,
        "completed_rate": 0.5,
        "censored_rate": 0.5,
    }


def test_wdl_summary_reports_zero_rates_for_empty_input():
    summary = wdl_summary([])

    assert summary["completed_count"] == 0
    assert summary["censored_count"] == 0
    assert summary["completed_rate"] == 0.0
    assert summary["censored_rate"] == 0.0


def test_wdl_orientation_distance_uses_an_exact_equal_weight_mean():
    real = {
        "A_WHITE": [{"result": "1-0", "censored": False, "orientation": "A_WHITE"}],
        "B_WHITE": [{"result": "1-0", "censored": False, "orientation": "B_WHITE"}],
    }
    generated = {
        "A_WHITE": [{"result": "0-1", "censored": False, "orientation": "A_WHITE"}],
        "B_WHITE": [{"result": "1-0", "censored": False, "orientation": "B_WHITE"}],
    }

    report = wdl_orientation_distance(generated, real)

    assert report["A_WHITE"]["distance"] == 1.0
    assert report["B_WHITE"]["distance"] == 0.0
    assert report["mean"] == 0.5
    assert report["A_WHITE"]["generated"] == {
        "distribution": {A_WIN: 0.0, DRAW: 0.0, B_WIN: 1.0},
        "completed_count": 1,
        "censored_count": 0,
        "completed_rate": 1.0,
        "censored_rate": 0.0,
    }
    assert report["A_WHITE"]["real"] == {
        "distribution": {A_WIN: 1.0, DRAW: 0.0, B_WIN: 0.0},
        "completed_count": 1,
        "censored_count": 0,
        "completed_rate": 1.0,
        "censored_rate": 0.0,
    }


def test_wdl_summary_rejects_a_non_censored_invalid_result():
    with pytest.raises(ValueError, match="result"):
        wdl_summary([{"result": "*", "censored": False, "orientation": "A_WHITE"}])


def test_wdl_orientation_distance_rejects_an_orientation_without_completed_games():
    generated = {
        "A_WHITE": [{"result": None, "censored": True, "orientation": "A_WHITE"}],
        "B_WHITE": [{"result": "1-0", "censored": False, "orientation": "B_WHITE"}],
    }
    real = {
        "A_WHITE": [{"result": "1-0", "censored": False, "orientation": "A_WHITE"}],
        "B_WHITE": [{"result": "1-0", "censored": False, "orientation": "B_WHITE"}],
    }

    with pytest.raises(ValueError, match="no completed games"):
        wdl_orientation_distance(generated, real)


def test_opening_index_retains_pin_and_classifies_latest_recognized_epd(tmp_path):
    path = tmp_path / "all.tsv"
    _write_openings(path, [
        {"eco": "A00", "name": "Comma Family, Quiet Line", "pgn": "", "uci": "", "epd": chess.Board().epd()},
        {"eco": "C20", "name": "King's Pawn Game: Wayward", "pgn": "", "uci": "e2e4", "epd": _epd(["e2e4"])},
        {"eco": "C20", "name": "Open Game: King's Pawn", "pgn": "", "uci": "e2e4 e7e5", "epd": _epd(["e2e4", "e7e5"])},
    ])
    index = load_opening_index(path, "4b8622759e7ae6f93f011cc6c83a3823401ab45e")

    assert index["commit"] == "4b8622759e7ae6f93f011cc6c83a3823401ab45e"
    assert classify_opening_family(["e2e4", "e7e5"], index) == "Open Game"
    assert classify_opening_family([], index) == "Comma Family, Quiet Line"
    with pytest.raises(ValueError, match="40-character hexadecimal"):
        load_opening_index(path, "master")


def test_opening_classification_handles_transpositions_and_unknown(tmp_path):
    path = tmp_path / "all.tsv"
    transposed = ["g1f3", "d7d5", "d2d4", "g8f6"]
    _write_openings(path, [{
        "eco": "A00", "name": "Réti Opening: Transposed", "pgn": "", "uci": "", "epd": _epd(transposed),
    }])
    index = load_opening_index(path, "4b8622759e7ae6f93f011cc6c83a3823401ab45e")

    assert classify_opening_family(["d2d4", "g8f6", "g1f3", "d7d5"], index) == "Réti Opening"
    assert classify_opening_family(["e2e4"], index) == UNKNOWN_OPENING_FAMILY


def test_opening_distribution_and_distance_use_union_support_and_equal_orientations(tmp_path):
    path = tmp_path / "all.tsv"
    _write_openings(path, [
        {"eco": "C20", "name": "King's Pawn Game: Line", "pgn": "", "uci": "e2e4", "epd": _epd(["e2e4"])},
        {"eco": "B20", "name": "Sicilian Defense: Line", "pgn": "", "uci": "e2e4 c7c5", "epd": _epd(["e2e4", "c7c5"])},
    ])
    index = load_opening_index(path, "4b8622759e7ae6f93f011cc6c83a3823401ab45e")
    generated = {
        "A_WHITE": [{"moves": ["e2e4"]}],
        "B_WHITE": [{"moves": ["e2e4", "c7c5"]}],
    }
    real = {
        "A_WHITE": [{"moves": ["e2e4", "c7c5"]}],
        "B_WHITE": [{"moves": ["e2e4", "c7c5"]}],
    }

    assert opening_distribution(generated["A_WHITE"], index) == {"King's Pawn Game": 1.0}
    assert opening_orientation_distance(generated, real, index) == {
        "A_WHITE": 1.0,
        "B_WHITE": 0.0,
        "mean": 0.5,
    }


@pytest.mark.parametrize("empty_sample", ("generated", "real"))
@pytest.mark.parametrize("orientation", ("A_WHITE", "B_WHITE"))
def test_opening_orientation_distance_rejects_empty_samples(
    tmp_path, empty_sample, orientation
):
    path = tmp_path / "all.tsv"
    _write_openings(path, [])
    index = load_opening_index(path, "4b8622759e7ae6f93f011cc6c83a3823401ab45e")
    generated = {
        "A_WHITE": [{"moves": []}],
        "B_WHITE": [{"moves": []}],
    }
    real = {
        "A_WHITE": [{"moves": []}],
        "B_WHITE": [{"moves": []}],
    }
    (generated if empty_sample == "generated" else real)[orientation] = []

    with pytest.raises(ValueError, match=rf"{orientation}.*{empty_sample}.*no games"):
        opening_orientation_distance(generated, real, index)
