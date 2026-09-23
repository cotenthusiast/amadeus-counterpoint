import chess

from amadeus_counterpoint.evaluation.bootstrap_wiring import (
    bootstrap_dyad,
    opening_metric_for_index,
    wdl_metric,
)
from amadeus_counterpoint.evaluation.metrics.openings import (
    load_opening_index,
    opening_orientation_distance,
)


def _real_games():
    return {
        "A_WHITE": [
            {"result": "1-0", "censored": False, "orientation": "A_WHITE"},
            {"result": "0-1", "censored": False, "orientation": "A_WHITE"},
        ],
        "B_WHITE": [
            {"result": "1-0", "censored": False, "orientation": "B_WHITE"},
            {"result": "1/2-1/2", "censored": False, "orientation": "B_WHITE"},
        ],
    }


def _generated_by_condition():
    fixed = {
        "A_WHITE": [{"result": "1-0", "censored": False, "orientation": "A_WHITE"}],
        "B_WHITE": [{"result": "1-0", "censored": False, "orientation": "B_WHITE"}],
    }
    return {condition: fixed for condition in ("GG", "AG", "GB", "AB")}


def test_bootstrap_dyad_is_deterministic_under_a_fixed_seed():
    real = _real_games()
    generated = _generated_by_condition()

    result_a = bootstrap_dyad(real, generated, wdl_metric, replicates=50, seed=7)
    result_b = bootstrap_dyad(real, generated, wdl_metric, replicates=50, seed=7)

    assert result_a == result_b


def test_bootstrap_dyad_returns_point_estimate_interval_and_contrasts():
    real = _real_games()
    generated = _generated_by_condition()

    result = bootstrap_dyad(real, generated, wdl_metric, replicates=50, seed=1)

    assert result["replicates"] == 50
    assert set(result["point_estimate"]) == {"GG", "AG", "GB", "AB"}
    assert set(result["contrast_point_estimate"]) == {"GG_minus_AB", "AG_minus_AB", "GB_minus_AB"}
    assert set(result["distance_interval"]) == {"GG", "AG", "GB", "AB"}

    # Every condition generated the exact same fixed games here, so every
    # contrast (a difference between two identical point estimates) is zero.
    assert all(value == 0.0 for value in result["contrast_point_estimate"].values())

    for lo, hi in result["distance_interval"].values():
        assert lo <= hi


def test_bootstrap_dyad_resamples_whole_games_not_moves():
    # A metric that only ever looks at how many real games it received --
    # if bootstrap resampled moves instead of whole games, this count would
    # not stay fixed at the real sample size across replicates.
    real = _real_games()
    real_sample_size = len(real["A_WHITE"]) + len(real["B_WHITE"])

    seen_sizes = set()

    def counting_metric(generated, resampled_real):
        seen_sizes.add(len(resampled_real["A_WHITE"]) + len(resampled_real["B_WHITE"]))
        return 0.0

    bootstrap_dyad(real, _generated_by_condition(), counting_metric, replicates=20, seed=3)

    assert seen_sizes == {real_sample_size}


def _epd(moves):
    board = chess.Board()
    for move in moves:
        board.push_uci(move)
    return board.epd()


def _write_openings(path, rows):
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("eco", "name", "pgn", "uci", "epd"), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_opening_metric_for_index_matches_uncached_orientation_distance(tmp_path):
    """`opening_metric_for_index`'s cached metric must be numerically identical
    to calling `opening_orientation_distance` directly, for every distinct
    generated-games object and every distinct real resample -- proving the
    cache-by-object-identity performance fix changes nothing about the
    statistic, only how many times the fixed synthetic side is recomputed.
    """
    path = tmp_path / "all.tsv"
    _write_openings(path, [
        {"eco": "C20", "name": "King's Pawn Game: Line", "pgn": "", "uci": "e2e4", "epd": _epd(["e2e4"])},
        {"eco": "B20", "name": "Sicilian Defense: Line", "pgn": "", "uci": "e2e4 c7c5", "epd": _epd(["e2e4", "c7c5"])},
        {"eco": "D00", "name": "Queen's Pawn Game: Line", "pgn": "", "uci": "d2d4", "epd": _epd(["d2d4"])},
    ])
    index = load_opening_index(path, "4b8622759e7ae6f93f011cc6c83a3823401ab45e")

    generated_a = {
        "A_WHITE": [{"moves": ["e2e4"]}, {"moves": ["e2e4", "c7c5"]}, {"moves": ["e2e4"]}],
        "B_WHITE": [{"moves": ["d2d4"]}, {"moves": ["e2e4", "c7c5"]}],
    }
    generated_b = {
        "A_WHITE": [{"moves": ["d2d4"]}],
        "B_WHITE": [{"moves": ["e2e4"]}, {"moves": ["d2d4"]}],
    }
    real_samples = [
        {
            "A_WHITE": [{"moves": ["e2e4", "c7c5"]}],
            "B_WHITE": [{"moves": ["d2d4"]}, {"moves": ["e2e4"]}],
        },
        {
            "A_WHITE": [{"moves": ["e2e4"]}, {"moves": ["d2d4"]}],
            "B_WHITE": [{"moves": ["e2e4", "c7c5"]}],
        },
        {
            "A_WHITE": [{"moves": ["d2d4"]}, {"moves": ["d2d4"]}, {"moves": ["e2e4"]}],
            "B_WHITE": [{"moves": ["e2e4", "c7c5"]}, {"moves": ["e2e4"]}],
        },
    ]

    cached_metric = opening_metric_for_index(index)

    for generated in (generated_a, generated_b):
        for real in real_samples:
            expected = opening_orientation_distance(generated, real, index)["mean"]
            actual = cached_metric(generated, real)
            assert actual == expected

    # Re-querying an already-cached generated object with a fresh real
    # sample still matches the uncached computation -- the cache must not
    # go stale or leak state between distinct generated objects.
    for real in reversed(real_samples):
        expected = opening_orientation_distance(generated_a, real, index)["mean"]
        assert cached_metric(generated_a, real) == expected
