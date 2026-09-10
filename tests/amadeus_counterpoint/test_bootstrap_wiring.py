from amadeus_counterpoint.evaluation.bootstrap_wiring import bootstrap_dyad, wdl_metric


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
