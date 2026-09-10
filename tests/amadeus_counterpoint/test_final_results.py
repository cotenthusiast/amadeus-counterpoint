import pytest

from amadeus_counterpoint.evaluation.final_results import build_final_results, read_final_results, write_final_results


def _provenance(**overrides):
    base = {
        "base_checkpoint_identity": "base-1",
        "personalization_checkpoint_identity": "m1-1",
        "representative_elo_identity": "elo-1",
        "opening_taxonomy_commit": "4b8622759e7ae6f93f011cc6c83a3823401ab45e",
        "root_seed": "1",
        "games_per_orientation": 5000,
        "protocol_version": "1",
        "code_commit": "abc",
    }
    base.update(overrides)
    return base


def _results_table():
    def report(distance):
        return {"mean": distance}

    return {
        "0__1": {
            condition: {"wdl": report(0.1), "opening": report(0.2)}
            for condition in ("GG", "AG", "GB", "AB")
        },
    }


def test_build_final_results_requires_provenance_keys():
    with pytest.raises(ValueError, match="root_seed"):
        build_final_results("method1", _results_table(), {}, {"base_checkpoint_identity": "x"})


def test_build_final_results_computes_equal_weight_aggregate():
    results = build_final_results("method1", _results_table(), {}, _provenance())

    assert results["method"] == "method1"
    assert results["equal_weight_aggregate"]["GG"]["wdl"] == 0.1
    assert results["equal_weight_aggregate"]["AB"]["opening"] == 0.2
    assert results["provenance"]["opening_taxonomy_commit"] == _provenance()["opening_taxonomy_commit"]


def test_final_results_round_trip_through_json(tmp_path):
    results = build_final_results("method1", _results_table(), {"0__1": {"note": "example"}}, _provenance())

    path = tmp_path / "results.json"
    write_final_results(results, path)
    read_back = read_final_results(path)

    assert read_back == results
