import pytest

from amadeus_counterpoint.evaluation.bootstrap import paired_bootstrap


def test_paired_bootstrap_is_seeded_and_resamples_whole_rows_for_all_conditions():
    real_games = {
        "A_WHITE": [
            {"game_id": "a1", "result": 1, "moves_uci": ["e2e4"]},
            {"game_id": "a2", "result": -1, "moves_uci": ["d2d4"]},
        ],
        "B_WHITE": [
            {"game_id": "b1", "result": 0, "moves_uci": ["c2c4"]},
            {"game_id": "b2", "result": 1, "moves_uci": ["g1f3"]},
            {"game_id": "b3", "result": -1, "moves_uci": ["b1c3"]},
        ],
    }
    generated_games = {
        condition: {"A_WHITE": [{"score": score}], "B_WHITE": [{"score": score}]}
        for condition, score in {"GG": 9, "AG": 5, "GB": 3, "AB": 1}.items()
    }
    calls = []

    def distance(generated, real):
        calls.append((generated["A_WHITE"][0]["score"], id(real), real))
        assert [len(real[orientation]) for orientation in ("A_WHITE", "B_WHITE")] == [2, 3]
        for games in real.values():
            for game in games:
                assert game["moves_uci"] == {
                    "a1": ["e2e4"], "a2": ["d2d4"], "b1": ["c2c4"],
                    "b2": ["g1f3"], "b3": ["b1c3"],
                }[game["game_id"]]
        return generated["A_WHITE"][0]["score"]

    result = paired_bootstrap(real_games, generated_games, distance, replicates=4, seed=17)
    repeat_calls = []

    def repeat_distance(generated, real):
        repeat_calls.append(
            tuple(game["game_id"] for orientation in ("A_WHITE", "B_WHITE") for game in real[orientation])
        )
        return generated["A_WHITE"][0]["score"]

    repeat = paired_bootstrap(
        real_games, generated_games, repeat_distance, replicates=4, seed=17
    )

    assert result["distances"] == repeat["distances"]
    assert {condition: len(values) for condition, values in result["distances"].items()} == {
        "GG": 4, "AG": 4, "GB": 4, "AB": 4,
    }
    assert result["contrasts"] == {
        "GG_minus_AB": [8, 8, 8, 8],
        "AG_minus_AB": [4, 4, 4, 4],
        "GB_minus_AB": [2, 2, 2, 2],
    }
    for replicate in range(4):
        assert len({calls[replicate * 4 + offset][1] for offset in range(4)}) == 1
    assert [
        tuple(
            game["game_id"]
            for orientation in ("A_WHITE", "B_WHITE")
            for game in calls[replicate * 4][2][orientation]
        )
        for replicate in range(4)
    ] == repeat_calls[::4]


def test_paired_bootstrap_rejects_an_empty_real_orientation():
    generated_games = {
        condition: {"A_WHITE": [], "B_WHITE": []}
        for condition in ("GG", "AG", "GB", "AB")
    }

    with pytest.raises(ValueError, match="A_WHITE"):
        paired_bootstrap({"A_WHITE": [], "B_WHITE": [{"game_id": "b"}]}, generated_games, lambda *_: 0)


def test_paired_bootstrap_supports_only_available_conditions():
    real_games = {
        "A_WHITE": [{"game_id": "a"}],
        "B_WHITE": [{"game_id": "b"}],
    }
    generated_games = {
        condition: {"A_WHITE": [{"score": score}], "B_WHITE": [{"score": score}]}
        for condition, score in {"GG": 3, "AG": 2, "AB": 1}.items()
    }

    result = paired_bootstrap(
        real_games,
        generated_games,
        lambda generated, _real: generated["A_WHITE"][0]["score"],
        replicates=2,
        seed=7,
    )

    assert set(result["distances"]) == {"GG", "AG", "AB"}
    assert result["contrasts"] == {
        "GG_minus_AB": [2, 2],
        "AG_minus_AB": [1, 1],
    }
