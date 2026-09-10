import json

from amadeus_counterpoint.data.broadcast_targets import load_quarantined_game_urls, match_target

TARGETS = [
    {
        "player_id": 0,
        "canonical_name": "Player Zero",
        "fide_id": "1000001",
        "aliases": ["Zero, Player", "Player Zero"],
    },
    {
        "player_id": 1,
        "canonical_name": "Player One",
        "fide_id": "1000002",
        "aliases": ["One, Player", "Player One"],
    },
]


def test_exact_fide_id_match():
    assert match_target("anything, really", "1000001", TARGETS) == 0


def test_exact_alias_match_when_fide_id_missing():
    assert match_target("Zero, Player", "", TARGETS) == 0


def test_fide_id_takes_priority_over_a_matching_alias_string():
    # A name string that happens to match player 0's alias, but tagged with
    # player 1's FIDE ID, must resolve to player 1 -- FIDE ID first, always.
    assert match_target("Zero, Player", "1000002", TARGETS) == 1


def test_unknown_fide_id_falls_through_to_alias_check():
    assert match_target("Player One", "5555555", TARGETS) == 1


def test_no_match_returns_none():
    assert match_target("Someone Else", "9999999", TARGETS) is None


def test_no_substring_match():
    assert match_target("Player Zero Junior", "", TARGETS) is None


def test_no_fuzzy_match():
    assert match_target("Zero, Playr", "", TARGETS) is None


# --- load_quarantined_game_urls -------------------------------------------


def test_load_quarantined_game_urls_pools_all_groups(tmp_path):
    path = tmp_path / "quarantine.json"
    path.write_text(
        json.dumps(
            {
                "groups": [
                    {"opponent_handle": "a", "game_urls": ["url1", "url2"]},
                    {"opponent_handle": "b", "game_urls": ["url3"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_quarantined_game_urls(path) == frozenset({"url1", "url2", "url3"})


def test_load_quarantined_game_urls_empty_groups_is_empty_set(tmp_path):
    path = tmp_path / "quarantine.json"
    path.write_text(json.dumps({"groups": []}), encoding="utf-8")

    assert load_quarantined_game_urls(path) == frozenset()
