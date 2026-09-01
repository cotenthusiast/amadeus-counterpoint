import json

from amadeus_counterpoint.data.target_exclusion import (
    is_target_game,
    load_target_alias_report,
    load_target_aliases,
    normalize_username,
)


def test_normalize_username_strips_and_case_folds():
    assert normalize_username("  DrNykterstein ") == "drnykterstein"


def test_load_target_aliases_pools_both_tiers_and_normalizes_across_targets(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "canonical_name": "Player One",
                        "verified_usernames": ["FooBar"],
                        "conservative_candidate_usernames": [" baz "],
                    },
                    {"canonical_name": "Player Two", "verified_usernames": ["Qux"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    aliases = load_target_aliases(path)

    # Both tiers are excluded identically -- the leakage risk is
    # asymmetric, so a conservative_candidate is pooled exactly like a
    # verified alias.
    assert aliases == frozenset({"foobar", "baz", "qux"})


def test_load_target_aliases_with_no_usernames_yields_empty_set(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps({"targets": [{"canonical_name": "Player One", "verified_usernames": []}]}),
        encoding="utf-8",
    )

    assert load_target_aliases(path) == frozenset()


def test_load_target_alias_report_preserves_tiers_for_audit(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps(
            {
                "targets": [
                    {
                        "canonical_name": "Player One",
                        "verified_usernames": ["FooBar"],
                        "conservative_candidate_usernames": ["Baz", "Qux"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = load_target_alias_report(path)

    assert report == [
        {
            "canonical_name": "Player One",
            "verified_usernames": ["FooBar"],
            "conservative_candidate_usernames": ["Baz", "Qux"],
        }
    ]


def test_is_target_game_matches_either_player_case_insensitively():
    aliases = frozenset({"drnykterstein"})

    assert is_target_game("DrNykterstein", "someone", aliases)
    assert is_target_game("someone", "DRNYKTERSTEIN", aliases)
    assert not is_target_game("alice", "bob", aliases)


def test_is_target_game_canonical_real_name_is_not_a_match():
    # Canonical player names are provenance labels, never usernames -- a
    # game between "Magnus Carlsen" (as a literal PGN player name) and
    # someone else must not match an alias list of real Lichess usernames.
    aliases = frozenset({"drnykterstein"})

    assert not is_target_game("Magnus Carlsen", "someone", aliases)
