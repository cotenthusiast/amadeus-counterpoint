from amadeus_counterpoint.data.sealed_dyads import (
    deduplicate_sealed_games,
    dyad_key,
    find_repeated_content,
    find_repeated_game_urls,
    group_sealed_games_by_dyad,
    iter_sealed_dyad_games,
)

TARGETS = [
    {
        "player_id": 0,
        "canonical_name": "Player Zero",
        "fide_id": "1000001",
        "aliases": ["Zero, Player"],
    },
    {
        "player_id": 1,
        "canonical_name": "Player One",
        "fide_id": "1000002",
        "aliases": ["One, Player"],
    },
    {
        "player_id": 2,
        "canonical_name": "Player Two",
        "fide_id": "1000003",
        "aliases": ["Two, Player"],
    },
]


def _game(headers, moves="1. e4 e5 2. Nf3 Nc6"):
    """Build minimal PGN text for one game from a headers dict and movetext."""
    header_lines = "\n".join(f'[{k} "{v}"]' for k, v in headers.items())
    return f"{header_lines}\n\n{moves}\n\n"


BASE_HEADERS = {
    "Event": "Test",
    "White": "Some Nontarget",
    "Black": "Another Nontarget",
    "Date": "2021.01.01",
    "Result": "1-0",
    "WhiteElo": "2500",
    "BlackElo": "2400",
}


# --- iter_sealed_dyad_games ------------------------------------------------


def test_zero_target_game_is_ignored(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(_game(BASE_HEADERS), encoding="utf-8")

    stats = {}
    games = list(iter_sealed_dyad_games([path], TARGETS, stats=stats))

    assert games == []
    assert stats["games_scanned"] == 1
    assert stats["two_target_games"] == 0


def test_one_target_game_is_ignored(tmp_path):
    headers = {**BASE_HEADERS, "White": "Zero, Player", "WhiteFideId": "1000001"}
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    stats = {}
    games = list(iter_sealed_dyad_games([path], TARGETS, stats=stats))

    assert games == []
    assert stats["two_target_games"] == 0


def test_two_target_game_is_emitted(tmp_path):
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "One, Player",
        "BlackFideId": "1000002",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    stats = {}
    games = list(iter_sealed_dyad_games([path], TARGETS, stats=stats))

    assert len(games) == 1
    game = games[0]
    assert game["white_player_id"] == 0
    assert game["black_player_id"] == 1
    assert game["player_id_a"] == 0
    assert game["player_id_b"] == 1
    assert game["orientation"] == "A_WHITE"
    assert game["result"] == "1-0"
    assert game["moves"] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert game["white_elo"] == 2500
    assert game["black_elo"] == 2400
    assert game["date"] == "2021.01.01"

    assert stats["games_scanned"] == 1
    assert stats["two_target_games"] == 1
    assert stats["invalid_sealed_games"] == 0
    assert stats["self_pair_games"] == 0


def test_invalid_two_target_game_is_counted_not_emitted(tmp_path):
    # Missing BlackElo makes game_to_record() reject the whole game.
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "One, Player",
        "BlackFideId": "1000002",
    }
    del headers["BlackElo"]
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    stats = {}
    games = list(iter_sealed_dyad_games([path], TARGETS, stats=stats))

    assert games == []
    assert stats["two_target_games"] == 1
    assert stats["invalid_sealed_games"] == 1


def test_self_pair_is_excluded_and_counted_separately(tmp_path):
    # Both sides tagged with the same target's FIDE ID: a data anomaly, not
    # a real dyad game.
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "Zero, Player",
        "BlackFideId": "1000001",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    stats = {}
    games = list(iter_sealed_dyad_games([path], TARGETS, stats=stats))

    assert games == []
    assert stats["self_pair_games"] == 1
    assert stats["two_target_games"] == 0
    assert stats["invalid_sealed_games"] == 0


def test_both_color_orientations_are_handled(tmp_path):
    a_white = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "One, Player",
        "BlackFideId": "1000002",
    }
    b_white = {
        **BASE_HEADERS,
        "White": "One, Player",
        "WhiteFideId": "1000002",
        "Black": "Zero, Player",
        "BlackFideId": "1000001",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(a_white) + _game(b_white), encoding="utf-8")

    games = list(iter_sealed_dyad_games([path], TARGETS))

    assert [game["orientation"] for game in games] == ["A_WHITE", "B_WHITE"]
    assert games[0]["white_player_id"] == 0
    assert games[1]["white_player_id"] == 1
    # Both games are still the same dyad regardless of who was White.
    assert dyad_key(games[0]["player_id_a"], games[0]["player_id_b"]) == dyad_key(
        games[1]["player_id_a"], games[1]["player_id_b"]
    )


def test_identity_matching_reuses_exact_fide_and_alias_semantics(tmp_path):
    # No WhiteFideId at all -> must fall back to the exact alias string,
    # exactly like data.broadcast_targets.match_target.
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "Black": "One, Player",
        "BlackFideId": "1000002",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    games = list(iter_sealed_dyad_games([path], TARGETS))

    assert len(games) == 1
    assert games[0]["white_player_id"] == 0
    assert games[0]["black_player_id"] == 1


def test_output_contains_fields_wdl_and_opening_metrics_need(tmp_path):
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "One, Player",
        "BlackFideId": "1000002",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    games = list(iter_sealed_dyad_games([path], TARGETS))

    assert games[0]["result"] == "1-0"
    assert games[0]["moves"] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert games[0]["orientation"] in ("A_WHITE", "B_WHITE")


# --- dyad_key ---------------------------------------------------------------


def test_dyad_key_is_stable_regardless_of_who_is_white():
    assert dyad_key(0, 1) == dyad_key(1, 0) == "0__1"


# --- group_sealed_games_by_dyad ---------------------------------------------


def test_group_sealed_games_by_dyad_splits_by_orientation_and_never_double_counts(
    tmp_path,
):
    dyad_a_b = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "One, Player",
        "BlackFideId": "1000002",
    }
    dyad_a_c = {
        **BASE_HEADERS,
        "White": "Two, Player",
        "WhiteFideId": "1000003",
        "Black": "Zero, Player",
        "BlackFideId": "1000001",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(dyad_a_b) + _game(dyad_a_c), encoding="utf-8")

    games = list(iter_sealed_dyad_games([path], TARGETS))
    grouped = group_sealed_games_by_dyad(games)

    assert set(grouped.keys()) == {"0__1", "0__2"}
    assert len(grouped["0__1"]["A_WHITE"]) == 1
    assert len(grouped["0__1"]["B_WHITE"]) == 0
    # dyad_a_c has White=player 2 (the higher id), so it lands in B_WHITE.
    assert len(grouped["0__2"]["A_WHITE"]) == 0
    assert len(grouped["0__2"]["B_WHITE"]) == 1

    total_grouped_games = sum(
        len(buckets["A_WHITE"]) + len(buckets["B_WHITE"]) for buckets in grouped.values()
    )
    assert total_grouped_games == len(games) == 2


# --- duplicate detection -----------------------------------------------------


def test_find_repeated_game_urls_reports_but_does_not_remove():
    games = [
        {"game_url": "https://example.com/g1"},
        {"game_url": "https://example.com/g1"},
        {"game_url": "https://example.com/g2"},
        {"game_url": None},
    ]

    repeated = find_repeated_game_urls(games)

    assert repeated == {"https://example.com/g1": 2}


def test_find_repeated_content_catches_same_game_under_different_url():
    shared = {
        "white_player_id": 0,
        "black_player_id": 1,
        "date": "2021.01.01",
        "result": "1-0",
        "moves": ["e2e4", "e7e5"],
    }
    games = [
        {**shared, "game_url": "https://example.com/g1"},
        {**shared, "game_url": "https://example.com/g2"},
        {**shared, "moves": ["d2d4"], "game_url": "https://example.com/g3"},
    ]

    repeated = find_repeated_content(games)

    assert len(repeated) == 1
    assert list(repeated.values()) == [2]


# --- deduplicate_sealed_games ------------------------------------------------


def _sealed_game(**overrides):
    game = {
        "player_id_a": 0,
        "player_id_b": 1,
        "orientation": "A_WHITE",
        "white_player_id": 0,
        "black_player_id": 1,
        "white_elo": 2800,
        "black_elo": 2750,
        "result": "1-0",
        "moves": ["e2e4", "e7e5"],
        "date": "2021.01.01",
        "game_url": "https://example.com/g1",
    }
    game.update(overrides)
    return game


def test_deduplicate_keeps_the_first_occurrence_and_drops_later_duplicates():
    first = _sealed_game(game_url="https://example.com/g1")
    duplicate = _sealed_game(game_url="https://example.com/g2")

    deduplicated, report = deduplicate_sealed_games([first, duplicate])

    assert deduplicated == [first]
    assert report == {
        "total_before": 2,
        "total_after": 1,
        "duplicate_groups": 1,
        "extra_removed": 1,
    }


def test_deduplicate_is_deterministic_regardless_of_which_copy_is_listed_first():
    first = _sealed_game(game_url="https://example.com/g1")
    duplicate = _sealed_game(game_url="https://example.com/g2")

    forward, _ = deduplicate_sealed_games([first, duplicate])
    reversed_order, _ = deduplicate_sealed_games([duplicate, first])

    assert forward == [first]
    assert reversed_order == [duplicate]  # whichever is listed first is canonical


def test_deduplicate_preserves_non_duplicate_games_untouched():
    a = _sealed_game(game_url="https://example.com/g1", moves=["e2e4", "e7e5"])
    b = _sealed_game(game_url="https://example.com/g2", moves=["d2d4", "d7d5"])

    deduplicated, report = deduplicate_sealed_games([a, b])

    assert deduplicated == [a, b]
    assert report["duplicate_groups"] == 0
    assert report["extra_removed"] == 0


def test_deduplicate_does_not_merge_games_from_different_dyads_or_orientations():
    # Same moves/date/result, but different player identities -- must never collapse.
    dyad_a_b = _sealed_game(white_player_id=0, black_player_id=1)
    dyad_a_c = _sealed_game(white_player_id=0, black_player_id=2, player_id_b=2)
    swapped_orientation = _sealed_game(white_player_id=1, black_player_id=0)

    deduplicated, report = deduplicate_sealed_games(
        [dyad_a_b, dyad_a_c, swapped_orientation]
    )

    assert deduplicated == [dyad_a_b, dyad_a_c, swapped_orientation]
    assert report["extra_removed"] == 0


def test_deduplicate_never_empties_a_dyad(tmp_path):
    # A dyad's only two games happen to be exact-content duplicates; after
    # deduplication the dyad must still have exactly one surviving game, not zero.
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "One, Player",
        "BlackFideId": "1000002",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers) + _game(headers), encoding="utf-8")

    games = list(iter_sealed_dyad_games([path], TARGETS))
    assert len(games) == 2  # both copies are valid, identical-content games

    deduplicated, report = deduplicate_sealed_games(games)
    grouped = group_sealed_games_by_dyad(deduplicated)

    assert report == {
        "total_before": 2,
        "total_after": 1,
        "duplicate_groups": 1,
        "extra_removed": 1,
    }
    assert len(grouped["0__1"]["A_WHITE"]) == 1
