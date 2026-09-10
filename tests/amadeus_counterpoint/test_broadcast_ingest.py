from amadeus_counterpoint.data.broadcast_ingest import classify_game, iter_target_game_records

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
]


def _game(headers, moves="1. e4 e5 2. Nf3 Nc6"):
    """Build minimal PGN text for one game from a headers dict and movetext."""
    header_lines = "\n".join(f'[{k} "{v}"]' for k, v in headers.items())
    return f"{header_lines}\n\n{moves}\n\n"


BASE_HEADERS = {
    "Event": "Test",
    "White": "Some Nontarget",
    "Black": "Another Nontarget",
    "Result": "1-0",
    "WhiteElo": "2500",
    "BlackElo": "2400",
}


# --- classify_game -----------------------------------------------------


def test_classify_zero_target_game():
    classification, player_id, mover_color = classify_game(
        "Some Nontarget", "Another Nontarget", "", "", TARGETS
    )
    assert classification == "zero_target"
    assert player_id is None
    assert mover_color is None


def test_classify_one_target_game_white_mover():
    classification, player_id, mover_color = classify_game(
        "Zero, Player", "Another Nontarget", "1000001", "", TARGETS
    )
    assert classification == "one_target"
    assert player_id == 0
    assert mover_color == "white"


def test_classify_one_target_game_black_mover():
    classification, player_id, mover_color = classify_game(
        "Another Nontarget", "One, Player", "", "1000002", TARGETS
    )
    assert classification == "one_target"
    assert player_id == 1
    assert mover_color == "black"


def test_classify_target_vs_target_game_is_excluded():
    classification, player_id, mover_color = classify_game(
        "Zero, Player", "One, Player", "1000001", "1000002", TARGETS
    )
    assert classification == "target_vs_target"
    assert player_id is None
    assert mover_color is None


# --- iter_target_game_records --------------------------------------------


def test_iter_target_game_records_end_to_end(tmp_path):
    games = [
        _game({**BASE_HEADERS, "White": "Zero, Player", "WhiteFideId": "1000001"}),
        _game({**BASE_HEADERS, "White": "Some Nontarget", "Black": "Another Nontarget"}),
        _game(
            {
                **BASE_HEADERS,
                "White": "Zero, Player",
                "WhiteFideId": "1000001",
                "Black": "One, Player",
                "BlackFideId": "1000002",
            }
        ),
    ]
    path = tmp_path / "games.pgn"
    path.write_text("".join(games), encoding="utf-8")

    stats = {}
    records = list(iter_target_game_records([path], TARGETS, stats=stats))

    assert len(records) == 1
    assert records[0]["player_id"] == 0
    assert records[0]["mover_color"] == "white"
    assert records[0]["moves"] == ["e2e4", "e7e5", "g1f3", "b8c6"]
    assert records[0]["white_elo"] == 2500
    assert records[0]["black_elo"] == 2400

    assert stats["games_scanned"] == 3
    assert stats["one_target_games"] == 1
    assert stats["zero_target_games"] == 1
    assert stats["target_vs_target_games"] == 1
    assert stats["invalid_games"] == 0


def test_invalid_one_target_game_is_counted_not_yielded(tmp_path):
    # Missing BlackElo makes game_to_record() reject the whole game.
    headers = {**BASE_HEADERS, "White": "Zero, Player", "WhiteFideId": "1000001"}
    del headers["BlackElo"]
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    stats = {}
    records = list(iter_target_game_records([path], TARGETS, stats=stats))

    assert records == []
    assert stats["one_target_games"] == 1
    assert stats["invalid_games"] == 1


def test_alias_fallback_used_when_fide_id_missing(tmp_path):
    headers = {**BASE_HEADERS, "White": "Zero, Player"}  # no WhiteFideId at all
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    records = list(iter_target_game_records([path], TARGETS))

    assert len(records) == 1
    assert records[0]["player_id"] == 0


# --- quarantined_game_urls ------------------------------------------------


def test_quarantined_game_is_excluded_even_though_one_side_is_a_target(tmp_path):
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "Black": "unverified_handle",
        "GameURL": "https://lichess.org/broadcast/example/-/abc/def",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    stats = {}
    records = list(
        iter_target_game_records(
            [path],
            TARGETS,
            stats=stats,
            quarantined_game_urls=frozenset({"https://lichess.org/broadcast/example/-/abc/def"}),
        )
    )

    assert records == []
    assert stats["quarantined_games"] == 1
    assert stats["one_target_games"] == 0
    assert stats["games_scanned"] == 1


def test_quarantine_check_is_opt_in_and_defaults_to_excluding_nothing(tmp_path):
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "GameURL": "https://lichess.org/broadcast/example/-/abc/def",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    records = list(iter_target_game_records([path], TARGETS))

    assert len(records) == 1


def test_non_quarantined_game_with_a_quarantine_list_configured_is_unaffected(tmp_path):
    headers = {
        **BASE_HEADERS,
        "White": "Zero, Player",
        "WhiteFideId": "1000001",
        "GameURL": "https://lichess.org/broadcast/example/-/abc/other",
    }
    path = tmp_path / "games.pgn"
    path.write_text(_game(headers), encoding="utf-8")

    stats = {}
    records = list(
        iter_target_game_records(
            [path],
            TARGETS,
            stats=stats,
            quarantined_game_urls=frozenset({"https://lichess.org/broadcast/example/-/abc/def"}),
        )
    )

    assert len(records) == 1
    assert stats["quarantined_games"] == 0
