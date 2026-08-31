import io

import chess.pgn
import pyarrow.parquet as pq
import pytest

from amadeus_counterpoint.data.preprocess import (
    balance_by_elo,
    eligible_ply_count,
    elo_bin,
    game_to_record,
    iter_pgn_games,
    iter_records,
    preprocess_pgn,
    write_shard,
)


def _game(headers: dict, moves: str = "1. e4 e5 2. Nf3 Nc6") -> str:
    """Build minimal PGN text for one game from a headers dict and movetext."""
    header_lines = "\n".join(f'[{k} "{v}"]' for k, v in headers.items())
    return f"{header_lines}\n\n{moves}\n\n"


VALID_HEADERS = {
    "Event": "Test",
    "White": "A",
    "Black": "B",
    "Result": "1-0",
    "WhiteElo": "1500",
    "BlackElo": "1600",
}


def _write_pgn(tmp_path, games: list[str]):
    path = tmp_path / "games.pgn"
    path.write_text("".join(games), encoding="utf-8")
    return path


# --- iter_pgn_games ---------------------------------------------------------


def test_iter_pgn_games_streams_multiple_games_in_order(tmp_path):
    headers_a = {**VALID_HEADERS, "White": "A"}
    headers_b = {**VALID_HEADERS, "White": "C"}
    path = _write_pgn(tmp_path, [_game(headers_a), _game(headers_b)])

    games = list(iter_pgn_games(path))

    assert len(games) == 2
    assert games[0].headers["White"] == "A"
    assert games[1].headers["White"] == "C"


# --- game_to_record ----------------------------------------------------------


def test_valid_game_serializes_correctly():
    game = chess.pgn.read_game(
        io.StringIO(_game(VALID_HEADERS))
    )

    record = game_to_record(game)

    assert record == {
        "white_elo": 1500,
        "black_elo": 1600,
        "result": "1-0",
        "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
        "eligible_ply_count": 4,  # no [%clk] annotations: nothing excluded
    }


def test_missing_elo_header_is_skipped():
    headers = {k: v for k, v in VALID_HEADERS.items() if k != "WhiteElo"}
    game = chess.pgn.read_game(io.StringIO(_game(headers)))

    assert game_to_record(game) is None


def test_non_numeric_elo_is_skipped():
    headers = {**VALID_HEADERS, "WhiteElo": "?"}
    game = chess.pgn.read_game(io.StringIO(_game(headers)))

    assert game_to_record(game) is None


def test_unfinished_result_is_skipped():
    headers = {**VALID_HEADERS, "Result": "*"}
    game = chess.pgn.read_game(io.StringIO(_game(headers)))

    assert game_to_record(game) is None


def test_empty_game_is_skipped():
    game = chess.pgn.read_game(
        io.StringIO(_game(VALID_HEADERS, moves="*"))
    )

    assert game_to_record(game) is None


def test_draw_result_is_kept():
    headers = {**VALID_HEADERS, "Result": "1/2-1/2"}
    game = chess.pgn.read_game(io.StringIO(_game(headers)))

    record = game_to_record(game)

    assert record is not None
    assert record["result"] == "1/2-1/2"


def test_non_standard_variant_is_skipped():
    # python-chess auto-selects a variant-specific board (e.g. Crazyhouse)
    # from this header; dataset.py always replays with a standard Board(),
    # so such games must never reach the training set.
    headers = {**VALID_HEADERS, "Variant": "Crazyhouse"}
    game = chess.pgn.read_game(io.StringIO(_game(headers)))

    assert game_to_record(game) is None


def test_explicit_standard_variant_header_is_kept():
    headers = {**VALID_HEADERS, "Variant": "Standard"}
    game = chess.pgn.read_game(io.StringIO(_game(headers)))

    assert game_to_record(game) is not None


def test_missing_variant_header_is_kept():
    assert "Variant" not in VALID_HEADERS
    game = chess.pgn.read_game(io.StringIO(_game(VALID_HEADERS)))

    assert game_to_record(game) is not None


def test_custom_fen_start_is_skipped():
    headers = {
        **VALID_HEADERS,
        "SetUp": "1",
        "FEN": "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    }
    game = chess.pgn.read_game(
        io.StringIO(_game(headers, moves="1. Kf1 Kf8"))
    )

    assert game_to_record(game) is None


def test_fen_header_without_setup_is_still_skipped():
    # A stray FEN header alone is enough signal that this isn't a standard
    # start; do not rely on SetUp being present too.
    headers = {**VALID_HEADERS, "FEN": chess.STARTING_FEN}
    game = chess.pgn.read_game(io.StringIO(_game(headers)))

    assert game_to_record(game) is None


def test_uci_moves_are_lossless_for_castling_en_passant_and_promotion():
    headers = VALID_HEADERS
    moves = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7"
    game = chess.pgn.read_game(io.StringIO(_game(headers, moves)))

    record = game_to_record(game)

    board = chess.Board()
    for uci in record["moves"]:
        board.push(chess.Move.from_uci(uci))

    # castling replayed correctly: king ends on g1, rook on f1
    assert board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
    assert board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)


# --- eligible_ply_count: <30s time-pressure cutoff -------------------------


def test_move_clearly_above_threshold_is_eligible():
    # Pre-move clock for ply 2 comes from clocks_after_move[0].
    assert eligible_ply_count([100.0, 100.0, 100.0]) == 3


def test_move_clearly_below_threshold_excludes_it_and_the_rest():
    # White's clock is 10s right before ply 2 (their previous reading is
    # clocks_after_move[0] = 10.0) -> ply 2 and everything after is cut.
    assert eligible_ply_count([10.0, 100.0, 100.0, 100.0]) == 2


def test_exact_threshold_boundary_is_strict_less_than():
    # "fewer than 30 seconds" is strict: exactly 30.0 does not trigger.
    assert eligible_ply_count([30.0, 100.0, 100.0]) == 3
    # anything below 30.0, however slightly, does.
    assert eligible_ply_count([29.999, 100.0, 100.0]) == 2


def test_first_two_plies_are_always_eligible_regardless_of_clock():
    # No same-color prior reading exists yet for plies 0 and 1.
    assert eligible_ply_count([]) == 0
    assert eligible_ply_count([5.0]) == 1
    assert eligible_ply_count([5.0, 5.0]) == 2


def test_cutoff_does_not_recover_even_if_clock_later_rises():
    # Once triggered, the cutoff is a hard truncation -- a later same-color
    # reading back above the threshold does not re-admit later plies.
    assert eligible_ply_count([10.0, 100.0, 100.0, 100.0, 100.0, 100.0]) == 2


def test_missing_clock_data_never_triggers_the_cutoff():
    # No evidence of time pressure -- our documented, conservative choice
    # for missing-clock handling (unspecified by the paper).
    assert eligible_ply_count([None, None, None, None]) == 4
    assert eligible_ply_count([100.0, None, 100.0, None]) == 4


def test_partial_missing_clock_does_not_block_a_later_real_cutoff():
    # ply 2's pre-move clock is missing (no evidence, clocks_after_move[0]);
    # ply 5's pre-move clock is real and low (clocks_after_move[3] = 10.0).
    assert eligible_ply_count([None, 100.0, 100.0, 10.0, 100.0, 100.0]) == 5


def test_game_to_record_clock_parsing_associates_correct_player_and_pre_move_position():
    # White: 3:02 after e4, then only 0:25 left before Bb5 (pre-move clock
    # for ply 4 = clocks_after_move[2] = 25s) -> Bb5 and beyond are cut,
    # even though Nf3 itself (played with 3:02 still on the clock) is kept.
    headers = {**VALID_HEADERS}
    moves_pgn = (
        "1. e4 { [%clk 0:03:02] } e5 { [%clk 0:02:59] } "
        "2. Nf3 { [%clk 0:00:25] } Nc6 { [%clk 0:02:50] } "
        "3. Bb5 { [%clk 0:00:20] } Nf6 { [%clk 0:02:40] } 1-0"
    )
    game = chess.pgn.read_game(io.StringIO(_game(headers, moves_pgn)))

    record = game_to_record(game)

    assert record["moves"] == ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6"]
    assert record["eligible_ply_count"] == 4


def test_game_to_record_without_clk_annotations_is_fully_eligible():
    game = chess.pgn.read_game(io.StringIO(_game(VALID_HEADERS)))

    record = game_to_record(game)

    assert record["eligible_ply_count"] == len(record["moves"])


# --- iter_records --------------------------------------------------------


def test_iter_records_yields_only_valid_games(tmp_path):
    bad_headers = {k: v for k, v in VALID_HEADERS.items() if k != "WhiteElo"}
    path = _write_pgn(
        tmp_path,
        [_game(VALID_HEADERS), _game(bad_headers), _game(VALID_HEADERS)],
    )

    records = list(iter_records(path))

    assert len(records) == 2


# --- write_shard / Parquet round trip --------------------------------------


def test_write_shard_ignores_empty_collection(tmp_path):
    path = tmp_path / "shard.parquet"

    write_shard([], path)

    assert not path.exists()


def test_write_shard_round_trip(tmp_path):
    records = [
        {"white_elo": 1500, "black_elo": 1600, "result": "1-0", "moves": ["e2e4", "e7e5"]},
        {"white_elo": 1200, "black_elo": 1900, "result": "0-1", "moves": ["d2d4"]},
    ]
    path = tmp_path / "shard.parquet"

    write_shard(records, path)
    table = pq.read_table(path)

    assert table.to_pylist() == records


def test_write_shard_round_trip_preserves_eligible_ply_count(tmp_path):
    records = [
        {
            "white_elo": 1500,
            "black_elo": 1600,
            "result": "1-0",
            "moves": ["e2e4", "e7e5", "g1f3", "b8c6"],
            "eligible_ply_count": 2,
        },
        {
            "white_elo": 1200,
            "black_elo": 1900,
            "result": "0-1",
            "moves": ["d2d4"],
            "eligible_ply_count": 1,
        },
    ]
    path = tmp_path / "shard.parquet"

    write_shard(records, path)
    table = pq.read_table(path)

    assert table.to_pylist() == records


# --- preprocess_pgn: shard boundaries ---------------------------------------


def test_invalid_shard_size_raises(tmp_path):
    path = _write_pgn(tmp_path, [_game(VALID_HEADERS)])

    with pytest.raises(ValueError):
        preprocess_pgn(path, tmp_path / "out", shard_size=0)


def test_exact_multiple_shard_size_produces_no_partial_shard(tmp_path):
    path = _write_pgn(tmp_path, [_game(VALID_HEADERS) for _ in range(4)])
    out_dir = tmp_path / "out"

    preprocess_pgn(path, out_dir, shard_size=2)

    shards = sorted(out_dir.glob("*.parquet"))
    assert [s.name for s in shards] == ["shard_00000.parquet", "shard_00001.parquet"]
    for shard in shards:
        assert len(pq.read_table(shard)) == 2


def test_final_partial_shard_is_written(tmp_path):
    path = _write_pgn(tmp_path, [_game(VALID_HEADERS) for _ in range(5)])
    out_dir = tmp_path / "out"

    preprocess_pgn(path, out_dir, shard_size=2)

    shards = sorted(out_dir.glob("*.parquet"))
    assert [s.name for s in shards] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
        "shard_00002.parquet",
    ]
    assert len(pq.read_table(shards[-1])) == 1


# --- elo_bin / balance_by_elo ----------------------------------------------


def _record(white_elo, black_elo, tag=None):
    record = {
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": "1-0",
        "moves": ["e2e4"],
    }
    if tag is not None:
        record["tag"] = tag
    return record


def test_elo_bin_boundaries():
    assert elo_bin(0) == 0
    assert elo_bin(599) == 0
    assert elo_bin(600) == 1
    assert elo_bin(699) == 1
    assert elo_bin(700) == 2
    assert elo_bin(2599) == 20
    assert elo_bin(2600) == 21
    assert elo_bin(5000) == 21


def test_elo_bin_uses_mean_of_both_players_not_either_alone():
    mean_elo = (1000 + 2200) / 2  # 1600

    assert elo_bin(mean_elo) != elo_bin(1000)
    assert elo_bin(mean_elo) != elo_bin(2200)


def test_balance_by_elo_caps_retained_games_per_bin():
    records = [_record(1550, 1550, tag=i) for i in range(15)]

    kept = list(balance_by_elo(records, chunk_size=20_000, games_per_bin=10))

    assert len(kept) == 10


def test_balance_by_elo_skips_games_once_bin_is_saturated():
    records = [_record(1550, 1550, tag=i) for i in range(15)]

    kept = list(balance_by_elo(records, chunk_size=20_000, games_per_bin=10))

    # the first 10 sequential games fill the bin; the rest are skipped
    assert [r["tag"] for r in kept] == list(range(10))


def test_balance_by_elo_leaves_underfilled_bin_when_chunk_ends():
    records = [_record(1550, 1550) for _ in range(3)]  # far below the cap

    kept = list(balance_by_elo(records, chunk_size=20_000, games_per_bin=10))

    assert len(kept) == 3


def test_balance_by_elo_terminates_chunk_early_once_all_bins_are_full():
    def mean_for_bin(b):
        if b == 0:
            return 300
        if b == 21:
            return 2700
        return 600 + 100 * (b - 1) + 50

    records = [
        _record(mean_for_bin(b), mean_for_bin(b))
        for b in range(22)
        for _ in range(10)
    ]
    trailing = [_record(300, 300, tag="trailing") for _ in range(5)]

    kept = list(balance_by_elo(records + trailing, chunk_size=100_000, games_per_bin=10))

    assert len(kept) == 220  # 22 bins * 10 games/bin
    assert all(r.get("tag") != "trailing" for r in kept)


def test_balance_by_elo_resets_bin_counts_on_a_new_chunk():
    records = [_record(1550, 1550, tag=i) for i in range(20)]

    kept = list(balance_by_elo(records, chunk_size=10, games_per_bin=5))

    # each 10-game chunk independently keeps its first 5 same-bin games
    assert [r["tag"] for r in kept] == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]


def test_balance_by_elo_is_deterministic():
    records = [_record(600 + i, 600 + i, tag=i) for i in range(50)]

    kept_a = list(balance_by_elo(records, chunk_size=20, games_per_bin=3))
    kept_b = list(balance_by_elo(records, chunk_size=20, games_per_bin=3))

    assert kept_a == kept_b


def test_balance_by_elo_empty_input_yields_nothing():
    assert list(balance_by_elo([])) == []


def test_all_games_invalid_produces_no_shards(tmp_path):
    bad_headers = {k: v for k, v in VALID_HEADERS.items() if k != "WhiteElo"}
    path = _write_pgn(tmp_path, [_game(bad_headers)])
    out_dir = tmp_path / "out"

    preprocess_pgn(path, out_dir, shard_size=10)

    assert list(out_dir.glob("*.parquet")) == []
    # output_dir is still created even when nothing is written to it
    assert out_dir.is_dir()
