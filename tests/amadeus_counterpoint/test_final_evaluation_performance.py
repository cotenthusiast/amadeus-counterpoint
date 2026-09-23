import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[2]))
from amadeus_counterpoint.evaluation import bootstrap_wiring
from amadeus_counterpoint.evaluation.bootstrap_wiring import (
    bootstrap_dyad,
    opening_metric_for_index,
    prepare_evaluation_records,
    wdl_metric,
)
from amadeus_counterpoint.evaluation.metrics.openings import (
    classify_opening_family,
    load_opening_index,
    opening_orientation_distance,
)
from amadeus_counterpoint.evaluation.metrics.wdl import wdl_orientation_distance
from scripts.compute_final_results import load_real_games_by_dyad

COMMIT = "4b8622759e7ae6f93f011cc6c83a3823401ab45e"


def _targets(path):
    path.write_text(json.dumps({
        "targets": [
            {"player_id": 0, "fide_id": "a", "aliases": ["A"]},
            {"player_id": 1, "fide_id": "b", "aliases": ["B"]},
        ]
    }))


def _normalized_game(game_id, *, white, black, white_id, black_id, result="1-0",
                     moves=None, variant="Standard", fen=None, setup=None,
                     white_elo=2800, black_elo=2800, date="2020.01.01"):
    return {
        "game_id": game_id,
        "source_month": "2020-01",
        "game_index_in_file": 0,
        "white": white,
        "black": black,
        "white_fide_id": white_id,
        "black_fide_id": black_id,
        "variant": variant,
        "fen": fen,
        "setup": setup,
        "result": result,
        "white_elo": white_elo,
        "black_elo": black_elo,
        "moves_uci": moves or ["e2e4"],
        "date": date,
        "game_url": game_id,
        "parse_ok": True,
        "headers_json": json.dumps({
            "White": white,
            "Black": black,
            "WhiteFideId": white_id,
            "BlackFideId": black_id,
            "Variant": variant,
            **({"FEN": fen} if fen is not None else {}),
            **({"SetUp": setup} if setup is not None else {}),
            "WhiteElo": str(white_elo) if white_elo is not None else "?",
            "BlackElo": str(black_elo) if black_elo is not None else "2800",
            "Result": result,
            "Date": date,
            "GameURL": game_id,
        }),
    }


def test_normalized_loader_matches_exact_filtering_and_reuses_cache(tmp_path):
    normalized_root = tmp_path / "normalized"
    month = normalized_root / "month=2020-01"
    month.mkdir(parents=True)
    rows = [
        _normalized_game("valid", white="A", black="B", white_id="a", black_id="b"),
        _normalized_game("duplicate", white="A", black="B", white_id="a", black_id="b"),
        _normalized_game("wrong-variant", white="A", black="B", white_id="a", black_id="b", variant="Chess960"),
        _normalized_game("missing-elo", white="A", black="B", white_id="a", black_id="b", white_elo=None),
        _normalized_game("one-target", white="A", black="other", white_id="a", black_id=""),
    ]
    pq.write_table(pa.Table.from_pylist(rows), month / "part-0.parquet")
    targets_path = tmp_path / "targets.json"
    _targets(targets_path)
    cache_path = tmp_path / "sealed-real.json"

    loaded = load_real_games_by_dyad(
        tmp_path / "unused-raw",
        targets_path,
        normalized_root=normalized_root,
        cache_path=cache_path,
    )

    assert set(loaded) == {"0__1"}
    assert [game["game_url"] for game in loaded["0__1"]["A_WHITE"]] == ["valid"]
    assert loaded["0__1"]["A_WHITE"][0]["censored"] is False

    normalized_root.rename(tmp_path / "normalized-away")
    cached = load_real_games_by_dyad(
        tmp_path / "unused-raw",
        targets_path,
        normalized_root=tmp_path / "missing-normalized",
        cache_path=cache_path,
    )
    assert cached == loaded


def _write_openings(path):
    path.write_text("eco\tname\tpgn\tuci\tepd\n")


def _game(result="1-0", orientation="A_WHITE", moves=None):
    return {
        "result": result,
        "censored": False,
        "orientation": orientation,
        "moves": moves or ["e2e4"],
    }


def test_prepared_labels_are_reused_by_both_metrics(monkeypatch, tmp_path):
    opening_path = tmp_path / "all.tsv"
    _write_openings(opening_path)
    opening_index = load_opening_index(opening_path, COMMIT)
    real = {
        "A_WHITE": [_game()],
        "B_WHITE": [_game(orientation="B_WHITE")],
    }
    generated = {
        condition: {
            "A_WHITE": [_game()],
            "B_WHITE": [_game(orientation="B_WHITE")],
        }
        for condition in ("GG", "AG", "GB", "AB")
    }

    prepare_evaluation_records({"0__1": generated}, {"0__1": real}, opening_index)
    assert all("_opening_family" in game and "_wdl_outcome" in game
               for games in generated.values()
               for orientation in games.values()
               for game in orientation)
    assert all("_opening_family" in game and "_wdl_outcome" in game
               for games in real.values()
               for game in games)

    def fail_if_reclassified(*args, **kwargs):
        raise AssertionError("opening classification repeated after preparation")

    monkeypatch.setattr(
        "amadeus_counterpoint.evaluation.bootstrap_wiring.classify_opening_family",
        fail_if_reclassified,
    )
    bootstrap_dyad(real, generated, wdl_metric, replicates=4, seed=3)
    bootstrap_dyad(real, generated, opening_metric_for_index(opening_index), replicates=4, seed=3)


def test_prepared_bootstrap_matches_uncached_bootstrap(tmp_path):
    opening_path = tmp_path / "all.tsv"
    _write_openings(opening_path)
    opening_index = load_opening_index(opening_path, COMMIT)
    real = {
        "A_WHITE": [_game(), _game("0-1", moves=["e2e4", "e7e5"])],
        "B_WHITE": [_game(orientation="B_WHITE"), _game("1/2-1/2", orientation="B_WHITE")],
    }
    generated = {
        condition: {
            "A_WHITE": [_game(), _game("0-1", moves=["e2e4", "e7e5"])],
            "B_WHITE": [_game(orientation="B_WHITE"), _game("1/2-1/2", orientation="B_WHITE")],
        }
        for condition in ("GG", "AG", "GB", "AB")
    }

    uncached = bootstrap_dyad(
        real,
        generated,
        lambda generated_games, real_games: opening_orientation_distance(
            generated_games, real_games, opening_index
        )["mean"],
        replicates=12,
        seed=17,
    )
    prepare_evaluation_records({"0__1": generated}, {"0__1": real}, opening_index)
    cached = bootstrap_dyad(
        real,
        generated,
        opening_metric_for_index(opening_index),
        replicates=12,
        seed=17,
    )

    assert cached == uncached


def test_opening_index_contains_exact_position_key_cache(tmp_path):
    opening_path = tmp_path / "all.tsv"
    _write_openings(opening_path)
    opening_index = load_opening_index(opening_path, COMMIT)

    assert "_position_key_to_name" in opening_index
    assert classify_opening_family(["e2e4"], opening_index) == "Unknown"


def test_prepared_wdl_bootstrap_matches_uncached_bootstrap(tmp_path):
    opening_path = tmp_path / "all.tsv"
    _write_openings(opening_path)
    opening_index = load_opening_index(opening_path, COMMIT)
    real = {
        "A_WHITE": [_game(), _game("0-1", moves=["e2e4", "e7e5"])],
        "B_WHITE": [_game(orientation="B_WHITE"), _game("1/2-1/2", orientation="B_WHITE")],
    }
    generated = {
        condition: {
            "A_WHITE": [_game(), _game("0-1", moves=["e2e4", "e7e5"])],
            "B_WHITE": [_game(orientation="B_WHITE"), _game("1/2-1/2", orientation="B_WHITE")],
        }
        for condition in ("GG", "AG", "GB", "AB")
    }

    uncached = bootstrap_dyad(
        real,
        generated,
        lambda generated_games, real_games: wdl_orientation_distance(
            generated_games, real_games
        )["mean"],
        replicates=12,
        seed=17,
    )
    prepare_evaluation_records({"0__1": generated}, {"0__1": real}, opening_index)
    cached = bootstrap_dyad(real, generated, wdl_metric, replicates=12, seed=17)

    assert cached == uncached


def test_wdl_metric_caches_fixed_generated_summaries(monkeypatch):
    real = {"A_WHITE": [_game()], "B_WHITE": [_game(orientation="B_WHITE")]}
    generated = {"A_WHITE": [_game()], "B_WHITE": [_game(orientation="B_WHITE")]}
    calls = []
    original = bootstrap_wiring.wdl_summary if hasattr(bootstrap_wiring, "wdl_summary") else None

    def spy(games):
        calls.append(id(games))
        return original(games)

    monkeypatch.setattr(bootstrap_wiring, "wdl_summary", spy, raising=False)
    bootstrap_wiring.wdl_metric(generated, real)
    bootstrap_wiring.wdl_metric(generated, real)

    assert calls.count(id(generated["A_WHITE"])) == 1
    assert calls.count(id(generated["B_WHITE"])) == 1
    assert calls.count(id(real["A_WHITE"])) == 2
    assert calls.count(id(real["B_WHITE"])) == 2
