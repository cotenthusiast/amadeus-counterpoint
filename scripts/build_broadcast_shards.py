#!/usr/bin/env python3
"""Build target-free Broadcast adaptation training/validation shards.

Source of truth for target exclusion: the sibling amadeus-broadcast-data
workspace's manifests/target_free_elite_adaptation.parquet -- an
already-audited, mechanically-tested zero-target game_id set (FIDE-ID
clustering across 177,436 distinct name/fide_id pairs + manual review +
decoy-identity resolution + quarantine enforcement, 10/10 invariant tests
passing; see that repo's README). This is used INSTEAD of
amadeus-counterpoint's configs/broadcast_targets.json, which was curated
against a smaller local-machine copy of the corpus and is missing several
verified aliases (MagzyBogues, FantasticStar, STL_Caruana, STL_Nakamura)
the more thorough Kelvin2 audit found -- using it directly would under-exclude
relative to the best available identity resolution.

For every target-free game, validity is checked with the EXACT SAME rule
population training used (amadeus_counterpoint.data.preprocess.game_to_record):
Standard variant, no custom FEN/SetUp, both Elo fields present, a terminal
result, at least one move. The already-parsed `variant`/`fen`/`setup`/
`white_elo`/`black_elo`/`result`/`moves_uci` columns in the normalized
parquet are used directly for this (they were extracted from the exact same
source PGN); only the clock-based eligible_ply_count computation needs a
fresh, targeted parse -- `movetext` (SAN + [%clk] comments, originally
exported BY python-chess) is re-parsed via chess.pgn.read_game() to recover
per-move clocks, then handed to the unmodified eligible_ply_count() so the
paper-verified <30s time-pressure cutoff logic is never reimplemented.

Elo-bin balancing (balance_by_elo) is deliberately NOT applied: it exists to
flatten an organic population's natural Elo concentration for population
training. A Broadcast/elite adaptation corpus is already concentrated at the
strong end on purpose (that's the point of this stage), and balancing it
would throw away most of the high-Elo games this stage exists to use.

Deterministic train/validation split: sha256(game_id) mod 100 < VAL_PCT ->
validation, else train. Pure function of game_id, so it is a strict
partition with no possibility of leakage between the two.

This script only IMPORTS amadeus_counterpoint (eligible_ply_count,
write_shard) -- it does not modify it. It also depends on `duckdb` and
`pandas`, which are not amadeus-counterpoint dependencies -- run it with the
amadeus-broadcast-data venv (which already has both, plus python-chess and
pyarrow), not the amadeus-counterpoint venv.
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chess.pgn
import duckdb
import pandas as pd

from amadeus_counterpoint.data.preprocess import eligible_ply_count, write_shard

VAL_PCT = 2  # percent of target-free games held out for validation
SHARD_SIZE = 10_000


def is_validation(game_id: str) -> bool:
    digest = hashlib.sha256(game_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100 < VAL_PCT


def process_month(args):
    (month, normalized_glob, manifest_path, train_dir, val_dir, shard_index_base) = args

    con = duckdb.connect()
    query = f"""
        SELECT n.game_id, n.variant, n.fen, n.setup, n.white_elo, n.black_elo,
               n.result, n.moves_uci, n.movetext
        FROM read_parquet('{normalized_glob}') n
        JOIN (SELECT game_id FROM read_parquet('{manifest_path}') WHERE source_month = '{month}') t
        ON n.game_id = t.game_id
        WHERE n.source_month = '{month}'
    """
    df = con.execute(query).fetchdf()

    stats = {
        "month": month,
        "raw_target_free_games": len(df),
        "rejected_non_standard_variant": 0,
        "rejected_custom_start": 0,
        "rejected_missing_or_invalid_elo": 0,
        "rejected_bad_result": 0,
        "rejected_no_moves": 0,
        "retained_train": 0,
        "retained_val": 0,
        "positions_train": 0,
        "positions_val": 0,
    }

    train_buffer = []
    val_buffer = []
    train_shard_idx = 0
    val_shard_idx = 0

    for row in df.itertuples(index=False):
        raw_variant = row.variant
        variant = "standard" if pd.isna(raw_variant) else raw_variant.strip().lower()
        if variant != "standard":
            stats["rejected_non_standard_variant"] += 1
            continue

        has_fen = not pd.isna(row.fen)
        setup_flag = (not pd.isna(row.setup)) and row.setup == "1"
        if has_fen or setup_flag:
            stats["rejected_custom_start"] += 1
            continue

        if pd.isna(row.white_elo) or pd.isna(row.black_elo):
            stats["rejected_missing_or_invalid_elo"] += 1
            continue
        white_elo = int(row.white_elo)
        black_elo = int(row.black_elo)

        if pd.isna(row.result) or row.result not in {"1-0", "0-1", "1/2-1/2"}:
            stats["rejected_bad_result"] += 1
            continue

        moves = list(row.moves_uci)
        if not moves:
            stats["rejected_no_moves"] += 1
            continue

        movetext = "" if pd.isna(row.movetext) else row.movetext
        game = chess.pgn.read_game(io.StringIO(movetext)) if movetext else None
        nodes = list(game.mainline()) if game is not None else []
        clocks_after_move = [node.clock() for node in nodes]
        eligible = eligible_ply_count(clocks_after_move) if clocks_after_move else len(moves)

        record = {
            "white_elo": white_elo,
            "black_elo": black_elo,
            "result": row.result,
            "moves": moves,
            "eligible_ply_count": eligible,
        }

        positions = min(eligible, 32)

        if is_validation(row.game_id):
            val_buffer.append(record)
            stats["retained_val"] += 1
            stats["positions_val"] += positions
            if len(val_buffer) >= SHARD_SIZE:
                write_shard(val_buffer, val_dir / f"broadcast_{month}_val_{val_shard_idx:03d}.parquet")
                val_buffer = []
                val_shard_idx += 1
        else:
            train_buffer.append(record)
            stats["retained_train"] += 1
            stats["positions_train"] += positions
            if len(train_buffer) >= SHARD_SIZE:
                write_shard(train_buffer, train_dir / f"broadcast_{month}_train_{train_shard_idx:03d}.parquet")
                train_buffer = []
                train_shard_idx += 1

    if train_buffer:
        write_shard(train_buffer, train_dir / f"broadcast_{month}_train_{train_shard_idx:03d}.parquet")
    if val_buffer:
        write_shard(val_buffer, val_dir / f"broadcast_{month}_val_{val_shard_idx:03d}.parquet")

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-glob", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    months = con.execute(
        f"SELECT DISTINCT source_month FROM read_parquet('{args.manifest}') ORDER BY source_month"
    ).fetchdf()["source_month"].tolist()

    print(f"months to process: {len(months)}")

    tasks = [
        (month, args.normalized_glob, args.manifest, train_dir, val_dir, i)
        for i, month in enumerate(months)
    ]

    with multiprocessing.Pool(args.num_workers) as pool:
        results = pool.map(process_month, tasks)

    totals = {
        "raw_target_free_games": 0,
        "rejected_non_standard_variant": 0,
        "rejected_custom_start": 0,
        "rejected_missing_or_invalid_elo": 0,
        "rejected_bad_result": 0,
        "rejected_no_moves": 0,
        "retained_train": 0,
        "retained_val": 0,
        "positions_train": 0,
        "positions_val": 0,
    }
    for r in results:
        for k in totals:
            totals[k] += r[k]

    totals["retained_total"] = totals["retained_train"] + totals["retained_val"]
    totals["positions_total"] = totals["positions_train"] + totals["positions_val"]
    totals["per_month"] = results

    Path(args.report).write_text(json.dumps(totals, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in totals.items() if k != "per_month"}, indent=2))


if __name__ == "__main__":
    main()
