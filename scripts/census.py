#!/usr/bin/env python3
"""Run the production preprocessing pipeline over a PGN stream and report
detailed raw/filter/balancing/position/storage counters.

This script makes NO filtering, validation, or balancing decisions of its
own -- every accept/reject/balance/write decision is made by the real
functions in amadeus_counterpoint.data.preprocess and
amadeus_counterpoint.data.target_exclusion. The only extra work here is
read-only header inspection to attribute *why* game_to_record rejected a
game (for the raw/filter breakdown) and simple arithmetic aggregation.

Usage:
    python scripts/census.py INPUT.pgn OUTPUT_DIR [--target-aliases PATH]
        [--pilot] [--max-raw-games N] [--report PATH]

INPUT.pgn may be a regular file or a FIFO (e.g. fed by `zstd -dc` so the
compressed archive is never permanently decompressed to disk).
"""

import argparse
import json
import sys
import time
from pathlib import Path

from amadeus_counterpoint.data.dataset import DEFAULT_MAX_POSITIONS_PER_GAME
from amadeus_counterpoint.data.preprocess import (
    ELO_BIN_HIGH,
    ELO_BIN_LOW,
    NUM_ELO_BINS,
    balance_by_elo,
    elo_bin,
    game_to_record,
    iter_pgn_games,
    write_shard,
)
from amadeus_counterpoint.data.target_exclusion import (
    TargetExclusionRequiredError,
    is_target_game,
    load_target_aliases,
)

ELO_CHUNK_SIZE = 20_000
GAMES_PER_ELO_BIN = 10
SHARD_SIZE = 10_000


def classify_rejection(headers: dict) -> str:
    """Best-effort, read-only attribution of why game_to_record likely
    rejected a game, by re-reading the same headers it consults. Does not
    influence the accept/reject decision -- that's made by game_to_record
    itself; this only labels it for reporting.
    """
    if headers.get("Variant", "Standard").strip().lower() != "standard":
        return "non_standard_variant"
    if "FEN" in headers or headers.get("SetUp") == "1":
        return "custom_start"
    try:
        int(headers["WhiteElo"])
        int(headers["BlackElo"])
    except (KeyError, ValueError):
        return "missing_or_invalid_elo"
    if headers.get("Result") not in {"1-0", "0-1", "1/2-1/2"}:
        return "bad_result"
    return "other_malformed"


def run_census(
    input_path: str,
    output_dir: str,
    target_aliases: frozenset[str],
    allow_missing_target_exclusion: bool,
    max_raw_games: int | None,
) -> dict:
    if not target_aliases and not allow_missing_target_exclusion:
        raise TargetExclusionRequiredError(
            "No target-cohort aliases configured; pass --pilot to run "
            "this census in explicit pilot mode."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "raw_games_encountered": 0,
        "standard_games": 0,
        "rejected_non_standard_variant": 0,
        "rejected_custom_start": 0,
        "rejected_missing_or_invalid_elo": 0,
        "rejected_bad_result": 0,
        "rejected_other_malformed": 0,
        "target_excluded_games": 0 if target_aliases else None,
        "valid_non_excluded_games": 0,  # enters Elo balancing
        "retained_after_balancing": 0,
    }
    elo_bin_occupancy = [0] * NUM_ELO_BINS
    total_plies = 0
    total_eligible_plies = 0
    total_selected_positions = 0

    def tagged_records():
        for game in iter_pgn_games(input_path):
            counts["raw_games_encountered"] += 1
            if max_raw_games is not None and counts["raw_games_encountered"] > max_raw_games:
                return

            headers = game.headers
            white = headers.get("White", "")
            black = headers.get("Black", "")

            if target_aliases and is_target_game(white, black, target_aliases):
                counts["target_excluded_games"] += 1
                continue

            record = game_to_record(game)
            if record is None:
                reason = classify_rejection(headers)
                counts[f"rejected_{reason}"] += 1
                continue

            counts["standard_games"] += 1
            counts["valid_non_excluded_games"] += 1
            yield record

    buffer = []
    shard_index = 0
    start = time.perf_counter()

    for record in balance_by_elo(
        tagged_records(), chunk_size=ELO_CHUNK_SIZE, games_per_bin=GAMES_PER_ELO_BIN
    ):
        counts["retained_after_balancing"] += 1
        mean_elo = (record["white_elo"] + record["black_elo"]) / 2
        elo_bin_occupancy[elo_bin(mean_elo)] += 1

        moves = len(record["moves"])
        eligible = record.get("eligible_ply_count", moves)
        total_plies += moves
        total_eligible_plies += eligible
        total_selected_positions += min(eligible, DEFAULT_MAX_POSITIONS_PER_GAME)

        buffer.append(record)
        if len(buffer) >= SHARD_SIZE:
            write_shard(buffer, output_dir / f"shard_{shard_index:05d}.parquet")
            buffer = []
            shard_index += 1

    if buffer:
        write_shard(buffer, output_dir / f"shard_{shard_index:05d}.parquet")

    elapsed = time.perf_counter() - start

    parquet_bytes = sum(p.stat().st_size for p in output_dir.glob("*.parquet"))
    num_chunks = max(1, -(-counts["raw_games_encountered"] // ELO_CHUNK_SIZE))
    theoretical_cap = num_chunks * NUM_ELO_BINS * GAMES_PER_ELO_BIN

    retained = counts["retained_after_balancing"]
    return {
        "counts": counts,
        "elo_bin_occupancy": {
            "bins": elo_bin_occupancy,
            "bin_edges": f"<{ELO_BIN_LOW}, 100-wide to {ELO_BIN_HIGH}, >={ELO_BIN_HIGH}",
            "under_filled_bins": sum(1 for c in elo_bin_occupancy if c < num_chunks * GAMES_PER_ELO_BIN),
            "num_elo_chunks_approx": num_chunks,
            "theoretical_cap_approx": theoretical_cap,
            "retained_vs_theoretical_cap_pct": round(100 * retained / theoretical_cap, 2) if theoretical_cap else None,
        },
        "retention_pct_of_raw": round(100 * retained / counts["raw_games_encountered"], 4)
        if counts["raw_games_encountered"] else None,
        "positions": {
            "total_plies_in_retained_games": total_plies,
            "total_eligible_plies_after_clock_cutoff": total_eligible_plies,
            "total_selected_after_32_cap": total_selected_positions,
            "avg_selected_positions_per_retained_game": round(total_selected_positions / retained, 3)
            if retained else None,
        },
        "storage_and_throughput": {
            "output_parquet_bytes": parquet_bytes,
            "wall_clock_seconds": round(elapsed, 3),
            "games_per_sec": round(counts["raw_games_encountered"] / elapsed, 2) if elapsed else None,
            "positions_per_sec": round(total_selected_positions / elapsed, 2) if elapsed else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_dir")
    parser.add_argument("--target-aliases", default="configs/target_aliases.json")
    parser.add_argument("--pilot", action="store_true",
                         help="Explicit opt-in to run without target exclusion (PILOT-ONLY).")
    parser.add_argument("--max-raw-games", type=int, default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    target_aliases = load_target_aliases(args.target_aliases)
    if not target_aliases:
        if not args.pilot:
            print(
                "No target-cohort aliases configured. Pass --pilot for a "
                "non-production run (output is PILOT-ONLY).",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            "PILOT-ONLY: running without target-cohort exclusion. This "
            "census and any output data are NOT VALID FOR FINAL "
            "POPULATION TRAINING.",
            file=sys.stderr,
        )

    report = run_census(
        args.input_path,
        args.output_dir,
        target_aliases=target_aliases,
        allow_missing_target_exclusion=args.pilot,
        max_raw_games=args.max_raw_games,
    )
    report["pilot_mode"] = not bool(target_aliases)

    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
