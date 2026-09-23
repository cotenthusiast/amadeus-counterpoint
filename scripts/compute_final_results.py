"""Compute final WDL/opening-family metrics, bootstrap uncertainty, and
assemble one machine-readable results file for ONE personalization method.

Reads:
    - sealed real target-vs-target games (raw Broadcast PGN corpus)
    - that method's production synthetic cell artifacts (one shard per
      dyad x condition x orientation, from generate_method{1,2}_production.py)
    - a pinned opening-family taxonomy (path + exact commit SHA -- never
      fetched or defaulted here)

Writes one JSON results file via evaluation.final_results.

    python scripts/compute_final_results.py \\
        --method method1 \\
        --synthetic-root artifacts/synthetic \\
        --broadcast-root /home/cotenthusiast/Data/lichess_broadcasts \\
        --opening-tsv-path data/chess_openings/dist/all.tsv \\
        --opening-taxonomy-commit <40-char-sha> \\
        --representative-elos configs/representative_elos.json \\
        --output-path results/method1_results.json \\
        --root-seed 20260910 --games-per-orientation 5000 \\
        --base-checkpoint-identity broadcast-adapt-step-40000 \\
        --personalization-checkpoint-identity method1-2026-09-10 \\
        --representative-elo-identity representative_elos.json@<sha> \\
        --protocol-version 1 --code-commit <git-sha>
"""

import argparse
import glob
import hashlib
import json
from pathlib import Path

from amadeus_counterpoint.data.broadcast_targets import (
    load_broadcast_targets,
    match_target,
)
from amadeus_counterpoint.data.sealed_dyads import (
    deduplicate_sealed_games,
    group_sealed_games_by_dyad,
    iter_sealed_dyad_games,
)
from amadeus_counterpoint.evaluation import artifacts
from amadeus_counterpoint.evaluation.aggregate import CONDITIONS, build_results_table
from amadeus_counterpoint.evaluation.bootstrap_wiring import (
    bootstrap_dyad,
    opening_metric_for_index,
    prepare_evaluation_records,
    wdl_metric,
)
from amadeus_counterpoint.evaluation.final_results import (
    build_final_results,
    write_final_results,
)
from amadeus_counterpoint.evaluation.generation.production import cell_artifact_path
from amadeus_counterpoint.evaluation.metrics.openings import load_opening_index


def _with_censored_false(games: list[dict]) -> list[dict]:
    """Real sealed games are always completed, never ply-capped -- the WDL
    metric primitive still expects an explicit `censored` field per game."""
    return [{**game, "censored": False} for game in games]


_NORMALIZED_COLUMNS = (
    "game_id", "source_month", "game_index_in_file", "white", "black",
    "white_fide_id", "black_fide_id", "variant", "fen", "setup", "result",
    "white_elo", "black_elo", "moves_uci", "date", "game_url", "headers_json",
    "parse_ok",
)
_REAL_CACHE_FORMAT = "amadeus_sealed_real_games_v1"


def _normalized_sealed_games(normalized_root, targets) -> list[dict]:
    """Load exact sealed records from the canonical normalized corpus.

    The normalized rows were produced by the same ``chess.pgn.read_game``
    and ``game.mainline()`` operations as the raw-PGN path.  Header JSON is
    consulted only for target candidates so the validity checks below retain
    the raw header-presence semantics of ``game_to_record`` (notably FEN and
    Variant placeholders).
    """
    import pyarrow.parquet as pq

    paths = sorted(Path(normalized_root).glob("month=*/part-0.parquet"))
    if not paths:
        raise FileNotFoundError(f"no normalized Parquet shards under {normalized_root}")

    sealed_games = []
    for path in paths:
        table = pq.read_table(path, columns=list(_NORMALIZED_COLUMNS))
        for row in table.to_pylist():
            white_player_id = match_target(
                row["white"] or "", row["white_fide_id"] or "", targets,
            )
            black_player_id = match_target(
                row["black"] or "", row["black_fide_id"] or "", targets,
            )
            if (
                white_player_id is None
                or black_player_id is None
                or white_player_id == black_player_id
            ):
                continue

            headers = json.loads(row["headers_json"])
            white_player_id = match_target(
                headers.get("White", ""), headers.get("WhiteFideId", ""), targets,
            )
            black_player_id = match_target(
                headers.get("Black", ""), headers.get("BlackFideId", ""), targets,
            )
            if (
                white_player_id is None
                or black_player_id is None
                or white_player_id == black_player_id
            ):
                continue
            if row["parse_ok"] is False:
                raise ValueError(f"normalized row cannot be replayed: {row['game_id']}")

            if headers.get("Variant", "Standard").strip().lower() != "standard":
                continue
            if "FEN" in headers or headers.get("SetUp") == "1":
                continue
            try:
                white_elo = int(headers["WhiteElo"])
                black_elo = int(headers["BlackElo"])
                result = headers["Result"]
            except (KeyError, ValueError):
                continue
            if result not in {"1-0", "0-1", "1/2-1/2"} or not row["moves_uci"]:
                continue

            sealed_games.append({
                "player_id_a": min(white_player_id, black_player_id),
                "player_id_b": max(white_player_id, black_player_id),
                "orientation": "A_WHITE" if white_player_id < black_player_id else "B_WHITE",
                "white_player_id": white_player_id,
                "black_player_id": black_player_id,
                "white_elo": white_elo,
                "black_elo": black_elo,
                "result": result,
                "moves": row["moves_uci"],
                "date": headers.get("Date"),
                "game_url": headers.get("GameURL"),
            })
    return sealed_games


def _read_real_cache(cache_path, targets_config_path) -> dict:
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(Path(targets_config_path).read_bytes()).hexdigest()
    if payload.get("format") != _REAL_CACHE_FORMAT:
        raise ValueError(f"unsupported sealed-real cache format: {cache_path}")
    if payload.get("targets_config_sha256") != expected_hash:
        raise ValueError("sealed-real cache was built from a different targets config")
    grouped = group_sealed_games_by_dyad(payload["games"])
    return {
        dyad: {
            orientation: _with_censored_false(games)
            for orientation, games in by_orientation.items()
        }
        for dyad, by_orientation in grouped.items()
    }


def _write_real_cache(cache_path, games, targets_config_path) -> None:
    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _REAL_CACHE_FORMAT,
        "targets_config_sha256": hashlib.sha256(Path(targets_config_path).read_bytes()).hexdigest(),
        "game_count": len(games),
        "games": games,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(destination)


def load_real_games_by_dyad(
    broadcast_root, targets_config_path, *, normalized_root=None, cache_path=None,
) -> dict:
    targets = load_broadcast_targets(targets_config_path)
    if cache_path is not None and Path(cache_path).exists():
        return _read_real_cache(cache_path, targets_config_path)

    if normalized_root is not None:
        sealed_games = _normalized_sealed_games(normalized_root, targets)
        sealed_games, _dedup_report = deduplicate_sealed_games(sealed_games)
        if cache_path is not None:
            _write_real_cache(cache_path, sealed_games, targets_config_path)
        grouped = group_sealed_games_by_dyad(sealed_games)
        return {
            dyad: {
                orientation: _with_censored_false(games)
                for orientation, games in by_orientation.items()
            }
            for dyad, by_orientation in grouped.items()
        }
    if cache_path is not None:
        raise ValueError("cache_path requires normalized_root when the cache does not exist")

    pgn_paths = sorted(glob.glob(str(Path(broadcast_root) / "lichess_db_broadcast_*.pgn")))

    sealed_games = list(iter_sealed_dyad_games(pgn_paths, targets))
    sealed_games, _dedup_report = deduplicate_sealed_games(sealed_games)
    grouped = group_sealed_games_by_dyad(sealed_games)

    return {
        dyad: {
            orientation: _with_censored_false(games)
            for orientation, games in by_orientation.items()
        }
        for dyad, by_orientation in grouped.items()
    }


def load_synthetic_games(
    synthetic_root, method: str, dyads: list[str], *, allow_missing_cells: bool = False,
) -> tuple[dict, list[str]]:
    """`{dyad: {condition: {"A_WHITE": [...], "B_WHITE": [...]}}}` read back
    from the one-shard-per-cell production artifacts."""
    synthetic = {}
    missing_cells = []
    for dyad in dyads:
        synthetic[dyad] = {}
        for condition in CONDITIONS:
            by_orientation = {}
            for orientation in ("A_WHITE", "B_WHITE"):
                path = cell_artifact_path(synthetic_root, method, dyad, condition, orientation)
                if not path.exists():
                    missing_cells.append(f"{dyad}/{condition}_{orientation}")
                    by_orientation = None
                    break
                by_orientation[orientation] = artifacts.read_games(path).to_pylist()
            if by_orientation is not None:
                synthetic[dyad][condition] = by_orientation
            elif not allow_missing_cells:
                raise FileNotFoundError(f"missing synthetic cell shard: {path}")
    return synthetic, missing_cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("method1", "method2", "method3_hybrid"), required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--broadcast-root", type=Path, required=True)
    parser.add_argument(
        "--normalized-broadcast-root", type=Path,
        help="Canonical normalized Broadcast Parquet root; avoids rescanning raw PGNs.",
    )
    parser.add_argument(
        "--real-cache-path", type=Path,
        help="Optional immutable JSON cache for the exact sealed real-game records.",
    )
    parser.add_argument(
        "--targets-config", type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "broadcast_targets.json",
    )
    parser.add_argument("--opening-tsv-path", type=Path, required=True)
    parser.add_argument("--opening-taxonomy-commit", type=str, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--dyad", dest="dyads", action="append",
        help="Restrict evaluation to one dyad; repeat for a matched subset.",
    )
    parser.add_argument(
        "--allow-missing-cells", action="store_true",
        help="Skip incomplete dyad-condition cells and record their exclusion.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--root-seed", type=str, required=True)
    parser.add_argument("--games-per-orientation", type=int, required=True)
    parser.add_argument("--base-checkpoint-identity", type=str, required=True)
    parser.add_argument("--personalization-checkpoint-identity", type=str, required=True)
    parser.add_argument("--representative-elo-identity", type=str, required=True)
    parser.add_argument("--protocol-version", type=str, required=True)
    parser.add_argument("--code-commit", type=str, required=True)
    args = parser.parse_args()

    opening_index = load_opening_index(args.opening_tsv_path, args.opening_taxonomy_commit)

    real_games_by_dyad = load_real_games_by_dyad(
        args.broadcast_root,
        args.targets_config,
        normalized_root=args.normalized_broadcast_root,
        cache_path=args.real_cache_path,
    )
    dyads = sorted(args.dyads or real_games_by_dyad)
    unknown_dyads = sorted(set(dyads) - set(real_games_by_dyad))
    if unknown_dyads:
        raise ValueError(f"unknown dyad(s): {unknown_dyads}")
    real_games_by_dyad = {dyad: real_games_by_dyad[dyad] for dyad in dyads}

    synthetic_games, missing_cells = load_synthetic_games(
        args.synthetic_root, args.method, dyads,
        allow_missing_cells=args.allow_missing_cells,
    )
    prepare_evaluation_records(synthetic_games, real_games_by_dyad, opening_index)

    results_table = build_results_table(synthetic_games, real_games_by_dyad, opening_index)

    opening_metric = opening_metric_for_index(opening_index)
    bootstrap_results = {}
    for dyad in dyads:
        bootstrap_results[dyad] = {
            "wdl": bootstrap_dyad(
                real_games_by_dyad[dyad], synthetic_games[dyad], wdl_metric,
                replicates=args.bootstrap_replicates, seed=args.bootstrap_seed,
            ),
            "opening": bootstrap_dyad(
                real_games_by_dyad[dyad], synthetic_games[dyad], opening_metric,
                replicates=args.bootstrap_replicates, seed=args.bootstrap_seed,
            ),
        }
        print(f"bootstrapped {dyad}")

    provenance = {
        "base_checkpoint_identity": args.base_checkpoint_identity,
        "personalization_checkpoint_identity": args.personalization_checkpoint_identity,
        "representative_elo_identity": args.representative_elo_identity,
        "opening_taxonomy_commit": args.opening_taxonomy_commit,
        "root_seed": args.root_seed,
        "games_per_orientation": args.games_per_orientation,
        "protocol_version": args.protocol_version,
        "code_commit": args.code_commit,
        "bootstrap_replicates": args.bootstrap_replicates,
        "missing_cells_skipped": missing_cells,
        "evaluated_dyads": dyads,
    }

    final_results = build_final_results(args.method, results_table, bootstrap_results, provenance)
    write_final_results(final_results, args.output_path)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
