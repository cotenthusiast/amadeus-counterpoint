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
from pathlib import Path

from amadeus_counterpoint.data.broadcast_targets import load_broadcast_targets
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
    wdl_metric,
)
from amadeus_counterpoint.evaluation.final_results import build_final_results, write_final_results
from amadeus_counterpoint.evaluation.generation.production import cell_artifact_path
from amadeus_counterpoint.evaluation.metrics.openings import load_opening_index


def _with_censored_false(games: list[dict]) -> list[dict]:
    """Real sealed games are always completed, never ply-capped -- the WDL
    metric primitive still expects an explicit `censored` field per game."""
    return [{**game, "censored": False} for game in games]


def load_real_games_by_dyad(broadcast_root, targets_config_path) -> dict:
    targets = load_broadcast_targets(targets_config_path)
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


def load_synthetic_games(synthetic_root, method: str, dyads: list[str]) -> dict:
    """`{dyad: {condition: {"A_WHITE": [...], "B_WHITE": [...]}}}` read back
    from the one-shard-per-cell production artifacts."""
    synthetic = {}
    for dyad in dyads:
        synthetic[dyad] = {}
        for condition in CONDITIONS:
            by_orientation = {}
            for orientation in ("A_WHITE", "B_WHITE"):
                path = cell_artifact_path(synthetic_root, method, dyad, condition, orientation)
                by_orientation[orientation] = artifacts.read_games(path).to_pylist()
            synthetic[dyad][condition] = by_orientation
    return synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("method1", "method2"), required=True)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--broadcast-root", type=Path, required=True)
    parser.add_argument(
        "--targets-config", type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "broadcast_targets.json",
    )
    parser.add_argument("--opening-tsv-path", type=Path, required=True)
    parser.add_argument("--opening-taxonomy-commit", type=str, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
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

    real_games_by_dyad = load_real_games_by_dyad(args.broadcast_root, args.targets_config)
    dyads = sorted(real_games_by_dyad)

    synthetic_games = load_synthetic_games(args.synthetic_root, args.method, dyads)

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
    }

    final_results = build_final_results(args.method, results_table, bootstrap_results, provenance)
    write_final_results(final_results, args.output_path)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
