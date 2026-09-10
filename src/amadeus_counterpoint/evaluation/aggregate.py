"""Wire the existing WDL/opening-family metric primitives into per-dyad,
per-condition results, and aggregate across dyads with equal weight.

Frozen: only these two metrics (`evaluation.metrics.wdl`/`openings`),
computed per orientation and combined 50/50 -- exactly what
`wdl_orientation_distance`/`opening_orientation_distance` already do. Dyad
aggregation gives every dyad equal weight regardless of how many historical
real games it has (no pooling, no weighting by real-game count).
"""

from amadeus_counterpoint.evaluation.metrics.openings import opening_orientation_distance
from amadeus_counterpoint.evaluation.metrics.wdl import wdl_orientation_distance

CONDITIONS = ("GG", "AG", "GB", "AB")


def compute_condition_metrics(
    synthetic_by_orientation: dict, real_by_orientation: dict, opening_index: dict,
) -> dict:
    """One dyad, one condition: WDL and opening-family distance reports,
    each already combining A_WHITE/B_WHITE with equal 50/50 weight.

    `synthetic_by_orientation`/`real_by_orientation`: `{"A_WHITE": [...],
    "B_WHITE": [...]}` game-record lists (synthetic: one condition's cell
    pair; real: `data.sealed_dyads.group_sealed_games_by_dyad`'s per-dyad
    value).
    """
    return {
        "wdl": wdl_orientation_distance(synthetic_by_orientation, real_by_orientation),
        "opening": opening_orientation_distance(synthetic_by_orientation, real_by_orientation, opening_index),
    }


def build_results_table(
    synthetic_games: dict, real_games_by_dyad: dict, opening_index: dict,
) -> dict:
    """Compute per-dyad, per-condition metrics for every dyad present.

    `synthetic_games[dyad][condition]` -> `{"A_WHITE": [...], "B_WHITE": [...]}`.
    `real_games_by_dyad[dyad]` -> `{"A_WHITE": [...], "B_WHITE": [...]}`
    (sealed real games for that dyad).

    Returns `{dyad: {condition: {"wdl": ..., "opening": ...}}}`.
    """
    results = {}
    for dyad, by_condition in synthetic_games.items():
        real_by_orientation = real_games_by_dyad[dyad]
        results[dyad] = {}
        for condition, synthetic_by_orientation in by_condition.items():
            results[dyad][condition] = compute_condition_metrics(
                synthetic_by_orientation, real_by_orientation, opening_index,
            )
    return results


def aggregate_equal_dyad_weight(results_table: dict, condition: str, metric: str) -> float:
    """Equal-weight mean, across every dyad in `results_table`, of one
    condition's already-orientation-combined metric (`"wdl"` or
    `"opening"`). Every dyad counts once, regardless of real-sample size.
    """
    values = [
        per_dyad[condition][metric]["mean"]
        for per_dyad in results_table.values()
    ]
    return sum(values) / len(values)
