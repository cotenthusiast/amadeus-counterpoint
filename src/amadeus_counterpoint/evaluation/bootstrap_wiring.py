"""Experiment-level wiring for `evaluation.bootstrap.paired_bootstrap`.

Bootstraps ONE dyad, ONE metric (WDL or opening-family TV) at a time -- the
bootstrap unit is a whole sealed real game, never an individual move or
position (see `bootstrap.paired_bootstrap`). Synthetic games are treated as
fixed (a production cell holds 5,000 generated games -- one dyad, condition,
and orientation, per `evaluation.generation.production`); only the finite
real sealed sample is resampled.
"""

import random

from amadeus_counterpoint.evaluation.bootstrap import paired_bootstrap
from amadeus_counterpoint.evaluation.metrics.openings import (
    classify_opening_family,
    opening_distribution,
)
from amadeus_counterpoint.evaluation.metrics.wdl import (
    ORIENTATIONS,
    WDL_OUTCOMES,
    outcome_for_a_b,
    total_variation,
    wdl_summary,
)

DEFAULT_REPLICATES = 10_000
_GENERATED_WDL_CACHE: dict[int, dict[str, dict[str, object]]] = {}


def _prepare_games(games_by_orientation, opening_index) -> None:
    for games in games_by_orientation.values():
        for game in games:
            if "_opening_family" not in game:
                game["_opening_family"] = classify_opening_family(game["moves"], opening_index)
            if not game["censored"] and "_wdl_outcome" not in game:
                game["_wdl_outcome"] = outcome_for_a_b(game["result"], game["orientation"])


def prepare_evaluation_records(synthetic_games, real_games_by_dyad, opening_index) -> None:
    """Compute immutable per-game WDL/opening labels once before metrics/bootstrap."""
    for by_condition in synthetic_games.values():
        for games_by_orientation in by_condition.values():
            _prepare_games(games_by_orientation, opening_index)
    for real_by_orientation in real_games_by_dyad.values():
        _prepare_games(real_by_orientation, opening_index)


def wdl_metric(generated_by_orientation, real_by_orientation) -> float:
    """`paired_bootstrap`-compatible WDL TV metric: the orientation-combined mean."""
    key = id(generated_by_orientation)
    generated = _GENERATED_WDL_CACHE.get(key)
    if generated is None:
        generated = {
            orientation: wdl_summary(generated_by_orientation[orientation])
            for orientation in ORIENTATIONS
        }
        _GENERATED_WDL_CACHE[key] = generated

    distances = {
        orientation: total_variation(
            generated[orientation]["distribution"],
            wdl_summary(real_by_orientation[orientation])["distribution"],
        )
        for orientation in ORIENTATIONS
    }
    return (distances["A_WHITE"] + distances["B_WHITE"]) / 2


wdl_metric._fast_kind = "wdl"


def opening_metric_for_index(opening_index: dict):
    """Return a `paired_bootstrap`-compatible opening-family TV metric bound
    to one pinned `opening_index` (see `metrics.openings.load_opening_index`).

    Caches each fixed synthetic sample's opening-family distribution by
    object identity. `paired_bootstrap` calls this metric once per
    replicate with the exact same `generated_by_orientation` object every
    time for a given dyad/condition -- only `real_by_orientation` is
    resampled -- so recomputing a 5,000-game chess.Board() replay on every
    one of `DEFAULT_REPLICATES` replicates would be purely redundant work
    with zero effect on the statistic (same input, same output, every
    time). The real side is deliberately never cached: it is a different
    object on every replicate by construction (`paired_bootstrap` builds a
    fresh resampled list each time), so caching it would be both useless
    and wrong. This produces numerically identical results to calling
    `opening_orientation_distance` directly on every replicate -- see
    `test_opening_metric_for_index_matches_uncached_orientation_distance` --
    it only avoids recomputing an invariant.
    """
    generated_distribution_cache: dict[int, dict[str, dict[str, float]]] = {}

    def cached_generated_distribution(games_by_orientation):
        key = id(games_by_orientation)
        cached = generated_distribution_cache.get(key)
        if cached is None:
            cached = {
                orientation: opening_distribution(games_by_orientation[orientation], opening_index)
                for orientation in ORIENTATIONS
            }
            generated_distribution_cache[key] = cached
        return cached

    def metric(generated_by_orientation, real_by_orientation) -> float:
        generated_distribution = cached_generated_distribution(generated_by_orientation)
        distances = {
            orientation: total_variation(
                generated_distribution[orientation],
                opening_distribution(real_by_orientation[orientation], opening_index),
            )
            for orientation in ORIENTATIONS
        }
        return (distances["A_WHITE"] + distances["B_WHITE"]) / 2

    metric._fast_kind = "opening"
    metric._opening_index = opening_index
    return metric


def _fast_bootstrap_raw(
    real_by_orientation, generated_by_condition, kind, replicates, seed, opening_index=None,
):
    """Bootstrap cached integer/string labels without rebuilding game records."""
    real_labels = {}
    for orientation in ORIENTATIONS:
        games = real_by_orientation[orientation]
        opening_labels = []
        wdl_labels = []
        for game in games:
            if "_opening_family" not in game:
                return None
            opening_labels.append(game["_opening_family"])
            if game["censored"]:
                wdl_labels.append(None)
            else:
                outcome = game.get("_wdl_outcome")
                if outcome is None:
                    outcome = outcome_for_a_b(game["result"], game["orientation"])
                wdl_labels.append(outcome)
        real_labels[orientation] = {"opening": opening_labels, "wdl": wdl_labels}

    fixed = {}
    for condition, generated_by_orientation in generated_by_condition.items():
        if kind == "opening":
            fixed[condition] = {
                orientation: opening_distribution(
                    generated_by_orientation[orientation], opening_index,
                )
                for orientation in ORIENTATIONS
            }
        else:
            fixed[condition] = {
                orientation: wdl_summary(generated_by_orientation[orientation])
                for orientation in ORIENTATIONS
            }

    def tv_from_counts(distribution, counts, total):
        if kind == "wdl":
            other = {
                outcome: counts.get(outcome, 0) / total if total else 0.0
                for outcome in WDL_OUTCOMES
            }
        else:
            other = {
                label: count / total
                for label, count in counts.items()
            } if total else {}
        return total_variation(distribution, other)

    conditions = tuple(generated_by_condition)
    distances = {condition: [] for condition in conditions}
    contrast_pairs = (
        ("GG_minus_AB", "GG", "AB"),
        ("AG_minus_AB", "AG", "AB"),
        ("GB_minus_AB", "GB", "AB"),
    )
    available_contrasts = tuple(
        (name, left, right)
        for name, left, right in contrast_pairs
        if left in conditions and right in conditions
    )
    contrasts = {name: [] for name, _, _ in available_contrasts}
    random_source = random.Random(seed)

    for _ in range(replicates):
        sampled = {}
        for orientation in ORIENTATIONS:
            labels = real_labels[orientation][kind]
            indices = [random_source.randrange(len(labels)) for _ in labels]
            counts = {}
            for index in indices:
                label = labels[index]
                if kind == "wdl" and label is None:
                    continue
                counts[label] = counts.get(label, 0) + 1
            sampled[orientation] = (counts, sum(counts.values()))

        for condition in conditions:
            distance = 0.0
            for orientation in ORIENTATIONS:
                counts, total = sampled[orientation]
                if kind == "wdl":
                    distribution = fixed[condition][orientation]["distribution"]
                else:
                    distribution = fixed[condition][orientation]
                distance += tv_from_counts(distribution, counts, total)
            distances[condition].append(distance / 2)

        for name, left, right in available_contrasts:
            contrasts[name].append(distances[left][-1] - distances[right][-1])

    return {"distances": distances, "contrasts": contrasts}


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    """A simple 95% percentile interval over bootstrap replicate values."""
    ordered = sorted(values)
    n = len(ordered)
    lo_index = max(0, min(n - 1, int(0.025 * n)))
    hi_index = max(0, min(n - 1, int(0.975 * n) - 1))
    return ordered[lo_index], ordered[hi_index]


def bootstrap_dyad(
    real_by_orientation: dict,
    generated_by_condition: dict,
    metric,
    replicates: int = DEFAULT_REPLICATES,
    seed: int | None = None,
) -> dict:
    """Paired whole-game bootstrap for one dyad, under one metric.

    `generated_by_condition[condition]` -> `{"A_WHITE": [...], "B_WHITE": [...]}`,
    for each of `"GG"`, `"AG"`, `"GB"`, `"AB"`.

    Returns:
        point_estimate: {condition: metric(condition's games, real)} on the
            UNRESAMPLED real sample.
        contrast_point_estimate: {"GG_minus_AB": ..., "AG_minus_AB": ...,
            "GB_minus_AB": ...} from the same unresampled point estimates.
        distance_interval / contrast_interval: 95% percentile intervals
            over the bootstrap replicates.
        replicates: the replicate count used.
    """
    point_estimate = {
        condition: metric(generated_by_condition[condition], real_by_orientation)
        for condition in generated_by_condition
    }
    contrast_point_estimate = {
        name: point_estimate[left] - point_estimate[right]
        for name, left, right in (
            ("GG_minus_AB", "GG", "AB"),
            ("AG_minus_AB", "AG", "AB"),
            ("GB_minus_AB", "GB", "AB"),
        )
        if left in point_estimate and right in point_estimate
    }

    fast_kind = getattr(metric, "_fast_kind", None)
    raw = (
        _fast_bootstrap_raw(
            real_by_orientation, generated_by_condition, fast_kind, replicates, seed,
            getattr(metric, "_opening_index", None),
        )
        if fast_kind is not None
        else None
    )
    if raw is None:
        raw = paired_bootstrap(
            real_by_orientation, generated_by_condition, metric,
            replicates=replicates, seed=seed,
        )

    return {
        "point_estimate": point_estimate,
        "contrast_point_estimate": contrast_point_estimate,
        "distance_interval": {
            condition: _percentile_interval(values)
            for condition, values in raw["distances"].items()
        },
        "contrast_interval": {
            name: _percentile_interval(values)
            for name, values in raw["contrasts"].items()
        },
        "replicates": replicates,
    }
