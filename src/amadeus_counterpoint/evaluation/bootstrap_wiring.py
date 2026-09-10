"""Experiment-level wiring for `evaluation.bootstrap.paired_bootstrap`.

Bootstraps ONE dyad, ONE metric (WDL or opening-family TV) at a time -- the
bootstrap unit is a whole sealed real game, never an individual move or
position (see `bootstrap.paired_bootstrap`). Synthetic games are treated as
fixed (a production cell holds 5,000 generated games -- one dyad, condition,
and orientation, per `evaluation.generation.production`); only the finite
real sealed sample is resampled.
"""

from amadeus_counterpoint.evaluation.bootstrap import paired_bootstrap
from amadeus_counterpoint.evaluation.metrics.openings import opening_orientation_distance
from amadeus_counterpoint.evaluation.metrics.wdl import wdl_orientation_distance

DEFAULT_REPLICATES = 10_000


def wdl_metric(generated_by_orientation, real_by_orientation) -> float:
    """`paired_bootstrap`-compatible WDL TV metric: the orientation-combined mean."""
    return wdl_orientation_distance(generated_by_orientation, real_by_orientation)["mean"]


def opening_metric_for_index(opening_index: dict):
    """Return a `paired_bootstrap`-compatible opening-family TV metric bound
    to one pinned `opening_index` (see `metrics.openings.load_opening_index`).
    """

    def metric(generated_by_orientation, real_by_orientation) -> float:
        return opening_orientation_distance(
            generated_by_orientation, real_by_orientation, opening_index
        )["mean"]

    return metric


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
        "GG_minus_AB": point_estimate["GG"] - point_estimate["AB"],
        "AG_minus_AB": point_estimate["AG"] - point_estimate["AB"],
        "GB_minus_AB": point_estimate["GB"] - point_estimate["AB"],
    }

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
