"""Model-independent WDL distributions and Total Variation calculations.

The public summaries intentionally keep censored games separate from WDL:
their distribution is calculated only from completed games, while counts and
rates for completed and censored games remain visible in the returned report.
"""

from collections.abc import Mapping, Sequence

A_WHITE = "A_WHITE"
B_WHITE = "B_WHITE"
ORIENTATIONS = (A_WHITE, B_WHITE)

A_WIN = "A_WIN"
DRAW = "DRAW"
B_WIN = "B_WIN"
WDL_OUTCOMES = (A_WIN, DRAW, B_WIN)

_PGN_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2"})


def outcome_for_a_b(result: str, orientation: str) -> str:
    """Map a completed PGN result to A's/B's perspective for one orientation.

    ``orientation`` must explicitly be ``A_WHITE`` or ``B_WHITE``; a white
    win is assigned to whoever has White in that orientation.  Invalid PGN
    result strings are rejected rather than being silently counted as draws.
    """
    if orientation not in ORIENTATIONS:
        raise ValueError(f"unknown orientation: {orientation!r}")
    if result not in _PGN_RESULTS:
        raise ValueError(f"invalid completed result: {result!r}")
    if result == "1/2-1/2":
        return DRAW
    white_won = result == "1-0"
    a_won = white_won == (orientation == A_WHITE)
    return A_WIN if a_won else B_WIN


def wdl_summary(games: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return completed-game WDL probabilities and explicit censoring counts.

    Each game record must contain ``result``, ``censored``, and
    ``orientation``.  Censored records must use ``result=None`` and do not
    enter the WDL denominator; a no-completed-games report uses zero mass for
    each WDL outcome rather than treating censored games as draws.
    """
    counts = {outcome: 0 for outcome in WDL_OUTCOMES}
    censored_count = 0

    for game in games:
        try:
            censored = game["censored"]
            result = game["result"]
            orientation = game["orientation"]
        except KeyError as error:
            raise ValueError(f"game record missing {error.args[0]!r}") from error
        if not isinstance(censored, bool):
            raise TypeError("censored must be a bool")
        if censored:
            if result is not None:
                raise ValueError("censored games must have result=None")
            censored_count += 1
            continue
        if not isinstance(result, str):
            raise TypeError(f"completed result must be a string: {result!r}")
        outcome = game.get("_wdl_outcome")
        if outcome is None:
            outcome = outcome_for_a_b(result, str(orientation))
        elif outcome not in WDL_OUTCOMES:
            raise ValueError(f"invalid cached WDL outcome: {outcome!r}")
        counts[outcome] += 1

    completed_count = sum(counts.values())
    total_count = completed_count + censored_count
    distribution = {
        outcome: counts[outcome] / completed_count if completed_count else 0.0
        for outcome in WDL_OUTCOMES
    }
    return {
        "distribution": distribution,
        "completed_count": completed_count,
        "censored_count": censored_count,
        "completed_rate": completed_count / total_count if total_count else 0.0,
        "censored_rate": censored_count / total_count if total_count else 0.0,
    }


def total_variation(
    first: Mapping[str, float], second: Mapping[str, float]
) -> float:
    """Compute Total Variation distance over the union of both supports."""
    return 0.5 * sum(
        abs(first.get(value, 0.0) - second.get(value, 0.0))
        for value in first.keys() | second.keys()
    )


def wdl_orientation_distance(
    generated_by_orientation: Mapping[str, Sequence[Mapping[str, object]]],
    real_by_orientation: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Report WDL TV and both explicit WDL summaries for each orientation.

    The report's ``A_WHITE`` and ``B_WHITE`` entries each contain
    ``distance``, ``generated``, and ``real``.  The latter two are exactly
    :func:`wdl_summary` reports, so completed-only denominators and censoring
    rates cannot disappear when reporting a comparison.  Both samples must
    have at least one completed game in each orientation, because TV is not
    defined for the zero-mass report used to represent all-censored games.
    ``mean`` is the exact unpooled 50/50 mean of the two orientation distances.
    """
    reports = {}
    for orientation in ORIENTATIONS:
        try:
            generated = generated_by_orientation[orientation]
            real = real_by_orientation[orientation]
        except KeyError as error:
            raise ValueError(f"missing orientation: {error.args[0]!r}") from error
        generated_summary = wdl_summary(generated)
        real_summary = wdl_summary(real)
        if not generated_summary["completed_count"] or not real_summary["completed_count"]:
            raise ValueError(f"{orientation} has no completed games")
        reports[orientation] = {
            "distance": total_variation(
                generated_summary["distribution"], real_summary["distribution"]
            ),
            "generated": generated_summary,
            "real": real_summary,
        }
    return {
        **reports,
        "mean": (reports[A_WHITE]["distance"] + reports[B_WHITE]["distance"]) / 2,
    }
