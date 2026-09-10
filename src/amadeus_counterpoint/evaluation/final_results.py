"""Assemble one serializable results object: WDL/opening metrics, bootstrap
uncertainty, and provenance. No plotting, no tables, no paper figures --
this is machine-readable output for a later analysis session.
"""

import json
from pathlib import Path

from amadeus_counterpoint.evaluation.aggregate import CONDITIONS, aggregate_equal_dyad_weight

REQUIRED_PROVENANCE_KEYS = frozenset({
    "base_checkpoint_identity",
    "personalization_checkpoint_identity",
    "representative_elo_identity",
    "opening_taxonomy_commit",
    "root_seed",
    "games_per_orientation",
    "protocol_version",
    "code_commit",
})


def build_final_results(
    method: str,
    results_table: dict,
    bootstrap_results: dict,
    provenance: dict,
) -> dict:
    """Combine per-dyad metrics, bootstrap results, and provenance.

    `results_table`: `aggregate.build_results_table`'s output.
    `bootstrap_results[dyad][condition]` -> `bootstrap_wiring.bootstrap_dyad`
    output for that dyad/condition -- assembled by the caller, not computed
    here.
    `provenance` must contain at least `REQUIRED_PROVENANCE_KEYS` (base
    checkpoint identity, personalization checkpoint identity, representative
    Elo mapping identity, opening taxonomy commit SHA, root seed, game
    count, protocol version, code commit) -- see
    EVALUATION_PROTOCOL.local.md Sec. 14. Extra keys are preserved as-is.

    Returns one dict: `{"method", "provenance", "per_dyad", "bootstrap",
    "equal_weight_aggregate"}`.
    """
    missing = REQUIRED_PROVENANCE_KEYS - provenance.keys()
    if missing:
        raise ValueError(f"provenance is missing required keys: {sorted(missing)}")

    equal_weight_aggregate = {
        condition: {
            "wdl": aggregate_equal_dyad_weight(results_table, condition, "wdl"),
            "opening": aggregate_equal_dyad_weight(results_table, condition, "opening"),
        }
        for condition in CONDITIONS
    }

    return {
        "method": method,
        "provenance": provenance,
        "per_dyad": results_table,
        "bootstrap": bootstrap_results,
        "equal_weight_aggregate": equal_weight_aggregate,
    }


def write_final_results(results: dict, path) -> None:
    Path(path).write_text(json.dumps(results, indent=2, sort_keys=True))


def read_final_results(path) -> dict:
    return json.loads(Path(path).read_text())
