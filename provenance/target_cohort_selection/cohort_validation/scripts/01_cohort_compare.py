#!/usr/bin/env python3
"""Sections 1 & 2: cohort-size comparison table + full matrix outputs for the best
6/7/8/9 cohorts. Reads only pairs.csv / players_normalized.csv (already-computed,
read-only derived views over games.parquet) -- touches no source PGNs.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import BEST_COHORTS, NAME_TO_KEY, PLAYERS_NORM, cohort_matrix, MATRICES_DIR, CV_BASE

os.makedirs(MATRICES_DIR, exist_ok=True)

EDGE_BUCKETS = [20, 30, 40, 50, 100]


def stats_for_cohort(k, names):
    mat, edges = cohort_matrix(names)
    edge_vals = np.array(list(edges.values()), dtype=float)
    keys = [NAME_TO_KEY[n] for n in names]
    prow = PLAYERS_NORM.set_index("id_key").loc[keys]
    total_games = dict(zip(names, prow.total_games))
    internal_for = {}
    for a in names:
        internal_for[a] = sum(v for (x, y), v in edges.items() if a in (x, y))
    outside_for = {a: total_games[a] - internal_for[a] for a in names}

    first_dates = pd.to_datetime(prow.first_date)
    last_dates = pd.to_datetime(prow.last_date)

    n_lt = {t: int((edge_vals < t).sum()) for t in EDGE_BUCKETS}
    n_ge50 = int((edge_vals >= 50).sum())
    n_ge100 = int((edge_vals >= 100).sum())

    return {
        "size": k,
        "players": names,
        "n_dyads": len(edges),
        "min_edge": int(edge_vals.min()),
        "p10_edge": float(np.percentile(edge_vals, 10)),
        "median_edge": float(np.median(edge_vals)),
        "mean_edge": float(np.mean(edge_vals)),
        "max_edge": int(edge_vals.max()),
        "total_internal_games": int(edge_vals.sum()),
        "min_outside_cohort_games": int(min(outside_for.values())),
        "median_outside_cohort_games": float(np.median(list(outside_for.values()))),
        "min_total_games_any_player": int(min(total_games.values())),
        "date_coverage_start": str(first_dates.min().date()),
        "date_coverage_end": str(last_dates.max().date()),
        "edges_lt_20": n_lt[20],
        "edges_lt_30": n_lt[30],
        "edges_lt_40": n_lt[40],
        "edges_lt_50": n_lt[50],
        "edges_ge_50": n_ge50,
        "edges_ge_100": n_ge100,
        "matrix": mat,
        "edges_dict": edges,
        "total_games": total_games,
        "internal_games": internal_for,
        "outside_games": outside_for,
    }


results = {k: stats_for_cohort(k, names) for k, names in BEST_COHORTS.items()}

# ---- comparison table (section 1) ----
rows = []
for k, r in results.items():
    rows.append({
        "cohort_size": k,
        "players": ", ".join(r["players"]),
        "n_dyads": r["n_dyads"],
        "min_edge": r["min_edge"],
        "p10_edge": round(r["p10_edge"], 1),
        "median_edge": r["median_edge"],
        "mean_edge": round(r["mean_edge"], 1),
        "max_edge": r["max_edge"],
        "total_internal_games": r["total_internal_games"],
        "min_outside_cohort_games": r["min_outside_cohort_games"],
        "median_outside_cohort_games": r["median_outside_cohort_games"],
        "min_total_games_any_player": r["min_total_games_any_player"],
        "date_coverage": f"{r['date_coverage_start']} .. {r['date_coverage_end']}",
        "edges_<20": r["edges_lt_20"],
        "edges_<30": r["edges_lt_30"],
        "edges_<40": r["edges_lt_40"],
        "edges_<50": r["edges_lt_50"],
        "edges_>=50": r["edges_ge_50"],
        "edges_>=100": r["edges_ge_100"],
    })
comp_df = pd.DataFrame(rows)
comp_df.to_csv(os.path.join(CV_BASE, "cohort_size_comparison.csv"), index=False)
print(comp_df.to_string(index=False))

# ---- full matrix outputs (section 2) per size ----
for k, r in results.items():
    r["matrix"].to_csv(os.path.join(MATRICES_DIR, f"matrix_{k}.csv"))
    sorted_edges = sorted(r["edges_dict"].items(), key=lambda x: x[1])
    edge_df = pd.DataFrame(
        [{"player_A": a, "player_B": b, "games": v} for (a, b), v in sorted_edges]
    )
    edge_df.to_csv(os.path.join(MATRICES_DIR, f"edges_sorted_{k}.csv"), index=False)

    per_player = pd.DataFrame([
        {
            "player": p,
            "total_games": r["total_games"][p],
            "internal_games": r["internal_games"][p],
            "outside_cohort_games": r["outside_games"][p],
        }
        for p in r["players"]
    ])
    per_player.to_csv(os.path.join(MATRICES_DIR, f"per_player_{k}.csv"), index=False)

    print(f"\n=== size {k} ===")
    print("weakest 5:", sorted_edges[:5])
    print("strongest 5:", sorted_edges[-5:])

comp_df.to_json(os.path.join(CV_BASE, "cohort_size_comparison.json"), orient="records", indent=2)
print(f"\nwrote {os.path.join(CV_BASE, 'cohort_size_comparison.csv')}")
print(f"wrote matrices/ under {MATRICES_DIR}")
