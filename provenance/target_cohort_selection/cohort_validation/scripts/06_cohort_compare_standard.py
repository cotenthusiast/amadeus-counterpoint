#!/usr/bin/env python3
"""Rebuild the section-1/2 comparison table and matrices on the CORRECTED
(dedup + Variant='Standard') basis, using the re-search results from
05_cohort_research_standard.py. This supersedes 01_cohort_compare.py's numbers
as the authoritative comparison for the final report (see 03_dedup_and_variant.py
for why the correction matters)."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import CV_BASE, MATRICES_DIR

with open(os.path.join(CV_BASE, "cohort_search_standard_results.pkl"), "rb") as f:
    results_by_size = pickle.load(f)

dates = pd.read_csv(os.path.join(CV_BASE, "player_dates_standard_dedup.csv"))
dates_map = dates.set_index("id_key")[["first_date", "last_date"]].to_dict("index")

EDGE_BUCKETS = [20, 30, 40, 50, 100]
rows = []
for k, records in results_by_size.items():
    rec = records[0]  # best candidate
    edge_vals = np.array(list(rec["edges"].values()), dtype=float)
    keys = rec["keys"]
    names = rec["names"]
    first_dates = [dates_map[key]["first_date"] for key in keys if key in dates_map]
    last_dates = [dates_map[key]["last_date"] for key in keys if key in dates_map]

    n_lt = {t: int((edge_vals < t).sum()) for t in EDGE_BUCKETS}

    rows.append({
        "cohort_size": k,
        "players": ", ".join(names),
        "n_dyads": len(edge_vals),
        "min_edge": int(edge_vals.min()),
        "p10_edge": float(np.percentile(edge_vals, 10)),
        "median_edge": float(np.median(edge_vals)),
        "mean_edge": round(float(np.mean(edge_vals)), 1),
        "max_edge": int(edge_vals.max()),
        "total_internal_games": int(edge_vals.sum()),
        "min_outside_cohort_games": int(min(rec["outside_games"].values())),
        "median_outside_cohort_games": float(np.median(list(rec["outside_games"].values()))),
        "min_total_games_any_player": int(min(rec["total_games"].values())),
        "date_coverage": f"{min(first_dates)} .. {max(last_dates)}" if first_dates else "n/a",
        "edges_<20": n_lt[20], "edges_<30": n_lt[30], "edges_<40": n_lt[40], "edges_<50": n_lt[50],
        "edges_>=50": int((edge_vals >= 50).sum()), "edges_>=100": int((edge_vals >= 100).sum()),
    })

    # matrix + per-player CSVs (standard basis)
    mat = pd.DataFrame(0, index=names, columns=names)
    for (ka, kb), v in rec["edges"].items():
        na = names[keys.index(ka)]
        nb = names[keys.index(kb)]
        mat.loc[na, nb] = v
        mat.loc[nb, na] = v
    mat.to_csv(os.path.join(MATRICES_DIR, f"matrix_{k}_standard.csv"))

    sorted_edges = sorted(rec["edges"].items(), key=lambda x: x[1])
    edge_rows = [{"player_A": names[keys.index(a)], "player_B": names[keys.index(b)], "games": v}
                 for (a, b), v in sorted_edges]
    pd.DataFrame(edge_rows).to_csv(os.path.join(MATRICES_DIR, f"edges_sorted_{k}_standard.csv"), index=False)

    per_player = pd.DataFrame([
        {"player": names[keys.index(kk)], "total_games_standard_dedup": rec["total_games"][kk],
         "internal_games": rec["total_games"][kk] - rec["outside_games"][kk],
         "outside_cohort_games": rec["outside_games"][kk]}
        for kk in keys
    ])
    per_player.to_csv(os.path.join(MATRICES_DIR, f"per_player_{k}_standard.csv"), index=False)

    print(f"size {k}: weakest 5 = {sorted_edges[:5]}")
    print(f"size {k}: strongest 5 = {sorted_edges[-5:]}")

comp_df = pd.DataFrame(rows)
comp_df.to_csv(os.path.join(CV_BASE, "cohort_size_comparison_standard.csv"), index=False)
print(comp_df.to_string(index=False))
print(f"\nwrote {os.path.join(CV_BASE, 'cohort_size_comparison_standard.csv')}")
