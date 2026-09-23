#!/usr/bin/env python3
"""Section 7: leave-one-opponent-out fold sizing, for every ordered pair A[-B]
among the current best 8-player cohort (56 folds). Basis: deduplicated,
Variant='Standard' games only (the corrected basis from 03/05/06)."""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from identity_resolution import build_resolved_map
from lib import TARGET_8, NAME_TO_KEY, CV_BASE, connect

con = connect()
target_keys = {n: NAME_TO_KEY[n] for n in TARGET_8}

resolved = build_resolved_map(con, "games")
resolved_df = pd.DataFrame([{"raw_name": k, "id_key": v[0]} for k, v in resolved.items()])
con.register("resolved_df", resolved_df)
con.execute("""
CREATE TEMP TABLE g_std AS
SELECT g.*, rw.id_key AS white_key, rb.id_key AS black_key
FROM games_dedup g
JOIN resolved_df rw ON g.White = rw.raw_name
JOIN resolved_df rb ON g.Black = rb.raw_name
WHERE g.Variant = 'Standard'
""")

# per-player: total games, total plies, unique opponents (standard+dedup basis)
per_player = {}
for name, key in target_keys.items():
    row = con.execute(f"""
    WITH appear AS (
      SELECT black_key AS opp, ply_count FROM g_std WHERE white_key = '{key}'
      UNION ALL
      SELECT white_key AS opp, ply_count FROM g_std WHERE black_key = '{key}'
    )
    SELECT COUNT(*) n, COUNT(DISTINCT opp) n_opp, SUM(ply_count) total_plies
    FROM appear
    """).fetchone()
    per_player[name] = {"total_games": row[0], "n_opponents": row[1], "total_plies": row[2]}

rows = []
for a, b in itertools.permutations(TARGET_8, 2):
    fa, fb = target_keys[a], target_keys[b]
    a_total = per_player[a]["total_games"]
    a_opp_total = per_player[a]["n_opponents"]

    # games A had against B specifically (standard+dedup)
    n_vs_b = con.execute(f"""
    SELECT COUNT(*) FROM g_std
    WHERE (white_key='{fa}' AND black_key='{fb}') OR (white_key='{fb}' AND black_key='{fa}')
    """).fetchone()[0]

    # of A's remaining (non-B) games, plies, and how many are vs other cohort members vs outside
    other_cohort = [target_keys[p] for p in TARGET_8 if p not in (a, b)]
    other_cohort_sql = ",".join(f"'{k}'" for k in other_cohort)
    row = con.execute(f"""
    WITH appear AS (
      SELECT black_key AS opp, ply_count FROM g_std WHERE white_key = '{fa}' AND black_key <> '{fb}'
      UNION ALL
      SELECT white_key AS opp, ply_count FROM g_std WHERE black_key = '{fa}' AND white_key <> '{fb}'
    )
    SELECT COUNT(*) n, COUNT(DISTINCT opp) n_opp, SUM(ply_count) plies,
           COUNT(*) FILTER (WHERE opp IN ({other_cohort_sql})) n_vs_other_cohort
    FROM appear
    """).fetchone()

    remaining_games, remaining_opponents, remaining_plies, vs_other_cohort = row
    remaining_plies = remaining_plies or 0
    vs_outside_cohort = remaining_games - vs_other_cohort

    rows.append({
        "player_A": a, "held_out_opponent_B": b,
        "A_total_usable_games_std_dedup": a_total,
        "A_total_unique_opponents": a_opp_total,
        "games_removed_because_opponent_is_B": n_vs_b,
        "remaining_training_games_for_A": remaining_games,
        "remaining_unique_opponents_for_A": remaining_opponents,
        "remaining_games_vs_other_6_cohort_members": vs_other_cohort,
        "remaining_games_vs_outside_cohort_players": vs_outside_cohort,
        "remaining_approx_plies": int(remaining_plies),
    })

folds = pd.DataFrame(rows)
folds.to_csv(os.path.join(CV_BASE, "loo_folds.csv"), index=False)
print(folds.to_string(index=False))
print(f"\n{len(folds)} folds written")
print(f"\nminimum remaining_training_games_for_A across all folds: {folds.remaining_training_games_for_A.min()}")
print(f"minimum remaining_approx_plies across all folds: {folds.remaining_approx_plies.min()}")
