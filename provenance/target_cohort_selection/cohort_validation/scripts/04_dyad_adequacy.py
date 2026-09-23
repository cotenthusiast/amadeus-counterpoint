#!/usr/bin/env python3
"""Section 6: per-dyad real-data adequacy for the current best 8-player cohort's
28 unordered dyads. Metrics computed directly from games_dedup (standard-chess
+ dedup is the baseline used for cohort selection); missing-Elo/missing-Date and
dedup/variant proportions are computed against the corresponding raw rows.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from identity_resolution import build_resolved_map
from lib import TARGET_8, NAME_TO_KEY, CV_BASE, connect

con = connect()

target_keys = {n: NAME_TO_KEY[n] for n in TARGET_8}

# Resolve identities the SAME way pairs.csv does (shared valid FIDE ID, or
# unambiguous canonical name), then join back onto games_dedup so a game counts
# for a player even on rows where that row's own FIDE-ID tag is missing/invalid
# -- matching pairs.csv's actual counting method (07_pairs.py), not a naive
# per-row FIDE-tag filter (which would undercount by ~2/3, since ~33% of rows
# have no FIDE ID tag at all).
# Built once on "games WHERE has_moves" (a strict superset of every distinct raw
# name in games_dedup, since dedup only removes exact-duplicate rows, never
# introduces a new name) and reused for both views below -- avoids rebuilding
# the corpus-wide union-find twice.
resolved = build_resolved_map(con, "games")
resolved_df = pd.DataFrame([{"raw_name": k, "id_key": v[0]} for k, v in resolved.items()])
con.register("resolved_df", resolved_df)
con.execute("""
CREATE TEMP TABLE g_resolved AS
SELECT g.*, rw.id_key AS white_key, rb.id_key AS black_key
FROM games_dedup g
JOIN resolved_df rw ON g.White = rw.raw_name
JOIN resolved_df rb ON g.Black = rb.raw_name
""")
con.execute("""
CREATE TEMP TABLE raw_resolved AS
SELECT g.*, rw.id_key AS white_key, rb.id_key AS black_key
FROM games g
JOIN resolved_df rw ON g.White = rw.raw_name
JOIN resolved_df rb ON g.Black = rb.raw_name
WHERE g.has_moves
""")

rows = []
for a, b in itertools.combinations(TARGET_8, 2):
    fa, fb = target_keys[a], target_keys[b]
    df_all = con.execute(f"""
    SELECT
      CASE WHEN white_key = '{fa}' THEN 'A' ELSE 'B' END AS a_color,
      Result AS result, WhiteElo_i, BlackElo_i, date_norm, Variant
    FROM g_resolved
    WHERE (white_key = '{fa}' AND black_key = '{fb}')
       OR (white_key = '{fb}' AND black_key = '{fa}')
    """).df()
    n_all = len(df_all)
    if n_all == 0:
        continue
    # STANDARD-CHESS ONLY subset -- this is the authoritative view (matches
    # sections 1/2/5's corrected min-edge=29 basis), since Chess960/Freestyle
    # and from-position games are not part of the paper's standard-chess dataset.
    df = df_all[df_all.Variant == "Standard"].copy()
    n = len(df)
    a_white = int((df.a_color == 'A').sum())
    b_white = n - a_white
    draws = int((df.result == '1/2-1/2').sum())
    a_wins = int((((df.a_color == 'A') & (df.result == '1-0')) | ((df.a_color == 'B') & (df.result == '0-1'))).sum())
    b_wins = int((((df.a_color == 'A') & (df.result == '0-1')) | ((df.a_color == 'B') & (df.result == '1-0'))).sum())
    other_result = n - draws - a_wins - b_wins
    missing_elo = int((df.WhiteElo_i.isna() | df.BlackElo_i.isna()).sum()) if n else 0
    missing_date = int(df.date_norm.isna().sum()) if n else 0
    ratings = pd.concat([df.WhiteElo_i, df.BlackElo_i]).dropna()

    # dedup impact: how many raw (pre-dedup), standard-chess rows collapsed into
    # this dyad's deduplicated, standard-chess rows
    df_raw = con.execute(f"""
    SELECT COUNT(*) n FROM raw_resolved
    WHERE Variant = 'Standard' AND (
      (white_key = '{fa}' AND black_key = '{fb}')
       OR (white_key = '{fb}' AND black_key = '{fa}')
    )
    """).fetchone()[0]

    rows.append({
        "player_A": a, "player_B": b,
        "games_standard_dedup": n,
        "games_all_variant_dedup": n_all,
        "pct_nonstandard_variant": round(100 * (n_all - n) / n_all, 1),
        "A_as_white": a_white, "B_as_white": b_white,
        "draws": draws, "A_wins": a_wins, "B_wins": b_wins, "other_result_token": other_result,
        "first_date": str(df.date_norm.min()) if n and df.date_norm.notna().any() else None,
        "last_date": str(df.date_norm.max()) if n and df.date_norm.notna().any() else None,
        "rating_min": float(ratings.min()) if len(ratings) else None,
        "rating_median": float(ratings.median()) if len(ratings) else None,
        "rating_max": float(ratings.max()) if len(ratings) else None,
        "pct_missing_elo": round(100 * missing_elo / n, 1) if n else None,
        "pct_missing_date": round(100 * missing_date / n, 1) if n else None,
        "raw_pre_dedup_standard_rows": int(df_raw),
        "pct_removed_by_dedup": round(100 * (df_raw - n) / df_raw, 1) if df_raw else 0.0,
    })

adequacy = pd.DataFrame(rows).sort_values("games_standard_dedup")


def flag(row):
    # threshold rationale (see report): >=60 real standard-chess games between
    # the two individuals = STRONG (enough to see the draw/decisive split settle
    # down and support a real held-out evaluation distribution); 40-59 = USABLE
    # (the cohort's own selection floor was 42, so this band is "as good as the
    # cohort search could find" but thin); <40 = WEAK.
    if row.games_standard_dedup >= 60:
        return "STRONG"
    if row.games_standard_dedup >= 40:
        return "USABLE"
    return "WEAK"


adequacy["flag"] = adequacy.apply(flag, axis=1)
adequacy.to_csv(os.path.join(CV_BASE, "dyad_adequacy_8cohort.csv"), index=False)
print(adequacy.to_string(index=False))
print("\nflag counts:")
print(adequacy.flag.value_counts())
