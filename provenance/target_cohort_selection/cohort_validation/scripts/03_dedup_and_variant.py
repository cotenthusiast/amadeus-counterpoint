#!/usr/bin/env python3
"""Sections 4 & 5: dedup-view validation and standard-chess variant filtering,
focused on impact on the current best 8-player cohort. Read-only; recomputes
pair counts under different row filters using the same identity-resolution
rules as pairs.csv (see identity_resolution.py)."""
import itertools
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from identity_resolution import pair_counts
from lib import TARGET_8, NAME_TO_KEY, CV_BASE, connect

con = connect()

# ---- Section 4: dedup view validation ----
n_raw = con.execute("SELECT COUNT(*) FROM games").fetchone()[0]
n_raw_has_moves = con.execute("SELECT COUNT(*) FROM games WHERE has_moves").fetchone()[0]
n_dedup = con.execute("SELECT COUNT(*) FROM games_dedup").fetchone()[0]
n_removed_by_dedup = n_raw_has_moves - n_dedup

print(f"raw row count (all parsed records): {n_raw}")
print(f"raw, has_moves=True: {n_raw_has_moves}")
print(f"deduplicated (games_dedup) row count: {n_dedup}")
print(f"removed by content-hash dedup: {n_removed_by_dedup}")

pairs_raw = pair_counts(con, "games", extra_where="has_moves")
pairs_dedup = pair_counts(con, "games_dedup")

target_keys = {NAME_TO_KEY[n] for n in TARGET_8}


def cohort_pairs_only(df):
    return df[df.player_A_key.isin(target_keys) & df.player_B_key.isin(target_keys)].copy()


cp_raw = cohort_pairs_only(pairs_raw).set_index(["player_A_key", "player_B_key"])
cp_dedup = cohort_pairs_only(pairs_dedup).set_index(["player_A_key", "player_B_key"])

merged = cp_dedup[["player_A", "player_B", "total_games"]].join(
    cp_raw[["total_games"]], rsuffix="_raw", how="outer"
)
merged = merged.rename(columns={"total_games": "games_dedup", "total_games_raw": "games_raw_beforedup"})
merged["delta"] = merged["games_raw_beforedup"] - merged["games_dedup"]
merged = merged.sort_values("delta", ascending=False)
merged.to_csv(os.path.join(CV_BASE, "dedup_impact_on_8cohort.csv"))
print("\n8-cohort dyad dedup impact (raw has_moves count minus deduped count):")
print(merged.to_string())

n_affected = int((merged["delta"] != 0).sum())
print(f"\ndyads with any change from dedup: {n_affected} / {len(merged)}")

# ---- Section 5: variant filtering ----
variant_counts = con.execute("SELECT Variant, COUNT(*) n FROM games GROUP BY 1 ORDER BY 2 DESC").df()
print("\nVariant tag counts (all parsed records, before any filtering):")
print(variant_counts.to_string(index=False))

standard_where = "Variant = 'Standard'"
n_standard_dedup = con.execute(f"SELECT COUNT(*) FROM games_dedup WHERE {standard_where}").fetchone()[0]
n_nonstandard_dedup = n_dedup - n_standard_dedup
print(f"\nOf the {n_dedup} deduplicated rows: standard-chess = {n_standard_dedup}, non-standard/null-variant = {n_nonstandard_dedup}")

# does restricting to Variant='Standard' change the 8x8 matrix at all?
pairs_dedup_standard = pair_counts(con, "games_dedup", extra_where=standard_where)
cp_dedup_std = cohort_pairs_only(pairs_dedup_standard).set_index(["player_A_key", "player_B_key"])

merged2 = cp_dedup[["player_A", "player_B", "total_games"]].join(
    cp_dedup_std[["total_games"]], rsuffix="_std", how="outer"
).fillna(0)
merged2 = merged2.rename(columns={"total_games": "games_dedup_allvariants", "total_games_std": "games_dedup_standardonly"})
merged2["delta"] = merged2["games_dedup_allvariants"] - merged2["games_dedup_standardonly"]
merged2.to_csv(os.path.join(CV_BASE, "variant_filter_impact_on_8cohort.csv"))
print("\n8-cohort dyad impact of restricting to Variant='Standard':")
print(merged2.to_string())
n_affected2 = int((merged2["delta"] != 0).sum())
print(f"\ndyads with any change from variant filtering: {n_affected2} / {len(merged2)}")

# any target-player games at all in a non-standard variant? (identity-resolved,
# not a raw per-row FIDE-tag filter, which would undercount -- a player's total
# games includes rows where their own occurrence lacks a FIDE tag but their
# opponent's, or neither, is tagged; joining via resolved names as in
# 04_dyad_adequacy.py's g_resolved avoids that undercount)
from identity_resolution import build_resolved_map  # noqa: E402
resolved_all = build_resolved_map(con, "games")
resolved_all_df = pd.DataFrame([{"raw_name": k, "id_key": v[0]} for k, v in resolved_all.items()])
con.register("resolved_all_df", resolved_all_df)
con.execute("""
CREATE TEMP TABLE g_resolved_all AS
SELECT g.*, rw.id_key AS white_key, rb.id_key AS black_key
FROM games_dedup g
JOIN resolved_all_df rw ON g.White = rw.raw_name
JOIN resolved_all_df rb ON g.Black = rb.raw_name
""")
target_keys_sql = ",".join(f"'{k}'" for k in target_keys)
n_target_nonstandard = con.execute(f"""
SELECT COUNT(*) FROM g_resolved_all
WHERE Variant <> 'Standard' AND (white_key IN ({target_keys_sql}) OR black_key IN ({target_keys_sql}))
""").fetchone()[0]
print(f"\ndeduplicated games involving any of the 8 target players (identity-resolved) tagged with a non-Standard variant: {n_target_nonstandard}")

# ---- persist full-corpus standard+dedup pair table and per-player totals, for
# re-running the cohort search on the corrected basis (script 01b) ----
pairs_dedup_standard.to_csv(os.path.join(CV_BASE, "pairs_standard_dedup.csv"), index=False)
player_totals_std = con.execute("""
SELECT white_key AS id_key, COUNT(*) n FROM g_resolved_all WHERE Variant='Standard' GROUP BY 1
UNION ALL
SELECT black_key AS id_key, COUNT(*) n FROM g_resolved_all WHERE Variant='Standard' GROUP BY 1
""").df().groupby("id_key")["n"].sum().reset_index().rename(columns={"n": "total_games_standard_dedup"})
player_totals_std.to_csv(os.path.join(CV_BASE, "player_totals_standard_dedup.csv"), index=False)
print(f"wrote pairs_standard_dedup.csv ({len(pairs_dedup_standard)} pairs) and player_totals_standard_dedup.csv ({len(player_totals_std)} players)")

summary = {
    "n_raw_all_records": n_raw,
    "n_raw_has_moves": n_raw_has_moves,
    "n_dedup": n_dedup,
    "n_removed_by_content_dedup": n_removed_by_dedup,
    "n_8cohort_dyads_affected_by_dedup": n_affected,
    "n_standard_dedup": int(n_standard_dedup),
    "n_nonstandard_or_null_dedup": int(n_nonstandard_dedup),
    "n_8cohort_dyads_affected_by_variant_filter": n_affected2,
    "n_target_player_games_nonstandard_variant": int(n_target_nonstandard),
}
with open(os.path.join(CV_BASE, "dedup_variant_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {os.path.join(CV_BASE, 'dedup_variant_summary.json')}")
