#!/usr/bin/env python3
"""Reusable identity-resolution routine, factored out of audit/scripts/07_pairs.py
so sections 4/5/6/7 can recompute pair/appearance tables under different row
filters (raw vs dedup, standard-only vs all-variants) using the IDENTICAL
merge rules already used to build the frozen pairs.csv (shared valid FIDE ID,
or unambiguous canonical-name match). No new merge logic is introduced.
"""
import os
import sys
from collections import defaultdict

import pandas as pd
from text_unidecode import unidecode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from common import connect  # noqa: E402


def canon_key(name: str) -> str:
    s = unidecode(str(name)).lower()
    s = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in s)
    return " ".join(sorted(t for t in s.split() if t))


def build_resolved_map(con, source_table="games_dedup"):
    """Returns dict: raw_name -> (id_key, canonical_name), using ALL raw names
    appearing in `source_table` (so it's consistent whichever row-filter view
    is passed in)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    name_fide_pairs = con.execute(f"""
    SELECT WhiteFideId_valid AS fide, White AS name FROM {source_table} WHERE WhiteFideId_valid IS NOT NULL
    UNION
    SELECT BlackFideId_valid AS fide, Black AS name FROM {source_table} WHERE BlackFideId_valid IS NOT NULL
    """).df()

    for fide, grp in name_fide_pairs.groupby("fide")["name"]:
        names = list(grp.unique())
        for n in names[1:]:
            union(names[0], n)

    canon_to_fides = defaultdict(set)
    for _, row in name_fide_pairs.iterrows():
        canon_to_fides[canon_key(row["name"])].add(row["fide"])
    ambiguous_canon_keys = {k for k, fides in canon_to_fides.items() if len(fides) > 1}

    all_names = con.execute(f"""
    SELECT DISTINCT name FROM (
      SELECT White AS name FROM {source_table} WHERE White IS NOT NULL
      UNION ALL SELECT Black FROM {source_table} WHERE Black IS NOT NULL
    )
    """).df()["name"].tolist()

    key_to_names = defaultdict(list)
    for n in all_names:
        key_to_names[canon_key(n)].append(n)

    for key, names in key_to_names.items():
        if not key or len(names) < 2 or key in ambiguous_canon_keys:
            continue
        for n in names[1:]:
            union(names[0], n)

    name_game_counts = con.execute(f"""
    WITH allnames AS (
      SELECT White AS name FROM {source_table} WHERE White IS NOT NULL
      UNION ALL SELECT Black FROM {source_table} WHERE Black IS NOT NULL
    )
    SELECT name, COUNT(*) n FROM allnames GROUP BY 1
    """).df().set_index("name")["n"].to_dict()

    component_names = defaultdict(list)
    for n in all_names:
        component_names[find(n)].append(n)

    # per-name fide occurrence counts, precomputed once (vectorized) instead of a
    # per-component isin() scan over the whole name_fide_pairs table -- that scan
    # is O(n_components * len(name_fide_pairs)) and was the actual runtime
    # bottleneck (component_names has ~1 entry per distinct raw name, i.e. can be
    # in the hundreds of thousands).
    name_fide_counts = (
        name_fide_pairs.groupby(["name", "fide"]).size().rename("cnt").reset_index()
    )
    per_name_fide_counter = defaultdict(dict)
    for row in name_fide_counts.itertuples(index=False):
        per_name_fide_counter[row.name][row.fide] = row.cnt

    component_fide = {}
    for row in name_fide_pairs.itertuples(index=False):
        root = find(row.name)
        component_fide.setdefault(root, set()).add(row.fide)

    resolved = {}
    for root, members in component_names.items():
        fides = component_fide.get(root, set())
        canonical_name = max(members, key=lambda n: name_game_counts.get(n, 0))
        if len(fides) >= 1:
            combined = defaultdict(int)
            for m in members:
                for fide, cnt in per_name_fide_counter.get(m, {}).items():
                    combined[fide] += cnt
            chosen_fide = max(combined, key=combined.get)
            id_key = f"F:{chosen_fide}"
        else:
            id_key = f"N:{canonical_name}"
        for n in members:
            resolved[n] = (id_key, canonical_name)
    return resolved


def pair_counts(con, source_table="games_dedup", extra_where="TRUE"):
    """Returns a DataFrame of unordered-pair game counts over `source_table`,
    filtered additionally by `extra_where`, using the standard identity
    resolution. Columns: player_A_key, player_B_key, player_A, player_B, total_games."""
    resolved = build_resolved_map(con, source_table)
    resolved_df = pd.DataFrame(
        [{"raw_name": k, "id_key": v[0], "id_name": v[1]} for k, v in resolved.items()]
    )
    con.register("resolved_df_tmp", resolved_df)
    df = con.execute(f"""
    WITH appearances AS (
      SELECT g.row_id, rw.id_key AS a_key, rb.id_key AS b_key, rw.id_name AS a_name, rb.id_name AS b_name
      FROM {source_table} g
      JOIN resolved_df_tmp rw ON g.White = rw.raw_name
      JOIN resolved_df_tmp rb ON g.Black = rb.raw_name
      WHERE g.White IS NOT NULL AND g.Black IS NOT NULL AND ({extra_where})
    ),
    ordered AS (
      SELECT LEAST(a_key,b_key) AS player_A_key, GREATEST(a_key,b_key) AS player_B_key,
             CASE WHEN a_key < b_key THEN a_name ELSE b_name END AS player_A,
             CASE WHEN a_key < b_key THEN b_name ELSE a_name END AS player_B
      FROM appearances WHERE a_key <> b_key
    )
    SELECT player_A_key, player_B_key, MODE(player_A) player_A, MODE(player_B) player_B, COUNT(*) total_games
    FROM ordered GROUP BY player_A_key, player_B_key
    """).df()
    con.unregister("resolved_df_tmp")
    return df


if __name__ == "__main__":
    con = connect()
    df = pair_counts(con, "games_dedup")
    print(df.sort_values("total_games", ascending=False).head())
    print(f"total pairs: {len(df)}")
