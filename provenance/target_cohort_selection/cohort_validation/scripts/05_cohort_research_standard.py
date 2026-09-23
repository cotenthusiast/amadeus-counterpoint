#!/usr/bin/env python3
"""Re-run the exhaustive cohort search (originally audit/scripts/08_cohort.py)
on the CORRECTED basis: content-hash-deduplicated AND Variant='Standard' only.

Motivation: 03_dedup_and_variant.py found that restricting to Variant='Standard'
changes every one of the current 8-cohort's 28 dyad counts, some by >60%
(e.g. Carlsen-Aronian 94 -> 32), because a non-trivial number of these elite
players' games are Chess960/Freestyle-Chess events or from-position "Playzone"
games, not standard chess from the start position. The original cohort search
(08_cohort.py) was run on dedup-but-all-variants pairs.csv, so its headline
numbers (min edge 42, etc.) do not reflect the standard-chess-only reality the
paper's dataset actually needs. This script re-derives the best 6/7/8/9-player
cohorts on the corrected pair table, for an apples-to-apples comparison against
the original search.
"""
import itertools
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import AUDIT_BASE, CV_BASE

EXACT_POOL_SIZE = 24
WIDE_POOL_SIZE = 70
SIZES = [6, 7, 8, 9]
random.seed(0)

players_norm = pd.read_csv(os.path.join(AUDIT_BASE, "players_normalized.csv"))
totals_std = pd.read_csv(os.path.join(CV_BASE, "player_totals_standard_dedup.csv"))
pairs_std = pd.read_csv(os.path.join(CV_BASE, "pairs_standard_dedup.csv"))

players = players_norm.merge(totals_std, on="id_key", how="inner")
players = players[players.id_key.str.startswith("F:") & (players.title.fillna("") != "BOT")]
players = players.sort_values("total_games_standard_dedup", ascending=False).reset_index(drop=True)

wide_pool = players.head(WIDE_POOL_SIZE).reset_index(drop=True)
wide_keys = wide_pool.id_key.tolist()
wide_index = {k: i for i, k in enumerate(wide_keys)}
n_wide = len(wide_keys)

A = np.zeros((n_wide, n_wide), dtype=np.int64)
sub = pairs_std[pairs_std.player_A_key.isin(wide_index) & pairs_std.player_B_key.isin(wide_index)]
for r in sub.itertuples():
    i, j = wide_index[r.player_A_key], wide_index[r.player_B_key]
    A[i, j] = r.total_games
    A[j, i] = r.total_games

name_of = dict(zip(wide_pool.id_key, wide_pool.player_name))
total_games_of = dict(zip(wide_pool.id_key, wide_pool.total_games_standard_dedup))
title_of = dict(zip(wide_pool.id_key, wide_pool.title))


def score(idxs):
    edges = [A[i, j] for a, i in enumerate(idxs) for j in idxs[a + 1:]]
    edges = np.array(edges)
    return (int(edges.min()), float(np.median(edges)), int(edges.sum()))


def cohort_record(idxs):
    keys = [wide_keys[i] for i in idxs]
    edges = {}
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            edges[(keys[a], keys[b])] = int(A[idxs[a], idxs[b]])
    edge_vals = list(edges.values())
    outside = {}
    for k in keys:
        i = wide_index[k]
        internal_for_k = sum(A[i, wide_index[k2]] for k2 in keys if k2 != k)
        outside[k] = total_games_of[k] - internal_for_k
    return {
        "keys": keys, "names": [name_of[k] for k in keys], "edges": edges,
        "min_edge": min(edge_vals), "median_edge": float(np.median(edge_vals)),
        "mean_edge": float(np.mean(edge_vals)), "total_internal": sum(edge_vals),
        "total_games": {k: total_games_of[k] for k in keys}, "outside_games": outside,
    }


results = {}
for k in SIZES:
    exact_idxs = list(range(min(EXACT_POOL_SIZE, n_wide)))
    best = [(score(combo), combo) for combo in itertools.combinations(exact_idxs, k)]
    best.sort(key=lambda x: x[0], reverse=True)
    top_exact = best[:20]

    def local_search(seed_combo, iters=4000):
        current = set(seed_combo)
        cur_score = score(tuple(sorted(current)))
        rng = random.Random(42 + k)
        pool_idxs = list(range(n_wide))
        for _ in range(iters):
            i_out = rng.choice(list(current))
            i_in = rng.choice([i for i in pool_idxs if i not in current])
            trial = (current - {i_out}) | {i_in}
            trial_score = score(tuple(sorted(trial)))
            if trial_score > cur_score:
                current, cur_score = trial, trial_score
        return tuple(sorted(current)), cur_score

    ls_combo, ls_score = local_search(top_exact[0][1])
    if ls_score > top_exact[0][0]:
        top_exact.insert(0, (ls_score, ls_combo))
        top_exact.sort(key=lambda x: x[0], reverse=True)

    seen, distinct = set(), []
    for s, combo in top_exact:
        ck = tuple(sorted(combo))
        if ck in seen:
            continue
        seen.add(ck)
        distinct.append((s, combo))
        if len(distinct) >= 5:
            break

    results[k] = [cohort_record(combo) for s, combo in distinct]
    print(f"k={k}: best (standard+dedup) min_edge={distinct[0][0][0]}, median={distinct[0][0][1]}, total={distinct[0][0][2]}")
    print(f"       players: {results[k][0]['names']}")


def fmt_cohort_md(k, records):
    lines = [f"# Cohort search (STANDARD-CHESS + DEDUP basis) -- size {k}\n"]
    lines.append(f"Candidate pool: top {EXACT_POOL_SIZE} FIDE-identified, non-BOT players by "
                 f"total STANDARD-variant, deduplicated games (exhaustive C({EXACT_POOL_SIZE},{k})), "
                 f"cross-checked with swap-based local search over top {WIDE_POOL_SIZE}.\n")
    for rank, rec in enumerate(records, 1):
        lines.append(f"\n## Candidate #{rank}\n")
        lines.append(f"- min pairwise edge: **{rec['min_edge']}**")
        lines.append(f"- median pairwise edge: **{rec['median_edge']}**")
        lines.append(f"- mean pairwise edge: **{rec['mean_edge']:.1f}**")
        lines.append(f"- total internal games: **{rec['total_internal']}**")
        lines.append(f"- players: {', '.join(rec['names'])}")
        mat = pd.DataFrame(0, index=rec["names"], columns=rec["names"])
        for (ka, kb), v in rec["edges"].items():
            na, nb = name_of[ka], name_of[kb]
            mat.loc[na, nb] = v
            mat.loc[nb, na] = v
        lines.append("\n**Pairwise matrix**\n")
        lines.append(mat.to_markdown())
        min_outside = min(rec["outside_games"].values())
        min_outside_player = [name_of[k2] for k2, v in rec["outside_games"].items() if v == min_outside][0]
        lines.append(f"\n- minimum outside-cohort (standard+dedup) games for any player: **{min_outside}** ({min_outside_player})")
    return "\n".join(lines) + "\n"


for k in SIZES:
    path = os.path.join(CV_BASE, f"cohort_{k}_standard.md")
    with open(path, "w") as f:
        f.write(fmt_cohort_md(k, results[k]))
    print(f"wrote {path}")

import pickle
with open(os.path.join(CV_BASE, "cohort_search_standard_results.pkl"), "wb") as f:
    pickle.dump(results, f)
