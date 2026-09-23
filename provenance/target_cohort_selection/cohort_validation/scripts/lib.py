"""Shared helpers for cohort_validation scripts. READ-ONLY against the audit corpus."""
import os
import sys
import json

import numpy as np
import pandas as pd

AUDIT_BASE = "/home/cotenthusiast/Data/lichess_broadcasts/audit"
CV_BASE = os.path.join(AUDIT_BASE, "cohort_validation")
MATRICES_DIR = os.path.join(CV_BASE, "matrices")

sys.path.insert(0, os.path.join(AUDIT_BASE, "scripts"))
from common import connect  # noqa: E402

# Best candidate (#1) player lists, taken verbatim from audit/cohort_{6,7,8,9}.md
# (already-computed exhaustive-search winners; reproduced here to avoid re-deriving
# via a slower path). These are cross-checked against pairs.csv/players_normalized.csv
# below, not trusted blindly.
BEST_COHORTS = {
    6: ["Carlsen, Magnus", "So, Wesley", "Aronian, Levon", "Caruana, Fabiano",
        "Vachier-Lagrave, Maxime", "Nakamura, Hikaru"],
    7: ["Carlsen, Magnus", "So, Wesley", "Aronian, Levon", "Caruana, Fabiano",
        "Vachier-Lagrave, Maxime", "Nakamura, Hikaru", "Nepomniachtchi, Ian"],
    8: ["Carlsen, Magnus", "So, Wesley", "Aronian, Levon", "Caruana, Fabiano",
        "Vachier-Lagrave, Maxime", "Nakamura, Hikaru", "Nepomniachtchi, Ian", "Firouzja, Alireza"],
    9: ["Carlsen, Magnus", "So, Wesley", "Aronian, Levon", "Caruana, Fabiano",
        "Vachier-Lagrave, Maxime", "Giri, Anish", "Nepomniachtchi, Ian",
        "Duda, Jan-Krzysztof", "Firouzja, Alireza"],
}

TARGET_8 = BEST_COHORTS[8]

PLAYERS_NORM = pd.read_csv(os.path.join(AUDIT_BASE, "players_normalized.csv"))
PAIRS = pd.read_csv(os.path.join(AUDIT_BASE, "pairs.csv"))

NAME_TO_KEY = dict(zip(PLAYERS_NORM.player_name, PLAYERS_NORM.id_key))
KEY_TO_NAME = dict(zip(PLAYERS_NORM.id_key, PLAYERS_NORM.player_name))


def edge_games(key_a, key_b):
    """Look up total_games for an unordered pair of id_keys from pairs.csv."""
    a, b = sorted([key_a, key_b])
    row = PAIRS[(PAIRS.player_A_key == a) & (PAIRS.player_B_key == b)]
    if len(row) == 0:
        return 0
    return int(row.iloc[0].total_games)


def cohort_matrix(names):
    keys = [NAME_TO_KEY[n] for n in names]
    n = len(names)
    mat = pd.DataFrame(0, index=names, columns=names)
    edges = {}
    for i in range(n):
        for j in range(i + 1, n):
            v = edge_games(keys[i], keys[j])
            mat.loc[names[i], names[j]] = v
            mat.loc[names[j], names[i]] = v
            edges[(names[i], names[j])] = v
    return mat, edges


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"wrote {path}", file=sys.stderr)
