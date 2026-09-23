#!/usr/bin/env python3
"""Section 8: build the frozen cohort manifest (JSON, machine-readable) from the
already-computed, already-verified numbers in this validation stage. No new
computation -- this script only assembles prior outputs into one document."""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from lib import TARGET_8, CV_BASE, AUDIT_BASE

adequacy = pd.read_csv(os.path.join(CV_BASE, "dyad_adequacy_8cohort.csv"))
totals_std = pd.read_csv(os.path.join(CV_BASE, "player_totals_standard_dedup.csv"))
players_norm = pd.read_csv(os.path.join(AUDIT_BASE, "players_normalized.csv"))
players_norm = players_norm[players_norm.player_name.isin(TARGET_8)]
totals_std = totals_std.merge(players_norm[["id_key", "player_name"]], on="id_key")

FIDE_IDS = {
    "Carlsen, Magnus": "1503014",
    "So, Wesley": "5202213",
    "Aronian, Levon": "13300474",
    "Caruana, Fabiano": "2020009",
    "Vachier-Lagrave, Maxime": "623539",
    "Nakamura, Hikaru": "2016192",
    "Nepomniachtchi, Ian": "4168119",
    "Firouzja, Alireza": "12573981",
}

ALIASES = {
    "Carlsen, Magnus": ["Carlsen, Magnus", "Magnus Carlsen", "MagnusCarlsen", "Carlsen Magnus (NOR)"],
    "So, Wesley": ["So, Wesley", "Wesley So", "GMWSO", "So Wesley (USA)"],
    "Aronian, Levon": ["Aronian, Levon", "Levon Aronian", "Aronian Levon"],
    "Caruana, Fabiano": ["Caruana, Fabiano", "Caruana Fabiano (USA)"],
    "Vachier-Lagrave, Maxime": ["Vachier-Lagrave, Maxime", "Maxime Vachier-Lagrave", "Vachier-Lagrave Maxime (FRA)", "LyonBeast"],
    "Nakamura, Hikaru": ["Nakamura, Hikaru", "Hikaru Nakamura", "Nakamura Hikaru (USA)"],
    "Nepomniachtchi, Ian": ["Nepomniachtchi, Ian", "lachesisQ", "Nepomniachtchi Ian (FID)"],
    "Firouzja, Alireza": ["Firouzja, Alireza", "Alireza Firouzja"],
}

IDENTITY_CLASS = {
    "Carlsen, Magnus": "VERIFIED_HIGH_CONFIDENCE",
    "So, Wesley": "VERIFIED_WITH_CAVEAT",
    "Aronian, Levon": "VERIFIED_HIGH_CONFIDENCE",
    "Caruana, Fabiano": "VERIFIED_HIGH_CONFIDENCE",
    "Vachier-Lagrave, Maxime": "VERIFIED_WITH_CAVEAT",
    "Nakamura, Hikaru": "VERIFIED_HIGH_CONFIDENCE",
    "Nepomniachtchi, Ian": "VERIFIED_WITH_CAVEAT",
    "Firouzja, Alireza": "VERIFIED_HIGH_CONFIDENCE",
}

pairwise = []
for _, r in adequacy.iterrows():
    pairwise.append({
        "player_A": r.player_A, "player_B": r.player_B,
        "games_standard_dedup": int(r.games_standard_dedup),
        "games_all_variant_dedup": int(r.games_all_variant_dedup),
        "flag": r.flag,
    })

manifest = {
    "manifest_version": "1.0",
    "generated_by": "audit/cohort_validation/scripts/08_manifest.py",
    "dataset_source": {
        "description": "Lichess broadcast PGN corpus (lichess_db_broadcast_*.pgn), 79 monthly files",
        "location": "/home/cotenthusiast/Data/lichess_broadcasts/",
        "source_date_range": "2020-01 .. 2026-07 (per-file coverage; earliest single-player observed date 2019-11-23)",
        "audit_version_date": "stage-1 audit under audit/ (audit_report.md, reports/01-06); this manifest is the stage-2 cohort_validation pass",
    },
    "target_players": [
        {
            "canonical_name": name,
            "canonical_id_key": f"F:{FIDE_IDS[name]}",
            "fide_id": FIDE_IDS[name],
            "title": "GM",
            "identity_classification": IDENTITY_CLASS[name],
            "accepted_raw_aliases": ALIASES[name],
            "total_games_all_variant_dedup": int(players_norm.set_index("player_name").loc[name, "total_games"]),
            "total_games_standard_dedup": int(totals_std.set_index("player_name").loc[name, "total_games_standard_dedup"]),
        }
        for name in TARGET_8
    ],
    "deduplication_rule": {
        "description": "One row per distinct content_hash (sha1 of normalized White|Black|Date|Result|SAN-move-list) among rows with has_moves=true, keeping the first-occurring record ordered by (src_file, src_game_index). Removes exact-republication broadcast-chapter duplicates only; never removes a game with distinct content.",
        "rows_removed_corpuswide": 5607,
        "rows_removed_from_8cohort_dyads": 4,
        "dyads_affected_in_8cohort": 3,
    },
    "standard_chess_variant_rule": {
        "description": "Keep rows where Variant = 'Standard' exactly (case-sensitive; the lowercase 'fischerandom' tag variant is still Chess960 and is excluded, not a typo of Standard). Null/unparseable Variant (3 rows corpus-wide) also excluded, unverified.",
        "excluded_variants": {
            "Chess960": 5048, "From Position": 4407, "Fischerandom": 1116,
            "fischerandom": 425, "Crazyhouse": 59, "King of the Hill": 25,
            "Antichess": 14, "null_or_unparseable": 3,
        },
        "rows_excluded_corpuswide": 11097,
        "games_involving_8cohort_excluded": 2155,
        "impact_on_8cohort_matrix": "min pairwise edge fell from 42 (all-variant) to 29 (standard-only); all 28 dyads changed, range 4.5%-66.0% games removed per dyad",
    },
    "pairwise_real_game_counts": pairwise,
    "cohort_size_recommendation": {
        "recommended_size": 8,
        "authoritative_min_edge": 29,
        "alternative_considered": "7 players (drop Nepomniachtchi), min edge 32, if a stricter per-dyad floor is required",
        "rejected": "9 players -- best-9 combination has LOWER total internal games (1581) than best-8 (1682), drops two well-established cohort members for three much-younger players with shorter track records, and its own runner-up gap is a fragile isolated peak (min edge falls from 22 to 16 for candidate #2)",
    },
    "known_caveats": [
        "So/Wesley (GMWSO), Vachier-Lagrave/Maxime (LyonBeast), and Nepomniachtchi/Ian (lachesisQ) were auto-flagged 'needs_review' by the corpus's Tier-A alias check (their Lichess-username aliases share no text token/letter-signature with the conventional name spelling); manually verified via consistent FIDE ID + plausible rating/date/title -- see 03_identity_verification.md.",
        "Aronian (~30 games) and Firouzja (~21 games) each have a small, immaterial set of country-code-suffixed raw name spellings (e.g. 'Aronian Levon (ARM)') that carry no FIDE tag and were NOT merged into their main identity by the conservative canonical-key rule (the appended country code changes the sorted key). Not included in any total_games figure in this manifest. Disclosed, not silently corrected.",
        "7 of 28 dyads are WEAK (<40 standard-chess games): all involve Aronian and/or Nepomniachtchi. Weakest is Aronian-Nepomniachtchi at 29 games. Treat any per-dyad point estimate from these as high-variance.",
        "Missing-Elo and missing-Date rates vary by dyad from ~2% to ~38%; see 06_dyad_adequacy.md.",
        "This manifest defines the cohort/filter protocol only; it does not copy or transform the underlying games. Downstream code must apply the deduplication_rule and standard_chess_variant_rule exactly as stated to reproduce these counts.",
    ],
}

out_path = os.path.join(CV_BASE, "cohort_manifest.json")
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote {out_path}")
