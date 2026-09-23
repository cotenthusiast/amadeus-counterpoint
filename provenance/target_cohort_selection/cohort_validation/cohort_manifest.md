# Cohort Manifest (Human-Readable) — Freeze Candidate

Machine-readable version: `cohort_manifest.json`. This document explains it.

## Status: cleared to freeze

All 8 identities passed manual verification (Section 3) — none is
NEEDS_MANUAL_REVIEW. 5 are VERIFIED_HIGH_CONFIDENCE; 3 (So, Vachier-Lagrave,
Nepomniachtchi) are VERIFIED_WITH_CAVEAT because their Lichess-username aliases
(GMWSO / LyonBeast / lachesisQ) tripped the corpus's automated "needs review"
heuristic (a token/letter-signature mismatch, not evidence of a real conflict)
and required a manual rating/date/title cross-check to clear. Two players
(Aronian, Firouzja) have a small (~1–2%), disclosed undercount from unmerged
country-code-suffixed name spellings. None of this blocks freezing.

## The 8 players

| player | FIDE ID | classification | total games (standard+dedup) |
|---|---|---|---:|
| Carlsen, Magnus | 1503014 | VERIFIED_HIGH_CONFIDENCE | 1,456 |
| So, Wesley | 5202213 | VERIFIED_WITH_CAVEAT | 1,409 |
| Aronian, Levon | 13300474 | VERIFIED_HIGH_CONFIDENCE | 1,241 |
| Caruana, Fabiano | 2020009 | VERIFIED_HIGH_CONFIDENCE | 1,561 |
| Vachier-Lagrave, Maxime | 623539 | VERIFIED_WITH_CAVEAT | 1,396 |
| Nakamura, Hikaru | 2016192 | VERIFIED_HIGH_CONFIDENCE | 1,226 |
| Nepomniachtchi, Ian | 4168119 | VERIFIED_WITH_CAVEAT | 1,158 |
| Firouzja, Alireza | 12573981 | VERIFIED_HIGH_CONFIDENCE | 999 |

## The two filtering rules that define this dataset

1. **Deduplication**: one row per distinct `content_hash` (normalized
   White|Black|Date|Result|move-list) among rows with moves, keeping the
   first-occurring record. Removes 5,607 exact broadcast-chapter
   re-publications corpus-wide; touches only 3 of the 8-cohort's 28 dyads,
   by 1–2 games each.
2. **Standard-chess filter**: `Variant = 'Standard'` exactly. Excludes 11,097
   corpus-wide rows (Chess960/Freestyle Chess, "From Position" games, a few
   fringe variants, 3 unparseable). This is the **dominant** filtering effect
   on the cohort: it touches all 28 dyads, removing 4.5%–66.0% of games per
   dyad, and drops the cohort's minimum edge from 42 to **29**.

Apply both rules in the order given (dedup first, then variant filter) to
reproduce every count in this manifest.

## Frozen pairwise real-game counts (standard+dedup basis)

See `cohort_manifest.json`'s `pairwise_real_game_counts` for all 28, or
`dyad_adequacy_8cohort.csv` for the full metadata-quality breakdown. Headline:
min 29 (Aronian–Nepomniachtchi), median 56, max 123 (Carlsen–Nakamura), total
1,682 internal games.

## Why 8 and not 6/7/9

Re-running the exhaustive cohort search on the corrected (dedup+standard) basis
reproduced the *same* 8 players as the best-8 combination — this is not a
carry-over of the original, uncorrected pick. 7 (dropping Nepomniachtchi) is a
legitimate, more conservative fallback (min edge 32, no dyad below 29). 9 is
rejected on the evidence: its best combination has *fewer* total internal games
than the 8-cohort (1,581 vs 1,682) despite adding a 9th player, because it must
drop two well-connected veterans (Nakamura, Nepomniachtchi) for three much
younger players with shorter broadcast track records, and its own runner-up
gap is a steep, fragile cliff (min edge 22 → 16). Full comparison:
`01_cohort_size_comparison.md`.

## What this manifest is NOT

It does not copy, transform, or re-store any game content. It is a protocol
definition: which 8 identities, which raw spellings are accepted as the same
person, and which two filters to apply to `games.parquet` to reproduce the
exact pairwise counts above. Source PGNs and `audit/data/games.parquet` are
untouched.
