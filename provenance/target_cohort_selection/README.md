# Target Cohort Selection — Provenance

This directory is a curated copy of the read-only research workspace that
selected Amadeus Counterpoint's 8-player target cohort. The original,
complete workspace lives outside this repository at
`~/Data/lichess_broadcasts/audit/` and also contains broader corpus-wide
audits (full player-identity resolution, duplicate detection, rating
distributions) not specific to cohort selection; only the
cohort-selection-relevant subset is copied here for citability and
reproducibility.

## What this shows

The 8 target players (`configs/broadcast_targets.json` in the main repo)
were not selected individually by rating or fame. They were selected by an
exhaustive **maximin (bottleneck) combinatorial search**: over all
$\binom{24}{8}=735{,}471$ 8-player subsets of the 24 most broadcast-active,
FIDE-identified (non-bot) players in the Lichess Broadcast corpus, the
subset that maximizes the *minimum* pairwise mutual-game count across all 28
dyads was selected (median/mean/total mutual games used as secondary
criteria), corroborated by a swap-based local search over the top 70
candidates. Cohort sizes 6, 7, 8, and 9 were compared; 8 gave the best
coverage/size trade-off (`01_cohort_size_comparison.md`). The same 8 players
were independently re-selected after correcting the corpus for duplicate
broadcast records and restricting to standard (non-Chess960/Freestyle)
games (`cohort_validation/FINAL_REPORT.md`), which also revised the
weakest-dyad floor from 42 to 29 games.

## Layout

- `audit_report.md` — top-level summary of the stage-1 corpus audit (parsing,
  dedup, identity resolution, ratings) that this cohort search was built on.
- `cohort_6.md` / `cohort_7.md` / `cohort_8.md` / `cohort_9.md` — the
  stage-1 exhaustive-search output (top 5 candidate subsets per cohort
  size, on the raw, non-deduplicated, all-variant corpus).
- `cohort_validation/` — the stage-2 corrected re-run (deduplicated,
  `Variant=Standard` only): `FINAL_REPORT.md` is the summary; the numbered
  `*.md` files are per-topic detail (size comparison, full matrix, identity
  verification, dedup impact, variant-filter impact, per-dyad adequacy,
  leave-one-opponent-out fold sizing); `cohort_manifest.json`/`.md` is the
  frozen, machine-readable cohort definition; the CSVs are the underlying
  data; `scripts/` are the read-only analysis scripts that produced them
  (run against `audit/data/games.parquet`, which is not copied here).

## Not copied here

`preliminary_behavior/` (post-selection WDL/opening exploratory analysis),
the full corpus-wide `players.csv`/`pairs.csv`/`duplicates.csv`/
`suspected_aliases.csv`, cohort sizes 6/7/9's `cohort_validation` detail
beyond the summary comparison, and `data/games.parquet` itself (the parsed
corpus) — these are broader-scope or non-reproducible-without-the-raw-PGNs
artifacts, not needed to cite or verify the 8-player selection criterion
specifically. The complete original workspace remains at
`~/Data/lichess_broadcasts/audit/` if deeper verification is needed.
