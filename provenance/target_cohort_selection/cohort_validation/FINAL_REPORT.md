# Second-Stage Cohort Validation & Freeze Report

Source PGNs, `audit/data/games.parquet`, and all stage-1 audit outputs were not
modified. No identities were silently merged (every merge decision traces to
`audit/scripts/07_pairs.py`'s existing rule, reused unchanged). No model code
was written. All new outputs are under `audit/cohort_validation/`.

## 1–4. Best cohort per size (corrected: dedup + `Variant='Standard'` basis)

| size | players | dyads | min edge | median edge | mean edge | total internal games |
|---|---|---:|---:|---:|---:|---:|
| 6 | Caruana, Carlsen, So, Vachier-Lagrave, Nakamura, Firouzja | 15 | 44 | 68.0 | 73.8 | 1,107 |
| 7 | + Aronian | 21 | 32 | 59.0 | 65.6 | 1,377 |
| **8 (current)** | + Nepomniachtchi | 28 | **29** | 56.0 | 60.1 | 1,682 |
| 9 | Caruana, Carlsen, Abdusattorov, So, Vachier-Lagrave, Giri, Gukesh D, Praggnanandhaa R, Firouzja | 36 | 22 | 35.0 | 43.9 | 1,581 |

Note the 8→9 step is the odd one out: size 9's best combination has *fewer*
total internal games than size 8's, because it must drop two well-connected
veterans (Nakamura, Nepomniachtchi) to reach its own local optimum. Full
comparison, runner-ups, and edge-count buckets: `01_cohort_size_comparison.md`.

## 5. Recommendation for cohort size

**Keep 8 players**, but adopt **min edge = 29** (not the originally reported 42)
as the honest floor — the difference is entirely explained by the standard-chess
variant correction (item 7 below). This was not a rubber-stamp: the exhaustive
search was re-run from scratch on the corrected pair-count table, and the same 8
players came out on top independently. **7 players** (drop Nepomniachtchi, min
edge 32) is a legitimate, more conservative fallback if a reviewer wants a
stricter per-dyad floor. **9 players is not recommended** — it buys more dyads
but less total data and a fragile, non-robust optimum (see `01_cohort_size_comparison.md`).

## 6. Identity verification: clean enough to freeze?

**Yes.** All 8 identities cleared, none NEEDS_MANUAL_REVIEW:
- 5 **VERIFIED_HIGH_CONFIDENCE**: Carlsen, Aronian, Caruana, Nakamura, Firouzja.
- 3 **VERIFIED_WITH_CAVEAT**: So (alias GMWSO), Vachier-Lagrave (alias
  LyonBeast), Nepomniachtchi (alias lachesisQ) — each player's Lichess-username
  alias tripped the corpus's automated "shares no name token" heuristic; manual
  cross-check of FIDE ID + rating + date + title confirmed each is the same
  person, not a mixup (see `03_identity_verification.md` for the exact evidence
  per alias). No unrelated raw identity shares any of the 8 FIDE IDs.
- Two minor, disclosed (not silently fixed) undercounts: ~30 Aronian games and
  ~21 Firouzja games under a country-code-suffixed spelling with no FIDE tag
  didn't merge into the main identity (conservative-by-design canonical-key
  rule); immaterial to any conclusion here.

## 7. Changes to the 8×8 matrix after dedup + standard-variant filtering

- **Deduplication**: negligible. Only 3 of 28 dyads change at all, by 1–2 games
  each (e.g. Carlsen–So: 170→169). 25 of 28 dyads are byte-identical.
- **Standard-chess variant filtering: substantial.** All 28 dyads change,
  4.5%–66.0% of games removed per dyad (Chess960/Freestyle-Chess tournament
  rounds and "From Position" games are real, non-trivial fractions of some
  pairs' broadcast history — e.g. Carlsen–Aronian drops from 94 to 32, since
  both are prominent Freestyle Chess tour players). Min edge: **42 → 29**.
  Median: 72.0 → 56.0. Total internal games: 2,241 → 1,682. This is the single
  biggest correction this validation stage made. Full detail: `05_variant_filtering.md`.

## 8. Weakest real dyad after final filtering

**Aronian – Nepomniachtchi: 29 games** (standard, deduplicated). All 7 WEAK
dyads (<40 games) involve Aronian and/or Nepomniachtchi; 21 of 28 dyads are
USABLE or STRONG. See `06_dyad_adequacy.md` for the transparent STRONG/USABLE/WEAK
threshold and full per-dyad breakdown (color split, W/D/L, missing-metadata
rates).

## 9. Minimum remaining training games across all 56 A[-B] folds

**902** (Firouzja's training pool after holding out Carlsen: 999 total − 97
vs-Carlsen games). Every fold retains ≥173 unique training opponents and ≥227
games against the other 6 cohort members, so the training side of the
leave-one-opponent-out design is never the limiting factor anywhere in this
cohort — the limiting factor is always the held-out dyad's own sample size
(item 8). Full 56-row table: `loo_folds.csv` (`07_loo_folds.md` for the
summary).

## 10. Generated files

All under `/home/cotenthusiast/Data/lichess_broadcasts/audit/cohort_validation/`:

**Narrative reports**
- `01_cohort_size_comparison.md` — sizes 6/7/8/9 compared, runner-ups, tradeoff, recommendation
- `02_full_matrix_outputs.md` — full matrix + weakest/strongest-5 for the current 8-cohort
- `03_identity_verification.md` — per-player alias/FIDE/rating evidence, classifications
- `04_dedup_validation.md` — dedup-view sanity check
- `05_variant_filtering.md` — the standard-chess filter and its (large) impact
- `06_dyad_adequacy.md` — 28-dyad adequacy table, STRONG/USABLE/WEAK
- `07_loo_folds.md` — 56-fold leave-one-opponent-out sizing summary
- `cohort_manifest.md` — human-readable freeze manifest
- `FINAL_REPORT.md` — this file

**Machine-readable**
- `cohort_manifest.json` — the frozen cohort/filter protocol definition
- `cohort_size_comparison.csv` / `cohort_size_comparison_standard.csv` — sizes 6-9, pre/post correction
- `cohort_{6,7,8,9}_standard.md` — full re-run search output (top 5 candidates each), corrected basis
- `matrices/matrix_{N}[_standard].csv`, `edges_sorted_{N}[_standard].csv`, `per_player_{N}[_standard].csv` — for N=6,7,8,9
- `dedup_impact_on_8cohort.csv`, `variant_filter_impact_on_8cohort.csv`, `dedup_variant_summary.json`
- `dyad_adequacy_8cohort.csv` — the 28-dyad adequacy table with flags
- `loo_folds.csv` — the 56-row leave-one-opponent-out fold table
- `pairs_standard_dedup.csv`, `player_totals_standard_dedup.csv`, `player_dates_standard_dedup.csv` — corpus-wide corrected basis (reusable for future cohort work)

**Scripts** (`scripts/`, all read-only against `games.parquet`)
- `lib.py`, `identity_resolution.py` — shared helpers (identity resolution verified to reproduce the frozen `pairs.csv` exactly before use)
- `01_cohort_compare.py`, `03_dedup_and_variant.py`, `04_dyad_adequacy.py`, `05_cohort_research_standard.py`, `06_cohort_compare_standard.py`, `07_loo_folds.py`, `08_manifest.py`
