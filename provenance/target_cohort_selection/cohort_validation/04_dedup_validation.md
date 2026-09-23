# Section 4 — Deduplication View Validation

No rows were deleted anywhere; `games_dedup` remains a logical DuckDB view over
`games.parquet` (see `audit/scripts/common.py`), reproduced read-only here.

| metric | value |
|---|---:|
| raw row count (all parsed records) | 1,186,338 |
| raw, has_moves=True (the dedup view's input population) | 1,098,863 |
| deduplicated row count (`games_dedup`) | 1,093,256 |
| exact number removed by content-hash deduplication | **5,607** |

This matches `audit/reports/02_duplicates.md` exactly (5,607 "duplicate records
beyond the first occurrence" across 3,461 content-hash groups) — the dedup view
behaves as documented.

## Impact on the current 8-player cohort's 28 pair counts

Recomputed pair counts before vs. after content-hash dedup, using the identical
identity-resolution rule that built the frozen `pairs.csv`
(`identity_resolution.build_resolved_map`, verified to reproduce `pairs.csv`'s
936,377 pairs exactly before use here). Full table: `dedup_impact_on_8cohort.csv`.

| player_A | player_B | games (raw, pre-dedup) | games (post-dedup) | delta |
|---|---|---:|---:|---:|
| Aronian, Levon | So, Wesley | 71 | 69 | 2 |
| Carlsen, Magnus | So, Wesley | 170 | 169 | 1 |
| Nakamura, Hikaru | So, Wesley | 94 | 93 | 1 |
| *(all other 25 dyads)* | | *(unchanged)* | *(unchanged)* | 0 |

**3 of 28 dyads change, by 1–2 games each; 25 of 28 are byte-identical
before/after dedup.** No dyad's pairwise count materially changes. The
broadcast-chapter-repeat duplicates found in `audit/reports/02_duplicates.md`
(mostly lower-level events, e.g. the German women's blitz championship example
cited there) essentially never touch top-elite-vs-top-elite games — consistent
with duplicates being a broadcast-authoring artifact of specific events, not a
systematic effect that would bias the target cohort's data.

**Conclusion: the deduplicated view used for pair/cohort counts behaves exactly
as expected and documented.** No further action needed on this axis; the material
finding this validation stage surfaced is the variant filter (Section 5), not
deduplication.
