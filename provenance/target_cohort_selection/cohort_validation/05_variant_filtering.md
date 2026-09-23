# Section 5 — Variant Filtering for the Standard-Chess Dataset

## Exact Variant tag values and counts (all 1,186,338 parsed records)

| Variant | count |
|---|---:|
| Standard | 1,175,241 |
| Chess960 | 5,048 |
| From Position | 4,407 |
| Fischerandom | 1,116 |
| fischerandom (lowercase — same variant, inconsistent capitalization in source PGNs) | 425 |
| Crazyhouse | 59 |
| King of the Hill | 25 |
| Antichess | 14 |
| *(null/unparseable)* | 3 |

Non-`'Standard'` total: **11,094** (5,048+4,407+1,116+425+59+25+14), matching the
stage-1 audit's "~11,094" figure exactly. Adding the 3 null-Variant rows gives
11,097 rows that are not affirmatively tagged `Standard`.

## Conservative standard-chess filter

**Filter: `Variant = 'Standard'` exactly** (case-sensitive; do not also accept the
lowercase `'fischerandom'` variant spelling — it is still Chess960, just an
inconsistently-cased tag, not a standard-chess row). Treat the 3 null-Variant rows
as excluded too (unverified, not affirmatively standard) — negligible either way.

- Of the 1,093,256 deduplicated rows: **1,082,369 are standard-chess**, **10,887
  are non-standard-or-null**.

## Does this affect the current 8-player cohort?

**Yes, substantially — this is the most important finding of this validation
stage.** 2,155 deduplicated games involving at least one of the 8 target players
(identity-resolved, i.e. this counts a player's games even on rows where their own
FIDE tag happens to be missing) carry a non-`Standard` Variant tag. Concretely
these are: Chess960/Freestyle-Chess tournament rounds (e.g. "FIDE Freestyle Chess
World Championship RR", "Freestyle Chess G.O.A.T. Challenge Rapid" — a real,
growing elite tour, not noise) and `From Position` "Playzone game" entries
(informal from-a-set-position games, not full games from the start position).

**All 28 of the current cohort's dyads change** when the filter is applied —
see `variant_filter_impact_on_8cohort.csv` for the full table; representative
examples:

| player_A | player_B | all-variant, dedup | standard-only, dedup | delta | % removed |
|---|---|---:|---:|---:|---:|
| Aronian, Levon | Carlsen, Magnus | 94 | 32 | 62 | 66.0% |
| Nakamura, Hikaru | Carlsen, Magnus | 158 | 123 | 35 | 22.2% |
| So, Wesley | Firouzja, Alireza | 69 | 46 | 23 | 33.3% |
| Vachier-Lagrave, Maxime | Firouzja, Alireza | 48 | 44 | 4 | 8.3% |
| Caruana, Fabiano | So, Wesley | 89 | 85 | 4 | 4.5% |

The impact ranges from negligible (~4%) to severe (~66%, Aronian–Carlsen), because
some pairs happened to play a large fraction of their broadcast head-to-head games
in Freestyle Chess events specifically (Carlsen and Aronian are both prominent
Freestyle/Chess960 tour participants) while others rarely did.

## Before/after 8×8 matrix

See `matrices/matrix_8.csv` (all-variant, dedup — the stage-1 basis) vs.
`matrices/matrix_8_standard.csv` (corrected basis) for the full before/after
matrices, and `01_cohort_size_comparison.md` / `02_full_matrix_outputs.md` for the
corrected matrix inline. Headline: **min edge falls from 42 to 29**, median from
72.0 to 56.0, total internal games from 2,241 to 1,682.

## Consequence for the cohort search itself

Because the change is this large, patching the old numbers would not have been
sound — a materially different pair-count table can change which combination of
players is optimal. The full exhaustive cohort search (`08_cohort.py`'s method)
was therefore re-run from scratch on the corrected (dedup + standard-only) pair
table for sizes 6–9 (`05_cohort_research_standard.py`). Result: **the best-8
combination is unchanged** — it is still exactly Carlsen/So/Aronian/Caruana/
Vachier-Lagrave/Nakamura/Nepomniachtchi/Firouzja — but the best-6, best-7, and
best-9 combinations **do** shift membership (see `01_cohort_size_comparison.md`).
This is a meaningful robustness result: the 8-cohort's composition survives the
correction; its headline min-edge number did not.
