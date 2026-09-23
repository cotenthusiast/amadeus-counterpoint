# Section 6 — Target-Pair Real-Data Adequacy (28 dyads, current 8-cohort)

Basis: deduplicated, `Variant='Standard'`-only games (the corrected basis from
Sections 4–5). Full per-dyad table: `dyad_adequacy_8cohort.csv`.

## Threshold used (transparent, not "statistically magical")

- **STRONG**: ≥60 real standard-chess games between the two individuals. Reasoning:
  60 games is roughly double the cohort's own search-floor requirement (the search
  targeted ≥40ish edges) and, at the observed ~45–55% draw rates for these dyads,
  is enough real decisive-vs-draw outcomes (roughly 25–35 decisive games) to see
  the win/draw/loss split stabilize somewhat rather than being dominated by a
  handful of results.
- **USABLE**: 40–59 games. This is "as good as the corrected cohort search could
  find" for several dyads — thin, but not vanishingly small, and still usable as a
  held-out evaluation distribution with appropriately wide uncertainty bars.
- **WEAK**: <40 games. Below the level where a two-outcome-category split (e.g.
  win rate) has a standard error much better than ±9 percentage points
  (√(0.25/29) ≈ 0.093 at n=29); treat any point estimate from these dyads as
  indicative only.
- This is a floor chosen for **interpretability of the report**, not a claim that
  40 or 60 is where some sharp discontinuity in statistical validity occurs — per
  the task's instruction, no fabricated "magic number" precision is implied.

## Results

| flag | count | dyads |
|---|---:|---|
| STRONG | 12 | Aronian–Nakamura(64), MVL–Nepomniachtchi(65), Nakamura–Firouzja(65), So–Nakamura(68), Caruana–Nakamura(71), So–MVL(77), Carlsen–Nepomniachtchi(78), So–Caruana(85), Carlsen–Firouzja(97), Carlsen–Caruana(102), Carlsen–So(107), Carlsen–Nakamura(123) |
| USABLE | 9 | Nepomniachtchi–Firouzja(42), MVL–Firouzja(44), So–Firouzja(46), So–Aronian(50), MVL–Nakamura(54), Caruana–MVL(54), Carlsen–MVL(55), Aronian–Caruana(57), Caruana–Firouzja(59) |
| WEAK | 7 | Aronian–Nepomniachtchi(29), So–Nepomniachtchi(30), Caruana–Nepomniachtchi(30), Nakamura–Nepomniachtchi(31), Carlsen–Aronian(32), Aronian–Firouzja(33), Aronian–MVL(34) |

All 7 WEAK dyads involve either **Nepomniachtchi** (4 of 7) or **Aronian** (6 of
7, including 3 shared with Nepomniachtchi) — these two players' pairwise
connections to the rest of the cohort are the cohort's real soft spot, consistent
with Section 1's finding that Aronian–Nepomniachtchi (29) is the global minimum
edge. Carlsen–Aronian is notable: 94 games all-variant but only 32 standard-chess
(66% were Freestyle/Chess960 rounds — both are prominent Freestyle Chess tour
players), so its WEAK classification is a direct, mechanical consequence of the
Section 5 variant correction, not a data-quality problem with the games
themselves.

## Metadata quality, missing data, dedup/variant impact

- **Missing Elo:** ranges 1.6%–23.5% across dyads (median ≈ 9%); worst is
  Aronian–Vachier-Lagrave (23.5%). No dyad exceeds a quarter of rows missing Elo.
- **Missing Date:** ranges 3.8%–38.2% (median ≈ 13%); worst is again
  Aronian–Vachier-Lagrave (38.2%). Missing dates do not remove games from the
  count — a game with an unknown date is still a real recorded game — but they
  do limit how finely first/last-date coverage or chronological splits can be
  computed for that dyad.
- **Affected by deduplication:** only 2 of 28 dyads lose anything to content-hash
  dedup at all (So–Aronian: 52→50, 3.8%; So–Nakamura: 69→68, 1.4%); the other 26
  are exact matches pre/post dedup. Consistent with Section 4's finding that
  dedup barely touches this cohort.
- **Affected by variant filtering:** all 28 dyads lose games; the proportion
  removed ranges from 4.5% (So–Caruana) to 66.0% (Carlsen–Aronian) — see Section 5
  for the full table. This is the dominant data-quality effect on this cohort, by
  a wide margin over dedup or missing-metadata effects.
- **Color balance:** every dyad's white/black split is close to even (no dyad
  worse than roughly 60/40), so no player is structurally starved of one color
  against a given opponent.
- **Rating plausibility:** every dyad's rating range sits inside 2650–3370, i.e.
  entirely within elite/classical-to-blitz range for these 8 players — no
  rating-based red flags.

## Bottom line

The sealed empirical distribution for the cohort's weakest dyad
(Aronian–Nepomniachtchi, 29 games) will carry real, non-trivial sampling
uncertainty — do not treat any downstream synthetic augmentation as removing
that uncertainty (see Section 9 framing). 21 of 28 dyads (75%) are at least
USABLE; 12 (43%) are STRONG. The 7 WEAK dyads are concentrated on two players
(Aronian, Nepomniachtchi) rather than spread evenly, which is useful to know when
designing per-dyad evaluation weighting.
