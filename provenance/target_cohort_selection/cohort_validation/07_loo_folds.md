# Section 7 — Leave-One-Opponent-Out Fold Sizing (56 folds)

Basis: deduplicated, `Variant='Standard'`-only games. Full table (one row per
ordered A[-B] fold): `loo_folds.csv`. No training was run — this is sizing only.

## Method

For every ordered pair (A, B) among the 8 cohort players (8×7 = 56 folds):
- `A_total_usable_games_std_dedup`: all of A's standard+dedup games, any opponent.
- `games_removed_because_opponent_is_B`: A vs B games (the held-out real
  interaction distribution for evaluation).
- `remaining_training_games_for_A` = the above minus the B-games (A's training set
  under "leave B out").
- Split of the remainder into games against the other 6 cohort members vs. games
  against outside-cohort opponents.
- `remaining_approx_plies`: sum of `ply_count` over A's remaining training games
  (an approximate move-decision budget; both players' plies are counted once each
  as part of a shared game, so this is not "decisions made by A" specifically but
  total half-moves played in games A appears in).

## Headline numbers

- **Minimum remaining training games for any A[-B] fold: 902** (Firouzja
  holding out Carlsen: 999 total − 97 vs-Carlsen = 902).
- **Maximum removed by holding out one opponent: 123** (Carlsen holding out
  Nakamura), i.e. even the biggest single-opponent removal only costs ~8.4% of
  Carlsen's 1,456-game training pool.
- **Minimum remaining approx. plies for any fold: ≈88,058** (Firouzja[-Carlsen]).
- Every fold retains 173–295 unique training opponents for A (median opponent
  pool size in the low 200s), and every fold retains the other 6 cohort members
  as training opponents too (range: 227–562 games vs. the other 6, per fold) —
  see `loo_folds.csv` for the `remaining_games_vs_other_6_cohort_members` column.

## Interpretation (Section 9 framing)

Priority 3 from the research framing ("enough non-target-opponent data to learn
each individual") is comfortably met: even in the worst fold, A retains 902
standard-chess games against 173 distinct opponents after removing all games vs.
the held-out B. Priority 4 ("enough held-out A-B data to evaluate the real
interaction") is the binding constraint, not this section — see Section 6's WEAK
flags (as low as 29 games) for where the real risk sits. This section confirms
that fold-availability for the *training* side of the leave-one-opponent-out
design is not a limiting factor anywhere in the 8-cohort; the limiting factor is
always the held-out dyad's own small sample size.
