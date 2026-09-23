# Amadeus Final Results Summary

This summary uses the finalized sealed M1, M2, and complete M3 result JSONs. The historical Croatia GCT / Sinquefield tournament experiment is excluded.

## Main sealed interaction results

Lower total-variation distance is better; values are equal-weighted across 28 dyads with 50/50 color-orientation aggregation.

| Method | WDL-TV GG | WDL-TV AB | WDL AB−GG | Opening-TV GG | Opening-TV AB | Opening AB−GG |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.164171 | 0.150148 | -0.014024 | 0.554433 | 0.536973 | -0.017461 |
| M2 | 0.163166 | 0.163408 | +0.000241 | 0.554711 | 0.437202 | -0.117509 |
| M3 | 0.163073 | 0.152036 | -0.011037 | 0.554395 | 0.444428 | -0.109967 |

## Dyad-level support

- M1: WDL-TV improved for 22/28 dyads; opening-family TV improved for 22/28.
- M2: WDL-TV improved for 13/28; opening-family TV improved for 27/28.
- M3: WDL-TV improved for 19/28; opening-family TV improved for 27/28.

Interpretation: M3 combines an M1-like WDL improvement with an M2-like opening-family improvement, while remaining exploratory and post-hoc. These are descriptive results, not a confirmatory comparison.

## Individual validation

The nonsealed identity diagnostic recovered the correct player representation as rank 1 for all 8 players for both M1 and M2. Mean generic-minus-correct NLL was 0.006802 for M1 and 0.009278 for M2; the NLL definitions differ and should not be treated as directly comparable effect sizes.

## Provenance and caveats

- 8 players, 28 dyads, 4 conditions, 2 orientations, 5,000 games per cell.
- M1/M2/M3 each have 1,120,000 synthetic games represented in the finalized pairwise evaluation.
- Bootstrap: 10,000 whole-real-game replicates per dyad and metric.
- Censored games were excluded from denominators rather than counted as draws.
- Aggregate bootstrap confidence intervals were not saved in the final JSONs; only per-dyad intervals are available in the source artifacts.
- M3 is a post-hoc exploratory hybrid and must be labeled as such in the paper.

See [results_audit.md](../docs/audits/results_audit.md) for the full dyad-level audit and [results_cheatsheet.md](results_cheatsheet.md) for a compact lookup.
