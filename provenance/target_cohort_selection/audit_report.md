# Lichess Broadcast Corpus -- Data Audit Report

Scope: 147,663 raw player-name identities, 936,377 unordered
player pairs, 79 monthly PGN files (2020-01 .. 2026-07), 1,186,338 parsed game
records. This audit is READ-ONLY: no source PGN file was modified, no duplicate
was deleted, and no player identity was merged without disclosed evidence. All
scripts live in `audit/scripts/`; all outputs live in `audit/`.

## How to reproduce

```
cd audit/scripts
python3 01_parse.py        # PGN -> data/games.parquet (~3 min)
python3 02_sanity.py       # -> reports/01_sanity.md
python3 03_duplicates.py   # -> reports/02_duplicates.md, duplicates.csv
python3 04_players.py      # -> players.csv, reports/03_leaderboard.md
python3 05_aliases.py      # -> suspected_aliases.csv, reports/04_aliases.md
python3 06_ratings.py      # -> reports/05_ratings.md
python3 07_pairs.py        # -> players_normalized.csv, pairs.csv, reports/06_pairs.md
python3 08_cohort.py       # -> cohort_6.md, cohort_7.md, cohort_8.md, cohort_9.md
python3 09_report.py       # -> audit_report.md (this file)
```

## 1. Corpus sanity -- see [reports/01_sanity.md](reports/01_sanity.md)

- 1,186,338 parsed games across 79 files, 0 structurally unparseable (all had an
  `[Event ...]` tag), 771 empty placeholder chapters (no players, no moves).
- Missing tags: White 787 (0.07%), Black 931 (0.08%), Result 3, Date 358,871
  (30.25%, includes PGN `?`-placeholders and 3,285 corrupted `ø`-placeholder dates
  in a handful of broadcasts), WhiteElo/BlackElo ~17.5% each, WhiteTitle/BlackTitle
  ~66% each (most broadcast games are untitled amateur players), FIDE IDs ~33% each.
- Results: 1-0 447,145 / 0-1 371,475 / 1/2-1/2 278,866 / `*` 88,716 / ~120 games
  with a non-standard result string (e.g. `½-½`, `+--`).
- Non-Standard variant games: 11,094 (Chess960, From Position, Fischerandom,
  Crazyhouse, King of the Hill, Antichess) -- excluded from ratings/cohort analysis
  is NOT automatic; they are included unless a downstream script filters on Variant.
  For a research cohort we recommend restricting to `Variant='Standard'`.
- Unusable games: 87,475 zero-ply, 68,989 of those are also `Result='*'` (i.e.
  chapters set up but never played), 7,901 short games of 1-9 plies (likely
  resignations/no-shows/misclicks, not abandoned uploads).

## 2. Duplicates -- see [reports/02_duplicates.md](reports/02_duplicates.md), [duplicates.csv](duplicates.csv)

- GameURL is present on 1,186,335/1,186,338 games and unique for all but 2 pairs
  (2 games got re-published under an identical GameURL in a later monthly file).
- **9,068 rows are exact content duplicates** (identical White, Black, Date,
  Result, and full move list) spread across 3,461 groups in ~variable number of
  different broadcasts -- concentrated in round-robin-style events where the
  identical game recording was apparently copy-pasted across many "round"
  chapters (e.g. the German Blitz Chess Championship (Women), 2023-07, where
  several pairings show the exact same game repeated for all ~21 rounds). This
  looks like a broadcast-authoring artifact, not real distinct games.
- **We did not delete anything.** All downstream player/pair/cohort analysis uses
  a `games_dedup` view that keeps only the first-occurring row per unique
  content_hash (see `scripts/common.py`) -- an analysis-time choice, fully
  reversible, documented, and separate from `games.parquet` and the source PGNs.

## 3. Player identity -- see [players.csv](players.csv), [suspected_aliases.csv](suspected_aliases.csv), [reports/04_aliases.md](reports/04_aliases.md)

- 147,663 distinct raw name strings observed (White+Black union),
  never merged in players.csv.
- suspected_aliases.csv has 57,740 rows across 26,164
  evidence groups in 3 tiers:
  - **Tier A -- shared valid FIDE ID** (6-9 digit numeric only; the raw FIDE ID
    tags also contain junk placeholders like `-`, `-1`, `1`, or country codes like
    `IND`/`CZE` which are excluded from "valid"): 2,761 consistent groups + 235
    "needs_review" groups where the shared-ID name variants don't share a
    token/letter -- usually a Lichess nickname of the real title-holder (confirmed
    examples: `LyonBeast`=Vachier-Lagrave Maxime, `GMWSO`=Wesley So,
    `lachesisQ`=Nepomniachtchi Ian) but occasionally a genuine data-entry mixup
    (confirmed example: FIDE ID 21834008 is shared by IM Niedbala, Bartlomiej
    (314 games) and a 5-game "Pinter, Filip" who is very likely a different,
    unrelated player wrongly tagged with Niedbala's ID in one small event).
  - **Tier B -- identical after case/diacritic/punctuation/word-order
    normalization** ("Magnus Carlsen" == "Carlsen, Magnus"): 22,812 groups.
  - **Tier C -- fuzzy match (token_sort_ratio>=92)**, scoped to names with no
    valid FIDE ID and >=5 games (27,112 candidates; full O(n^2) over all 147k
    names was not attempted): 356 groups.
- **players.csv rows are never merged.** For pairwise/cohort analysis
  (players_normalized.csv, pairs.csv, cohort_*.md) we DO normalize identity, but
  only via a union-find over two certain-equivalence rules: (1) shared valid FIDE
  ID, (2) identical canonical name form -- and only when that canonical form maps
  to at most one distinct FIDE ID across the whole corpus (389 canonical-name
  groups were SKIPPED as ambiguous common-name collisions, e.g. several unrelated
  "Molnar, Laszlo"s each have their own real FIDE ID). 336 resulting identity
  components still contain >1 distinct FIDE ID (typically because one raw name
  string was inconsistently tagged with more than one FIDE ID across different
  broadcasts) -- a residual data-quality limitation, not a merge choice.

## 4. Rating / title distribution -- see [reports/05_ratings.md](reports/05_ratings.md)

- Combined White/Black Elo: median 2071, p90 2526, p99 3008 (before excluding
  the data-quality issues below).
- **Data quality caveats**: 26,777 Elo values of exactly 0 (placeholder, not a
  real rating), a handful of Elo values are FIDE-ID-sized numbers apparently
  miskeyed into the Elo field (e.g. "5327130"), and BOT-titled players can show
  Elo 3700-3900+ (their real Lichess bot rating, not comparable to human Elo).
  Recommend filtering to 1000<=Elo<=3200 and title!='BOT' for any rating-based
  cohort scoring.
- Both players GM: 59,703 games. Both players >=2700: 16,159 games.
- Title field has a long non-standard tail (`k`, `*`, `1N`, numeric-looking
  values); treat anything outside the standard FIDE/Lichess title set as unreliable.

## 5. Leaderboard -- see [reports/03_leaderboard.md](reports/03_leaderboard.md) (raw-name basis, top 100)

Top raw names by deduplicated game count are essentially all elite GMs
(Caruana, Carlsen, Abdusattorov, Aronian, Giri, Vachier-Lagrave, Keymer, ...),
each in the 900-1500 game range on a single dominant spelling. FIDE-normalized
totals (players_normalized.csv) are higher once nickname/format variants are
merged -- e.g. Carlsen's raw-name total is 1,486 games but his FIDE-normalized
total is 2,157 games.

## 6. Pairwise interactions -- see [reports/06_pairs.md](reports/06_pairs.md), [pairs.csv](pairs.csv)

- 119,134 resolved identities (85,340
  FIDE-normalized), 936,377 unordered pairs with >=1 shared game.
- Pair-count distribution is heavily skewed: median pair has exactly 1 shared
  game; only 935 pairs have >=10 games, and
  43 pairs have >=50 games. The corpus's
  interaction density is concentrated almost entirely among the handful of elite
  players who appear in many recurring broadcasts (Titled Tuesday-style rapid/
  blitz events, Grand Chess Tour, Norway Chess, etc.) -- exactly the players a
  dense cohort should draw from.
- Games touched by a "needs_review" FIDE ID: 21,646 / 1,093,256 deduplicated
  games (1.98%) -- immaterial at the corpus level, but worth a per-player glance
  for any chosen cohort (see section 8 below: all clear for our top-8 pick).

## 7 & 8. Dense cohort search -- see [cohort_6.md](cohort_6.md), [cohort_7.md](cohort_7.md), [cohort_8.md](cohort_8.md), [cohort_9.md](cohort_9.md)

Method: candidate pool = the 24 most active FIDE-normalized, non-BOT identities
(all happen to be GMs); every combination of size 6/7/8/9 from that pool was
exhaustively scored (primary objective = maximize the minimum pairwise game
count, tie-break by median edge, then total internal games), cross-checked with
a swap-based local search over the top 70 players to make sure nothing better
lay just outside the exact pool. This is a "promising cohorts" search, not a
proof of global optimality over the full corpus.

### Best 8-player cohort

## Candidate #1

- min pairwise edge: **42**
- median pairwise edge: **72.0**
- mean pairwise edge: **80.0**
- total internal games (sum of all edges): **2241**
- players: Carlsen, Magnus, So, Wesley, Aronian, Levon, Caruana, Fabiano, Vachier-Lagrave, Maxime, Nakamura, Hikaru, Nepomniachtchi, Ian, Firouzja, Alireza

**Player summary**

| player                  | title   |   mean_rating |   total_games |   games_vs_cohort |   games_outside_cohort |
|:------------------------|:--------|--------------:|--------------:|------------------:|-----------------------:|
| Carlsen, Magnus         | GM      |        2900.3 |          2157 |               849 |                   1308 |
| So, Wesley              | GM      |        2788.7 |          1795 |               640 |                   1155 |
| Aronian, Levon          | GM      |        2765.1 |          1714 |               471 |                   1243 |
| Caruana, Fabiano        | GM      |        2788.3 |          1714 |               529 |                   1185 |
| Vachier-Lagrave, Maxime | GM      |        2804.3 |          1640 |               502 |                   1138 |
| Nakamura, Hikaru        | GM      |        2860.8 |          1524 |               617 |                    907 |
| Nepomniachtchi, Ian     | GM      |        2777.6 |          1440 |               425 |                   1015 |
| Firouzja, Alireza       | GM      |        2773.7 |          1176 |               449 |                    727 |

**Pairwise matrix**

|                         |   Carlsen, Magnus |   So, Wesley |   Aronian, Levon |   Caruana, Fabiano |   Vachier-Lagrave, Maxime |   Nakamura, Hikaru |   Nepomniachtchi, Ian |   Firouzja, Alireza |
|:------------------------|------------------:|-------------:|-----------------:|-------------------:|--------------------------:|-------------------:|----------------------:|--------------------:|
| Carlsen, Magnus         |                 0 |          169 |               94 |                120 |                        78 |                158 |                   123 |                 107 |
| So, Wesley              |               169 |            0 |               69 |                 89 |                       105 |                 93 |                    46 |                  69 |
| Aronian, Levon          |                94 |           69 |                0 |                 71 |                        63 |                 86 |                    46 |                  42 |
| Caruana, Fabiano        |               120 |           89 |               71 |                  0 |                        59 |                 81 |                    43 |                  66 |
| Vachier-Lagrave, Maxime |                78 |          105 |               63 |                 59 |                         0 |                 77 |                    72 |                  48 |
| Nakamura, Hikaru        |               158 |           93 |               86 |                 81 |                        77 |                  0 |                    50 |                  72 |
| Nepomniachtchi, Ian     |               123 |           46 |               46 |                 43 |                        72 |                 50 |                     0 |                  45 |
| Firouzja, Alireza       |               107 |           69 |               42 |                 66 |                        48 |                 72 |                    45 |                   0 |

**All 28 pairwise counts, sorted**

| player_A                | player_B                |   games |
|:------------------------|:------------------------|--------:|
| Aronian, Levon          | Firouzja, Alireza       |      42 |
| Caruana, Fabiano        | Nepomniachtchi, Ian     |      43 |
| Nepomniachtchi, Ian     | Firouzja, Alireza       |      45 |
| So, Wesley              | Nepomniachtchi, Ian     |      46 |
| Aronian, Levon          | Nepomniachtchi, Ian     |      46 |
| Vachier-Lagrave, Maxime | Firouzja, Alireza       |      48 |
| Nakamura, Hikaru        | Nepomniachtchi, Ian     |      50 |
| Caruana, Fabiano        | Vachier-Lagrave, Maxime |      59 |
| Aronian, Levon          | Vachier-Lagrave, Maxime |      63 |
| Caruana, Fabiano        | Firouzja, Alireza       |      66 |
| So, Wesley              | Aronian, Levon          |      69 |
| So, Wesley              | Firouzja, Alireza       |      69 |
| Aronian, Levon          | Caruana, Fabiano        |      71 |
| Vachier-Lagrave, Maxime | Nepomniachtchi, Ian     |      72 |
| Nakamura, Hikaru        | Firouzja, Alireza       |      72 |
| Vachier-Lagrave, Maxime | Nakamura, Hikaru        |      77 |
| Carlsen, Magnus         | Vachier-Lagrave, Maxime |      78 |
| Caruana, Fabiano        | Nakamura, Hikaru        |      81 |
| Aronian, Levon          | Nakamura, Hikaru        |      86 |
| So, Wesley              | Caruana, Fabiano        |      89 |
| So, Wesley              | Nakamura, Hikaru        |      93 |
| Carlsen, Magnus         | Aronian, Levon          |      94 |
| So, Wesley              | Vachier-Lagrave, Maxime |     105 |
| Carlsen, Magnus         | Firouzja, Alireza       |     107 |
| Carlsen, Magnus         | Caruana, Fabiano        |     120 |
| Carlsen, Magnus         | Nepomniachtchi, Ian     |     123 |
| Carlsen, Magnus         | Nakamura, Hikaru        |     158 |
| Carlsen, Magnus         | So, Wesley              |     169 |

- weakest 5 edges: [('Aronian, Levon', 'Firouzja, Alireza', 42), ('Caruana, Fabiano', 'Nepomniachtchi, Ian', 43), ('Nepomniachtchi, Ian', 'Firouzja, Alireza', 45), ('So, Wesley', 'Nepomniachtchi, Ian', 46), ('Aronian, Levon', 'Nepomniachtchi, Ian', 46)]

- strongest 5 edges: [('Carlsen, Magnus', 'Firouzja, Alireza', 107), ('Carlsen, Magnus', 'Caruana, Fabiano', 120), ('Carlsen, Magnus', 'Nepomniachtchi, Ian', 123), ('Carlsen, Magnus', 'Nakamura, Hikaru', 158), ('Carlsen, Magnus', 'So, Wesley', 169)]

- minimum outside-cohort games for any player: **727** (Firouzja, Alireza)



**Identity-quality check on this cohort**: 3 of the 8 players (So, Wesley;
Vachier-Lagrave, Maxime; Nepomniachtchi, Ian) have a "needs_review" FIDE-ID
group in suspected_aliases.csv. In all 3 cases the divergent name is a confirmed
Lichess username of that same player (GMWSO, LyonBeast, lachesisQ respectively)
-- not a different person. No other identity-quality issue was found for this
cohort (no ambiguous-common-name collisions, no multi-FIDE component, all 8 are
titled GM with plausible Elo).

See cohort_6.md / cohort_7.md / cohort_9.md for the other sizes and 4 runner-up
alternatives per size.

## Known limitations / uncertainty (read before use)

1. Duplicate handling is an analysis-time deduplication (content_hash), not a
   removal from source data; see section 2. It affects ~0.8% of games corpus-wide
   and appears immaterial to the chosen elite cohort's pairwise counts (spot-check
   this if precision matters for the paper).
2. FIDE ID is trusted as identity evidence corpus-wide, including 235
   "needs_review" groups where the name variance is unexplained by simple
   normalization. We manually confirmed the 3 instances relevant to the top-8
   cohort are legitimate nicknames, not mixups; we did NOT manually check the
   other ~230 groups (they don't affect the recommended cohort).
3. Canonical-name-based merging (Tier B) was applied only when non-ambiguous
   (<=1 distinct FIDE ID per canonical form); 389 ambiguous common-name groups
   were deliberately left unmerged.
4. Non-standard chess variants (11,094 games) are NOT filtered out anywhere by
   default; none of the top-24 candidate pool's games were checked variant-by-variant.
5. Rating field mixes different time controls/contexts (classical/rapid/blitz/
   bullet arena, and bot ratings) with no reliable column to separate them for
   most rows -- mean_rating in players_normalized.csv should be read as a rough
   indicator, not a rigorous classical Elo.
6. Cohort search used a fixed candidate pool of 24 (cross-checked against a
   swap search over 70); a wider or different pool could in principle surface a
   marginally denser cohort we did not find.
