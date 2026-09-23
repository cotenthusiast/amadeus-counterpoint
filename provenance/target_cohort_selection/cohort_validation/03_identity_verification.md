# Section 3 — Manual/Conservative Identity Verification (Current 8)

Method: for each target player, pull every raw name string that the corpus's
identity-resolution union-find (see `audit/scripts/07_pairs.py`, reused verbatim in
`identity_resolution.py`) merged into their identity, then independently check the
title/rating/date evidence attached to each raw spelling. A valid FIDE ID (strict
6–9-digit numeric; junk placeholders like "-", "0", "IND" are rejected — see
`audit/scripts/common.py`) is treated as the strongest identity evidence, per the
research brief. Nothing here is fuzzy-name-only: every merge below is backed by
either a shared valid FIDE ID or an exact post-normalization name match, and every
alias's rating/date/title profile was checked for a plausible, non-conflicting
story before being accepted.

## Per-player evidence

### Carlsen, Magnus
- Canonical id: `F:1503014`, FIDE ID **1503014**, title GM (consistent, n_distinct_titles=1 on raw canonical spelling)
- Raw variants (Tier A, shared FIDE ID, evidence_type=`same_fide_id_consistent`, group 1618): `Carlsen, Magnus` (1486 games raw), `Magnus Carlsen` (501), `MagnusCarlsen` (113, no-space Lichess-username form), `Carlsen Magnus (NOR)` (44)
- Per-alias rating range (rows where that row's own FIDE tag is present): `Carlsen, Magnus` 2808–2953 (μ2854); `Magnus Carlsen` 2829–3331 (μ3263, includes a Sept-2023 elite blitz/bullet event with much higher time-control rating — not a discrepancy, a different time control); `MagnusCarlsen` 3198–3358; `Carlsen Magnus (NOR)` 2835 flat.
- First/last observed date (identity-resolved, all variants): 2020-01-11 .. 2026-06-21.
- No conflicting metadata found; no evidence any other unrelated raw identity shares FIDE 1503014.
- **VERIFIED_HIGH_CONFIDENCE**

### So, Wesley
- Canonical id: `F:5202213`, FIDE ID **5202213**, title GM
- Raw variants (Tier A, flagged `same_fide_id_needs_review` by the automated check, group 1624 — see below): `So, Wesley` (1128), `Wesley So` (608), `GMWSO` (39), `So Wesley (USA)` (10)
- Why the automated pipeline flagged this "needs review": `GMWSO` shares no whitespace-token or letter-signature with the other spellings (it's an initialism, not a spelling variant), so the Tier-A consistency check couldn't auto-confirm it. Manual check: `GMWSO` games carry rating 3059–3125 (a Sept-2023 blitz/bullet exhibition, same date window and title context as the other high-rating outlier aliases above — consistent with a single well-known elite blitz event, not a different person), dates fall entirely inside Wesley So's active broadcast window, and the FIDE ID match (a real, hard-to-guess 7-digit ID) is exact and shared by no unrelated name in the corpus. `GMWSO` is the well-known "GM Wesley SO" Lichess-username convention.
- **VERIFIED_WITH_CAVEAT** — caveat: the automated needs-review flag was resolved manually here (token/signature mismatch on a nickname, not a genuine identity conflict); no unresolved ambiguity remains.

### Aronian, Levon
- Canonical id: `F:13300474`, FIDE ID **13300474**, title GM
- Only ONE raw spelling merged via the FIDE-ID/canonical-key path used for pairs.csv counting: `Aronian, Levon` (1311) + reordered `Levon Aronian` (396) + `Aronian Levon` (7) — all under canonical-key group 3000 (sum 1714, matches players_normalized total exactly).
- **Known caveat:** a separate canonical-key group (11461) contains `Aronian Levon (ARM)` / `Aronian Levon  (ARM)` (29 + 1 = 30 games, all 2021, rating flat 2782, no FIDE tag on these rows). Because the appended "(ARM)" country-code token changes the sorted canonical key, these did NOT get merged into the main identity by the (deliberately conservative) merge rule, even though they are almost certainly the same player. This means Aronian's true total is ~30 games higher than reported everywhere in this analysis — negligible relative to his ~1714-game total and does not change any cohort/dyad conclusion, but is disclosed here rather than silently corrected (per the "do not silently merge identities" instruction).
- No conflicting metadata; FIDE ID not shared with any other raw identity.
- **VERIFIED_HIGH_CONFIDENCE** (with the above disclosed, immaterial undercount noted)

### Caruana, Fabiano
- Canonical id: `F:2020009`, FIDE ID **2020009**, title GM
- Raw variants (Tier A consistent, group 1497): `Caruana, Fabiano` (1508), `Caruana Fabiano (USA)` (25)
- No needs-review flag, no conflicting metadata, FIDE ID unique to this player.
- **VERIFIED_HIGH_CONFIDENCE**

### Vachier-Lagrave, Maxime
- Canonical id: `F:623539`, FIDE ID **623539**, title GM
- Raw variants (Tier A, flagged `same_fide_id_needs_review`, group 1589): `Vachier-Lagrave, Maxime` (1195), `Maxime Vachier-Lagrave` (330), `Vachier-Lagrave Maxime (FRA)` (23), `LyonBeast` (59)
- Why flagged: `LyonBeast` shares no token/signature with the conventional spellings. Manual check: `LyonBeast`'s single logged own-FIDE-tag row has rating 3146 on 2023-09-15 — the same date as the other players' elite blitz-event outlier ratings above (a shared broadcast/event window), and the FIDE ID match is exact. `LyonBeast` is Vachier-Lagrave's well-known Lichess handle (from his native Lyon, France).
- **VERIFIED_WITH_CAVEAT** — same resolution pattern as So/Wesley above: automated flag was a nickname/token mismatch, not a real conflict.

### Nakamura, Hikaru
- Canonical id: `F:2016192`, FIDE ID **2016192**, title GM
- Raw variants (Tier A consistent, group 360): `Nakamura, Hikaru` (1126), `Hikaru Nakamura` (386), `Nakamura Hikaru (USA)` (8)
- No needs-review flag, no conflicting metadata, FIDE ID unique to this player.
- **VERIFIED_HIGH_CONFIDENCE**

### Nepomniachtchi, Ian
- Canonical id: `F:4168119`, FIDE ID **4168119**, title GM
- Raw variants (Tier A, flagged `same_fide_id_needs_review`, group 1143): `Nepomniachtchi, Ian` (1125), `lachesisQ` (63), `Nepomniachtchi Ian (FID)` (16)
- Why flagged: `lachesisQ` shares no token/signature with the conventional spellings. Manual check: `lachesisQ`'s own-FIDE-tag rows show rating 3010–3105 on 2023-09-15 (same elite-blitz-event date window as the other nickname outliers above), exact FIDE ID match, dates inside his active broadcast window. `lachesisQ` is Nepomniachtchi's well-known Lichess handle.
- Note: `Nepomniachtchi Ian (FID)` uses the FIDE (international federation) flag code rather than a national federation, consistent with his 2022–2023 competition status — not a red flag.
- **VERIFIED_WITH_CAVEAT** — same nickname-resolution pattern as above.

### Firouzja, Alireza
- Canonical id: `F:12573981`, FIDE ID **12573981**, title GM
- Only one raw spelling merged via the counted path: `Firouzja, Alireza` (963) + reordered `Alireza Firouzja` (213), canonical-key group 3017 (sum 1176, matches players_normalized total).
- **Known caveat**, same pattern as Aronian: a separate group (15384) with `Firouzja Alireza (FRA)` / `Firouzja Alireza  (FRA)` (15 + 6 = 21 games, all 2021, flat ratings 2759–2770, no FIDE tag) did not merge into the main identity due to the appended country-code token. Immaterial (~21 games vs. 1176 total), disclosed rather than silently fixed.
- No conflicting metadata; FIDE ID not shared with any other raw identity.
- **VERIFIED_HIGH_CONFIDENCE** (with the above disclosed, immaterial undercount noted)

## Per-player first/last observed date and rating (identity-resolved, all-variant dedup basis)

| player | first_date | last_date | mean_rating | median_rating | total_games |
|---|---|---|---:|---:|---:|
| Carlsen, Magnus | 2020-01-11 | 2026-06-21 | 2900.3 | 2856.0 | 2157 |
| So, Wesley | 2020-01-11 | 2026-07-05 | 2788.7 | 2770.0 | 1795 |
| Aronian, Levon | 2020-05-07 | 2026-07-22 | 2765.1 | 2756.0 | 1714 |
| Caruana, Fabiano | 2020-01-11 | 2026-07-04 | 2788.3 | 2782.0 | 1714 |
| Vachier-Lagrave, Maxime | 2019-11-23 | 2026-07-03 | 2804.3 | 2758.0 | 1640 |
| Nakamura, Hikaru | 2020-04-27 | 2026-07-05 | 2860.8 | 2828.0 | 1524 |
| Nepomniachtchi, Ian | 2020-03-17 | 2026-07-22 | 2777.6 | 2778.0 | 1440 |
| Firouzja, Alireza | 2020-01-11 | 2026-07-22 | 2773.7 | 2767.0 | 1176 |

(These totals are the all-variant, deduplicated basis, i.e. `players_normalized.csv`
— matching the original stage-1 headline numbers; the standard-chess-only totals
used for sections 6/7/8 are lower, see `player_totals_standard_dedup.csv`.)

## Cross-check: any FIDE ID shared with an unrelated-looking raw identity?

For each of the 8 target canonical raw-name spellings, `players.csv` shows
`n_distinct_valid_fide_ids = 1` and `n_distinct_titles = 1` — i.e. that exact
spelling never appears in the corpus tagged with a second, different FIDE ID or a
conflicting title. Combined with the alias-group membership dumps above (every
group's membership list contains only that one player's own known spellings/
nicknames, nothing else), there is no evidence of a genuine identity mixup for
any of the 8.

## Classification summary

| player | classification |
|---|---|
| Carlsen, Magnus | VERIFIED_HIGH_CONFIDENCE |
| So, Wesley | VERIFIED_WITH_CAVEAT (GMWSO nickname resolved manually) |
| Aronian, Levon | VERIFIED_HIGH_CONFIDENCE (immaterial ~30-game unmerged ARM-suffix spelling disclosed) |
| Caruana, Fabiano | VERIFIED_HIGH_CONFIDENCE |
| Vachier-Lagrave, Maxime | VERIFIED_WITH_CAVEAT (LyonBeast nickname resolved manually) |
| Nakamura, Hikaru | VERIFIED_HIGH_CONFIDENCE |
| Nepomniachtchi, Ian | VERIFIED_WITH_CAVEAT (lachesisQ nickname resolved manually) |
| Firouzja, Alireza | VERIFIED_HIGH_CONFIDENCE (immaterial ~21-game unmerged FRA-suffix spelling disclosed) |

**No player is NEEDS_MANUAL_REVIEW.** All 8 identities are clean enough to freeze
(Section 8), subject to the two disclosed caveats above (minor, ~1–2% undercounts
from un-mergeable country-suffixed spellings, and three FIDE-nickname flags
resolved by manual rating/date/title cross-check rather than by the automated
token/signature heuristic).
