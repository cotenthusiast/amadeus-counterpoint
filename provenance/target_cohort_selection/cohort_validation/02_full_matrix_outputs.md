# Section 2 — Full Matrix Outputs (best cohort per size)

All figures use the corrected (dedup + `Variant='Standard'`) basis (see Section 1).
Machine-readable CSVs for every size are under `matrices/`:

- `matrices/matrix_{N}_standard.csv` — full pairwise count matrix
- `matrices/edges_sorted_{N}_standard.csv` — every dyad, sorted weakest→strongest
- `matrices/per_player_{N}_standard.csv` — total / internal / outside-cohort games per player

(All-variant, pre-correction equivalents without the `_standard` suffix are also
present, produced by `01_cohort_compare.py`, for before/after comparison.)

## Size 8 (current cohort) — full pairwise matrix, standard+dedup basis

|                         | Caruana | Carlsen | So | MVL | Aronian | Nakamura | Nepomniachtchi | Firouzja |
|:------------------------|--------:|--------:|---:|----:|--------:|---------:|---------------:|---------:|
| Caruana, Fabiano        |       0 |     102 | 85 |  54 |      57 |       71 |             30 |       59 |
| Carlsen, Magnus         |     102 |       0 |107 |  55 |      32 |      123 |             78 |       97 |
| So, Wesley              |      85 |     107 |  0 |  77 |      50 |       68 |             30 |       46 |
| Vachier-Lagrave, Maxime |      54 |      55 | 77 |   0 |      34 |       54 |             65 |       44 |
| Aronian, Levon          |      57 |      32 | 50 |  34 |       0 |       64 |             29 |       33 |
| Nakamura, Hikaru        |      71 |     123 | 68 |  54 |      64 |        0 |             31 |       65 |
| Nepomniachtchi, Ian     |      30 |      78 | 30 |  65 |      29 |       31 |              0 |       42 |
| Firouzja, Alireza       |      59 |      97 | 46 |  44 |      33 |       65 |             42 |        0 |

**Weakest 5 (standard+dedup):**
1. Aronian–Nepomniachtchi = 29
2. Caruana–Nepomniachtchi = 30
3. So–Nepomniachtchi = 30
4. Nakamura–Nepomniachtchi = 31
5. Carlsen–Aronian = 32

**Strongest 5 (standard+dedup):**
1. Carlsen–Nakamura = 123
2. Carlsen–So = 107
3. Carlsen–Caruana = 102
4. Carlsen–Firouzja = 97
5. So–Vachier-Lagrave = 77

**Per-player (size-8, standard+dedup):**

| player | total games | internal (vs cohort) | outside cohort |
|---|---:|---:|---:|
| Carlsen, Magnus | 1456 | 594 | 862 |
| Caruana, Fabiano | 1561 | 458 | 1103 |
| So, Wesley | 1409 | 463 | 946 |
| Vachier-Lagrave, Maxime | 1396 | 383 | 1013 |
| Aronian, Levon | 1241 | 299 | 942 |
| Nakamura, Hikaru | 1226 | 476 | 750 |
| Nepomniachtchi, Ian | 1158 | 305 | 853 |
| Firouzja, Alireza | 999 | 386 | 613 |

(Sizes 6, 7, 9 full matrices are in `matrices/matrix_{6,7,9}_standard.csv` and
`cohort_{6,7,9}_standard.md` — not reproduced inline here to keep this report
short; the weakest/strongest-5 and per-player CSVs cover them.)
