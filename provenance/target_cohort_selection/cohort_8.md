# Cohort search -- size 8

Candidate pool: top 24 FIDE-identified, non-BOT players by total games (exhaustively searched, all C(24,8) combinations), cross-checked with a swap-based local search over the top 70 players.


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

## Candidate #2

- min pairwise edge: **29**
- median pairwise edge: **48.0**
- mean pairwise edge: **59.8**
- total internal games (sum of all edges): **1673**
- players: Carlsen, Magnus, So, Wesley, Caruana, Fabiano, Vachier-Lagrave, Maxime, Abdusattorov, Nodirbek, Giri, Anish, Praggnanandhaa R, Firouzja, Alireza

**Player summary**

| player                  | title   |   mean_rating |   total_games |   games_vs_cohort |   games_outside_cohort |
|:------------------------|:--------|--------------:|--------------:|------------------:|-----------------------:|
| Carlsen, Magnus         | GM      |        2900.3 |          2157 |               643 |                   1514 |
| So, Wesley              | GM      |        2788.7 |          1795 |               569 |                   1226 |
| Caruana, Fabiano        | GM      |        2788.3 |          1714 |               446 |                   1268 |
| Vachier-Lagrave, Maxime | GM      |        2804.3 |          1640 |               404 |                   1236 |
| Abdusattorov, Nodirbek  | GM      |        2717.4 |          1563 |               306 |                   1257 |
| Giri, Anish             | GM      |        2727.6 |          1515 |               325 |                   1190 |
| Praggnanandhaa R        | GM      |        2689.7 |          1314 |               250 |                   1064 |
| Firouzja, Alireza       | GM      |        2773.7 |          1176 |               403 |                    773 |

**Pairwise matrix**

|                         |   Carlsen, Magnus |   So, Wesley |   Caruana, Fabiano |   Vachier-Lagrave, Maxime |   Abdusattorov, Nodirbek |   Giri, Anish |   Praggnanandhaa R |   Firouzja, Alireza |
|:------------------------|------------------:|-------------:|-------------------:|--------------------------:|-------------------------:|--------------:|-------------------:|--------------------:|
| Carlsen, Magnus         |                 0 |          169 |                120 |                        78 |                       58 |            67 |                 44 |                 107 |
| So, Wesley              |               169 |            0 |                 89 |                       105 |                       56 |            51 |                 30 |                  69 |
| Caruana, Fabiano        |               120 |           89 |                  0 |                        59 |                       44 |            34 |                 34 |                  66 |
| Vachier-Lagrave, Maxime |                78 |          105 |                 59 |                         0 |                       37 |            48 |                 29 |                  48 |
| Abdusattorov, Nodirbek  |                58 |           56 |                 44 |                        37 |                        0 |            38 |                 35 |                  38 |
| Giri, Anish             |                67 |           51 |                 34 |                        48 |                       38 |             0 |                 45 |                  42 |
| Praggnanandhaa R        |                44 |           30 |                 34 |                        29 |                       35 |            45 |                  0 |                  33 |
| Firouzja, Alireza       |               107 |           69 |                 66 |                        48 |                       38 |            42 |                 33 |                   0 |

**All 28 pairwise counts, sorted**

| player_A                | player_B                |   games |
|:------------------------|:------------------------|--------:|
| Vachier-Lagrave, Maxime | Praggnanandhaa R        |      29 |
| So, Wesley              | Praggnanandhaa R        |      30 |
| Praggnanandhaa R        | Firouzja, Alireza       |      33 |
| Caruana, Fabiano        | Giri, Anish             |      34 |
| Caruana, Fabiano        | Praggnanandhaa R        |      34 |
| Abdusattorov, Nodirbek  | Praggnanandhaa R        |      35 |
| Vachier-Lagrave, Maxime | Abdusattorov, Nodirbek  |      37 |
| Abdusattorov, Nodirbek  | Giri, Anish             |      38 |
| Abdusattorov, Nodirbek  | Firouzja, Alireza       |      38 |
| Giri, Anish             | Firouzja, Alireza       |      42 |
| Carlsen, Magnus         | Praggnanandhaa R        |      44 |
| Caruana, Fabiano        | Abdusattorov, Nodirbek  |      44 |
| Giri, Anish             | Praggnanandhaa R        |      45 |
| Vachier-Lagrave, Maxime | Giri, Anish             |      48 |
| Vachier-Lagrave, Maxime | Firouzja, Alireza       |      48 |
| So, Wesley              | Giri, Anish             |      51 |
| So, Wesley              | Abdusattorov, Nodirbek  |      56 |
| Carlsen, Magnus         | Abdusattorov, Nodirbek  |      58 |
| Caruana, Fabiano        | Vachier-Lagrave, Maxime |      59 |
| Caruana, Fabiano        | Firouzja, Alireza       |      66 |
| Carlsen, Magnus         | Giri, Anish             |      67 |
| So, Wesley              | Firouzja, Alireza       |      69 |
| Carlsen, Magnus         | Vachier-Lagrave, Maxime |      78 |
| So, Wesley              | Caruana, Fabiano        |      89 |
| So, Wesley              | Vachier-Lagrave, Maxime |     105 |
| Carlsen, Magnus         | Firouzja, Alireza       |     107 |
| Carlsen, Magnus         | Caruana, Fabiano        |     120 |
| Carlsen, Magnus         | So, Wesley              |     169 |

- weakest 5 edges: [('Vachier-Lagrave, Maxime', 'Praggnanandhaa R', 29), ('So, Wesley', 'Praggnanandhaa R', 30), ('Praggnanandhaa R', 'Firouzja, Alireza', 33), ('Caruana, Fabiano', 'Giri, Anish', 34), ('Caruana, Fabiano', 'Praggnanandhaa R', 34)]

- strongest 5 edges: [('So, Wesley', 'Caruana, Fabiano', 89), ('So, Wesley', 'Vachier-Lagrave, Maxime', 105), ('Carlsen, Magnus', 'Firouzja, Alireza', 107), ('Carlsen, Magnus', 'Caruana, Fabiano', 120), ('Carlsen, Magnus', 'So, Wesley', 169)]

- minimum outside-cohort games for any player: **773** (Firouzja, Alireza)

## Candidate #3

- min pairwise edge: **29**
- median pairwise edge: **47.5**
- mean pairwise edge: **59.8**
- total internal games (sum of all edges): **1674**
- players: Carlsen, Magnus, So, Wesley, Caruana, Fabiano, Vachier-Lagrave, Maxime, Giri, Anish, Praggnanandhaa R, Duda, Jan-Krzysztof, Firouzja, Alireza

**Player summary**

| player                  | title   |   mean_rating |   total_games |   games_vs_cohort |   games_outside_cohort |
|:------------------------|:--------|--------------:|--------------:|------------------:|-----------------------:|
| Carlsen, Magnus         | GM      |        2900.3 |          2157 |               675 |                   1482 |
| So, Wesley              | GM      |        2788.7 |          1795 |               552 |                   1243 |
| Caruana, Fabiano        | GM      |        2788.3 |          1714 |               437 |                   1277 |
| Vachier-Lagrave, Maxime | GM      |        2804.3 |          1640 |               397 |                   1243 |
| Giri, Anish             | GM      |        2727.6 |          1515 |               334 |                   1181 |
| Praggnanandhaa R        | GM      |        2689.7 |          1314 |               247 |                   1067 |
| Duda, Jan-Krzysztof     | GM      |        2752.5 |          1194 |               307 |                    887 |
| Firouzja, Alireza       | GM      |        2773.7 |          1176 |               399 |                    777 |

**Pairwise matrix**

|                         |   Carlsen, Magnus |   So, Wesley |   Caruana, Fabiano |   Vachier-Lagrave, Maxime |   Giri, Anish |   Praggnanandhaa R |   Duda, Jan-Krzysztof |   Firouzja, Alireza |
|:------------------------|------------------:|-------------:|-------------------:|--------------------------:|--------------:|-------------------:|----------------------:|--------------------:|
| Carlsen, Magnus         |                 0 |          169 |                120 |                        78 |            67 |                 44 |                    90 |                 107 |
| So, Wesley              |               169 |            0 |                 89 |                       105 |            51 |                 30 |                    39 |                  69 |
| Caruana, Fabiano        |               120 |           89 |                  0 |                        59 |            34 |                 34 |                    35 |                  66 |
| Vachier-Lagrave, Maxime |                78 |          105 |                 59 |                         0 |            48 |                 29 |                    30 |                  48 |
| Giri, Anish             |                67 |           51 |                 34 |                        48 |             0 |                 45 |                    47 |                  42 |
| Praggnanandhaa R        |                44 |           30 |                 34 |                        29 |            45 |                  0 |                    32 |                  33 |
| Duda, Jan-Krzysztof     |                90 |           39 |                 35 |                        30 |            47 |                 32 |                     0 |                  34 |
| Firouzja, Alireza       |               107 |           69 |                 66 |                        48 |            42 |                 33 |                    34 |                   0 |

**All 28 pairwise counts, sorted**

| player_A                | player_B                |   games |
|:------------------------|:------------------------|--------:|
| Vachier-Lagrave, Maxime | Praggnanandhaa R        |      29 |
| So, Wesley              | Praggnanandhaa R        |      30 |
| Vachier-Lagrave, Maxime | Duda, Jan-Krzysztof     |      30 |
| Praggnanandhaa R        | Duda, Jan-Krzysztof     |      32 |
| Praggnanandhaa R        | Firouzja, Alireza       |      33 |
| Caruana, Fabiano        | Giri, Anish             |      34 |
| Caruana, Fabiano        | Praggnanandhaa R        |      34 |
| Duda, Jan-Krzysztof     | Firouzja, Alireza       |      34 |
| Caruana, Fabiano        | Duda, Jan-Krzysztof     |      35 |
| So, Wesley              | Duda, Jan-Krzysztof     |      39 |
| Giri, Anish             | Firouzja, Alireza       |      42 |
| Carlsen, Magnus         | Praggnanandhaa R        |      44 |
| Giri, Anish             | Praggnanandhaa R        |      45 |
| Giri, Anish             | Duda, Jan-Krzysztof     |      47 |
| Vachier-Lagrave, Maxime | Giri, Anish             |      48 |
| Vachier-Lagrave, Maxime | Firouzja, Alireza       |      48 |
| So, Wesley              | Giri, Anish             |      51 |
| Caruana, Fabiano        | Vachier-Lagrave, Maxime |      59 |
| Caruana, Fabiano        | Firouzja, Alireza       |      66 |
| Carlsen, Magnus         | Giri, Anish             |      67 |
| So, Wesley              | Firouzja, Alireza       |      69 |
| Carlsen, Magnus         | Vachier-Lagrave, Maxime |      78 |
| So, Wesley              | Caruana, Fabiano        |      89 |
| Carlsen, Magnus         | Duda, Jan-Krzysztof     |      90 |
| So, Wesley              | Vachier-Lagrave, Maxime |     105 |
| Carlsen, Magnus         | Firouzja, Alireza       |     107 |
| Carlsen, Magnus         | Caruana, Fabiano        |     120 |
| Carlsen, Magnus         | So, Wesley              |     169 |

- weakest 5 edges: [('Vachier-Lagrave, Maxime', 'Praggnanandhaa R', 29), ('So, Wesley', 'Praggnanandhaa R', 30), ('Vachier-Lagrave, Maxime', 'Duda, Jan-Krzysztof', 30), ('Praggnanandhaa R', 'Duda, Jan-Krzysztof', 32), ('Praggnanandhaa R', 'Firouzja, Alireza', 33)]

- strongest 5 edges: [('Carlsen, Magnus', 'Duda, Jan-Krzysztof', 90), ('So, Wesley', 'Vachier-Lagrave, Maxime', 105), ('Carlsen, Magnus', 'Firouzja, Alireza', 107), ('Carlsen, Magnus', 'Caruana, Fabiano', 120), ('Carlsen, Magnus', 'So, Wesley', 169)]

- minimum outside-cohort games for any player: **777** (Firouzja, Alireza)

## Candidate #4

- min pairwise edge: **28**
- median pairwise edge: **64.5**
- mean pairwise edge: **69.4**
- total internal games (sum of all edges): **1944**
- players: Carlsen, Magnus, So, Wesley, Aronian, Levon, Caruana, Fabiano, Vachier-Lagrave, Maxime, Giri, Anish, Nepomniachtchi, Ian, Firouzja, Alireza

**Player summary**

| player                  | title   |   mean_rating |   total_games |   games_vs_cohort |   games_outside_cohort |
|:------------------------|:--------|--------------:|--------------:|------------------:|-----------------------:|
| Carlsen, Magnus         | GM      |        2900.3 |          2157 |               758 |                   1399 |
| So, Wesley              | GM      |        2788.7 |          1795 |               598 |                   1197 |
| Aronian, Levon          | GM      |        2765.1 |          1714 |               413 |                   1301 |
| Caruana, Fabiano        | GM      |        2788.3 |          1714 |               482 |                   1232 |
| Vachier-Lagrave, Maxime | GM      |        2804.3 |          1640 |               473 |                   1167 |
| Giri, Anish             | GM      |        2727.6 |          1515 |               320 |                   1195 |
| Nepomniachtchi, Ian     | GM      |        2777.6 |          1440 |               425 |                   1015 |
| Firouzja, Alireza       | GM      |        2773.7 |          1176 |               419 |                    757 |

**Pairwise matrix**

|                         |   Carlsen, Magnus |   So, Wesley |   Aronian, Levon |   Caruana, Fabiano |   Vachier-Lagrave, Maxime |   Giri, Anish |   Nepomniachtchi, Ian |   Firouzja, Alireza |
|:------------------------|------------------:|-------------:|-----------------:|-------------------:|--------------------------:|--------------:|----------------------:|--------------------:|
| Carlsen, Magnus         |                 0 |          169 |               94 |                120 |                        78 |            67 |                   123 |                 107 |
| So, Wesley              |               169 |            0 |               69 |                 89 |                       105 |            51 |                    46 |                  69 |
| Aronian, Levon          |                94 |           69 |                0 |                 71 |                        63 |            28 |                    46 |                  42 |
| Caruana, Fabiano        |               120 |           89 |               71 |                  0 |                        59 |            34 |                    43 |                  66 |
| Vachier-Lagrave, Maxime |                78 |          105 |               63 |                 59 |                         0 |            48 |                    72 |                  48 |
| Giri, Anish             |                67 |           51 |               28 |                 34 |                        48 |             0 |                    50 |                  42 |
| Nepomniachtchi, Ian     |               123 |           46 |               46 |                 43 |                        72 |            50 |                     0 |                  45 |
| Firouzja, Alireza       |               107 |           69 |               42 |                 66 |                        48 |            42 |                    45 |                   0 |

**All 28 pairwise counts, sorted**

| player_A                | player_B                |   games |
|:------------------------|:------------------------|--------:|
| Aronian, Levon          | Giri, Anish             |      28 |
| Caruana, Fabiano        | Giri, Anish             |      34 |
| Aronian, Levon          | Firouzja, Alireza       |      42 |
| Giri, Anish             | Firouzja, Alireza       |      42 |
| Caruana, Fabiano        | Nepomniachtchi, Ian     |      43 |
| Nepomniachtchi, Ian     | Firouzja, Alireza       |      45 |
| So, Wesley              | Nepomniachtchi, Ian     |      46 |
| Aronian, Levon          | Nepomniachtchi, Ian     |      46 |
| Vachier-Lagrave, Maxime | Giri, Anish             |      48 |
| Vachier-Lagrave, Maxime | Firouzja, Alireza       |      48 |
| Giri, Anish             | Nepomniachtchi, Ian     |      50 |
| So, Wesley              | Giri, Anish             |      51 |
| Caruana, Fabiano        | Vachier-Lagrave, Maxime |      59 |
| Aronian, Levon          | Vachier-Lagrave, Maxime |      63 |
| Caruana, Fabiano        | Firouzja, Alireza       |      66 |
| Carlsen, Magnus         | Giri, Anish             |      67 |
| So, Wesley              | Aronian, Levon          |      69 |
| So, Wesley              | Firouzja, Alireza       |      69 |
| Aronian, Levon          | Caruana, Fabiano        |      71 |
| Vachier-Lagrave, Maxime | Nepomniachtchi, Ian     |      72 |
| Carlsen, Magnus         | Vachier-Lagrave, Maxime |      78 |
| So, Wesley              | Caruana, Fabiano        |      89 |
| Carlsen, Magnus         | Aronian, Levon          |      94 |
| So, Wesley              | Vachier-Lagrave, Maxime |     105 |
| Carlsen, Magnus         | Firouzja, Alireza       |     107 |
| Carlsen, Magnus         | Caruana, Fabiano        |     120 |
| Carlsen, Magnus         | Nepomniachtchi, Ian     |     123 |
| Carlsen, Magnus         | So, Wesley              |     169 |

- weakest 5 edges: [('Aronian, Levon', 'Giri, Anish', 28), ('Caruana, Fabiano', 'Giri, Anish', 34), ('Aronian, Levon', 'Firouzja, Alireza', 42), ('Giri, Anish', 'Firouzja, Alireza', 42), ('Caruana, Fabiano', 'Nepomniachtchi, Ian', 43)]

- strongest 5 edges: [('So, Wesley', 'Vachier-Lagrave, Maxime', 105), ('Carlsen, Magnus', 'Firouzja, Alireza', 107), ('Carlsen, Magnus', 'Caruana, Fabiano', 120), ('Carlsen, Magnus', 'Nepomniachtchi, Ian', 123), ('Carlsen, Magnus', 'So, Wesley', 169)]

- minimum outside-cohort games for any player: **757** (Firouzja, Alireza)

## Candidate #5

- min pairwise edge: **28**
- median pairwise edge: **64.5**
- mean pairwise edge: **68.7**
- total internal games (sum of all edges): **1923**
- players: Carlsen, Magnus, So, Wesley, Aronian, Levon, Caruana, Fabiano, Vachier-Lagrave, Maxime, Nepomniachtchi, Ian, Duda, Jan-Krzysztof, Firouzja, Alireza

**Player summary**

| player                  | title   |   mean_rating |   total_games |   games_vs_cohort |   games_outside_cohort |
|:------------------------|:--------|--------------:|--------------:|------------------:|-----------------------:|
| Carlsen, Magnus         | GM      |        2900.3 |          2157 |               781 |                   1376 |
| So, Wesley              | GM      |        2788.7 |          1795 |               586 |                   1209 |
| Aronian, Levon          | GM      |        2765.1 |          1714 |               428 |                   1286 |
| Caruana, Fabiano        | GM      |        2788.3 |          1714 |               483 |                   1231 |
| Vachier-Lagrave, Maxime | GM      |        2804.3 |          1640 |               455 |                   1185 |
| Nepomniachtchi, Ian     | GM      |        2777.6 |          1440 |               403 |                   1037 |
| Duda, Jan-Krzysztof     | GM      |        2752.5 |          1194 |               299 |                    895 |
| Firouzja, Alireza       | GM      |        2773.7 |          1176 |               411 |                    765 |

**Pairwise matrix**

|                         |   Carlsen, Magnus |   So, Wesley |   Aronian, Levon |   Caruana, Fabiano |   Vachier-Lagrave, Maxime |   Nepomniachtchi, Ian |   Duda, Jan-Krzysztof |   Firouzja, Alireza |
|:------------------------|------------------:|-------------:|-----------------:|-------------------:|--------------------------:|----------------------:|----------------------:|--------------------:|
| Carlsen, Magnus         |                 0 |          169 |               94 |                120 |                        78 |                   123 |                    90 |                 107 |
| So, Wesley              |               169 |            0 |               69 |                 89 |                       105 |                    46 |                    39 |                  69 |
| Aronian, Levon          |                94 |           69 |                0 |                 71 |                        63 |                    46 |                    43 |                  42 |
| Caruana, Fabiano        |               120 |           89 |               71 |                  0 |                        59 |                    43 |                    35 |                  66 |
| Vachier-Lagrave, Maxime |                78 |          105 |               63 |                 59 |                         0 |                    72 |                    30 |                  48 |
| Nepomniachtchi, Ian     |               123 |           46 |               46 |                 43 |                        72 |                     0 |                    28 |                  45 |
| Duda, Jan-Krzysztof     |                90 |           39 |               43 |                 35 |                        30 |                    28 |                     0 |                  34 |
| Firouzja, Alireza       |               107 |           69 |               42 |                 66 |                        48 |                    45 |                    34 |                   0 |

**All 28 pairwise counts, sorted**

| player_A                | player_B                |   games |
|:------------------------|:------------------------|--------:|
| Nepomniachtchi, Ian     | Duda, Jan-Krzysztof     |      28 |
| Vachier-Lagrave, Maxime | Duda, Jan-Krzysztof     |      30 |
| Duda, Jan-Krzysztof     | Firouzja, Alireza       |      34 |
| Caruana, Fabiano        | Duda, Jan-Krzysztof     |      35 |
| So, Wesley              | Duda, Jan-Krzysztof     |      39 |
| Aronian, Levon          | Firouzja, Alireza       |      42 |
| Aronian, Levon          | Duda, Jan-Krzysztof     |      43 |
| Caruana, Fabiano        | Nepomniachtchi, Ian     |      43 |
| Nepomniachtchi, Ian     | Firouzja, Alireza       |      45 |
| So, Wesley              | Nepomniachtchi, Ian     |      46 |
| Aronian, Levon          | Nepomniachtchi, Ian     |      46 |
| Vachier-Lagrave, Maxime | Firouzja, Alireza       |      48 |
| Caruana, Fabiano        | Vachier-Lagrave, Maxime |      59 |
| Aronian, Levon          | Vachier-Lagrave, Maxime |      63 |
| Caruana, Fabiano        | Firouzja, Alireza       |      66 |
| So, Wesley              | Aronian, Levon          |      69 |
| So, Wesley              | Firouzja, Alireza       |      69 |
| Aronian, Levon          | Caruana, Fabiano        |      71 |
| Vachier-Lagrave, Maxime | Nepomniachtchi, Ian     |      72 |
| Carlsen, Magnus         | Vachier-Lagrave, Maxime |      78 |
| So, Wesley              | Caruana, Fabiano        |      89 |
| Carlsen, Magnus         | Duda, Jan-Krzysztof     |      90 |
| Carlsen, Magnus         | Aronian, Levon          |      94 |
| So, Wesley              | Vachier-Lagrave, Maxime |     105 |
| Carlsen, Magnus         | Firouzja, Alireza       |     107 |
| Carlsen, Magnus         | Caruana, Fabiano        |     120 |
| Carlsen, Magnus         | Nepomniachtchi, Ian     |     123 |
| Carlsen, Magnus         | So, Wesley              |     169 |

- weakest 5 edges: [('Nepomniachtchi, Ian', 'Duda, Jan-Krzysztof', 28), ('Vachier-Lagrave, Maxime', 'Duda, Jan-Krzysztof', 30), ('Duda, Jan-Krzysztof', 'Firouzja, Alireza', 34), ('Caruana, Fabiano', 'Duda, Jan-Krzysztof', 35), ('So, Wesley', 'Duda, Jan-Krzysztof', 39)]

- strongest 5 edges: [('So, Wesley', 'Vachier-Lagrave, Maxime', 105), ('Carlsen, Magnus', 'Firouzja, Alireza', 107), ('Carlsen, Magnus', 'Caruana, Fabiano', 120), ('Carlsen, Magnus', 'Nepomniachtchi, Ian', 123), ('Carlsen, Magnus', 'So, Wesley', 169)]

- minimum outside-cohort games for any player: **765** (Firouzja, Alireza)
