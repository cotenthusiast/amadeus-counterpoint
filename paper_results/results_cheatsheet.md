# Amadeus Results Cheatsheet

Primary sealed sources: /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method1_results.json; /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method2_results.json; /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method3_final_complete_results.json. Historical tournament outputs are excluded.

## Individual validation

- M1: correct representation rank 1 for 8/8; mean generic−correct NLL = 0.006802.
- M2: correct representation rank 1 for 8/8; mean candidate-matched generic−correct NLL = 0.009278; raw top-5 coverage range = 0.9416–0.9564.

## Main interaction results

### M1
- WDL-TV: GG 0.164171; AG 0.157753; GB 0.156141; AB 0.150148; AB−GG -0.014024.
- Opening-family TV: GG 0.554433; AG 0.547406; GB 0.544765; AB 0.536973; AB−GG -0.017461.

### M2
- WDL-TV: GG 0.163166; AG 0.162582; GB 0.162589; AB 0.163408; AB−GG 0.000241.
- Opening-family TV: GG 0.554711; AG 0.501370; GB 0.488343; AB 0.437202; AB−GG -0.117509.

### M3
- WDL-TV: GG 0.163073; AG 0.155709; GB 0.153723; AB 0.152036; AB−GG -0.011037.
- Opening-family TV: GG 0.554395; AG 0.500140; GB 0.488235; AB 0.444428; AB−GG -0.109967.

## Dyad-level

- M1 WDL-TV: 22/28 improve; median AB−GG -0.010900; range [-0.088700, 0.046300].
- M1 Opening-family TV: 22/28 improve; median AB−GG -0.018883; range [-0.058385, 0.027417].
- M2 WDL-TV: 13/28 improve; median AB−GG 0.001650; range [-0.018300, 0.015700].
- M2 Opening-family TV: 27/28 improve; median AB−GG -0.105987; range [-0.364176, 0.004373].
- M3 WDL-TV: 19/28 improve; median AB−GG -0.010500; range [-0.064400, 0.032800].
- M3 Opening-family TV: 27/28 improve; median AB−GG -0.090846; range [-0.347976, 0.006572].

## Counts

- 8 players, 28 dyads, 224 cells/method, 5,000 games/orientation, 1,120,000 games/method.
- 10,000 whole-real-game bootstrap replicates per dyad/metric.
- Sealed cache: 1,609 deduplicated real games.
