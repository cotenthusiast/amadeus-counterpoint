# Amadeus Results Audit

Scope: latest finalized M1/M2/M3 sealed pairwise results and finalized nonsealed individual-validation artifacts. The historical Croatia GCT / Sinquefield tournament experiment is deliberately excluded.

## Authority and source policy

The complete final sealed JSONs are authoritative for pairwise results. Small summary mirrors under paper_results/source are used only to build these outputs; production games and checkpoints remain on Kelvin2.

# 1. Individual Behavioural Validation

The authoritative individual-validation artifact is the nonsealed 8-way identity diagnostic. M1 uses legal-masked full-action policy NLL; M2 uses candidate-matched NLL on raw top-5 candidates with the observed move appended when absent. The NLL scales are not directly comparable.

| Method | Positions / players | Correct representation rank 1 | Mean generic-minus-correct NLL | Mean wrong-minus-correct NLL |
|---|---:|---:|---:|---:|
| M1 | 47,136 / 8 | 8/8 | 0.006802 | 0.004365 |
| M2 | 46,950 / 8 | 8/8 | 0.009278 | 0.020449 |

| Player | Positions | M1 ΔNLL | M1 top-1 | M2 ΔNLL | M2 raw top-5 | M2 deployable top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Magnus Carlsen | 5,985 | 0.005616 | 0.5888 | 0.006019 | 0.9514 | 0.5891 |
| Wesley So | 6,294 | 0.004831 | 0.6153 | 0.006440 | 0.9512 | 0.6220 |
| Levon Aronian | 6,432 | 0.003872 | 0.5998 | 0.009803 | 0.9470 | 0.5890 |
| Fabiano Caruana | 7,236 | 0.005446 | 0.6017 | 0.004084 | 0.9450 | 0.6058 |
| Maxime Vachier-Lagrave | 6,384 | 0.008366 | 0.6203 | 0.019544 | 0.9553 | 0.6360 |
| Hikaru Nakamura | 4,950 | 0.013230 | 0.5715 | 0.017752 | 0.9416 | 0.5731 |
| Ian Nepomniachtchi | 5,607 | 0.007295 | 0.6167 | 0.010069 | 0.9564 | 0.6124 |
| Alireza Firouzja | 4,248 | 0.005763 | 0.6156 | 0.000510 | 0.9506 | 0.6083 |

M1 personalized generation was closer to held-out real individual-player distributions than the generic reference for 8/8 players on both first-move TV and opening-family TV. This is a nonsealed individual-fidelity diagnostic using a self-defined preliminary opening taxonomy; it is not the primary sealed interaction metric.

Sources: /mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic.json and /mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/nonsealed_real_comparison.json.

# 2. Sealed Pairwise Interaction Fidelity

Total-variation distance is lower-is-better. Values use 50/50 orientation averaging and equal weighting across 28 dyads.

| Method | Metric | GG | AG | GB | AB | AB − GG |
|---|---|---:|---:|---:|---:|---:|
| M1 | WDL-TV | 0.164171 | 0.157753 | 0.156141 | 0.150148 | -0.014024 |
| M1 | Opening-family TV | 0.554433 | 0.547406 | 0.544765 | 0.536973 | -0.017461 |
| M2 | WDL-TV | 0.163166 | 0.162582 | 0.162589 | 0.163408 | 0.000241 |
| M2 | Opening-family TV | 0.554711 | 0.501370 | 0.488343 | 0.437202 | -0.117509 |
| M3 | WDL-TV | 0.163073 | 0.155709 | 0.153723 | 0.152036 | -0.011037 |
| M3 | Opening-family TV | 0.554395 | 0.500140 | 0.488235 | 0.444428 | -0.109967 |

The evaluator saved 10,000 whole-real-game bootstrap replicates per dyad and metric. Per-dyad intervals are present in the result JSONs; aggregate CI fields are not present and are not invented here.

Sources: /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method1_results.json, /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method2_results.json, /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method3_final_complete_results.json, and /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/PRE_EVAL_MANIFEST.md.

# 3. Dyad-Level Variation

The following tables report AB − GG; negative values indicate lower AB TV distance.

## M1
### WDL-TV

22/28 dyads (78.6%) improved; mean -0.014024, median -0.010900, range [-0.088700, 0.046300].

| Dyad | GG | AB | AB − GG |
|---|---:|---:|---:|
| Carlsen--So | 0.197725 | 0.191325 | -0.006400 |
| Carlsen--Aronian | 0.333733 | 0.320533 | -0.013200 |
| Carlsen--Caruana | 0.334339 | 0.315239 | -0.019100 |
| Carlsen--Vachier-Lagrave | 0.207988 | 0.183488 | -0.024500 |
| Carlsen--Nakamura | 0.127803 | 0.121494 | -0.006309 |
| Carlsen--Nepomniachtchi | 0.203511 | 0.194911 | -0.008600 |
| Carlsen--Firouzja | 0.264972 | 0.259772 | -0.005200 |
| So--Aronian | 0.101252 | 0.108552 | 0.007300 |
| So--Caruana | 0.087158 | 0.071193 | -0.015965 |
| So--Vachier-Lagrave | 0.026930 | 0.034230 | 0.007300 |
| So--Nakamura | 0.098368 | 0.144668 | 0.046300 |
| So--Nepomniachtchi | 0.070819 | 0.070519 | -0.000300 |
| So--Firouzja | 0.133924 | 0.129624 | -0.004300 |
| Aronian--Caruana | 0.035633 | 0.055533 | 0.019900 |
| Aronian--Vachier-Lagrave | 0.140033 | 0.113218 | -0.026815 |
| Aronian--Nakamura | 0.243984 | 0.209237 | -0.034747 |
| Aronian--Nepomniachtchi | 0.193300 | 0.166700 | -0.026600 |
| Aronian--Firouzja | 0.158233 | 0.156900 | -0.001333 |
| Caruana--Vachier-Lagrave | 0.087652 | 0.070952 | -0.016700 |
| Caruana--Nakamura | 0.200764 | 0.127800 | -0.072964 |
| Caruana--Nepomniachtchi | 0.128287 | 0.153187 | 0.024900 |
| Caruana--Firouzja | 0.226652 | 0.187852 | -0.038800 |
| Vachier-Lagrave--Nakamura | 0.274103 | 0.232203 | -0.041900 |
| Vachier-Lagrave--Nepomniachtchi | 0.164873 | 0.149143 | -0.015730 |
| Vachier-Lagrave--Firouzja | 0.080471 | 0.077771 | -0.002700 |
| Nakamura--Nepomniachtchi | 0.093765 | 0.097265 | 0.003500 |
| Nakamura--Firouzja | 0.200925 | 0.112225 | -0.088700 |
| Nepomniachtchi--Firouzja | 0.179600 | 0.148600 | -0.031000 |

Largest improvements: Nakamura--Firouzja (-0.088700), Caruana--Nakamura (-0.072964), Vachier-Lagrave--Nakamura (-0.041900).
Largest regressions: So--Nakamura (0.046300), Caruana--Nepomniachtchi (0.024900), Aronian--Caruana (0.019900).

### Opening-family TV

22/28 dyads (78.6%) improved; mean -0.017461, median -0.018883, range [-0.058385, 0.027417].

| Dyad | GG | AB | AB − GG |
|---|---:|---:|---:|
| Carlsen--So | 0.395150 | 0.357266 | -0.037884 |
| Carlsen--Aronian | 0.560852 | 0.531486 | -0.029367 |
| Carlsen--Caruana | 0.377791 | 0.356791 | -0.021000 |
| Carlsen--Vachier-Lagrave | 0.605862 | 0.547477 | -0.058385 |
| Carlsen--Nakamura | 0.412735 | 0.398609 | -0.014126 |
| Carlsen--Nepomniachtchi | 0.540821 | 0.507221 | -0.033600 |
| Carlsen--Firouzja | 0.369683 | 0.370711 | 0.001028 |
| So--Aronian | 0.521907 | 0.520507 | -0.001400 |
| So--Caruana | 0.497227 | 0.484694 | -0.012533 |
| So--Vachier-Lagrave | 0.534771 | 0.511186 | -0.023586 |
| So--Nakamura | 0.537680 | 0.549823 | 0.012143 |
| So--Nepomniachtchi | 0.617952 | 0.599986 | -0.017967 |
| So--Firouzja | 0.538990 | 0.537390 | -0.001600 |
| Aronian--Caruana | 0.545633 | 0.531733 | -0.013900 |
| Aronian--Vachier-Lagrave | 0.702420 | 0.681020 | -0.021400 |
| Aronian--Nakamura | 0.548028 | 0.550399 | 0.002371 |
| Aronian--Nepomniachtchi | 0.579100 | 0.559300 | -0.019800 |
| Aronian--Firouzja | 0.614983 | 0.574083 | -0.040900 |
| Caruana--Vachier-Lagrave | 0.622683 | 0.578083 | -0.044600 |
| Caruana--Nakamura | 0.427982 | 0.422779 | -0.005203 |
| Caruana--Nepomniachtchi | 0.602033 | 0.609133 | 0.007100 |
| Caruana--Firouzja | 0.480790 | 0.447281 | -0.033510 |
| Vachier-Lagrave--Nakamura | 0.742033 | 0.690433 | -0.051600 |
| Vachier-Lagrave--Nepomniachtchi | 0.658420 | 0.637220 | -0.021200 |
| Vachier-Lagrave--Firouzja | 0.672700 | 0.629400 | -0.043300 |
| Nakamura--Nepomniachtchi | 0.618150 | 0.615150 | -0.003000 |
| Nakamura--Firouzja | 0.541789 | 0.569207 | 0.027417 |
| Nepomniachtchi--Firouzja | 0.655967 | 0.666867 | 0.010900 |

Largest improvements: Carlsen--Vachier-Lagrave (-0.058385), Vachier-Lagrave--Nakamura (-0.051600), Caruana--Vachier-Lagrave (-0.044600).
Largest regressions: Nakamura--Firouzja (0.027417), So--Nakamura (0.012143), Nepomniachtchi--Firouzja (0.010900).

## M2
### WDL-TV

13/28 dyads (46.4%) improved; mean 0.000241, median 0.001650, range [-0.018300, 0.015700].

| Dyad | GG | AB | AB − GG |
|---|---:|---:|---:|
| Carlsen--So | 0.193525 | 0.199425 | 0.005900 |
| Carlsen--Aronian | 0.330733 | 0.323433 | -0.007300 |
| Carlsen--Caruana | 0.324039 | 0.333139 | 0.009100 |
| Carlsen--Vachier-Lagrave | 0.205288 | 0.199488 | -0.005800 |
| Carlsen--Nakamura | 0.125003 | 0.130503 | 0.005500 |
| Carlsen--Nepomniachtchi | 0.202211 | 0.206111 | 0.003900 |
| Carlsen--Firouzja | 0.276572 | 0.268172 | -0.008400 |
| So--Aronian | 0.094052 | 0.097252 | 0.003200 |
| So--Caruana | 0.086758 | 0.094758 | 0.008000 |
| So--Vachier-Lagrave | 0.031430 | 0.046389 | 0.014959 |
| So--Nakamura | 0.112568 | 0.094268 | -0.018300 |
| So--Nepomniachtchi | 0.072519 | 0.086019 | 0.013500 |
| So--Firouzja | 0.140824 | 0.140924 | 0.000100 |
| Aronian--Caruana | 0.031233 | 0.038333 | 0.007100 |
| Aronian--Vachier-Lagrave | 0.136133 | 0.129933 | -0.006200 |
| Aronian--Nakamura | 0.235037 | 0.239637 | 0.004600 |
| Aronian--Nepomniachtchi | 0.189100 | 0.196100 | 0.007000 |
| Aronian--Firouzja | 0.166133 | 0.159433 | -0.006700 |
| Caruana--Vachier-Lagrave | 0.088052 | 0.077552 | -0.010500 |
| Caruana--Nakamura | 0.195264 | 0.207664 | 0.012400 |
| Caruana--Nepomniachtchi | 0.116187 | 0.131887 | 0.015700 |
| Caruana--Firouzja | 0.219552 | 0.212452 | -0.007100 |
| Vachier-Lagrave--Nakamura | 0.271103 | 0.286703 | 0.015600 |
| Vachier-Lagrave--Nepomniachtchi | 0.169473 | 0.154973 | -0.014500 |
| Vachier-Lagrave--Firouzja | 0.078071 | 0.075171 | -0.002900 |
| Nakamura--Nepomniachtchi | 0.099265 | 0.091365 | -0.007900 |
| Nakamura--Firouzja | 0.195425 | 0.187025 | -0.008400 |
| Nepomniachtchi--Firouzja | 0.183100 | 0.167300 | -0.015800 |

Largest improvements: So--Nakamura (-0.018300), Nepomniachtchi--Firouzja (-0.015800), Vachier-Lagrave--Nepomniachtchi (-0.014500).
Largest regressions: Caruana--Nepomniachtchi (0.015700), Vachier-Lagrave--Nakamura (0.015600), So--Vachier-Lagrave (0.014959).

### Opening-family TV

27/28 dyads (96.4%) improved; mean -0.117509, median -0.105987, range [-0.364176, 0.004373].

| Dyad | GG | AB | AB − GG |
|---|---:|---:|---:|
| Carlsen--So | 0.395639 | 0.400012 | 0.004373 |
| Carlsen--Aronian | 0.563352 | 0.454943 | -0.108410 |
| Carlsen--Caruana | 0.383491 | 0.323809 | -0.059683 |
| Carlsen--Vachier-Lagrave | 0.594662 | 0.395592 | -0.199070 |
| Carlsen--Nakamura | 0.407135 | 0.322284 | -0.084851 |
| Carlsen--Nepomniachtchi | 0.547421 | 0.399700 | -0.147721 |
| Carlsen--Firouzja | 0.366583 | 0.308833 | -0.057750 |
| So--Aronian | 0.522607 | 0.460363 | -0.062244 |
| So--Caruana | 0.496799 | 0.370083 | -0.126716 |
| So--Vachier-Lagrave | 0.545657 | 0.366689 | -0.178968 |
| So--Nakamura | 0.536177 | 0.422437 | -0.113740 |
| So--Nepomniachtchi | 0.624086 | 0.447452 | -0.176633 |
| So--Firouzja | 0.536390 | 0.405690 | -0.130700 |
| Aronian--Caruana | 0.535533 | 0.467367 | -0.068167 |
| Aronian--Vachier-Lagrave | 0.704320 | 0.467062 | -0.237258 |
| Aronian--Nakamura | 0.551228 | 0.447664 | -0.103564 |
| Aronian--Nepomniachtchi | 0.582100 | 0.510500 | -0.071600 |
| Aronian--Firouzja | 0.613683 | 0.552167 | -0.061517 |
| Caruana--Vachier-Lagrave | 0.618883 | 0.460526 | -0.158357 |
| Caruana--Nakamura | 0.443136 | 0.370233 | -0.072903 |
| Caruana--Nepomniachtchi | 0.607667 | 0.569733 | -0.037933 |
| Caruana--Firouzja | 0.476590 | 0.447538 | -0.029052 |
| Vachier-Lagrave--Nakamura | 0.747333 | 0.610272 | -0.137062 |
| Vachier-Lagrave--Nepomniachtchi | 0.658920 | 0.421262 | -0.237658 |
| Vachier-Lagrave--Firouzja | 0.666300 | 0.302124 | -0.364176 |
| Nakamura--Nepomniachtchi | 0.618750 | 0.458265 | -0.160485 |
| Nakamura--Firouzja | 0.533189 | 0.458489 | -0.074700 |
| Nepomniachtchi--Firouzja | 0.654267 | 0.620567 | -0.033700 |

Largest improvements: Vachier-Lagrave--Firouzja (-0.364176), Vachier-Lagrave--Nepomniachtchi (-0.237658), Aronian--Vachier-Lagrave (-0.237258).
Largest regressions: Carlsen--So (0.004373), Caruana--Firouzja (-0.029052), Nepomniachtchi--Firouzja (-0.033700).

## M3
### WDL-TV

19/28 dyads (67.9%) improved; mean -0.011037, median -0.010500, range [-0.064400, 0.032800].

| Dyad | GG | AB | AB − GG |
|---|---:|---:|---:|
| Carlsen--So | 0.196225 | 0.195125 | -0.001100 |
| Carlsen--Aronian | 0.319533 | 0.334033 | 0.014500 |
| Carlsen--Caruana | 0.336139 | 0.308539 | -0.027600 |
| Carlsen--Vachier-Lagrave | 0.199788 | 0.181888 | -0.017900 |
| Carlsen--Nakamura | 0.123603 | 0.122523 | -0.001081 |
| Carlsen--Nepomniachtchi | 0.204211 | 0.195911 | -0.008300 |
| Carlsen--Firouzja | 0.273194 | 0.251272 | -0.021922 |
| So--Aronian | 0.092752 | 0.103452 | 0.010700 |
| So--Caruana | 0.085458 | 0.072758 | -0.012700 |
| So--Vachier-Lagrave | 0.025930 | 0.040989 | 0.015059 |
| So--Nakamura | 0.113968 | 0.131668 | 0.017700 |
| So--Nepomniachtchi | 0.079219 | 0.096019 | 0.016800 |
| So--Firouzja | 0.136924 | 0.135324 | -0.001600 |
| Aronian--Caruana | 0.029233 | 0.053033 | 0.023800 |
| Aronian--Vachier-Lagrave | 0.143504 | 0.097188 | -0.046315 |
| Aronian--Nakamura | 0.239137 | 0.213037 | -0.026100 |
| Aronian--Nepomniachtchi | 0.198600 | 0.169400 | -0.029200 |
| Aronian--Firouzja | 0.155533 | 0.169800 | 0.014267 |
| Caruana--Vachier-Lagrave | 0.090152 | 0.068052 | -0.022100 |
| Caruana--Nakamura | 0.185464 | 0.133764 | -0.051700 |
| Caruana--Nepomniachtchi | 0.125787 | 0.158587 | 0.032800 |
| Caruana--Firouzja | 0.218352 | 0.178952 | -0.039400 |
| Vachier-Lagrave--Nakamura | 0.277503 | 0.249803 | -0.027700 |
| Vachier-Lagrave--Nepomniachtchi | 0.165873 | 0.144043 | -0.021830 |
| Vachier-Lagrave--Firouzja | 0.079371 | 0.087471 | 0.008100 |
| Nakamura--Nepomniachtchi | 0.100765 | 0.097965 | -0.002800 |
| Nakamura--Firouzja | 0.189025 | 0.124625 | -0.064400 |
| Nepomniachtchi--Firouzja | 0.180800 | 0.141800 | -0.039000 |

Largest improvements: Nakamura--Firouzja (-0.064400), Caruana--Nakamura (-0.051700), Aronian--Vachier-Lagrave (-0.046315).
Largest regressions: Caruana--Nepomniachtchi (0.032800), Aronian--Caruana (0.023800), So--Nakamura (0.017700).

### Opening-family TV

27/28 dyads (96.4%) improved; mean -0.109967, median -0.090846, range [-0.347976, 0.006572].

| Dyad | GG | AB | AB − GG |
|---|---:|---:|---:|
| Carlsen--So | 0.390171 | 0.382765 | -0.007406 |
| Carlsen--Aronian | 0.563152 | 0.455976 | -0.107176 |
| Carlsen--Caruana | 0.384291 | 0.309478 | -0.074813 |
| Carlsen--Vachier-Lagrave | 0.600057 | 0.399487 | -0.200570 |
| Carlsen--Nakamura | 0.408072 | 0.341396 | -0.066676 |
| Carlsen--Nepomniachtchi | 0.538221 | 0.413937 | -0.124284 |
| Carlsen--Firouzja | 0.353883 | 0.360456 | 0.006572 |
| So--Aronian | 0.519707 | 0.455763 | -0.063944 |
| So--Caruana | 0.510627 | 0.343283 | -0.167344 |
| So--Vachier-Lagrave | 0.537386 | 0.370789 | -0.166597 |
| So--Nakamura | 0.535980 | 0.445688 | -0.090291 |
| So--Nepomniachtchi | 0.620052 | 0.467152 | -0.152900 |
| So--Firouzja | 0.536390 | 0.438852 | -0.097538 |
| Aronian--Caruana | 0.530533 | 0.468567 | -0.061967 |
| Aronian--Vachier-Lagrave | 0.709820 | 0.488862 | -0.220958 |
| Aronian--Nakamura | 0.550928 | 0.459527 | -0.091401 |
| Aronian--Nepomniachtchi | 0.580600 | 0.508700 | -0.071900 |
| Aronian--Firouzja | 0.609383 | 0.529200 | -0.080183 |
| Caruana--Vachier-Lagrave | 0.615083 | 0.438026 | -0.177057 |
| Caruana--Nakamura | 0.436385 | 0.367876 | -0.068509 |
| Caruana--Nepomniachtchi | 0.596233 | 0.569210 | -0.027023 |
| Caruana--Firouzja | 0.494290 | 0.418714 | -0.075576 |
| Vachier-Lagrave--Nakamura | 0.741133 | 0.588372 | -0.152762 |
| Vachier-Lagrave--Nepomniachtchi | 0.657320 | 0.440962 | -0.216358 |
| Vachier-Lagrave--Firouzja | 0.675400 | 0.327424 | -0.347976 |
| Nakamura--Nepomniachtchi | 0.621950 | 0.489565 | -0.132385 |
| Nakamura--Firouzja | 0.546255 | 0.519689 | -0.026566 |
| Nepomniachtchi--Firouzja | 0.659767 | 0.644267 | -0.015500 |

Largest improvements: Vachier-Lagrave--Firouzja (-0.347976), Aronian--Vachier-Lagrave (-0.220958), Vachier-Lagrave--Nepomniachtchi (-0.216358).
Largest regressions: Carlsen--Firouzja (0.006572), Carlsen--So (-0.007406), Nepomniachtchi--Firouzja (-0.015500).

Source per-dyad data: /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method1_results.json, /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method2_results.json, /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method3_final_complete_results.json.

# 4. Exploratory Hybrid Model (M3)

M3 is a post-hoc exploratory hybrid combining the already-trained M1 player vectors with the already-trained M2 candidate reranker. It introduced no new training or scientific hyperparameters.

| Metric | M3 GG | M3 AG | M3 GB | M3 AB | AB − GG | M1 AB | M2 AB |
|---|---:|---:|---:|---:|---:|---:|---:|
| WDL-TV | 0.163073 | 0.155709 | 0.153723 | 0.152036 | -0.011037 | 0.150148 | 0.163408 |
| Opening-family TV | 0.554395 | 0.500140 | 0.488235 | 0.444428 | -0.109967 | 0.536973 | 0.437202 |

M3 has its own generated GG baseline. The provisional file with missing 5__6 GB is superseded by the complete result file.

Sources: /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method3_final_complete_results.json and /mnt/scratch2/users/40482774/paper_artifacts/m3_provenance_2026-09-17.json.

# 5. Dataset / Evaluation Counts

- 8 target players and 28 unordered dyads.
- 4 conditions × 2 orientations × 5,000 games = 224 cells and 1,120,000 games per method.
- M1/M2 production: 224/224 cells each and 2,240,000 games combined. M3 final evaluation reports 28 evaluated dyads and no skipped cells.
- Sealed evaluator cache: 1,609 deduplicated real games; exact pre-dedup counts are not recoverable from the cache alone.
- Bootstrap: 10,000 whole-real-game replicates per dyad/metric.
- Censoring: censored synthetic games are excluded from denominators and never silently counted as draws.

Sources: /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/PRE_EVAL_MANIFEST.md and /mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/production_manifest.json.

# 6. Potential Conflicts / Things to Verify Manually

1. M3 provisional versus complete: use only method3_final_complete_results.json.
2. M1/M2 final evaluation commit 980405aa..., M3 generation provenance 51ff259..., and M3 final evaluator commit 5b56fb... should be preserved as separate provenance fields.
3. M1 nonsealed marginal opening TV uses a self-defined taxonomy; do not merge it with final sealed opening-family TV.
4. Aggregate bootstrap confidence intervals are not saved in the final JSONs; only per-dyad intervals are available.
5. Historical Croatia/Sinquefield tournament artifacts are excluded from all paper outputs.
