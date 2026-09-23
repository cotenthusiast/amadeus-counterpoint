# Individual Behavioral Validation — Read-Only Audit

Audit date: 2026-09-17. Scope is validation performed on nonsealed target-vs-outsider data before the final sealed compositional evaluation. This is an artifact reconstruction, not paper prose.

`VERIFIED FACT` means directly present in repository code or a saved Kelvin2 artifact. `INFERENCE` is a classification made from artifact role. `UNKNOWN` means the inspected sources do not establish it.

## Authoritative artifact inventory

1. `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/MANIFEST.md`
2. `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic.json`
3. `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic_9877842.out`
4. `/mnt/scratch2/users/40482774/lichess_data/method1_player{0..7}_987124{2..9}.out`
5. `/mnt/scratch2/users/40482774/lichess_data/method2_joint_9871250.out`
6. `/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/M1_COMPLETE_PRELIMINARY_REPORT.md`
7. `/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/nonsealed_real_comparison.json`
8. `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/verification_pass_provisional/section_d_state_dependent_differentiation.json`
9. `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/final_sampler_freeze/opening_distribution_inspection.md`
10. `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/final_sampler_freeze/player_differentiation_before_after.json`
11. `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/topk_strength_guardrail_diagnostic/TOPK_GUARDRAIL_REPORT.md`
12. `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/verification_pass_provisional/VERIFICATION_REPORT.md`

The identity archive was created on 2026-09-11 under commit `05c7e706...`; the final sealed pre-evaluation freeze is dated 2026-09-16. The identity artifact therefore predates the final sealed behavioral evaluation. Sources: identity `MANIFEST.md`; `/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/PRE_EVAL_MANIFEST.md`.

## M1 individual validation

### Protocol and data

- `VERIFIED FACT` — The input is `/mnt/scratch2/users/40482774/lichess_data/style_records.json`, containing 6,922 one-target records. The file is produced by `data.broadcast_ingest.iter_target_game_records`; target-target games are discarded upstream. Source: `evaluation.identity_diagnostic` module docstring; identity `MANIFEST.md`.
- `VERIFIED FACT` — M1 validation is a game-level 80/20 split with `split_seed=0`, followed by phase balancing with `balance_seed=0`. M1 filters to one `player_id` before calling `build_style_datasets()`. Sources: `src/amadeus_counterpoint/data/style_dataset.py`, `split_games_by_player()`, `build_style_datasets()`; `src/amadeus_counterpoint/evaluation/identity_diagnostic.py`, `build_method1_val_dataset()`.
- `VERIFIED FACT` — The held-out split is distinct from M1 training at the game level. The saved M1 logs report the eligible/train/validation game counts and pre-balancing decision counts below. The identity JSON reports the final post-phase-balancing validation-position count.

| player | eligible games | train games | held-out games | held-out decisions before balancing | identity-diagnostic positions after balancing |
|---|---:|---:|---:|---:|---:|
| Magnus Carlsen | 894 | 715 | 179 | 8,339 | 5,985 |
| Wesley So | 925 | 740 | 185 | 8,607 | 6,294 |
| Levon Aronian | 915 | 732 | 183 | 8,696 | 6,432 |
| Fabiano Caruana | 1,070 | 856 | 214 | 10,655 | 7,236 |
| Maxime Vachier-Lagrave | 911 | 729 | 182 | 8,732 | 6,384 |
| Hikaru Nakamura | 747 | 598 | 149 | 7,568 | 4,950 |
| Ian Nepomniachtchi | 825 | 660 | 165 | 7,823 | 5,607 |
| Alireza Firouzja | 635 | 508 | 127 | 5,633 | 4,248 |

Source for split counts: the eight `/mnt/scratch2/users/40482774/lichess_data/method1_player*_98712*.out` logs. Source for final position counts and all NLL values: `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic.json`.

### Metric and comparison mechanics

- `VERIFIED FACT` — M1 uses legal-move-masked cross-entropy/NLL over the full 4,352-action policy space. No candidate restriction and no value loss are used in `evaluation.identity_diagnostic._full_action_metrics()` and `training.method1_trainer.Method1Trainer._policy_loss()`.
- `VERIFIED FACT` — The generic baseline is the frozen Broadcast base evaluated with each position's recorded mover and opponent Elos. A representation column is evaluated with one existing M1 wrapper; `PersonalizedChessformer.forward()` ignores the supplied mover Elo and substitutes that wrapper's learned `z_player`, while the opponent Elo remains ordinary base conditioning. Sources: `src/amadeus_counterpoint/evaluation/identity_diagnostic.py`, `evaluate_generic()` and `evaluate_method1_representation()`; `src/amadeus_counterpoint/models/personalized_chessformer.py`, `forward()`.
- `VERIFIED FACT` — For each true player, all eight M1 wrappers are scored on that player's held-out positions. NLL is averaged over positions. The correct representation rank is the ascending rank of mean NLL; rank 1 is lowest NLL. `generic-minus-correct` and `mean-wrong-minus-correct` are computed from mean NLLs. Source: `src/amadeus_counterpoint/evaluation/identity_diagnostic.py`, `summarize_row()`.

### Correct vs. generic and correct vs. wrong-player results

Values below are from the saved JSON; shown to six decimal places. `mean wrong` is the unweighted mean of the seven wrong-player representations. Accuracy is full-action legal-masked top-1 accuracy for the correct representation.

| player | generic NLL | correct NLL | mean wrong NLL | generic − correct | wrong − correct | correct top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Magnus Carlsen | 1.153328 | 1.147712 | 1.150685 | 0.005616 | 0.002973 | 0.588805 |
| Wesley So | 1.076210 | 1.071379 | 1.074831 | 0.004831 | 0.003452 | 0.615348 |
| Levon Aronian | 1.135083 | 1.131212 | 1.136323 | 0.003872 | 0.005111 | 0.599813 |
| Fabiano Caruana | 1.139759 | 1.134313 | 1.139032 | 0.005446 | 0.004720 | 0.601714 |
| Maxime Vachier-Lagrave | 1.085394 | 1.077028 | 1.081512 | 0.008366 | 0.004484 | 0.620301 |
| Hikaru Nakamura | 1.210990 | 1.197760 | 1.203940 | 0.013230 | 0.006180 | 0.571515 |
| Ian Nepomniachtchi | 1.106990 | 1.099695 | 1.103328 | 0.007295 | 0.003633 | 0.616729 |
| Alireza Firouzja | 1.090038 | 1.084275 | 1.088640 | 0.005763 | 0.004365 | 0.615584 |

`VERIFIED FACT` — Overall mean generic-minus-correct NLL is `0.006802377438040991`; overall mean wrong-minus-correct NLL is `0.004364752412807621`; correct representation is rank 1 for 8/8 players. Source: `.../personalization_identity_diagnostic.json`, `method1.overall`.

### Full M1 8×8 NLL matrix

Rows are true-player held-out data; columns are representation used. Lower is better. Values are rounded for display; the JSON contains full precision and also contains the corresponding 8×8 accuracy matrix under `accuracy_by_representation`.

| true \\ representation | Magnus | Wesley | Levon | Fabiano | MVL | Hikaru | Ian | Alireza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Magnus | 1.1477 | 1.1539 | 1.1511 | 1.1490 | 1.1499 | 1.1516 | 1.1493 | 1.1501 |
| Wesley | 1.0742 | 1.0714 | 1.0738 | 1.0737 | 1.0730 | 1.0799 | 1.0715 | 1.0777 |
| Levon | 1.1371 | 1.1345 | 1.1312 | 1.1366 | 1.1321 | 1.1408 | 1.1352 | 1.1380 |
| Fabiano | 1.1377 | 1.1400 | 1.1393 | 1.1343 | 1.1385 | 1.1415 | 1.1379 | 1.1383 |
| MVL | 1.0796 | 1.0819 | 1.0821 | 1.0801 | 1.0770 | 1.0835 | 1.0796 | 1.0837 |
| Hikaru | 1.2007 | 1.2078 | 1.2053 | 1.2037 | 1.2046 | 1.1978 | 1.2039 | 1.2015 |
| Ian | 1.1021 | 1.1043 | 1.1037 | 1.1028 | 1.1020 | 1.1059 | 1.0997 | 1.1026 |
| Alireza | 1.0876 | 1.0916 | 1.0893 | 1.0863 | 1.0879 | 1.0921 | 1.0857 | 1.0843 |

Source: `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic.json`, `method1.rows[*].nll_by_representation`.

### M1 checkpoint-selection validation

- `VERIFIED FACT` — M1 optimization uses only `z_player`; the base is frozen. The objective is full legal-policy cross-entropy. The training script uses AdamW, learning rate `1e-3`, weight decay `1e-4`, maximum 100 epochs, patience 5, batch size 32, and the same 80/20 split parameters above. Sources: `src/amadeus_counterpoint/training/method1_trainer.py`; `scripts/train_method1_player.py`.
- `VERIFIED FACT` — Best validation epochs for players 0–7 are `10, 3, 21, 15, 14, 4, 10, 10`; corresponding best losses are `1.1492, 1.0723, 1.1312, 1.1361, 1.0780, 1.1967, 1.0984, 1.0841`. Sources: the eight saved M1 logs above.
- `VERIFIED FACT` — Production checkpoints are `/mnt/scratch2/users/40482774/checkpoints/method1/player_0.pt` through `player_7.pt`; the identity archive records those exact paths as inputs.

## M2 individual validation

### Protocol and data

- `VERIFIED FACT` — M2 uses the same nonsealed `style_records.json`, but calls `build_style_datasets()` jointly on all eight players and then partitions the resulting validation dataset by `player_id`. This is distinct from M1's filter-before-split sequence. Source: `src/amadeus_counterpoint/evaluation/identity_diagnostic.py`, `build_method2_val_datasets()`; `scripts/train_method2.py`.
- `VERIFIED FACT` — The split is game-level 80/20 with `split_seed=0`, then phase-balanced with `balance_seed=0`. M2 uses the 80/20 style-record split; the separate 2% hash split belongs to Broadcast adaptation, not M2 personalization.

| player | eligible games | train games | held-out games | held-out decisions before balancing | identity-diagnostic positions after balancing |
|---|---:|---:|---:|---:|---:|
| Magnus Carlsen | 894 | 715 | 179 | 8,339 | 5,985 |
| Wesley So | 925 | 740 | 185 | 8,737 | 6,297 |
| Levon Aronian | 915 | 732 | 183 | 8,378 | 6,222 |
| Fabiano Caruana | 1,070 | 856 | 214 | 11,082 | 7,323 |
| Maxime Vachier-Lagrave | 911 | 729 | 182 | 8,270 | 6,303 |
| Hikaru Nakamura | 747 | 598 | 149 | 7,125 | 4,983 |
| Ian Nepomniachtchi | 825 | 660 | 165 | 7,354 | 5,487 |
| Alireza Firouzja | 635 | 508 | 127 | 6,118 | 4,350 |

Source for split counts: `/mnt/scratch2/users/40482774/lichess_data/method2_joint_9871250.out`. Source for final positions: identity JSON `method2.rows[*].n_positions`.

### Metric and comparison mechanics

- `VERIFIED FACT` — M2 training/validation uses `K=5` raw legal candidates. If the human target is outside raw top-5, it is appended as a sixth candidate for training/validation; if already present, the sixth slot is invalid padding. Source: `src/amadeus_counterpoint/models/candidates.py`, `select_candidates()`; `src/amadeus_counterpoint/training/style_trainer.py`, `validate()`.
- `VERIFIED FACT` — The identity diagnostic evaluates generic, correct-style, and all wrong-style representations on the identical oracle-expanded candidate set: raw top-5 plus target append if absent. This is a candidate-matched NLL, not a full 4,352-action NLL. Source: `src/amadeus_counterpoint/evaluation/identity_diagnostic.py`, `evaluate_method2_row()`.
- `VERIFIED FACT` — The candidate-matched generic baseline is the frozen base's own candidate logits without style residual. Deltas and ranking use this candidate-matched baseline, not the descriptive full-action generic baseline. Source: `evaluate_personalization_identity.py` comments and `identity_diagnostic.py` module docstring.
- `VERIFIED FACT` — Deployable top-1 accuracy is separately evaluated with `target_index=None`, so only raw top-5 candidates are available; a human move outside raw top-5 is automatically counted wrong. Source: `evaluate_method2_row()`.

### Correct vs. generic and correct vs. wrong-player results

| player | candidate-matched generic NLL | correct NLL | mean wrong NLL | generic − correct | wrong − correct | raw top-5 coverage | correct deployable top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Magnus Carlsen | 1.095070 | 1.089051 | 1.107340 | 0.006019 | 0.018289 | 0.951378 | 0.589140 |
| Wesley So | 1.029250 | 1.022809 | 1.038603 | 0.006440 | 0.015794 | 0.951247 | 0.622042 |
| Levon Aronian | 1.097234 | 1.087431 | 1.107444 | 0.009803 | 0.020013 | 0.946962 | 0.589039 |
| Fabiano Caruana | 1.083274 | 1.079190 | 1.095417 | 0.004084 | 0.016227 | 0.944968 | 0.605763 |
| Maxime Vachier-Lagrave | 1.010589 | 0.991045 | 1.021324 | 0.019544 | 0.030279 | 0.955259 | 0.636046 |
| Hikaru Nakamura | 1.181498 | 1.163746 | 1.192857 | 0.017752 | 0.029111 | 0.941601 | 0.573149 |
| Ian Nepomniachtchi | 1.052151 | 1.042082 | 1.060758 | 0.010069 | 0.018677 | 0.956443 | 0.612356 |
| Alireza Firouzja | 1.069488 | 1.068978 | 1.084180 | 0.000510 | 0.015202 | 0.950575 | 0.608276 |

`VERIFIED FACT` — Overall mean generic-minus-correct candidate-matched NLL is `0.009277804541675441`; overall mean wrong-minus-correct NLL is `0.02044897915454176`; correct representation rank is 1 for 8/8 players. Source: identity JSON `method2.overall`.

### Full M2 8×8 candidate-matched NLL matrix

| true \\ representation | Magnus | Wesley | Levon | Fabiano | MVL | Hikaru | Ian | Alireza |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Magnus | 1.0891 | 1.1110 | 1.1075 | 1.1034 | 1.1114 | 1.1095 | 1.1057 | 1.1029 |
| Wesley | 1.0347 | 1.0228 | 1.0376 | 1.0343 | 1.0511 | 1.0359 | 1.0394 | 1.0373 |
| Levon | 1.1037 | 1.1041 | 1.0874 | 1.0981 | 1.1206 | 1.1137 | 1.1029 | 1.1090 |
| Fabiano | 1.0901 | 1.0967 | 1.0934 | 1.0792 | 1.1066 | 1.0984 | 1.0932 | 1.0895 |
| MVL | 1.0176 | 1.0257 | 1.0173 | 1.0208 | 0.9910 | 1.0446 | 1.0062 | 1.0170 |
| Hikaru | 1.1707 | 1.1960 | 1.1996 | 1.1917 | 1.2126 | 1.1637 | 1.1967 | 1.1827 |
| Ian | 1.0571 | 1.0654 | 1.0611 | 1.0531 | 1.0637 | 1.0660 | 1.0421 | 1.0589 |
| Alireza | 1.0775 | 1.0866 | 1.0866 | 1.0810 | 1.0875 | 1.0881 | 1.0820 | 1.0690 |

Source: identity JSON `method2.rows[*].nll_by_representation`.

### M2 checkpoint-selection validation

- `VERIFIED FACT` — All eight players share one jointly trained M2 style stack: `MoveStyleCNN`, `PlayerStyleTable`, and `StyleResidual`; the ChessFormer base is frozen. Source: `scripts/train_method2.py`; `src/amadeus_counterpoint/training/style_trainer.py`.
- `VERIFIED FACT` — The saved M2 training log selects epoch 4 with validation loss `1.0658` and raw top-5 coverage `0.9498`; training stops at epoch 9 after patience 5. Source: `/mnt/scratch2/users/40482774/lichess_data/method2_joint_9871250.out`.
- `VERIFIED FACT` — Production checkpoint is `/mnt/scratch2/users/40482774/checkpoints/method2/joint.pt`; identity `MANIFEST.md` records this exact input.

## 8-way player identification

- `VERIFIED FACT` — There was an explicit 8-way experiment. For each true player, the diagnostic scored the same held-out positions under the generic baseline and all eight learned representations, then ranked the eight by mean NLL. Source: `scripts/evaluate_personalization_identity.py`; `src/amadeus_counterpoint/evaluation/identity_diagnostic.py`, `summarize_row()`.
- `VERIFIED FACT` — It was run for both M1 and M2. M1 ranks by full-action legal-masked NLL; M2 ranks by candidate-matched oracle-expanded NLL. These scales must not be compared directly.
- `VERIFIED FACT` — Correct representation rank is 1 for every target for both methods: M1 `8/8`, M2 `8/8`; mean correct rank is `1.0` for both. Source: identity JSON `method1.overall` and `method2.overall`.
- `VERIFIED FACT` — No exact NLL ties are present in the saved matrices. The smallest gap between rank 1 and rank 2, computed from the stored values, is approximately `0.000115` nats for M1 (Wesley row) and `0.006974` nats for M2 (Hikaru row). Source: identity JSON matrices; gap is a read-only arithmetic check.
- `VERIFIED FACT` — The diagnostic did not save an MRR, entropy-of-ranking, confidence interval, or significance test. Source: identity JSON schema and `identity_diagnostic.py`.

## Candidate/policy sanity

### M1

- `VERIFIED FACT` — Full-action correct-player top-1 accuracy is `0.571515–0.620301` across players in the identity JSON. Generic and correct top-1 accuracy are both present under `generic_accuracy` and `correct_accuracy`; correct accuracy is not higher for every player, so NLL ranking is the primary identity metric in this artifact.
- `VERIFIED FACT` — M1 best held-out validation losses are the eight values listed above; these are full-action policy cross-entropies. Source: M1 logs and `Method1Trainer.validate()`.

| player | generic full-action top-1 | correct M1 top-1 |
|---|---:|---:|
| Magnus Carlsen | 0.584461 | 0.588805 |
| Wesley So | 0.613441 | 0.615348 |
| Levon Aronian | 0.602612 | 0.599813 |
| Fabiano Caruana | 0.603787 | 0.601714 |
| Maxime Vachier-Lagrave | 0.621867 | 0.620301 |
| Hikaru Nakamura | 0.569293 | 0.571515 |
| Ian Nepomniachtchi | 0.618869 | 0.616729 |
| Alireza Firouzja | 0.612053 | 0.615584 |

Source: identity JSON `method1.rows[*].generic_accuracy` and `correct_accuracy`.

### M2

- `VERIFIED FACT` — Raw top-5 coverage means the fraction of held-out human target moves already contained in the frozen base's legal raw top-5, before target append and before style reranking. Source: `src/amadeus_counterpoint/models/candidates.py`, `topk_target_coverage()`; identity JSON `raw_topk_coverage`.
- `VERIFIED FACT` — Per-player raw top-5 coverage is `0.941601–0.956443`; overall training-log coverage is `0.9498`. The identity diagnostic's per-player coverage values are in the M2 table above.
- `VERIFIED FACT` — Correct-player deployable top-1 accuracy over raw top-5 is `0.573149–0.636046` across players. This is not full-policy top-1 accuracy and is not the same objective as candidate-matched validation NLL.
- `VERIFIED FACT` — The saved M2 identity JSON contains generic and correct candidate-matched top-1 accuracies, plus full-action generic accuracy as a descriptive-only field. It does not contain a separate, independent top-1 experiment outside the identity diagnostic.

| player | candidate-matched generic top-1 | correct M2 top-1 over raw top-5 |
|---|---:|---:|
| Magnus Carlsen | 0.584461 | 0.589140 |
| Wesley So | 0.616166 | 0.622042 |
| Levon Aronian | 0.589521 | 0.589039 |
| Fabiano Caruana | 0.599345 | 0.605763 |
| Maxime Vachier-Lagrave | 0.631128 | 0.636046 |
| Hikaru Nakamura | 0.568132 | 0.573149 |
| Ian Nepomniachtchi | 0.610169 | 0.612356 |
| Alireza Firouzja | 0.608966 | 0.608276 |

Source: identity JSON `method2.rows[*].candidate_matched_generic_accuracy` and `deployable_top1_accuracy_correct`.

## Other individual-fidelity diagnostics

### M1 synthetic marginal vs. held-out real player behavior

`VERIFIED FACT` — A separate M1-only analysis compared generated marginal behavior with held-out nonsealed real target-vs-outsider games. It used the M1 split reconstruction (filter player, then split, 80/20, seeds 0/0), first moves from held-out white games, and opening-family distributions from all held-out games. Sources: `/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/m1_nonsealed_real_fidelity.py`; `nonsealed_real_comparison.json`.

The generated reference was the complete guarded M1 synthetic corpus: 70,000 personalized games per player and a pooled generic GG reference of 280,000 games. Source: `M1_COMPLETE_PRELIMINARY_REPORT.md`, Sections 1–5.

| player | held-out games | held-out white games | first-move TV personalized/real | first-move TV generic/real | opening-family TV personalized/real | opening-family TV generic/real |
|---|---:|---:|---:|---:|---:|---:|
| Magnus Carlsen | 179 | 85 | 0.0904 | 0.1267 | 0.2274 | 0.2637 |
| Wesley So | 185 | 99 | 0.0758 | 0.1074 | 0.1691 | 0.1697 |
| Levon Aronian | 183 | 99 | 0.2907 | 0.3279 | 0.2580 | 0.2727 |
| Fabiano Caruana | 214 | 109 | 0.1571 | 0.1834 | 0.1570 | 0.1677 |
| Maxime Vachier-Lagrave | 182 | 89 | 0.3028 | 0.3457 | 0.3237 | 0.3555 |
| Hikaru Nakamura | 149 | 75 | 0.2709 | 0.2956 | 0.2627 | 0.2837 |
| Ian Nepomniachtchi | 165 | 90 | 0.3378 | 0.3790 | 0.2582 | 0.2679 |
| Alireza Firouzja | 127 | 56 | 0.1155 | 0.1605 | 0.1679 | 0.2306 |

- `VERIFIED FACT` — Personalized first-move TV was lower than generic-vs-real TV for all 8/8 players.
- `VERIFIED FACT` — Personalized opening-family TV was lower for all 8/8 players, although Wesley's margin is only `0.0006` TV units.
- `VERIFIED FACT` — This artifact uses the report's self-defined M1 opening taxonomy, not the pinned final sealed-evaluation opening taxonomy. Source: `M1_COMPLETE_PRELIMINARY_REPORT.md`, Section 3; `m1_opening_taxonomy.py`.
- `INFERENCE` — This is individual-fidelity evidence, not a direct move-level likelihood test and not interaction validation. It compares marginal distributions from generated M1 games to held-out real individual games.
- `UNKNOWN` — No corresponding saved M2 generated-marginal-vs-real individual-fidelity table was found. The M2 identity diagnostic is the verified M2 individual validation artifact.

### M1 state-dependent representation differentiation

`VERIFIED FACT` — `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/verification_pass_provisional/section_d_state_dependent_differentiation.json` evaluates the same real nonsealed held-out board position under generic plus all eight M1 representations. It samples 20 positions per band from distinct held-out games, with seed `20260912`; the source script discards which player's game supplied the position before comparing distributions.

| band | positions | all-8 same top-1 | mean player/player TV | mean player/generic TV | mean player/player JS | mean top-5 overlap player/player |
|---|---:|---:|---:|---:|---:|---:|
| ply 0 | 20 | 1.00 | 0.028298 | 0.057875 | 0.032748 | 1.000000 |
| ply 4–6 | 20 | 0.90 | 0.030454 | 0.039558 | 0.035981 | 0.975357 |
| ply 8–10 | 20 | 0.90 | 0.034900 | 0.037517 | 0.042661 | 0.965000 |
| ply 16–20 | 20 | 0.95 | 0.025954 | 0.026824 | 0.033934 | 0.979643 |
| ply 24–30 | 20 | 0.85 | 0.030784 | 0.030544 | 0.035347 | 0.955357 |
| ply 40+ | 20 | 0.85 | 0.036084 | 0.032567 | 0.037421 | 0.975357 |

- `VERIFIED FACT` — This is a same-state distribution-difference diagnostic, not a correct-player-vs-observed-move test. It provides no per-player identity ranking.
- `INFERENCE` — It is supporting representation-differentiation evidence, not as strong as the identity diagnostic or M1 real-marginal comparison for individual behavioral validity.

### Fixed-opening inference inspection for M1 and M2

- `VERIFIED FACT` — `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/final_sampler_freeze/opening_distribution_inspection.md` evaluates fixed opening positions with opponent Elo fixed at 2800 and no sealed data. At the starting position, mean pairwise behavioral TV is `0.0275` for M1 before guardrail and `0.0254` after; for M2 it is `0.2335` before and `0.2328` after.
- `VERIFIED FACT` — At the starting position, all eight M1 top-5 candidate sets are identical and all eight M2 top-5 candidate sets are identical; the modal move is also the same for all eight within each method. This shows the representations can change probabilities substantially without necessarily changing the opening candidate set or modal move.
- `INFERENCE` — This is a fixed-state representation/style-conditioning sanity check, not evidence that the correct representation matches a particular player's observed data.

## Rollout-quality sanity

These are not core individual behavioral validation because they score generated move quality or infrastructure behavior rather than whether a representation predicts the correct player's held-out moves.

- `VERIFIED FACT` — The earlier unguarded generation-strength diagnostic used 40 M1 synthetic games / 80 sides and 40 held-out nonsealed real sides. Median capped ACPL was `64.8833` synthetic versus `18.2370` real. Source: `/mnt/scratch2/users/40482774/paper_artifacts/generation_strength_diagnostic_2026-09-12/SUMMARY.md`. This is a quality-gap diagnostic, not player-fidelity evidence.
- `VERIFIED FACT` — The complete guarded M1 preliminary report records a depth-20 sampled mean capped ACPL of `27.8` across its stratified quality sample and zero censored games in that report's full M1 corpus checks. Source: `/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/M1_COMPLETE_PRELIMINARY_REPORT.md`, Sections 1 and 7.
- `VERIFIED FACT` — The sampler-freeze validation used 480 nonsealed positions split into 240 DEV and 240 TEST by whole game, seed `20260914`. Frozen `lambda=2.0` had TEST mean loss `7.12` cp, median `4.29` cp, and zero TEST positions at or above 50 cp. Source: `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/final_sampler_freeze/FINAL_SAMPLER_REPORT.md`, Sections 7–10.
- `VERIFIED FACT` — The same sampler report records M1 mean pairwise policy TV `0.0328786` before and `0.0253692` after guardrail on 75 shared nonsealed positions. This is a guardrail-effect diagnostic, not a new M1 training/identity result. Source: `final_sampler_freeze/player_differentiation_before_after.json`.
- `VERIFIED FACT` — A provisional M2-vs-M1 rollout comparison used only 24 games, three dyads all involving Magnus, and was explicitly marked preliminary/nonrepresentative. It reported mean capped ACPL `76.2` for M2 versus `63.6` for matched M1. Source: `verification_pass_provisional/VERIFICATION_REPORT.md`, Section Q4.

## Leakage, sealing, and chronology

- `VERIFIED FACT` — `style_records.json` contains only one-target records. Target-target games are excluded by `data.broadcast_ingest.classify_game()` / `iter_target_game_records()` before records are written. Source: `src/amadeus_counterpoint/data/broadcast_ingest.py`; identity `MANIFEST.md`.
- `VERIFIED FACT` — Every M1/M2 validation result above that uses real held-out moves or positions is therefore target-vs-outsider data. No sealed A–B game is present in the validation input.
- `VERIFIED FACT` — The identity archive explicitly states that no sealed dyad data or target-target manifest was read. Source: identity `MANIFEST.md`.
- `VERIFIED FACT` — The M1 nonsealed real-fidelity script states the same boundary and reads only `style_records.json`. Source: `m1_nonsealed_real_fidelity.py`.
- `VERIFIED FACT` — The final pre-evaluation freeze states it was finalized before any sealed WDL, opening-family, TV, or bootstrap output was computed or inspected. Source: `/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/PRE_EVAL_MANIFEST.md`, opening sections.
- `VERIFIED FACT` — Artifact timestamps establish the identity diagnostic on September 11, the sampler/verification diagnostics on September 12, the complete M1 preliminary report on September 15, and the sealed pre-evaluation freeze on September 16. Source: Kelvin2 `stat` output for the named artifacts and the reports' embedded dates.
- `VERIFIED FACT` — Structural sealed information such as counts, IDs, and parse integrity was inspected separately for data validation. That is not the same as inspecting sealed behavioral results. Source: final `PRE_EVAL_MANIFEST.md`.

## What belongs in the main paper

Short factual shortlist only:

1. M1 and M2 were validated on held-out, nonsealed one-target games using game-level 80/20 splits; M1 split each player before splitting, while M2 split all players jointly before partitioning.
2. In the saved 8-way identity diagnostic, the correct player representation ranked first for all 8 players for both M1 and M2. The mean generic-minus-correct NLL deltas were `0.006802` for M1 and `0.009278` for M2; the M2 value is candidate-matched, not full-action.
3. M1's generated marginal first-move and opening-family distributions were closer to held-out real individual-player distributions than the generic reference for all 8 players in the saved nonsealed comparison.
4. M2's raw frozen-base top-5 contained `94.16%–95.64%` of held-out human moves across players; correct-style deployable top-1 accuracy was `57.31%–63.60%`.

## What belongs in the appendix

- Full M1/M2 8×8 NLL and accuracy matrices, per-player validation counts, and per-player deltas.
- M1 checkpoint-selection epochs/losses and M2 epoch-4 loss/coverage.
- M1 first-move/opening-family real-vs-synthetic TV table, including the self-defined taxonomy caveat.
- M1 state-dependent TV/JS/top-5-overlap diagnostic.
- Fixed-opening M1/M2 probability tables and representation-differentiation values.
- Raw top-5 candidate-quality, guardrail, ACPL, and closed-loop rollout diagnostics, with their sample sizes and preliminary labels.

`INFERENCE` — No repository artifact formally labels any result “main paper” or “appendix”; the split above is an artifact-role classification, not saved metadata.

## Remaining unknowns

- `UNKNOWN` — No separate saved M2 real-vs-synthetic marginal-fidelity analysis equivalent to M1's `nonsealed_real_comparison.json` was found.
- `UNKNOWN` — No separate M1/M2 calibration analysis, confidence intervals, significance tests, MRR, ranking entropy, or formal player-identification classifier beyond mean-NLL rank was found.
- `UNKNOWN` — The identity artifact does not preserve the validation game IDs or a standalone per-player post-balancing phase-count report; it preserves position counts and the split configuration.
- `UNKNOWN` — No evidence was found that any sealed behavioral result was used for model selection, hyperparameter selection, K selection, sampler tuning, or rollout diagnostics before the final sealed evaluation. The inspected manifests explicitly say the opposite, but independent provenance of every historical exploratory command is not available.
- `UNKNOWN` — The exact number of raw/valid/pre-dedup sealed games is irrelevant to these individual validations and was not needed by them; no sealed behavioral sample was used in the reported individual-validation metrics.
