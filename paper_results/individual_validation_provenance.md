# Conclusion

**A. The identity diagnostic is fully held out from personalization training.**

For both M1 and M2, the diagnostic positions come from the validation side of a deterministic **game-level 80/20 split**. No validation game contributes a training decision, and no training game contributes a diagnostic decision. The diagnostic therefore has zero overlap with the examples used for gradient-based fitting of the personalized parameters.

Important qualification: this is **not an untouched final test set**. The same validation split is used during training for validation loss, early stopping, and best-checkpoint selection. The diagnostic is therefore a post-training validation/checkpoint-selection diagnostic, not independent model-selection-free generalization evidence.

The input contains only nonsealed one-target games. It excludes target-vs-target games before the records are written.

# M1 provenance

## Personalization training data

- Input: `/mnt/scratch2/users/40482774/lichess_data/style_records.json`.
- Remote file facts checked: 6,922 records; SHA-256 `d746de9e184d76c9f0243345bdb5989bc182cd7a2e3c944cfd59b93b23cc9f94`.
- Each record is an exactly-one-target game. The saved record schema contains `player_id`, mover color, Elo values, result, and UCI moves, but no GameURL, source PGN filename, or persistent game ID.
- M1 runs one player at a time. `train_method1_player.py` filters records to the requested player before calling `build_style_datasets()` (`scripts/train_method1_player.py:85-93`).
- The base model is frozen; only that player's M1 `z_player` is optimized. The production checkpoints are the eight archived `player_0.pt` through `player_7.pt` files listed in the identity manifest.

## Diagnostic data

- Artifact: `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic.json`.
- Diagnostic job: SLURM `9877842`; evaluator commit `05c7e70600883100c7ce2a599834f3b748ce3312`.
- `build_method1_val_dataset()` applies the same player filter and then calls the same `build_style_datasets()` routine (`src/amadeus_counterpoint/evaluation/identity_diagnostic.py:75-90`).
- The diagnostic evaluates the resulting balanced validation decisions under the generic base and all eight frozen M1 representations.

## Split and overlap

The split is game-level, not position-level, with `train_fraction=0.8`, `split_seed=0`, and phase balancing after the split with `balance_seed=0`. One `random.Random(0)` split is used per M1 player run.

| Player | Eligible games | Train games | Diagnostic/validation games | Train decisions before balancing | Diagnostic positions after balancing |
|---|---:|---:|---:|---:|---:|
| Magnus Carlsen | 894 | 715 | 179 | 34,491 | 5,985 |
| Wesley So | 925 | 740 | 185 | 33,599 | 6,294 |
| Levon Aronian | 915 | 732 | 183 | 34,039 | 6,432 |
| Fabiano Caruana | 1,070 | 856 | 214 | 43,224 | 7,236 |
| Maxime Vachier-Lagrave | 911 | 729 | 182 | 32,579 | 6,384 |
| Hikaru Nakamura | 747 | 598 | 149 | 28,951 | 4,950 |
| Ian Nepomniachtchi | 825 | 660 | 165 | 30,032 | 5,607 |
| Alireza Firouzja | 635 | 508 | 127 | 25,319 | 4,248 |
| **Total** | **6,922** | **5,538** | **1,384** | **262,234** | **47,136** |

The exact overlap result is **zero at the source-game/decision level**. `split_games_by_player()` returns separate train and validation record lists, and `build_style_datasets()` enumerates decisions from those separate lists (`src/amadeus_counterpoint/data/style_dataset.py:39-91, 94-106, 242-290`). Phase balancing only downsamples within each already-separated split; it cannot move a training game into validation or vice versa.

| Comparison | Result |
|---|---:|
| Training games ∩ diagnostic games | 0 games; 0% of diagnostic games |
| Training decision occurrences ∩ diagnostic decision occurrences | 0 by construction; 0% of diagnostic occurrences |
| Persistent raw GameURL/position-ID intersection | Not computable: those IDs were not retained |

The identity JSON stores only aggregate per-player counts and NLL matrices. It does not store the `(game_id, ply)` rows. The compact `StyleGameRecord` also strips source headers, including `GameURL`. Therefore the zero-overlap result is proven from the exact split/dataflow code and saved split counts, not from a post-hoc raw-ID join.

# M2 provenance

## Personalization training data

- Input: the same 6,922-record `style_records.json` file.
- M2 trains one joint style stack for all eight players over a frozen base (`scripts/train_method2.py:84-99`).
- The M2 training log is `/mnt/scratch2/users/40482774/lichess_data/method2_joint_9871250.out`.
- The log records the 80/20 split and the per-player pre-balancing decision counts. The best checkpoint was selected by validation loss at epoch 4; later epochs were stopped by validation patience.

## Diagnostic data

- The same identity JSON and identity job are used.
- `build_method2_val_datasets()` calls `build_style_datasets()` once on all eight players and then partitions the resulting balanced validation dataset by `player_id` (`src/amadeus_counterpoint/evaluation/identity_diagnostic.py:93-115`).
- The M2 diagnostic uses candidate-matched NLL on the raw top-5 plus target append when needed. This metric definition does not affect the data split.

## Split and overlap

M2 uses the same nominal settings but a different split call sequence: game-level 80/20, `split_seed=0`, `balance_seed=0`, and one shared `random.Random(0)` consumed across all players in sorted player-ID order.

| Player | Eligible games | Train games | Diagnostic/validation games | Train decisions before balancing | Diagnostic positions after balancing |
|---|---:|---:|---:|---:|---:|
| Magnus Carlsen | 894 | 715 | 179 | 34,491 | 5,985 |
| Wesley So | 925 | 740 | 185 | 33,469 | 6,297 |
| Levon Aronian | 915 | 732 | 183 | 34,357 | 6,222 |
| Fabiano Caruana | 1,070 | 856 | 214 | 42,797 | 7,323 |
| Maxime Vachier-Lagrave | 911 | 729 | 182 | 33,041 | 6,303 |
| Hikaru Nakamura | 747 | 598 | 149 | 29,394 | 4,983 |
| Ian Nepomniachtchi | 825 | 660 | 165 | 30,501 | 5,487 |
| Alireza Firouzja | 635 | 508 | 127 | 24,834 | 4,350 |
| **Total** | **6,922** | **5,538** | **1,384** | **262,884** | **46,950** |

The exact overlap result is again **zero at the source-game/decision level**. M2's `train_dataset` and `val_dataset` are returned from the same disjoint game-level split, and the identity diagnostic reconstructs the latter split with the same call sequence and seeds.

| Comparison | Result |
|---|---:|
| Training games ∩ diagnostic games | 0 games; 0% of diagnostic games |
| Training decision occurrences ∩ diagnostic decision occurrences | 0 by construction; 0% of diagnostic occurrences |
| Persistent raw GameURL/position-ID intersection | Not computable: those IDs were not retained |

# Why M1 and M2 position counts differ

The diagnostic datasets are **not identical**, even though they use the same source file, 80/20 fraction, and seed values.

- M1 filters to one player before splitting. Each player's split starts from a fresh `random.Random(0)` state.
- M2 splits all eight players jointly. One shared RNG is consumed in sorted player-ID order.
- Consequently, player 0 begins from the same RNG state in both procedures, while later player splits use different shuffled game orders. The validation game memberships therefore differ by method for later players.
- Phase balancing then samples separately within each method's validation games.

The resulting diagnostic counts are 47,136 for M1 and 46,950 for M2, a difference of 186 positions. This is a consequence of different validation-game memberships and phase composition, not evidence of train/validation leakage. The count difference is also visible in the archived identity log and JSON.

# M1 generative sanity check

Artifact: `/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/nonsealed_real_comparison.json`.

This separate M1-only analysis is genuinely based on the M1 split's held-out nonsealed games:

- It reconstructs the M1 split by filtering to each player, then applying the same 80/20 game split with `split_seed=0` and `balance_seed=0`.
- It compares generated M1 marginal behavior against those held-out real games. The artifact's `split_report` records the same per-player validation-game counts and pre-balancing validation-decision counts as the M1 identity path.
- It uses the held-out games for first-move and opening-family distributions; it does not use M1 training games for the real-data reference.
- This is still not a completely untouched test set: those validation games were used for checkpoint selection during M1 training.
- Its opening-family analysis uses a self-defined preliminary taxonomy, not the pinned final sealed-evaluation taxonomy.

Therefore the correct description is “held-out from gradient fitting under the M1 game split,” not “independent test set.”

# Paper wording recommendation

Recommended wording:

> We performed a post-training eight-way identity diagnostic on nonsealed target-vs-outsider validation positions. For each method, positions were drawn from a deterministic game-level 80/20 split that was disjoint from the games used for gradient-based personalization fitting; the validation split was also used for checkpoint selection. The correct player representation ranked first by the method-specific NLL for all eight players.

Avoid saying simply “held-out test positions” or “untouched held-out data.” “Held-out nonsealed validation positions” is accurate if the checkpoint-selection qualification is included.

# Reviewer-risk note

- **Training leakage:** not supported for source-game/decision membership. The code splits games before generating positions, so a training game cannot contribute a diagnostic position.
- **In-sample/model-selection criticism:** reasonable. The validation positions were used to choose the saved checkpoint, so the identity diagnostic is not independent of checkpoint selection.
- **Generalization claim:** it supports within-corpus validation after game-level separation, but not a fully independent external test of personalization generalization.
- **Data identity auditability:** exact raw GameURL or position-ID intersection cannot be reproduced from the saved identity JSON or `style_records.json`, because those identifiers were discarded. The zero-overlap claim rests on deterministic reconstruction of the archived split procedure and saved split counts.
- **Sealed leakage:** not supported. `iter_target_game_records()` yields only one-target games and discards target-vs-target games before `StyleGameRecord` creation (`src/amadeus_counterpoint/data/broadcast_ingest.py:1-8, 32-55, 58-143`).

# Evidence inventory

- Repository split/dataflow: `src/amadeus_counterpoint/data/style_dataset.py`, `scripts/train_method1_player.py`, `scripts/train_method2.py`, `src/amadeus_counterpoint/evaluation/identity_diagnostic.py`, and `scripts/evaluate_personalization_identity.py`.
- Identity archive: `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/MANIFEST.md`, `.json`, and `.out`.
- M1 training logs: `/mnt/scratch2/users/40482774/lichess_data/method1_player{0..7}_98712*.out` as listed in the existing provenance audit.
- M2 training log: `/mnt/scratch2/users/40482774/lichess_data/method2_joint_9871250.out`.
- M1 generative sanity check: `/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/nonsealed_real_comparison.json` and its accompanying report/script.
- Existing supporting audit: `individual_behavioral_validation_audit.md`, especially its “Leakage, sealing, and chronology” and “Remaining unknowns” sections.
