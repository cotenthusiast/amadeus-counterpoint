# Amadeus Methodology — Read-Only Implementation Audit

Audit date: 2026-09-17

This is an implementation audit, not paper prose. `VERIFIED FACT` means directly established from repository code/configuration or a saved Kelvin2 artifact. `INFERENCE` is a conclusion that follows from verified evidence but is not itself a literal implementation field. `UNKNOWN` means the inspected sources do not establish it. Where sources disagree, both are retained.

The local repository had pre-existing uncommitted changes in the evaluator and tests when this audit began. Those files were not modified. Kelvin2 was inspected in place; no large corpus, checkpoint, or generation job was copied or launched.

## 1. Base ChessFormer

### 1.1 Input encoding and Elo conditioning

- `VERIFIED FACT` — `src/amadeus_counterpoint/encoding.py`, `encode_board()` canonicalizes every position with `board.mirror()` when Black is to move. The canonical frame therefore always has the mover as White. The 12 channels are one-hot white pawn/knight/bishop/rook/queen/king followed by black pawn/knight/bishop/rook/queen/king.
- `VERIFIED FACT` — `encoding.py`, `HISTORY_LENGTH = 8`; `encode_history()` retains the latest eight canonical boards, concatenates their 12-channel encodings in oldest-to-newest order, and prepends copies of the earliest available position when fewer than eight boards exist. Normal input is therefore current position plus the previous seven positions; early-game padding is repeated earliest-position padding.
- `VERIFIED FACT` — Raw board/history shape is `[B, 64, 96]` (`8 × 12` channels per square). `scripts/train.py` sets `RAW_INPUT_DIM = 96`.
- `VERIFIED FACT` — `models/chessformer.py`, `Chessformer.forward()`: mover and opponent Elo each produce a `[B,128]` embedding, expanded to `[B,64,128]`, then concatenated with `[B,64,96]` to form `[B,64,352]`. The input projection is `Linear(352,1024)`.
- `VERIFIED FACT` — `Chessformer.interpolate_elo()` clamps Elo to `[0,5000]` and computes

  `w_low = elo / 5000`, `w_high = 1 - w_low`,

  `embedding = w_low * elo_low + w_high * elo_high`.

  Both `elo_low` and `elo_high` are learned `nn.Embedding(1,128)` anchors. The variable names are counterintuitive: Elo 0 gives the `elo_high` anchor and Elo 5000 gives the `elo_low` anchor under the implemented equation.
- `VERIFIED FACT` — The claimed `96 + 128 + 128 = 352` feature count is correct.

### 1.2 Backbone

Production constants are in `scripts/train.py` and repeated by the generation scripts:

| quantity | implemented value |
|---|---:|
| `d_model` | 1024 |
| blocks | 8 |
| attention heads | 32 |
| head dimension | 32 (`1024 / 32`) |
| FFN hidden dimension | 2048 |
| dropout | 0.0 in production |
| normalization | RMSNorm in transformer blocks; LayerNorm in final encoder norm, GAB, and value head |
| ordering | post-norm residual blocks |

- `VERIFIED FACT` — `models/mha.py`, `MultiHeadAttention`: Q/K/V are bias-free `1024 → 1024` projections, reshaped to `[B,32,64,32]`; QK logits are `[B,32,64,64]` and scaled by `head_dim ** -0.5 = 32 ** -0.5`.
- `VERIFIED FACT` — `models/transformer_block.py`, `TransformerBlock.forward()` applies

  `x = RMSNorm1(x + dropout1(attention(x)))`

  followed by

  `x = RMSNorm2(x + dropout2(Linear2(GELU(Linear1(x)))))`.

  The production dropout values are zero, but the residual ordering is still post-norm.

### 1.3 GAB shape trace

`models/gab.py`, `GeometricAttentionBias`, with `d1=32`, `d2=128`, `d3=128`, `H=32`:

| stage | shape |
|---|---|
| block input | `[B,64,1024]` |
| `d1flattening: Linear(1024,32)` | `[B,64,32]` |
| flatten board | `[B,2048]` |
| `d1d2: Linear(2048,128)` | `[B,128]` |
| GELU, `LayerNorm(128)` | `[B,128]` |
| `d2Hd3: Linear(128,4096)` | `[B,4096]` |
| GELU, `LayerNorm(4096)` | `[B,4096]` |
| reshape | `[B,32,128]` |
| shared bank `d34096` | `[4096,128]` |
| einsum `bhi,oi->bho` | `[B,32,4096]` |
| reshape | `[B,32,64,64]` |

- `VERIFIED FACT` — The GAB is added directly to the QK attention logits before softmax in `models/mha.py`, `attn_logits = attn_logits + self.gab(x)`.
- `VERIFIED FACT` — The `d1flattening`, `d1d2`, and `d2Hd3` layers are separate in each of the eight blocks. The `d34096` bank is one `nn.Parameter` owned by `ChessFormer` and passed by reference into every block; it is shared across layers and heads. There is no separate final GAB projection per head or layer.
- `VERIFIED FACT` — The complete per-block attention/GAB trace is `[B,64,1024] → Q,K,V [B,32,64,32] → QK [B,32,64,64]`; GAB has the same `[B,32,64,64]` shape and is added elementwise; softmax/value mixing returns `[B,64,1024]`.

### 1.4 Policy head

- `VERIFIED FACT` — `models/heads.py`, `PolicyHead`: `proj_sq_from` and `proj_sq_to` are bias-free `1024 → 1024` projections. Their pairwise product is `from @ to.T / sqrt(1024)`, shape `[B,64,64]`, reshaped to 4096 ordinary-move logits.
- `VERIFIED FACT` — Promotion logits use the eight rank-7 source squares and eight rank-8 destination squares. For each of `8 × 8` source/destination pairs, four promotion biases (Q/R/B/N) are added to the ordinary pair score, producing `8 × 8 × 4 = 256` promotion logits. The final policy shape is `[B,4352] = [B,4096+256]`.
- `VERIFIED FACT` — `encoding.py`, `move_to_policy_index()` uses ordinary index `from_square*64 + to_square`; promotion indices start at 4096 and encode file/promotion type as 256 slots.

### 1.5 Value head

- `VERIFIED FACT` — `models/heads.py`, `ValueHead`: `[B,64,1024]` → mean over squares `[B,1024]` → `LayerNorm(1024)` → `Linear(1024,1024)` → ReLU → `Linear(1024,3)`.
- `VERIFIED FACT` — `data/dataset.py`, `result_to_value_target()`: class 0 is loss for the side to move, class 1 draw, class 2 win. The head therefore emits W/D/L logits in side-to-move perspective, not a scalar value.

### 1.6 Parameter count

- `VERIFIED FACT` — Summing the parameter shapes instantiated by the production constants in `scripts/train.py` and the module constructors gives **77,849,091 trainable parameters**. The shared `d34096` bank is counted once, while each block’s GAB projection/norm parameters are counted separately.
- `VERIFIED FACT` — The saved Broadcast checkpoint is `/mnt/scratch2/users/40482774/checkpoints/chessformer-79m-broadcast/step_00200000.pt`; the final-evaluation manifest records size 934,407,626 bytes. `79m` is the run label; the exact module count above is the implementation count.

## 2. Target-clean base training

### 2.1 Stage 1 population corpus

- `VERIFIED FACT` — The Kelvin2 launch scripts `/mnt/scratch2/users/40482774/lichess_data/preprocess_2017-06.sh` through `preprocess_2021-06.sh` process exactly five public Lichess Standard Rated snapshots: June 2017, June 2018, June 2019, June 2020, and June 2021. There is no production Stage-1 June-2022 training script in the inspected artifacts.
- `VERIFIED FACT` — `src/amadeus_counterpoint/data/preprocess.py`, `game_to_record()`: a usable game must be Standard, have no custom FEN/SetUp, parse both Elo headers as integers, have a result in `{1-0,0-1,1/2-1/2}`, and contain at least one move. Clock data is used for eligible-position filtering but is not required for game validity.
- `VERIFIED FACT` — `preprocess.py`, `balance_by_elo()`: 22 bins, width 100, with boundaries `(<600)`, `[600,700)`, …, `[2500,2600)`, and `(>=2600)`. It processes records in sequential chunks of 20,000 and keeps the first at most 10 games per bin in each chunk. This is not a global random sample of 10 games per bin; the cap resets per 20,000-record chunk and preserves source order.
- `VERIFIED FACT` — The remote monthly census artifacts report:

  | month | raw PGN games | valid/standard reported | target-excluded | retained games |
  |---|---:|---:|---:|---:|
  | 2017-06 | 11,512,600 | 11,470,045 | 0 | 102,447 |
  | 2018-06 | 20,273,737 | 20,222,301 | 523 | 183,616 |
  | 2019-06 | 33,935,786 | 33,877,640 | 165 | 318,027 |
  | 2020-06 | 70,374,749 | 70,064,604 | 939 | 711,127 |
  | 2021-06 | 92,190,803 | 91,740,152 | 300 | 963,385 |
  | **reported total** | **228,287,675** | **227,374,742** | **1,927** | **2,278,602** |

  Source: `/mnt/scratch2/users/40482774/lichess_data/corpus_optionD/YYYY-06/census_report.json` and the corresponding `preprocess_YYYY-06.sh` scripts. The reports also record 65,954,319 retained positions after the per-game 32-position cap.
- `VERIFIED FACT` — Population target exclusion is username-based and exact after strip/casefold normalization: `src/amadeus_counterpoint/data/target_exclusion.py`, `exclude_game_if_target_involved()`, driven by `configs/target_aliases.json`.
- `VERIFIED FACT` — The production population-training run metadata is `/mnt/scratch2/users/40482774/checkpoints/chessformer-79m/run_metadata_1788720450.json`: 231 shards, 1,000,000 steps, batch 512, AdamW learning rate `5e-5` with minimum `1e-5`, weight decay `1e-6`, warmup 1,000, cycle 50,000, gradient-norm cap 3.5, value coefficient 0.1, AMP enabled, checkpoint every 1,000 steps.
- `VERIFIED FACT` — `scripts/train.py` constructs one `ChessDataset` from the supplied shard directory and has no production train/validation split or validation-based checkpoint selection. **Stage-1 train/validation split: UNKNOWN / not implemented in the inspected production path.**
- `UNKNOWN` — The inspected scripts and reports do not document a scientific reason for choosing June or for combining five years. It is established that the five June snapshots were used; a rationale such as seasonal control or temporal coverage would be an inference, not a verified implementation fact.

### 2.2 Stage 2 Broadcast adaptation

- `VERIFIED FACT` — Source is the Kelvin2 target-free Broadcast manifest `/mnt/scratch2/users/40482774/repos/amadeus-broadcast-data/manifests/target_free_elite_adaptation.parquet`, with the raw corpus represented in `/mnt/scratch2/users/40482774/lichess_data/broadcast_raw_pgn/`.
- `VERIFIED FACT` — `scripts/build_broadcast_shards.py` uses the zero-target manifest, so any game in which either side resolves to a target is excluded from the adaptation source. This is stronger than merely excluding target-vs-target games.
- `VERIFIED FACT` — `/mnt/scratch2/users/40482774/lichess_data/broadcast_shard_build_report.json`: raw target-free games 1,174,793; retained train games 833,119; validation games 16,821; train positions 26,237,458; validation positions 530,237; total positions 26,767,695. Rejections were reported for nonstandard variants 8,802, custom FEN 2,126, missing/invalid Elo 244,097, bad results 55,417, and no moves 14,411.
- `VERIFIED FACT` — The split is deterministic: `sha256(game_id) mod 100 < 2` is validation (2%); no Elo-bin balancing is used in Stage 2. This is documented by `scripts/build_broadcast_shards.py` and the shard-build report.
- `VERIFIED FACT` — `/mnt/scratch2/users/40482774/checkpoints/chessformer-79m-broadcast/run_metadata_1788959957.json`: initialized from `/mnt/scratch2/users/40482774/checkpoints/chessformer-79m/step_01000000.pt`, 200,000 adaptation steps, batch 512, learning rate `1e-5`, minimum `2e-6`, weight decay `1e-6`, warmup 1,000, cycle 50,000, value coefficient 0.1, AMP enabled, checkpoint every 10,000.
- `VERIFIED FACT` — `/mnt/scratch2/users/40482774/lichess_data/broadcast_val_sweep_report.json` selects **step 200,000**: policy loss 1.1935612554, value loss 0.7717853368, total loss 1.2707397891, move-matching accuracy 0.5870506962. The exact checkpoint used later is `/mnt/scratch2/users/40482774/checkpoints/chessformer-79m-broadcast/step_00200000.pt`.

## 3. Target cohort selection and sealed counts

### 3.1 How the eight were selected

- `VERIFIED FACT` — The candidate pool in the Kelvin2 Broadcast-data workspace is all distinct `(name,FIDE ID)` pairs from the canonical corpus plus free-text name candidates. `/mnt/scratch2/users/40482774/repos/amadeus-broadcast-data/README.md` reports 177,436 distinct `(name,FIDE)` pairs.
- `VERIFIED FACT` — `scripts/target_resolution_candidates.py` has a hard-coded `TARGETS` list containing the eight final canonical names, generates fuzzy/FIDE/name-order candidates for manual review, and does not optimize total mutual games, minimum dyad coverage, a densest subgraph, or a pair-count threshold.
- `VERIFIED FACT` — The cohort was manually imposed/curated, then identity candidates were reviewed. The available code does not establish that the eight were selected by mutual-game maximization or any graph criterion.

### 3.2 Final-evaluator sealed cache

- `VERIFIED FACT` — The final evaluator cache is `/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/sealed_real_games_cache.json`. It contains `game_count = 1609`, all 28 dyads, and therefore a minimum dyad count of 25 and maximum of 131. The counts below are the exact deduplicated records actually available to the saved final evaluator, mapped through local `configs/broadcast_targets.json` IDs.

| dyad | deduplicated valid games |
|---|---:|
| Magnus–Wesley | 73 |
| Magnus–Levon | 29 |
| Magnus–Fabiano | 86 |
| Magnus–MVL | 76 |
| Magnus–Hikaru | 131 |
| Magnus–Ian | 76 |
| Magnus–Alireza | 93 |
| Wesley–Levon | 47 |
| Wesley–Fabiano | 73 |
| Wesley–MVL | 69 |
| Wesley–Hikaru | 61 |
| Wesley–Ian | 29 |
| Wesley–Alireza | 41 |
| Levon–Fabiano | 55 |
| Levon–MVL | 53 |
| Levon–Hikaru | 62 |
| Levon–Ian | 50 |
| Levon–Alireza | 31 |
| Fabiano–MVL | 48 |
| Fabiano–Hikaru | 66 |
| Fabiano–Ian | **25** |
| Fabiano–Alireza | 52 |
| MVL–Hikaru | 50 |
| MVL–Ian | 61 |
| MVL–Alireza | 41 |
| Hikaru–Ian | 33 |
| Hikaru–Alireza | 60 |
| Ian–Alireza | 38 |

- `VERIFIED FACT` — The cache was created by `scripts/compute_final_results.py`, `load_real_games_by_dyad()`, which applies `deduplicate_sealed_games()` to valid two-target records before writing the cache. The cache stores only the post-dedup data; it does **not** store `total_before`, invalid counts, or the deduplication report.
- `UNKNOWN` — The exact raw two-target count and exact pre-dedup valid count for the final cache’s local target configuration are not recoverable from the saved cache itself. The final manifest explicitly says those counts were not independently re-derived before evaluation.

### 3.3 Conflicting Kelvin2 manifest count

- `VERIFIED FACT` — The separate manifest workspace `/mnt/scratch2/users/40482774/repos/amadeus-broadcast-data/manifests/sealed_dyads/summary.json` reports **2,506** sealed manifest rows, minimum dyad 43 (Alireza–Levon), maximum 213 (Hikaru–Magnus), and the full 28-dyad table. Those counts differ materially from the final evaluator cache’s 1,609 / 25 / 131.
- `VERIFIED FACT` — The two workspaces use different target configuration scopes: the Kelvin2 `target_names.json` includes both `verified_name_strings` and `conservative_candidate_name_strings`, including aliases not present in local `configs/broadcast_targets.json` (for example `Offerspill-M. Carlsen`, `Magnus Øen Carlsen`, and conservative `Hikaru`). The local evaluator config includes `Maxime ASN Vachier Lagrave` but not all Kelvin2 conservative aliases.
- `INFERENCE` — The count disagreement is consistent with different alias tiers and/or corpus-manifest validity scopes. It must not be silently presented as one number. For the final metric JSONs, 1,609 is the provenance-closest number; 2,506 is the separate Kelvin2 manifest census number.

## 4. Player identities and aliases

### 4.1 Canonical cohort

`VERIFIED FACT` — Canonical names and FIDE IDs are the eight entries in `configs/broadcast_targets.json` and Kelvin2 `configs/target_names.json`:

| ID | canonical name | FIDE ID |
|---:|---|---:|
| 0 | Magnus Carlsen | 1503014 |
| 1 | Wesley So | 5202213 |
| 2 | Levon Aronian | 13300474 |
| 3 | Fabiano Caruana | 2020009 |
| 4 | Maxime Vachier-Lagrave | 623539 |
| 5 | Hikaru Nakamura | 2016192 |
| 6 | Ian Nepomniachtchi | 4168119 |
| 7 | Alireza Firouzja | 12573981 |

### 4.2 Broadcast aliases

`VERIFIED FACT` — Local production Broadcast matching is exact FIDE-ID first, then exact alias string, in `data/broadcast_targets.match_target()`. `canonical_name` is not directly matched. The local aliases in `configs/broadcast_targets.json` are:

- Magnus: `Carlsen, Magnus`; `Magnus Carlsen`; `MagnusCarlsen`; `Carlsen Magnus`; `Carlsen Magnus (NOR)`; `GM Magnus Carlsen`; `Carlsen Magnus  (NOR)`; `Carlsen Magnus  *) (NOR)`; `MagzyBogues`.
- Wesley: `So, Wesley`; `Wesley So`; `So Wesley (USA)`; `GM Wesley So`; `So Wesley`; `GMWSO`.
- Levon: `Aronian, Levon`; `Levon Aronian`; `Aronian Levon`; `Aronian Levon (ARM)`; `LevonAronian`; `Aronian Levon  (ARM)`; `GM Levon Aronian`; `Aronian Levon  (USA)`.
- Fabiano: `Caruana, Fabiano`; `Fabiano Caruana`; `Caruana Fabiano`; `Caruana Fabiano (USA)`; `Caruana Fabiano  (USA)`; `GM Fabiano Caruana`; `STL_Caruana`.
- MVL: `Vachier-Lagrave, Maxime`; `Maxime Vachier-Lagrave`; `Vachier-Lagrave Maxime (FRA)`; `Vachier-Lagrave Maxime  (FRA)`; `GM Maxime Vachier-Lagrave`; `Maxime ASN Vachier Lagrave`; `LyonBeast`; `M. Vachier-Lagrave`.
- Hikaru: `Nakamura, Hikaru`; `Hikaru Nakamura`; `Nakamura Hikaru (USA)`; `GM Hikaru Nakamura`; `Nakamura Hikaru  (USA)`; `Nakamura, Nakamura, Hikaru`; `GMHikaru`; `STL_Nakamura`.
- Ian: `Nepomniachtchi, Ian`; `Nepomniachtchi Ian (FID)`; `Ian Nepomniachtchi`; `GM Ian Nepomniachtchi`; `Ian_Nepomniachtchi`; `Nepomniachtchi Ian  (CFR)`; `lachesisQ`.
- Alireza: `Firouzja, Alireza`; `Alireza Firouzja`; `Firouzja Alireza (FRA)`; `GM Alireza Firouzja`; `Firouzja Alireza  (FRA)`; `FantasticStar`.

`VERIFIED FACT` — Kelvin2’s broader `configs/target_names.json` additionally records conservative candidates, including Magnus `Carlsen Magnus  *) (NOR)`, `Magnus Øen Carlsen`, `Offerspill-M. Carlsen`; MVL `Maxime ASN Vachier Lagrave`; Hikaru `Hikaru`; and no conservative candidates for Wesley, Levon, Fabiano, Ian, or Alireza. These broader entries were used by the Kelvin2 manifest workspace, not necessarily by the final local evaluator configuration.

### 4.3 Population-training usernames and excluded accounts

`VERIFIED FACT` — `configs/target_aliases.json` is a separate Lichess-username map used for Stage-1 population exclusion, not Broadcast name matching:

- Magnus: verified `DrNykterstein`, `DannyTheDonkey`; conservative `DrDrunkenstein`, `MagnusCarlsen`, `manwithavan`, `DrChampionstein`, `DrGrekenstein`, `STL_Carlsen`, `damnsaltythatsport`.
- Wesley: verified `Wesley_So`.
- Levon: conservative `Aronian_Levon`.
- Fabiano: conservative `Bombegranate`, `FabianoCaruanaa`, `Fabiano_Caruana26`.
- MVL: verified `avalongamemaster`; conservative `UnVieuxMonsieur`.
- Hikaru: verified `TSMFTXH`.
- Ian: verified `Ian_Nepomniachtchi`; conservative `Jepetto`.
- Alireza: verified `alireza2003`.

### 4.4 Quarantined/unresolved identities

- `VERIFIED FACT` — `configs/quarantined_broadcast_games.json` quarantines 12 local games opposite unresolved handle `pedestrian`: the seven games covered by Kelvin2’s `ambiguous_decoy_identities.json` plus five additional local games opposite `GMHikaru`.
- `VERIFIED FACT` — The same file quarantines six local `STL_So` games opposite `STL_Nakamura`. `STL_So` was not resolved as Wesley; it was conservatively quarantined because the naming pattern is suggestive but not primary-source confirmed.
- `VERIFIED FACT` — Kelvin2’s independent identity audit resolves `MagzyBogues` to Magnus, `FantasticStar` to Alireza, `polborta` to non-target Peter Svidler, and leaves `pedestrian` permanently unresolved. `STL_So` is a local extension of that quarantine, not a resolved identity.
- `VERIFIED FACT` — Quarantine is used by Broadcast personalization ingestion (`broadcast_ingest`/style-record building) to prevent uncertain one-target games entering training. `iter_sealed_dyad_games()` itself requires both sides to resolve to distinct verified target IDs, so unresolved `pedestrian`/`STL_So` records do not enter its sealed set.

## 5. Interaction sealing

- `VERIFIED FACT` — `data.broadcast_ingest.classify_game()` defines `target_vs_target` as both sides resolving to distinct target identities. Such games are discarded from personalization records. Any game with exactly one target is eligible for the target’s non-sealed style records, subject to the validity and quarantine filters.
- `VERIFIED FACT` — `data.sealed_dyads.iter_sealed_dyad_games()` retains only valid games with two distinct target identities. `build_manifests.py` in the Kelvin2 Broadcast-data workspace enumerates all `itertools.combinations(targets,2)`, hence all 28 dyads.
- `VERIFIED FACT` — Stage-1 target usernames are excluded by `target_exclusion.py`; Stage-2 Broadcast adaptation uses the zero-target manifest and therefore excludes any game involving a target.
- `VERIFIED FACT` — Actual personalization records are `/mnt/scratch2/users/40482774/lichess_data/style_records.json`. The saved build log reports 1,186,335 games scanned, 1,174,827 zero-target, 9,041 one-target, 2,449 target-vs-target, 18 quarantined, 2,119 invalid, and 6,922 valid one-target records written. The 2,449 target-vs-target count is after the local 18-game quarantine accounting and differs from the separate sibling census count of 2,506.
- `VERIFIED FACT` — The target-specific train/validation split is per player, game-level, 80/20, `split_seed=0`; phase balancing is then performed separately over opening/middlegame/endgame. This is `data/style_dataset.split_games_by_player()` and `build_style_datasets()`.
- `VERIFIED FACT` — M1 filters one player before splitting. M2 performs one joint all-player split, then partitions validation by player. `evaluation.identity_diagnostic` explicitly reconstructs these two different split procedures.
- `VERIFIED FACT` — The training records contain target-vs-outsider games only because all target-vs-target games are discarded upstream. Thus target A and target B do not train on their mutual games; in fact, the implemented style-record path excludes all target-target games, not merely the dyad currently being evaluated.
- `VERIFIED FACT` — The inspected protocol and manifests document sealed data as unavailable for training, personalization, hyperparameter/temperature/calibration/method selection, metric/opening-granularity selection, generated sample-count selection, and sampler selection. Nonsealed validation was used for checkpoint selection and sampler calibration. No sealed behavioral metric was used before the final evaluator was frozen; `PRE_EVAL_MANIFEST.md` records this freeze.
- `VERIFIED FACT` — Structural sealed information such as counts, IDs, parse integrity, and dyad coverage was inspected for data validation. The exact final cache stores 1,609 deduplicated valid games; pre-dedup raw/valid counts remain UNKNOWN as noted in Section 3.

## 6. Method 1 — policy-level personalization

- `VERIFIED FACT` — `models/personalized_chessformer.py`, `PersonalizedChessformer`: one trainable `z_player ∈ R^128` per player, initialized from that player’s nominal-Elo interpolation. It replaces the mover Elo embedding through `player_emb_override`; it is not additive. The opponent still uses ordinary Elo interpolation. The `player_elo` argument is accepted for signature compatibility but is unused for the mover.
- `VERIFIED FACT` — The entire base ChessFormer is frozen. `training/method1_trainer.py` trains only `wrapper.z_player`; the value output is not used.
- `VERIFIED FACT` — Loss is full legal-move policy cross-entropy over 4,352 actions after illegal masking. There is no candidate restriction and no value loss.
- `VERIFIED FACT` — Optimizer is AdamW on `z_player` only, learning rate `1e-3`, weight decay `1e-4`. Defaults are in `training/method1_trainer.py` and `scripts/train_method1_player.py`.
- `VERIFIED FACT` — M1 players train independently. `scripts/train_method1_player.py` filters to one player and invokes one trainer/checkpoint per player. Maximum epochs 100, patience 5, batch size 32, train fraction 0.8, split seed 0, balance seed 0.
- `VERIFIED FACT` — Saved logs show best validation epochs: player IDs 0–7 respectively 10, 3, 21, 15, 14, 4, 10, 10. The corresponding best validation losses are 1.1492, 1.0723, 1.1312, 1.1361, 1.0780, 1.1967, 1.0984, and 1.0841. Sources: `/mnt/scratch2/users/40482774/lichess_data/method1_player{0..7}_987124{2..9}.out`.
- `VERIFIED FACT` — Production M1 checkpoints are `/mnt/scratch2/users/40482774/checkpoints/method1/player_0.pt` through `player_7.pt`. The final generation manifest confirms these exact adapters and identity/Elo mappings.
- `VERIFIED FACT` — Final generation used representative Elo values 2855 (Magnus), 2769 (Wesley), 2756 (Levon), 2786 (Fabiano), 2751 (MVL), 2829 (Hikaru), 2778 (Ian), and 2767 (Alireza), from `/mnt/scratch2/users/40482774/lichess_data/representative_elos.json`.
- `UNSUPPORTED / TODO` — The repository references the Maia-3 encoding/augmentation convention in `encoding.py` and `data/dataset.py`, but the inspected M1 code, reports, and history do not establish that M1 was genuinely inspired by Maia-2. Do not state that relationship as fact without another source.

## 7. Method 2 — candidate-level personalization

### 7.1 Candidate construction and training/inference difference

- `VERIFIED FACT` — `models/style_scoring.py`, `score_candidates()`: frozen base produces full policy logits; illegal actions are set to `-inf`; `models/candidates.select_candidates()` selects raw top-`K` legal actions.
- `VERIFIED FACT` — Production `K=5`. At training/validation, the human move is appended as a sixth candidate only when it is absent from raw top-5; if already present, the extra slot is invalid padding. At inference, `target_index=None`, so the candidate set is exactly the raw top-5 and moves outside it receive zero probability.
- `VERIFIED FACT` — The saved Method-2 checkpoint is `/mnt/scratch2/users/40482774/checkpoints/method2/joint.pt`, with `k=5`, `style_dim=32`, and an eight-player ID map.

### 7.2 Style representation

- `VERIFIED FACT` — `models/style_cnn.py`: the current canonical 12 board planes are extracted from the trailing 12 channels of `[B,64,96]`; each candidate adds one from-square plane, one to-square plane, and four promotion planes, giving `[B,K',18,8,8]`.
- `VERIFIED FACT` — `MoveStyleCNN`: `Conv2d(18,32,3,padding=1)` → GELU → `Conv2d(32,64,3,padding=1)` → GELU → `Conv2d(64,64,3,padding=1)` → GELU → flatten 4096 → `Linear(4096,32)`. There is no pooling, normalization, dropout, residual block, or final activation.
- `VERIFIED FACT` — `models/player_style.py`, `PlayerStyleTable`: one learned `[32]` vector per player, shape `[8,32]` in the experiment.
- `VERIFIED FACT` — `StyleResidual`: normalizes candidate features and player vector separately, computes their dot product, and returns

  `r_i = s * sqrt(32) * (normalize(phi_i) · normalize(z_u))`,

  with trainable scalar `s` initialized to 0.17. The scaling is `sqrt(32)`, not `1/sqrt(32)`.
- `VERIFIED FACT` — Final candidate score is exactly

  `score_i = base_logit_i + s * sqrt(32) * (normalize(phi_i) · normalize(z_u))`.

  The base logits are the raw base-policy logits at the selected candidates; the residual is added before the candidate softmax.

### 7.3 Optimization

- `VERIFIED FACT` — Only `MoveStyleCNN`, `PlayerStyleTable`, and `StyleResidual` are trainable; the ChessFormer base and its value head are frozen/unused by `score_candidates()`.
- `VERIFIED FACT` — `training/style_trainer.py`: candidate-restricted cross-entropy using target append; AdamW over the CNN/table/residual parameters, learning rate `1e-3`, weight decay `1e-4`; maximum epochs 100, patience 5, batch size 32, joint split/balance seeds 0.
- `VERIFIED FACT` — All eight players are trained jointly in one Method-2 run. The log `/mnt/scratch2/users/40482774/lichess_data/method2_joint_9871250.out` selects epoch 4, validation loss 1.0658, raw top-5 coverage 0.9498, and stops after epoch 9 with patience 5.
- `VERIFIED FACT` — Final Method-2 production uses `/mnt/scratch2/users/40482774/checkpoints/method2/joint.pt`.

## 8. Production Stockfish/sampling rule

- `VERIFIED FACT` — `evaluation/generation/strength_guardrail.py` implements

  `score_i = log(p_behavior_i) - lambda * (cheap_loss_i / 100.0)`

  and

  `q_i = softmax(score_i)` over the retained candidate set.

- `VERIFIED FACT` — `cheap_loss_i = max(0, best_candidate_eval - candidate_eval_i)`, where evaluations are raw centipawns from the mover’s perspective, at depth 8, relative to the best of the retained candidates, not the best move in the entire legal move set.
- `VERIFIED FACT` — Final production config is `StrengthGuardrailConfig(lam=2.0, k=5, cheap_depth=8, threads=1, hash_mb=128)`. Stockfish is version 19 (`sf_19`). The final production job logs and `/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/PRE_EVAL_MANIFEST.md` confirm the literal runtime values.
- `VERIFIED FACT` — `MATE_SCORE = 100000` is used when converting mate scores to centipawn-like integers. Losses are not capped; the only cap-like behavior is `max(0, best-eval)`. Terminal positions are handled without sending them to Stockfish.
- `VERIFIED FACT` — After softmax, probabilities are floored at `1e-12` and renormalized to prevent numerical zero. Sampling is stochastic `torch.multinomial`, temperature is 1.0, and each game has a deterministic `torch.Generator` seeded from `derive_seed(root_seed, dyad, condition, orientation, game_index)`.
- `VERIFIED FACT` — `p_behavior` differs by agent: generic uses the frozen base policy’s own top-5 probabilities; M1 uses the player wrapper’s top-5 probabilities; M2 uses raw base top-5 candidates but the existing style-reranked candidate softmax probabilities. The same strength reweighting is applied to GG, AG, GB, and AB and to both methods.
- `VERIFIED FACT` — The final guarded implementation uses a persistent pool of 16 Stockfish worker processes, Python `spawn`, one engine per worker, `Threads=1`, `Hash=128`, blocking order-preserving `pool.map`; engines are reused and are not restarted per candidate/position. The final integration report documents that per-position `ucinewgame` isolation was investigated but not adopted.
- `VERIFIED FACT` — Lambda selection used only nonsealed held-out data in `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/final_sampler_freeze/`. The rule selected the smallest tested lambda whose pooled DEV oracle CPL was no greater than the human reference and whose late-band criterion passed; lambda 2.0 passed, smaller tested values did not. No sealed data was touched.
- `VERIFIED FACT` — The final evaluation’s `production_manifest` and `PRE_EVAL_MANIFEST` confirm that all 448 final synthetic shards used the guarded sampler. The earlier `/mnt/scratch2/users/40482774/paper_artifacts/primary_generation_2026-09-11/MANIFEST.md` describes a separate protocol-version-1 primary run whose scripts did not pass `--use-strength-guardrail`; do not conflate those two artifact roots.

## 9. Individual behavioral validation

The following are the meaningful nonsealed individual-personalization validations found in code/reports. They are reported as validation provenance, not as scientific interpretation.

### 9.1 Cross-player identity diagnostic

- `Scientific question` — Does each player’s learned representation score that player’s held-out nonsealed moves better than generic or another player’s representation?
- `Dataset/split` — `style_records.json`, reconstructed exactly from the Method-1 and Method-2 training split procedures: game-level 80/20, split seed 0, balance seed 0. Method 1 filters before splitting; Method 2 splits jointly.
- `Sealed status` — Nonsealed one-target records only. The archive manifest explicitly says no sealed dyad data or sealed manifest was read.
- `Metric/protocol` — M1 legal full-action NLL and top-1 accuracy over 4,352 actions; M2 candidate-matched oracle-expanded K(+1) NLL/ranking plus separate raw-top-K coverage and deployable top-1 accuracy. M2 candidate-matched values are not directly comparable to M1 full-action NLL.
- `Exact result` — `/mnt/scratch2/users/40482774/paper_artifacts/personalization_identity_diagnostic_2026-09-11/personalization_identity_diagnostic.json`: both methods rank the correct representation first for all 8/8 players; mean correct rank 1.0. M1 mean generic-minus-correct NLL 0.0068023774 and wrong-minus-correct NLL 0.0043647524. M2 corresponding values 0.0092778045 and 0.0204489792; raw top-5 coverage 0.9498 in the archived headline. Per-player M2 coverage is in the JSON.
- `Method-selection role` — Post-hoc diagnostic of selected checkpoints; it did not select the method or use sealed interactions.

### 9.2 Personalization training validation

- `Scientific question` — Which checkpoint epoch minimizes held-out nonsealed move loss while training the adapter/style stack?
- `Protocol` — M1 validation is full legal-policy CE; M2 validation is target-appended candidate CE and separately reports raw top-5 coverage. Early stopping patience is 5.
- `Exact result` — M1 best epochs and losses are listed in Section 6; M2 best epoch 4, loss 1.0658, coverage 0.9498.
- `Method-selection role` — This is checkpoint selection on nonsealed data, not sealed evaluation.

### 9.3 Candidate coverage and sampler validation

- `Scientific question` — Does top-5 candidate restriction contain a usable move, and does the strength guardrail avoid catastrophic low-quality samples while retaining player signal?
- `Dataset/split` — The final sampler-freeze report uses 480 nonsealed positions, split by whole game into 240 DEV and 240 TEST, seed 20260914, stratified across 8 players and depth bands. Lambda is selected on DEV and evaluated once on untouched TEST.
- `Metrics/protocol` — Raw top-K coverage/best-in-K oracle CPL; human-move reference CPL; lambda sweep; M1 cross-player TV before/after; closed-loop small-game checks. No sealed data.
- `Exact results` — DEV pooled mean CPL at lambda 2.0 is 7.90 versus human reference 9.92; TEST frozen lambda-2.0 mean CPL 7.12, median 4.29, with zero TEST positions at or above 50/100/200/300cp in the saved table. Method-2 raw top-5 coverage in the identity diagnostic is 0.9498 overall. The report also records M1 mean pairwise TV 0.0329 before and 0.0254 after guardrail on 75 shared nonsealed positions.
- `Method-selection role` — Lambda/sampler selection and guardrail validation only; not a model-weight or method-selection result.

### 9.4 Generation-strength and integration checks

- `Scientific question` — Does the unguarded/guarded rollout regime produce legal, nonpathological games and preserve basic style diversity?
- `Artifacts` — `/mnt/scratch2/users/40482774/paper_artifacts/generation_strength_diagnostic_2026-09-12/SUMMARY.md`, `/mnt/scratch2/users/40482774/paper_artifacts/rollout_quality_exploratory_2026-09-12/final_sampler_freeze/PRODUCTION_INTEGRATION_REPORT.md`, and `/mnt/scratch2/users/40482774/paper_artifacts/guarded_m1_complete_preliminary_2026-09-14/`.
- `Exact results` — The earlier unguarded diagnostic reports synthetic median capped ACPL 64.88 versus held-out real 18.24 on a small 80-side comparison; this is a diagnostic artifact, not final evaluation. The guarded integration report records 8/8 smoke cells with zero illegal moves and zero censored games in the smoke sample, and the final pre-evaluation manifest records 224 complete cells per method at 5,000 rows each.
- `Method-selection role` — Quality/engineering validation. It did not use sealed behavioral metrics.

### 9.5 Main-paper versus appendix candidates

`INFERENCE` — Based on artifact role, the following classification is the least misleading one; it is not a writing recommendation about scientific importance.

**A. Likely main-paper validation facts**

- Nonsealed one-target train/validation separation and exact 80/20 game-level split.
- M1/M2 checkpoint-selection losses and M2 raw-top-5 coverage if the paper needs to establish that the learned representations were used rather than degenerate.
- Cross-player identity result: 8/8 correct representations rank first, with the explicit warning that M2 uses a candidate-matched metric.
- Guarded sampler selection was frozen on nonsealed DEV and checked on untouched nonsealed TEST, if the final paper describes the guardrail.

**B. Likely appendix-only validation facts**

- Full 8×8 NLL matrices and per-player accuracy matrices.
- Candidate-level CNN/coverage details, per-depth-band CPL, lambda sweep table, M1 TV-before/after table, and closed-loop sampler checks.
- Small generation-strength diagnostics, Stockfish-history dependence, worker-pool throughput, and smoke-generation details.
- The earlier unguarded-generation ACPL diagnostic and preliminary guarded M1 reports.

`UNKNOWN` — No inspected artifact identifies a formally named “main-paper validation subset” versus an “appendix subset”; the labels above are artifact-role classifications, not repository metadata.

## 10. Compositional generation protocol

- `VERIFIED FACT` — `evaluation/generation/experiment.py` defines 8 players, A/B as the lower/higher player ID in each unordered dyad, 28 dyads, conditions `GG`, `AG`, `GB`, `AB`, and orientations `A_WHITE`, `B_WHITE`.
- `VERIFIED FACT` — Semantics are: GG generic A/generic B; AG personalized A/generic B; GB generic A/personalized B; AB personalized A/personalized B. “Generic” still uses each player’s representative Elo; it means no player-specific representation.
- `VERIFIED FACT` — Each dyad/condition/orientation cell has 5,000 games. There are 224 cells per method and 1,120,000 games per method: `28 × 4 × 2 × 5000`. Both methods together contain 2,240,000 synthetic games.
- `VERIFIED FACT` — `/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/PRE_EVAL_MANIFEST.md` exhaustively verifies 224/224 cells and 5,000 rows per cell for both methods under `/mnt/scratch2/users/40482774/artifacts/synthetic_guarded_2026-09-13/`.
- `VERIFIED FACT` — Final generation root seed is 20260911, internal generation batch size 128, temperature metadata 1.0, and final protocol version 2. `evaluation/seeding.derive_seed()` hashes root seed, canonical dyad, condition, orientation, and game index with BLAKE2b-64, then seeds a per-game CPU `torch.Generator`.
- `VERIFIED FACT` — `generation.single.py`, `method1_personalized.py`, `method2_personalized.py`, and the guarded loop stop at normal `python-chess` outcomes with `claim_draw=True` (including claimable threefold/fifty-move draws) or after 500 plies. A 500-ply game is `censored=True`, `result=None`, not an adjudicated draw. Legal masks prevent illegal sampled moves; engine failure is intended to raise rather than silently fall back.
- `VERIFIED FACT` — Final output directories are `/mnt/scratch2/users/40482774/artifacts/synthetic_guarded_2026-09-13/method1/` and `/mnt/scratch2/users/40482774/artifacts/synthetic_guarded_2026-09-13/method2/`, with one parquet shard per dyad/condition/orientation. The final manifest records base, M1, M2 checkpoint identities and code commit `980405aa77cf7a59c68c18c4a2ff157a5545745a`.

## 11. Final sealed evaluation

### 11.1 WDL metric

- `VERIFIED FACT` — `evaluation/metrics/wdl.py`: each completed game is mapped to `A_WIN`, `DRAW`, or `B_WIN` using the dyad orientation. Draws remain draws; Black/White result strings are converted to the A/B perspective. Censored games are excluded from the WDL denominator and reported separately; they are not counted as draws.
- `VERIFIED FACT` — Total variation is `0.5 * sum(abs(p_i - q_i))` over the union of supports.
- `VERIFIED FACT` — WDL distributions are computed separately for A_WHITE and B_WHITE, then the dyad metric is the unpooled 50/50 mean of the two orientation distances.

### 11.2 Opening-family metric

- `VERIFIED FACT` — `evaluation/metrics/openings.py` loads a pinned TSV with columns `eco,name,pgn,uci,epd`; the final artifact is `/mnt/scratch2/users/40482774/lichess_data/chess_openings_all.tsv`, 3,811 rows, commit identity `4b8622759e7ae6f93f011cc6c83a3823401ab45e`.
- `VERIFIED FACT` — Each game is replayed from the standard start position in UCI. The deepest matching recognized EPD is selected, with transposition-safe position-key matching; the opening family is the part before the first colon in the full opening name. Games with no recognized opening are `Unknown` and remain an explicit distribution category.
- `VERIFIED FACT` — Opening TV uses the same TV equation, separately by orientation, then an exact 50/50 orientation mean.

### 11.3 Dyad aggregation and bootstrap

- `VERIFIED FACT` — `evaluation/aggregate.py` gives each of the 28 dyads equal weight in the final aggregate, regardless of the number of real sealed games in that dyad. It does not pool all real games across dyads.
- `VERIFIED FACT` — `evaluation/bootstrap.py` / `bootstrap_wiring.py`: one dyad and one metric at a time; each replicate resamples whole real games with replacement separately within each orientation, preserving that orientation’s real sample size. The same resampled real dyad is passed to GG/AG/GB/AB in the replicate; synthetic games are fixed. The bootstrap unit is a whole real game, not a move or position.
- `VERIFIED FACT` — Default and final artifact count is 10,000 replicates per dyad/metric. Intervals are simple percentile intervals using the implementation’s sorted-index rule at the 2.5% and 97.5% positions. Contrasts include GG−AB, AG−AB, and GB−AB.
- `VERIFIED FACT` — `scripts/compute_final_results.py` has `--bootstrap-seed` default 0 and uses the same seed for each dyad/metric call. The final JSON provenance records the replicate count but not the bootstrap seed; **the exact runtime seed is therefore not independently proven by the saved JSON, although the code default is 0**.

### 11.4 Saved aggregate results

`VERIFIED FACT` — The requested paths exist and contain these exact equal-dyad-weight aggregates:

| method / condition | WDL TV | opening TV |
|---|---:|---:|
| M1 GG | 0.1641713794 | 0.5544333952 |
| M1 AG | 0.1577525346 | 0.5474059232 |
| M1 GB | 0.1561414331 | 0.5447647159 |
| M1 AB | 0.1501476968 | 0.5369726580 |
| M2 GG | 0.1631661273 | 0.5547107189 |
| M2 AG | 0.1625818997 | 0.5013704617 |
| M2 GB | 0.1625892836 | 0.4883428362 |
| M2 AB | 0.1634075139 | 0.4372020003 |

Sources: `/mnt/scratch2/users/40482774/paper_artifacts/final_sealed_eval_2026-09-16/method1_results.json` and `method2_results.json`. Rounded to four decimals, these are the numbers in the question. Their saved provenance records 5,000 games/orientation, root seed 20260911, protocol version 2, guarded-generation code commit `980405aa...`, base checkpoint `chessformer-79m-broadcast/step_00200000.pt`, and 10,000 bootstrap replicates.

## 12. Classification of the current rough writing claims

| rough claim | classification | audit basis |
|---|---|---|
| “June games from 2017 to 2022” | **PARTIALLY CORRECT** | Verified production Stage 1 uses June 2017–2021. June 2022 appears as a separate evaluation/holdout artifact, not verified Stage-1 training. |
| “22 Elo buckets” | **VERIFIED** | `preprocess.balance_by_elo()` has 22 bins. |
| “maximum 10 games from each bucket” | **PARTIALLY CORRECT** | It is max 10 per bin **per sequential 20,000-record chunk**, not globally per source month or dataset. |
| “players were selected based on mutual-game coverage” | **INCORRECT / UNSUPPORTED** | The target list is hard-coded and manually reviewed; no coverage-maximizing algorithm is present. |
| “M1 replaces the player Elo vector” | **VERIFIED** | `PersonalizedChessformer.forward()` replaces the mover’s Elo-interpolated vector with `z_player`; opponent Elo remains. |
| “M2 analyzes style of the top-K moves” | **VERIFIED, with qualification** | M2 constructs candidate features for raw base top-5 moves; training appends the target if absent, inference does not. |
| “M2 player vector is dot-producted with candidate representation” | **VERIFIED, with qualification** | Normalized 32-dimensional vectors are dot-producted inside `StyleResidual`; the residual is scaled and added to base candidate logits. |
| “scaled by sqrt(32)” | **VERIFIED** | `StyleResidual.forward()` multiplies by `s * math.sqrt(style_dim)`, with `style_dim=32`. |
| “M1/M2 use a Stockfish depth-8 quality-weighted sampler” | **VERIFIED for final guarded production** | Final artifacts and live logs confirm Stockfish 19 depth 8, K=5, lambda=2.0 for all methods/conditions. Earlier primary-generation artifacts were a separate unguarded run. |
| “lambda is minimal to preserve player style” | **PARTIALLY CORRECT** | Lambda 2.0 is the smallest tested value passing the saved human-comparable DEV criteria; the report measures a nonzero but reduced M1 signal after guarding. It was not selected solely by a formal style-preservation objective. |
| “base model is retrained because public ChessFormer may contain target-player data” | **UNSUPPORTED / TODO** | The audit verifies a population pretraining run and Broadcast adaptation, but found no implementation/report establishing that this was the documented reason for retraining. |

# THINGS KARL SHOULD NOT WRITE YET

- Do not write that the cohort was algorithmically selected by mutual-game coverage, densest-subgraph optimization, or a minimum dyad threshold.
- Do not write a single sealed-game total without naming the source/configuration. The final cache has 1,609 post-dedup valid records; the separate Kelvin2 manifest census has 2,506 rows.
- Do not write that Stage 1 used June 2022 unless the separate holdout artifact is intentionally being described as training, which the inspected launch evidence does not support.
- Do not write “10 games per Elo bucket” without the 20,000-record chunk qualifier.
- Do not claim a Maia-2 origin for M1. The code supports a Maia-3-style encoding convention, not a verified Maia-2 inspiration claim.
- Do not describe M2 validation NLL as a full 4,352-action NLL; it is candidate-matched oracle-expanded K(+1) scoring.
- Do not say the Stockfish guardrail merely “tunes lambda to preserve style.” Its saved selection rule is a nonsealed human-comparable-strength criterion, and the report documents measurable signal compression.
- Do not state an exact final bootstrap seed from the JSON artifacts; the code default is 0 but the saved provenance omits the runtime seed.
- Do not claim the exact raw/pre-dedup sealed count for the final evaluator; that count was not retained in the cache or final JSON.

# FACTS SAFE TO FORMALIZE

- Production base input is 96 canonicalized history features plus 128-dimensional mover and opponent Elo embeddings, for 352 features per square.
- Production ChessFormer constants are `d_model=1024`, 8 blocks, 32 heads, head dimension 32, FFN dimension 2048, and 77,849,091 parameters by module-shape count.
- Policy output is 4,352 logits: 4,096 ordinary moves plus 256 promotion moves.
- Base training used June snapshots 2017–2021 with 22 Elo bins and chunk-local 10-per-bin balancing; exact documented scientific rationale for June is not available.
- Broadcast adaptation used zero-target games, a deterministic 98/2 game split, 200,000 steps, and production checkpoint `step_00200000.pt`.
- M1 is one independently trained 128-dimensional mover-embedding replacement per player with a frozen base and full legal-policy CE.
- M2 is a jointly trained 32-dimensional candidate style residual over raw top-5 base-policy legal candidates, with target append only during training/validation.
- Final generation used 28 dyads, GG/AG/GB/AB, two orientations, 5,000 games per cell, 1,120,000 games per method, root seed 20260911, and the guarded sampler.
- Final guarded sampler equation, K=5, depth=8, lambda=2.0, Stockfish 19, temperature 1.0, seeded stochastic multinomial sampling, and persistent 16-worker engine pool are directly supported.
- Final evaluation uses WDL TV and opening-family TV, equal 50/50 orientation weighting, equal dyad weighting, fixed synthetic samples, whole-real-game paired bootstrap, 10,000 replicates, and percentile intervals.
- The saved M1/M2 aggregate values in Section 11.4 are directly verified from the two final JSON artifacts.

# REMAINING OPEN QUESTIONS

- Exact raw two-target count, exact pre-dedup valid count, and exact duplicate-removal report for the final local evaluator cache.
- Which exact launch-time target-alias configuration generated every final sealed cache row, beyond the cache hash and local config path; the local and Kelvin2 sibling configurations are not identical.
- The scientific rationale for using June snapshots and multiple Stage-1 years.
- Whether the base was retrained specifically because a public ChessFormer checkpoint might contain target-player data; no direct source for that rationale was found.
- The exact bootstrap seed used by the final Kelvin2 commands; source default is 0, but final JSON provenance does not record it.
- A repository-native, formally named separation between “main-paper” and “appendix” validation facts.
- Any verified historical statement that M1 was inspired by Maia-2 rather than sharing only Maia-3-style encoding conventions.
