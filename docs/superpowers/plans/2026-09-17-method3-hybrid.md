# Method 3 Hybrid Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use $subagent-driven-development (recommended) or $executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generation-only M3 hybrid that uses loaded M1 personalized base logits before the unchanged M2 top-K/style/guarded sampler, then validate it without launching jobs.

**Architecture:** Extend M2 inference APIs with optional personalized base modules defaulting to `None`, preserving M2 behavior. A new experiment entry point routes GG through the generic path, AG/GB through one M1 wrapper plus the existing M2 stack, and AB through two wrappers plus the same stack. A new M3 production CLI loads the frozen base, eight M1 vectors, and M2 joint checkpoint and writes a distinct `method3_hybrid` tree.

**Tech Stack:** Python 3.12, PyTorch, python-chess, PyArrow/Parquet, pytest, existing Stockfish guardrail and SLURM cell-array convention.

## Global Constraints

- Do not regenerate or modify M1/M2 games, checkpoints, or sealed evaluation artifacts.
- Do not add an M3 training script, optimizer, new embedding, new hyperparameter, or protocol change.
- Keep `K=5`, the existing M2 style/reranking code, sampler, Stockfish settings, and metadata schema.
- M3 personalized base calls must use the existing `PersonalizedChessformer` override before legal masking and top-K selection.
- Use a distinct `method3_hybrid` output root and seed namespace.
- Do not submit or queue Kelvin2 jobs in this implementation plan; stop for a pre-launch user approval after smoke tests.

---

### Task 1: Add failing M3 routing and equivalence tests

**Files:**
- Create: `tests/amadeus_counterpoint/test_method3_hybrid.py`
- Read-only reference: `src/amadeus_counterpoint/evaluation/generation/method2_personalized.py`
- Read-only reference: `src/amadeus_counterpoint/evaluation/generation/guarded_generation.py`

**Interfaces:**
- Tests will require `play_games_method3`-equivalent routing through the existing M2 functions and a future `generate_method3_cell_batched`.
- Tests will use the existing fixed-logit and fake-engine patterns from `test_guarded_generation.py`.

- [ ] **Step 1: Write focused failing tests.** Cover: optional personalized base routing on unguarded single-sided and AB M2 functions; the same routing on guarded single-sided and AB functions; direct M1 wrapper logits/candidate indices; changed top-K; M2 default preservation; legal GG/AG/GB/AB short games; and M3 cell metadata/seed namespace.

- [ ] **Step 2: Run the new tests to verify RED.**

Run: `pytest -q tests/amadeus_counterpoint/test_method3_hybrid.py`

Expected: collection or assertion failures because the new optional arguments and `generate_method3_cell_batched` do not yet exist.

### Task 2: Implement additive M2 routing hooks

**Files:**
- Modify: `src/amadeus_counterpoint/evaluation/generation/method2_personalized.py`
- Modify: `src/amadeus_counterpoint/evaluation/generation/guarded_generation.py`

**Interfaces:**
- `play_games_method2(..., personalized_base=None)` and its guarded counterpart use `personalized_base` only on the personalized player’s turns.
- `play_games_method2_ab(..., personalized_base_a=None, personalized_base_b=None)` and its guarded counterpart select the wrapper matching the mover’s side.
- All new parameters default to `None`, preserving every existing M2 call and output path.

- [ ] **Step 1: Add the optional parameters and choose `candidate_base` only inside personalized branches.** Generic branches continue to call the original `base`; AB chooses `personalized_base_a` or `_b` by `a_color`.

- [ ] **Step 2: Run the focused routing/equivalence tests.**

Run: `pytest -q tests/amadeus_counterpoint/test_method3_hybrid.py`

Expected: the routing and direct equivalence tests pass; experiment/CLI tests remain red until Task 3.

- [ ] **Step 3: Run existing M2 and guarded tests.**

Run: `pytest -q tests/amadeus_counterpoint/test_method2_generation.py tests/amadeus_counterpoint/test_guarded_generation.py tests/amadeus_counterpoint/test_experiment.py`

Expected: all existing tests pass, demonstrating default M2 behavior is unchanged.

### Task 3: Add M3 experiment orchestration and provenance

**Files:**
- Modify: `src/amadeus_counterpoint/evaluation/generation/experiment.py`
- Modify: `tests/amadeus_counterpoint/test_method3_hybrid.py`

**Interfaces:**
- Add `generate_method3_cell_batched(wrapper_a, wrapper_b, base, cnn, table, residual, a_color, condition, elo_a, elo_b, k, dyad, n_games, root_seed, checkpoint_identity, chunk_size=None, guardrail=None, engine_pool=None) -> list[dict]`.
- Derive seeds with `dyad=f"method3_hybrid__{dyad}"`.
- Route GG through `batch.play_games` or `guarded_generation.play_games_guarded`; AG/GB/AB call existing M2 batched/guarded functions with the appropriate M1 wrappers.
- Keep the existing `GAME_SCHEMA` fields and representation identities (`global` or player IDs).

- [ ] **Step 1: Add a failing test for M3 cell count, seed namespace, representation identities, orientation, and all four conditions.**

- [ ] **Step 2: Run that test and confirm it fails because the function is absent.**

Run: `pytest -q tests/amadeus_counterpoint/test_method3_hybrid.py -k cell`

- [ ] **Step 3: Implement the M3 cell function by mirroring only the existing M2 orchestration and passing the new optional wrapper arguments.** Do not alter M1/M2 functions’ condition semantics or the artifact schema.

- [ ] **Step 4: Run the M3 test file and existing experiment/chunking tests.**

Run: `pytest -q tests/amadeus_counterpoint/test_method3_hybrid.py tests/amadeus_counterpoint/test_experiment.py tests/amadeus_counterpoint/test_generation_chunking.py`

Expected: all pass.

### Task 4: Add the generation-only M3 production CLI

**Files:**
- Create: `scripts/generate_method3_production.py`
- Modify: `tests/amadeus_counterpoint/test_method3_hybrid.py`

**Interfaces:**
- CLI accepts the existing M2 generation arguments plus `--m1-player-checkpoints` as a JSON mapping of player IDs to checkpoint paths and `--m1-checkpoint-identity-prefix` (or equivalent explicit identity mapping), and writes `args.output_root / "method3_hybrid"`.
- Loads the base with the exact M2 architecture, loads M1 wrappers with `load_method1_wrapper_for_generation`, freezes all wrapper parameters for inference, and loads M2 with `load_method2_style_stack_for_generation`.
- Validates the M2 checkpoint style dimension and frozen `K=5`; records base/M1/M2 checkpoint identities, M3 method identifier, guardrail/sampler settings, protocol version, code commit, and root seed in Parquet metadata.
- Supports the same dyad/orientation/condition filters and immutable cell skipping as the M2 script.

- [ ] **Step 1: Add a failing CLI test using tiny checkpoints and monkeypatched generation.** Assert the new script has no training/optimizer path, loads all requested checkpoints, writes under `method3_hybrid`, and passes the configured method identity/metadata.

- [ ] **Step 2: Run the CLI test to confirm RED.**

Run: `pytest -q tests/amadeus_counterpoint/test_method3_hybrid.py -k production`

- [ ] **Step 3: Implement the CLI with the existing M2 architecture constants and guardrail setup.** Keep output separate from `/mnt/scratch2/users/40482774/artifacts/synthetic_guarded_2026-09-13`.

- [ ] **Step 4: Run the CLI test and the production filtering tests.**

Run: `pytest -q tests/amadeus_counterpoint/test_method3_hybrid.py tests/amadeus_counterpoint/test_generate_production_filtering.py`

Expected: all pass.

### Task 5: Full verification and pre-launch audit

**Files:**
- Inspect only: all changed files, current main worktree, Kelvin2 paths/logs.

- [ ] **Step 1: Run formatting/lint for changed Python files.**

Run: `ruff check src/amadeus_counterpoint/evaluation/generation/method2_personalized.py src/amadeus_counterpoint/evaluation/generation/guarded_generation.py src/amadeus_counterpoint/evaluation/generation/experiment.py scripts/generate_method3_production.py tests/amadeus_counterpoint/test_method3_hybrid.py`

- [ ] **Step 2: Run the focused M3 and existing generation/checkpoint suites.**

Run: `pytest -q tests/amadeus_counterpoint/test_method3_hybrid.py tests/amadeus_counterpoint/test_method2_generation.py tests/amadeus_counterpoint/test_guarded_generation.py tests/amadeus_counterpoint/test_experiment.py tests/amadeus_counterpoint/test_generation_chunking.py tests/amadeus_counterpoint/test_checkpoint_loading.py`

- [ ] **Step 3: Inspect the diff and verify no M1/M2 production files, checkpoints, or Kelvin2 output roots changed.**

Run: `git diff main...HEAD --stat; git diff main...HEAD --check; git status --short`

- [ ] **Step 4: Run read-only Kelvin2 checks for exact checkpoint paths, existing M2 throughput logs, output-root availability, and absence of M3 jobs.** Do not invoke `sbatch`, `srun`, or a full generation command.

- [ ] **Step 5: Report files changed, exact inference path, checkpoints, smoke-test evidence, output directory, expected 224 cells/1,120,000 games, runtime estimate from existing M2 throughput, and the launch command. Stop for explicit launch approval.**
