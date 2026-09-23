# Method 3 Hybrid Generation Design

## Goal

Add a generation-only exploratory Method 3 (M3) that composes each existing
M1 personalized base policy with the existing frozen M2 candidate reranker and
sampling/Stockfish guard, without changing M1/M2 checkpoints, training, or
existing artifacts.

## Existing behavior verified

- `PersonalizedChessformer.forward()` replaces only the mover Elo embedding
  with its loaded `z_player`; the opponent Elo embedding remains the ordinary
  interpolated Elo representation.
- M2 `score_candidates()` calls its supplied base-like module first, masks
  illegal policy entries, selects legal top-K candidates (`K=5` in the frozen
  production checkpoint), and then applies the existing CNN/style-table/
  residual reranker.
- M2 generic turns call the plain shared base directly. Personalized turns
  call `score_candidates()`.
- The guarded path uses the same supplied base logits/candidate machinery
  before the frozen Stockfish depth-8 strength reweighting sampler.

## Recommended architecture

Extend the existing M2 generation functions with optional personalized-base
arguments whose defaults are `None`. The default path remains unchanged. For
M3, pass the already-loaded player-specific `PersonalizedChessformer` as the
base-like module only on that player's personalized M2 turns; generic turns
continue to use the shared plain base. This reuses the existing M2 candidate,
reranking, batching, and guarded-generation code instead of copying a second
pipeline.

The M3 production runner is a new generation-only script. It loads one frozen
base, the eight existing M1 player checkpoints as wrappers around that base,
and the existing M2 joint style checkpoint. It writes only to a new
`method3_hybrid` output directory with explicit M3 provenance metadata. No
optimizer or training loop is imported or invoked.

M3 uses a distinct seed namespace (`method3_hybrid__<dyad>`) and method
identifier. Although M3 GG uses the same generic inference semantics as M2 GG,
it is not silently substituted with the M2 artifact; separate M3 provenance
and seeds keep the exploratory corpus auditable. Reuse of M2 GG is not part of
this change.

## Files and interfaces

- Modify `evaluation/generation/method2_personalized.py`: add optional
  personalized base arguments to single/batched M2 inference functions; use
  them only for personalized turns.
- Modify `evaluation/generation/guarded_generation.py`: add the corresponding
  optional arguments to guarded M2 entry points; default behavior is identical.
- Modify `evaluation/generation/experiment.py`: add
  `generate_method3_cell_batched()` using the existing M2 generators, M1
  wrappers, M3 seed namespace, and current artifact schema.
- Create `scripts/generate_method3_production.py`: generation-only CLI with
  the exact M2 production settings, M1 checkpoint mapping, M2 checkpoint,
  guardrail options, cell filtering, resumability, and M3 provenance.
- Create M3 tests covering generic routing, M1 logits/candidate equivalence,
  M2 reranking preservation, changed candidate sets, frozen inference, all
  four conditions, legal moves, orientation, attribution, and cell metadata.

## Validation gates

1. Run the new focused tests in RED before production code is added.
2. Run the focused tests GREEN, then the existing generation/checkpoint test
   suite and lint on changed files.
3. Use deterministic fixed-logit smoke tests to show generic M3 routing equals
   M2, personalized M3 pre-rerank candidates equal direct M1 output, the M2
   reranker is unchanged, and M1 changes at least some top-K sets.
4. Run a handful of GG/AG/GB/AB short games through the unguarded and guarded
   paths with fake engine support; verify legal replay and correct metadata.
5. Before any Kelvin2 submission, inspect the final diff, checkpoint paths,
   output path, and command. Submit nothing until the pre-launch report is
   shown and the user explicitly approves the launch.

## Non-goals

No M3 training script, new embedding, parameter optimization, M1/M2 checkpoint
changes, M2 protocol changes, sampler changes, Stockfish setting changes,
sealed evaluation, or changes to existing M1/M2 artifacts.
