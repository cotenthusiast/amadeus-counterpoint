# Method-3 Hybrid Launch Provenance

`m3_provenance.json` is a rescued untracked file from a local git worktree
(`amadeus-counterpoint-m3`, branch `method3-hybrid`) that was used to stage
the Kelvin2 SLURM submission for the M3 hybrid production generation run.
It was never committed to any branch, so it is preserved here rather than
lost when the worktree/branch were cleaned up.

## What it shows

Checkpoint paths, generation config (28 dyads, 4 conditions, 2
orientations, 5,000 games/cell, K=5, root seed 20260911), the frozen
Stockfish strength-guardrail config (lambda 2.0, depth 8), and the SLURM
array/resource config (`scripts/submit_method3_guarded_array.sh`,
224-task array, `k2-gpu-a100mig` partition) used to launch M3 generation
on Kelvin2.

## What it does NOT resolve

Its `commit_hash` field is `51ff259963d1eef015203c2b3da6ed5f3444df3b` —
the commit this repo actually merged (see the `merge: bring in exploratory
Method-3 hybrid generation` commit). This does **not** match the
`code_commit` recorded in the final saved results
(`paper_results/source/method3_final_complete_results.json`'s
provenance: `5b56fb4695fd20faede241a6942f955b3713556d`), which does not
correspond to any commit in this repository's history.

Given this file's own `commit_hash` field and the branch's own commit
history (`chore: derive method3 provenance commit at launch`,
`chore: isolate Kelvin2 method3 checkout`), the most likely explanation is
that `5b56fb46...` was a commit hash computed on an isolated Kelvin2-side
checkout (`remote_worktree` below) at actual launch time, distinct from
any commit ever present in this local repository. This file is staged/
pre-launch provenance, not the literal record of what ran — it narrows the
gap but does not close it. `docs/audits/methodology_audit.md` predates M3
entirely and does not discuss it; there is currently no committed
methodology writeup for M3 in this repository.
