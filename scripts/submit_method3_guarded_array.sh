#!/bin/bash
# Submit the frozen exploratory M3 production array on Kelvin2.
# One task owns one (dyad, condition, orientation) cell.
#
# Submit only after reviewing the committed production state and confirming
# Kelvin2 authentication. This script does not touch existing M1/M2 roots.

#SBATCH --job-name=amadeus-gen-m3-hybrid
#SBATCH --partition=k2-gpu-a100mig
#SBATCH --gres=gpu:1
#SBATCH --array=0-223%8
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=/mnt/scratch2/users/40482774/lichess_data/m3_generation_jobs/method3_%A_%a.out

set -eo pipefail

REPO=/mnt/scratch2/users/40482774/repos/amadeus-counterpoint
MAPPING=/mnt/scratch2/users/40482774/paper_artifacts/primary_generation_2026-09-11/cell_mapping.tsv
OUTPUT_ROOT=/mnt/scratch2/users/40482774/artifacts/synthetic_m3_hybrid_2026-09-17
STOCKFISH=/mnt/scratch2/users/40482774/tools/stockfish/Stockfish-sf_19/src/stockfish
GCC_LIB=/opt/gridware/depots/54e7fb3c/el8/pkg/compilers/gcc/14.1.0/lib64

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$MAPPING")
PLAYER_A=$(echo "$LINE" | cut -f2)
PLAYER_B=$(echo "$LINE" | cut -f3)
CONDITION=$(echo "$LINE" | cut -f4)
ORIENTATION=$(echo "$LINE" | cut -f5)

echo "array_task=$SLURM_ARRAY_TASK_ID -> dyad=($PLAYER_A,$PLAYER_B) condition=$CONDITION orientation=$ORIENTATION"
hostname
nvidia-smi -L

source /etc/profile.d/zz-flight-starter.sh
flight env activate gridware
module load apps/miniconda3/24.4.0
module load compilers/gcc/14.1.0
source /mnt/scratch2/users/40482774/venvs/amadeus-counterpoint/bin/activate
cd "$REPO"

python3 scripts/generate_method3_production.py \
  --base-checkpoint /mnt/scratch2/users/40482774/checkpoints/chessformer-79m-broadcast/step_00200000.pt \
  --player-checkpoints '{"0":"/mnt/scratch2/users/40482774/checkpoints/method1/player_0.pt","1":"/mnt/scratch2/users/40482774/checkpoints/method1/player_1.pt","2":"/mnt/scratch2/users/40482774/checkpoints/method1/player_2.pt","3":"/mnt/scratch2/users/40482774/checkpoints/method1/player_3.pt","4":"/mnt/scratch2/users/40482774/checkpoints/method1/player_4.pt","5":"/mnt/scratch2/users/40482774/checkpoints/method1/player_5.pt","6":"/mnt/scratch2/users/40482774/checkpoints/method1/player_6.pt","7":"/mnt/scratch2/users/40482774/checkpoints/method1/player_7.pt"}' \
  --identities '{"0":"Magnus Carlsen","1":"Wesley So","2":"Levon Aronian","3":"Fabiano Caruana","4":"Maxime Vachier-Lagrave","5":"Hikaru Nakamura","6":"Ian Nepomniachtchi","7":"Alireza Firouzja"}' \
  --method2-checkpoint /mnt/scratch2/users/40482774/checkpoints/method2/joint.pt \
  --representative-elos /mnt/scratch2/users/40482774/lichess_data/representative_elos.json \
  --player-ids 0,1,2,3,4,5,6,7 \
  --num-players 8 --style-dim 32 --k 5 \
  --output-root "$OUTPUT_ROOT" \
  --root-seed 20260911 \
  --checkpoint-identity method3-hybrid-2026-09-17-guarded \
  --protocol-version 2 \
  --code-commit __PRODUCTION_COMMIT__ \
  --generation-batch-size 128 \
  --only-dyad-a "$PLAYER_A" --only-dyad-b "$PLAYER_B" \
  --only-condition "$CONDITION" --only-orientation "$ORIENTATION" \
  --use-strength-guardrail \
  --stockfish-path "$STOCKFISH" \
  --gcc-lib-path "$GCC_LIB" \
  --n-engine-workers 16

echo "EXPLORATORY_METHOD3_TASK_DONE"
