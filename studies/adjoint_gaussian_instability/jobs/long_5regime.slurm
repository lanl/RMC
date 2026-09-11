#!/bin/bash
#SBATCH --job-name=adj_g002_long
#SBATCH --partition=shared-gpu-ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=/vast/home/dmcbride/ROU-project/ICLR_ROU_samplers/RMC/studies/adjoint_gaussian_instability/results/long_run/logs/%x-%j.out
#SBATCH --error=/vast/home/dmcbride/ROU-project/ICLR_ROU_samplers/RMC/studies/adjoint_gaussian_instability/results/long_run/logs/%x-%j.err

set -euo pipefail

REPO="$HOME/ROU-project/ICLR_ROU_samplers/RMC"
STUDY="$REPO/studies/adjoint_gaussian_instability"
OUT="$STUDY/results/long_run"

cd "$REPO"

export PATH="$HOME/ROU-project/envs/rou-gpu-py311/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg
export OMP_NUM_THREADS=4

echo "===== JOB ====="
echo "host: $(hostname)"
echo "branch: $(git branch --show-current)"
echo "commit: $(git rev-parse HEAD)"
echo "start: $(date)"

python - <<'PY'
import jax
print("backend:", jax.default_backend())
print("devices:", jax.devices())
PY

python -u \
  studies/adjoint_gaussian_instability/scripts/run_long_experiment.py \
  --outer 100 \
  --inner 100 \
  --initial-inner 5000 \
  --train-samples 512 \
  --eval-samples 10000 \
  --oracle-samples 50000 \
  --batch-size 64 \
  --steps 500 \
  --seeds 0 1 2 \
  --eta 0.5 \
  --no-plots \
  --output "$OUT"

echo "finish: $(date)"
