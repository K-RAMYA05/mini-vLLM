#!/bin/bash
#SBATCH --job-name=mini-vllm-sweep
#SBATCH --account=jieyuz_1727
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch1/$USER/mini_vllm}
VENV_DIR=${VENV_DIR:-$SCRATCH_ROOT/venvs/mini_vllm}
HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_cache}
DATA_DIR=${DATA_DIR:-$SCRATCH_ROOT/distill_data}
SWEEP_DIR=${SWEEP_DIR:-$SCRATCH_ROOT/sweep_depth}
TEACHER=${TEACHER:-meta-llama/Llama-3.1-8B}
DEPTHS=${DEPTHS:-"4 6 8 10 12"}
GAMMA=${GAMMA:-4}
NUM_EVAL_SEQS=${NUM_EVAL_SEQS:-100}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
DTYPE=${DTYPE:-bfloat16}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.6.3}
GCC_MODULE=${GCC_MODULE:-gcc/13.3.0}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Submit with: HF_TOKEN=hf_xxx sbatch scripts/slurm_sweep_depth.sbatch" >&2
  exit 1
fi

export HF_TOKEN HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

mkdir -p "$SCRATCH_ROOT" "$HF_HOME" "$DATA_DIR" "$SWEEP_DIR"

echo "[$(date)] Job started on $(hostname)"
nvidia-smi

module reset
module load "$GCC_MODULE"
module load "$CUDA_MODULE"

source "$VENV_DIR/bin/activate"

cd "$REPO_ROOT"

# Teacher data: generate only if missing.
if [[ ! -f "$DATA_DIR/manifest.json" ]] && [[ -z "$(ls -A "$DATA_DIR" 2>/dev/null || true)" ]]; then
  echo "==> Generating teacher data (one-time, reused across all depths)"
  python -m mini_vllm.distill.generate_teacher_data \
    --teacher "$TEACHER" \
    --num-sequences 50000 --seq-len 512 \
    --output-dir "$DATA_DIR" \
    --topk 50 --batch-size 8 \
    2>&1 | tee "$SWEEP_DIR/teacher_data.log"
fi

DATA_DIR="$DATA_DIR" \
OUT_ROOT="$SWEEP_DIR" \
TEACHER="$TEACHER" \
DEPTHS="$DEPTHS" \
GAMMA="$GAMMA" \
NUM_EVAL_SEQS="$NUM_EVAL_SEQS" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
DTYPE="$DTYPE" \
bash scripts/sweep_draft_depth.sh

echo "[$(date)] Sweep done."
echo "Report: $SWEEP_DIR/report.md"
