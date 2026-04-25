#!/bin/bash
#SBATCH --job-name=mini-vllm-full
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ----------------------------
# User-configurable env vars
# ----------------------------
REPO_ROOT=${REPO_ROOT:-$SLURM_SUBMIT_DIR}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch1/$USER/mini_vllm}
VENV_DIR=${VENV_DIR:-$SCRATCH_ROOT/venvs/mini_vllm}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip_cache}
HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf_cache}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
DATA_DIR=${DATA_DIR:-$SCRATCH_ROOT/distill_data}
DRAFT_DIR=${DRAFT_DIR:-$SCRATCH_ROOT/outputs/llama-3.1-8b-draft-8layer}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
GPU_TYPE=${GPU_TYPE:-h100}                 # h100 | a100
CUDA_MODULE=${CUDA_MODULE:-cuda/12.6.3}
GCC_MODULE=${GCC_MODULE:-gcc/13.3.0}
RUN_DISTILL=${RUN_DISTILL:-1}              # 1 -> run distill pipeline
RUN_BENCH=${RUN_BENCH:-1}                  # 1 -> run benchmark pipeline
RUN_SPEC=${RUN_SPEC:-1}                    # forwarded to benchmark script
RUN_PROFILE=${RUN_PROFILE:-0}              # forwarded to benchmark script
INSTALL_DEPS=${INSTALL_DEPS:-1}            # pip install repo deps
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}# build/install flash attention package
RUN_SMOKE=${RUN_SMOKE:-0}                  # 1 -> quick 100-seq smoke teacher gen first
NUM_SEQUENCES=${NUM_SEQUENCES:-50000}
NUM_KEEP_LAYERS=${NUM_KEEP_LAYERS:-8}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_JOBS=${MAX_JOBS:-1}
TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
BLIS_NUM_THREADS=${BLIS_NUM_THREADS:-1}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export a read token before sbatch submit." >&2
  echo "Example: HF_TOKEN=hf_xxx sbatch scripts/slurm_full_pipeline.sbatch" >&2
  exit 1
fi

export HF_TOKEN
export HF_HOME
export TRANSFORMERS_CACHE
export HF_HUB_ENABLE_HF_TRANSFER=1
export PIP_CACHE_DIR
export TOKENIZERS_PARALLELISM
export MAX_JOBS
export OPENBLAS_NUM_THREADS
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export NUMEXPR_NUM_THREADS
export VECLIB_MAXIMUM_THREADS
export BLIS_NUM_THREADS

mkdir -p "$SCRATCH_ROOT" "$PIP_CACHE_DIR" "$HF_HOME" "$DATA_DIR" "$DRAFT_DIR" \
         "$SCRATCH_ROOT/logs/h100" "$SCRATCH_ROOT/logs/a100" "$SCRATCH_ROOT/logs/distill"

echo "[$(date)] Job started on $(hostname)"
echo "REPO_ROOT=$REPO_ROOT"
echo "SCRATCH_ROOT=$SCRATCH_ROOT"
echo "GPU_TYPE=$GPU_TYPE"
echo "MODEL=$MODEL"

module reset
module load "$GCC_MODULE"
module load "$CUDA_MODULE"
export CUDA_HOME
CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version
nvidia-smi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install -U pip "setuptools>=70.1,<82" wheel packaging ninja

cd "$REPO_ROOT"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  pip install -r requirements.txt
  pip install -e ".[serve]"
fi

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  if [[ "$GPU_TYPE" == "h100" ]]; then
    FA_DIR="$SCRATCH_ROOT/flash-attention"
    if [[ ! -d "$FA_DIR" ]]; then
      git clone https://github.com/Dao-AILab/flash-attention "$FA_DIR"
    fi
    cd "$FA_DIR/hopper"
    MAX_JOBS="$MAX_JOBS" pip install . --no-build-isolation
    cd "$REPO_ROOT"
    python -c "import flash_attn_interface; print('FA3 ok')"
  else
    pip install flash-attn --no-build-isolation
    python -c "from flash_attn import flash_attn_func; print('FA2 ok')"
  fi
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability(0))"
python -c "from huggingface_hub import whoami; import os; print(whoami(token=os.environ['HF_TOKEN'])['name'])"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='${MODEL}', filename='tokenizer_config.json', token=True); print('HF model access ok')"

if [[ "$RUN_SMOKE" == "1" ]]; then
  python -m mini_vllm.distill.generate_teacher_data \
    --teacher "$MODEL" \
    --num-sequences 100 \
    --seq-len 128 \
    --output-dir "$DATA_DIR/smoke" \
    --topk 50 \
    --batch-size 4
fi

DISTILL_RUN="$SCRATCH_ROOT/logs/distill/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DISTILL_RUN"

if [[ "$RUN_DISTILL" == "1" ]]; then
  python -m mini_vllm.distill.generate_teacher_data \
    --teacher "$MODEL" \
    --num-sequences "$NUM_SEQUENCES" \
    --seq-len 512 \
    --output-dir "$DATA_DIR" \
    --topk 50 \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$DISTILL_RUN/step1_teacher_data.log"

  python -m mini_vllm.distill.train_distill \
    --teacher "$MODEL" \
    --data-dir "$DATA_DIR" \
    --num-keep-layers "$NUM_KEEP_LAYERS" \
    --output-dir "$DRAFT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --use-bf16 \
    2>&1 | tee "$DISTILL_RUN/step2_train_distill.log"

  python -m mini_vllm.distill.eval_acceptance \
    --target "$MODEL" \
    --draft "$DRAFT_DIR/final" \
    --gamma 4 \
    --num-sequences 200 \
    --dtype bfloat16 \
    2>&1 | tee "$DISTILL_RUN/step3_eval_acceptance.log"
fi

if [[ "$RUN_BENCH" == "1" ]]; then
  if [[ "$RUN_SPEC" == "1" && ! -d "$DRAFT_DIR/final" ]]; then
    echo "RUN_SPEC=1 but draft checkpoint not found at $DRAFT_DIR/final" >&2
    exit 1
  fi

  if [[ "$GPU_TYPE" == "h100" ]]; then
    LOG_ROOT="$SCRATCH_ROOT/logs/h100" \
    MODEL="$MODEL" \
    RUN_SPEC="$RUN_SPEC" \
    RUN_PROFILE="$RUN_PROFILE" \
    DRAFT_MODEL="$DRAFT_DIR/final" \
    bash scripts/run_h100_all.sh
    LATEST_RUN="$(ls -td "$SCRATCH_ROOT"/logs/h100/* | head -n 1)"
  else
    LOG_ROOT="$SCRATCH_ROOT/logs/a100" \
    MODEL="$MODEL" \
    RUN_SPEC="$RUN_SPEC" \
    RUN_PROFILE="$RUN_PROFILE" \
    DRAFT_MODEL="$DRAFT_DIR/final" \
    bash scripts/run_a100_all.sh
    LATEST_RUN="$(ls -td "$SCRATCH_ROOT"/logs/a100/* | head -n 1)"
  fi

  bash scripts/summarize_logs.sh "$LATEST_RUN" > "$LATEST_RUN/summary.md"
  echo "Benchmark summary written to $LATEST_RUN/summary.md"
fi

echo "[$(date)] Done."
echo "Distill logs: $DISTILL_RUN"
echo "Draft model: $DRAFT_DIR/final"
