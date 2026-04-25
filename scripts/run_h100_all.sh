#!/bin/bash

set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
DTYPE=${DTYPE:-bfloat16}
NUM_GPU_BLOCKS=${NUM_GPU_BLOCKS:-8192}
BLOCK_SIZE=${BLOCK_SIZE:-16}
PROMPT_LEN=${PROMPT_LEN:-512}
MAX_TOKENS=${MAX_TOKENS:-128}
NUM_PROMPTS=${NUM_PROMPTS:-128}
REQUESTS_PER_WAVE=${REQUESTS_PER_WAVE:-32}
WAVES=${WAVES:-10}
DRAFT_MODEL=${DRAFT_MODEL:-}
RUN_SPEC=${RUN_SPEC:-0}
RUN_PROFILE=${RUN_PROFILE:-0}
PROFILE_BATCH_SIZE=${PROFILE_BATCH_SIZE:-32}
LOG_ROOT=${LOG_ROOT:-logs/h100}
RUN_NAME=${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
LOG_DIR=${LOG_DIR:-$LOG_ROOT/$RUN_NAME}

mkdir -p "$LOG_DIR"/{kernel,scaling,ablations,stress,spec,profile}

run_and_log() {
  local log_file=$1
  shift
  echo ">>> $*" | tee "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
}

echo "Writing logs to $LOG_DIR"
nvidia-smi | tee "$LOG_DIR/nvidia_smi.txt"

run_and_log "$LOG_DIR/kernel/seq512.log" \
  python benchmarks/bench_kernel.py \
    --batch-size 32 --num-heads 32 --num-kv-heads 8 \
    --head-dim 128 --seq-len 512 --block-size "$BLOCK_SIZE" --iters 100

run_and_log "$LOG_DIR/kernel/seq2048.log" \
  python benchmarks/bench_kernel.py \
    --batch-size 32 --num-heads 32 --num-kv-heads 8 \
    --head-dim 128 --seq-len 2048 --block-size "$BLOCK_SIZE" --iters 100

run_and_log "$LOG_DIR/ablations/throughput_flash_triton_on.log" \
  python benchmarks/bench_throughput.py \
    --model "$MODEL" --dtype "$DTYPE" \
    --batch-size 32 --num-prompts "$NUM_PROMPTS" \
    --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" \
    --block-size "$BLOCK_SIZE" --num-gpu-blocks "$NUM_GPU_BLOCKS" \
    --prefill-backend flash

run_and_log "$LOG_DIR/ablations/throughput_flash_attn_triton_off.log" \
  python benchmarks/bench_throughput.py \
    --model "$MODEL" --dtype "$DTYPE" \
    --batch-size 32 --num-prompts "$NUM_PROMPTS" \
    --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" \
    --block-size "$BLOCK_SIZE" --num-gpu-blocks "$NUM_GPU_BLOCKS" \
    --prefill-backend flash_attn --no-triton --skip-hf

run_and_log "$LOG_DIR/ablations/throughput_math_triton_off.log" \
  python benchmarks/bench_throughput.py \
    --model "$MODEL" --dtype "$DTYPE" \
    --batch-size 32 --num-prompts "$NUM_PROMPTS" \
    --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" \
    --block-size "$BLOCK_SIZE" --num-gpu-blocks "$NUM_GPU_BLOCKS" \
    --prefill-backend math --no-triton --skip-hf

if python -c "import flash_attn" >/dev/null 2>&1; then
  run_and_log "$LOG_DIR/ablations/throughput_flash_attn_triton_on.log" \
    python benchmarks/bench_throughput.py \
      --model "$MODEL" --dtype "$DTYPE" \
      --batch-size 32 --num-prompts "$NUM_PROMPTS" \
      --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" \
      --block-size "$BLOCK_SIZE" --num-gpu-blocks "$NUM_GPU_BLOCKS" \
      --prefill-backend flash_attn --skip-hf
fi

for B in 1 2 4 8 16 32 64 128; do
  run_and_log "$LOG_DIR/scaling/throughput_b${B}.log" \
    python benchmarks/bench_throughput.py \
      --model "$MODEL" --dtype "$DTYPE" \
      --batch-size "$B" --num-prompts "$NUM_PROMPTS" \
      --prompt-len "$PROMPT_LEN" --max-tokens "$MAX_TOKENS" \
      --block-size "$BLOCK_SIZE" --num-gpu-blocks "$NUM_GPU_BLOCKS" \
      --prefill-backend flash_attn --skip-hf
done

run_and_log "$LOG_DIR/stress/base.log" \
  python benchmarks/stress_long_run.py \
    --model "$MODEL" --dtype "$DTYPE" \
    --waves "$WAVES" \
    --requests-per-wave "$REQUESTS_PER_WAVE" \
    --prompt-len-words 256 \
    --max-tokens "$MAX_TOKENS" \
    --num-gpu-blocks "$NUM_GPU_BLOCKS" \
    --prefill-backend flash_attn

run_and_log "$LOG_DIR/ablations/prefix_cache.log" \
  python benchmarks/bench_prefix_cache.py \
    --model "$MODEL" --dtype "$DTYPE" \
    --batch-size 32 --num-prompts 33 \
    --prefix-len 256 --suffix-len 32 --max-tokens 64 \
    --block-size "$BLOCK_SIZE" --num-gpu-blocks "$NUM_GPU_BLOCKS" \
    --prefill-backend flash_attn

if [[ "$RUN_SPEC" == "1" ]]; then
  if [[ -z "$DRAFT_MODEL" ]]; then
    echo "RUN_SPEC=1 requires DRAFT_MODEL to be set" >&2
    exit 1
  fi
  run_and_log "$LOG_DIR/spec/gamma4.log" \
    python benchmarks/bench_speculative.py \
      --target "$MODEL" \
      --draft "$DRAFT_MODEL" \
      --gamma 4 \
      --dtype "$DTYPE"
fi

if [[ "$RUN_PROFILE" == "1" ]]; then
  if command -v nsys >/dev/null 2>&1; then
    nsys profile -o "$LOG_DIR/profile/throughput_b${PROFILE_BATCH_SIZE}" \
      python benchmarks/bench_throughput.py \
        --model "$MODEL" --dtype "$DTYPE" \
        --batch-size "$PROFILE_BATCH_SIZE" --num-prompts 32 \
        --prompt-len "$PROMPT_LEN" --max-tokens 64 \
        --block-size "$BLOCK_SIZE" --num-gpu-blocks "$NUM_GPU_BLOCKS" \
        --prefill-backend flash_attn --skip-hf \
      > "$LOG_DIR/profile/nsys_throughput.log" 2>&1
  fi

  if command -v ncu >/dev/null 2>&1; then
    ncu -o "$LOG_DIR/profile/kernel_seq512" \
      python benchmarks/bench_kernel.py \
        --batch-size 32 --num-heads 32 --num-kv-heads 8 \
        --head-dim 128 --seq-len 512 --block-size "$BLOCK_SIZE" \
      > "$LOG_DIR/profile/ncu_kernel.log" 2>&1
  fi
fi

echo "Done. Logs: $LOG_DIR"
