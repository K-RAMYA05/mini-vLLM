#!/bin/bash
# Run the core A100/H100 validation matrix on the current allocated GPU.

set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
DTYPE=${DTYPE:-bfloat16}
PREFILL_BACKEND=${PREFILL_BACKEND:-flash}
NUM_GPU_BLOCKS=${NUM_GPU_BLOCKS:-8192}
LOG_DIR=${LOG_DIR:-logs/gpu_matrix_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$LOG_DIR"

echo "Writing logs to $LOG_DIR"
nvidia-smi | tee "$LOG_DIR/nvidia_smi.txt"

python benchmarks/bench_kernel.py \
  --batch-size 32 --num-heads 32 --num-kv-heads 8 \
  --head-dim 128 --seq-len 512 --block-size 16 --iters 100 \
  | tee "$LOG_DIR/kernel_seq512.log"

python benchmarks/bench_kernel.py \
  --batch-size 32 --num-heads 32 --num-kv-heads 8 \
  --head-dim 128 --seq-len 2048 --block-size 16 --iters 100 \
  | tee "$LOG_DIR/kernel_seq2048.log"

python benchmarks/bench_throughput.py \
  --model "$MODEL" --dtype "$DTYPE" \
  --batch-size 4 --num-prompts 8 --prompt-len 128 --max-tokens 32 \
  --num-gpu-blocks "$NUM_GPU_BLOCKS" --prefill-backend "$PREFILL_BACKEND" \
  --skip-hf | tee "$LOG_DIR/throughput_smoke.log"

for B in 1 8 16 32 64 128; do
  python benchmarks/bench_throughput.py \
    --model "$MODEL" --dtype "$DTYPE" \
    --batch-size "$B" --num-prompts 128 --prompt-len 512 --max-tokens 128 \
    --block-size 16 --num-gpu-blocks "$NUM_GPU_BLOCKS" \
    --prefill-backend "$PREFILL_BACKEND" --skip-hf \
    | tee "$LOG_DIR/throughput_b${B}.log"
done

python benchmarks/stress_long_run.py \
  --model "$MODEL" --dtype "$DTYPE" \
  --waves 5 --requests-per-wave 32 --prompt-len-words 256 --max-tokens 128 \
  --num-gpu-blocks "$NUM_GPU_BLOCKS" --prefill-backend "$PREFILL_BACKEND" \
  | tee "$LOG_DIR/stress.log"

echo "Done. Logs: $LOG_DIR"
