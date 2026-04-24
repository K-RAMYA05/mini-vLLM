#!/bin/bash
# Full distillation pipeline — run on Discovery.
#
# Rough time budget for Llama-3.1-8B on A100/H100:
#   Step 1 (teacher data gen): several hours for 50K sequences
#   Step 2 (distillation):     several hours for 3 epochs
#   Step 3 (eval):             ~30 min
#
# Total: ~8-15 hours. Fits in a one-night SLURM job.
#
# Usage:
#   bash scripts/run_distillation.sh

set -euo pipefail

MODEL=${MODEL:-meta-llama/Llama-3.1-8B}
DATA_DIR=${DATA_DIR:-./distill_data}
DRAFT_DIR=${DRAFT_DIR:-./outputs/llama-3.1-8b-draft-8layer}
NUM_SEQUENCES=${NUM_SEQUENCES:-50000}
NUM_KEEP_LAYERS=${NUM_KEEP_LAYERS:-8}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}

echo "============================================"
echo "Step 1/3: Generate teacher distillation data"
echo "============================================"
python -m mini_vllm.distill.generate_teacher_data \
    --teacher "$MODEL" \
    --num-sequences "$NUM_SEQUENCES" \
    --seq-len 512 \
    --output-dir "$DATA_DIR" \
    --topk 50 \
    --batch-size "$BATCH_SIZE"

echo ""
echo "============================================"
echo "Step 2/3: Distill draft model"
echo "============================================"
python -m mini_vllm.distill.train_distill \
    --teacher "$MODEL" \
    --data-dir "$DATA_DIR" \
    --num-keep-layers "$NUM_KEEP_LAYERS" \
    --output-dir "$DRAFT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --use-bf16

echo ""
echo "============================================"
echo "Step 3/3: Evaluate acceptance rate"
echo "============================================"
python -m mini_vllm.distill.eval_acceptance \
    --target "$MODEL" \
    --draft "$DRAFT_DIR/final" \
    --gamma 4 \
    --num-sequences 200

echo ""
echo "Done. Draft model is at $DRAFT_DIR/final"
echo "Plug it into mini-vLLM with:"
echo "  EngineConfig(use_speculative=True, draft_model_name_or_path=\"$DRAFT_DIR/final\", ...)"
