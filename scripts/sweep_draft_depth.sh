#!/usr/bin/env bash
# Sweep draft-model depth (num-keep-layers) and measure acceptance rate.
# Produces one JSON per depth + an aggregated markdown report.
#
# Inputs (env vars):
#   TEACHER         HF id or path of the target model.   default: meta-llama/Llama-3.1-8B
#   DATA_DIR        Path to teacher logits already generated. required.
#   OUT_ROOT        Root dir for draft checkpoints + sweep outputs. required.
#   DEPTHS          Space-separated list of layer counts. default: "4 6 8 10 12"
#   GAMMA           Speculative γ for eval. default: 4
#   NUM_EVAL_SEQS   Eval prompt count. default: 100
#   EPOCHS          Distill epochs per depth. default: 1   (set higher for the full sweep)
#   BATCH_SIZE      Distill batch size. default: 4
#   DTYPE           bfloat16 | float16. default: bfloat16
#
# Usage:
#   DATA_DIR=/scratch1/$USER/mini_vllm/distill_data \
#   OUT_ROOT=/scratch1/$USER/mini_vllm/sweep_depth \
#   bash scripts/sweep_draft_depth.sh

set -euo pipefail

TEACHER=${TEACHER:-meta-llama/Llama-3.1-8B}
DATA_DIR=${DATA_DIR:?DATA_DIR is required}
OUT_ROOT=${OUT_ROOT:?OUT_ROOT is required}
DEPTHS=${DEPTHS:-"4 6 8 10 12"}
GAMMA=${GAMMA:-4}
NUM_EVAL_SEQS=${NUM_EVAL_SEQS:-100}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-4}
DTYPE=${DTYPE:-bfloat16}

mkdir -p "$OUT_ROOT"

for D in $DEPTHS; do
  CHECKPOINT_DIR="$OUT_ROOT/draft_${D}layer"
  EVAL_JSON="$OUT_ROOT/eval_${D}layer.json"
  TRAIN_LOG="$OUT_ROOT/train_${D}layer.log"
  EVAL_LOG="$OUT_ROOT/eval_${D}layer.log"

  if [[ ! -d "$CHECKPOINT_DIR/final" ]]; then
    echo "==> Training draft at depth=$D"
    python -m mini_vllm.distill.train_distill \
      --teacher "$TEACHER" \
      --data-dir "$DATA_DIR" \
      --num-keep-layers "$D" \
      --output-dir "$CHECKPOINT_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --use-bf16 \
      2>&1 | tee "$TRAIN_LOG"
  else
    echo "==> Skipping train (depth=$D, $CHECKPOINT_DIR/final exists)"
  fi

  if [[ ! -f "$EVAL_JSON" ]]; then
    echo "==> Evaluating draft at depth=$D"
    python -m mini_vllm.distill.eval_acceptance \
      --target "$TEACHER" \
      --draft "$CHECKPOINT_DIR/final" \
      --gamma "$GAMMA" \
      --num-sequences "$NUM_EVAL_SEQS" \
      --dtype "$DTYPE" \
      --json-out "$EVAL_JSON" \
      2>&1 | tee "$EVAL_LOG"
  else
    echo "==> Skipping eval (depth=$D, $EVAL_JSON exists)"
  fi
done

echo
echo "==> Aggregating results"
python -m mini_vllm.distill.analyze_sweep \
  --sweep-root "$OUT_ROOT" \
  --gamma "$GAMMA" \
  --report "$OUT_ROOT/report.md"

echo
echo "Sweep done. Report: $OUT_ROOT/report.md"
