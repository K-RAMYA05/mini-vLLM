# Self-distilled draft methodology — depth × acceptance Pareto

The headline original contribution of this project: rather than using an
off-the-shelf small model as the speculative-decoding draft, build the
draft in-repo and quantify its acceptance rate against the theoretical
prediction across draft depths.

## Setup

- **Target:** `meta-llama/Llama-3.1-8B` (32 layers, 8 KV heads, 128 head dim).
- **Draft construction:** layer-prune the target down to N evenly-spaced layers,
  inheriting hidden size, head counts, RoPE parameters, and tokenizer by
  construction. Initialize the pruned model from the target's own weights —
  only depth is reduced.
- **Distillation:** combined loss
  `α · KL(student || teacher_top50) + (1 − α) · CE(student, next_token)`
  with α=0.9, temperature T=2.0 (Hinton scaling). Trained on a 50k-sequence
  WikiText + CodeSearchNet corpus, 512 tokens per sequence.
- **Eval:** speculative decoding with γ=4 (4 draft tokens proposed per step),
  rejection sampling at temperature 0 (greedy) on 100 prompts split evenly
  between WikiText (natural text) and HumanEval (code).
- **Sweep:** train and evaluate at depths {4, 6, 8, 10, 12} layers.

## Theoretical prediction

Under Leviathan et al. (2022) the expected number of tokens accepted per
speculative step, assuming per-position acceptance is i.i.d. Bernoulli(α),
is:

    E[tok/step] = (1 − α^(γ+1)) / (1 − α)

This is an upper bound when accept events are positively autocorrelated
(streaks of easy positions) and a lower bound when they're negatively
autocorrelated. Comparing empirical to theoretical lets us validate (or
falsify) the i.i.d. assumption that nearly every spec-decode paper uses.

## Empirical procedure

For each draft depth `D`:

1. Distill: train the D-layer student against the target's logits.
2. Eval: measure `α` (fraction of draft tokens accepted) and empirical
   `tok/step` on the held-out prompt set.
3. Compute theoretical `tok/step` = `(1 − α^(γ+1)) / (1 − α)`.
4. Report residual = empirical − theoretical.

Adjusted speedup factors in the draft's forward cost:

    speedup ≈ tok/step / (1 + c · γ / tok/step)

where `c = D / 32` is the layer-count-based cost ratio between draft and
target (a crude but defensible proxy at this scale, since hidden size and
head counts are preserved).

## Results

(filled in by `mini_vllm.distill.analyze_sweep` after the sweep job finishes)

| depth | domain | α | tok/step (emp.) | tok/step (Leviathan) | residual | adj. speedup |
|---:|:--|---:|---:|---:|---:|---:|
| 4  | natural | … | … | … | … | … |
| 4  | code    | … | … | … | … | … |
| 6  | natural | … | … | … | … | … |
| 6  | code    | … | … | … | … | … |
| 8  | natural | … | … | … | … | … |
| 8  | code    | … | … | … | … | … |
| 10 | natural | … | … | … | … | … |
| 10 | code    | … | … | … | … | … |
| 12 | natural | … | … | … | … | … |
| 12 | code    | … | … | … | … | … |

## Discussion

(fill in after results land)

- **Pareto frontier:** which draft depth maximizes the *adjusted* speedup?
  Higher α at deeper drafts should be balanced against the higher draft-forward
  cost. The expected sweet spot is around D=8 for an 8B target on A100, but
  this is what the experiment is testing.
- **i.i.d. validation:** if |residual| < 0.05 across all (depth, domain) pairs,
  the i.i.d. assumption is empirically validated. Larger residuals — especially
  on code — would suggest acceptance is positively autocorrelated (long
  predictable runs of boilerplate) and motivate adaptive γ schedules tied to
  recent acceptance rate.
- **Domain transfer:** distillation generalizes from WikiText+CodeSearchNet
  *to* HumanEval if the code α stays comparable to the WikiText α at the same
  depth. Divergence indicates the distillation set was too narrow.

## Reproducing

```bash
DATA_DIR=/scratch1/$USER/mini_vllm/distill_data \
OUT_ROOT=/scratch1/$USER/mini_vllm/sweep_depth \
TEACHER=meta-llama/Llama-3.1-8B \
DEPTHS="4 6 8 10 12" \
GAMMA=4 NUM_EVAL_SEQS=100 EPOCHS=3 \
bash scripts/sweep_draft_depth.sh
```

The sweep is idempotent — checkpoints and JSON eval outputs are written
per-depth, and re-running picks up where it stopped.

## Files

- `scripts/sweep_draft_depth.sh` — driver script.
- `scripts/slurm_sweep_depth.sbatch` — SLURM submission for CARC.
- `mini_vllm/distill/train_distill.py` — KL+CE distillation loop.
- `mini_vllm/distill/eval_acceptance.py` — α measurement (writes JSON).
- `mini_vllm/distill/analyze_sweep.py` — aggregates JSON, computes Leviathan
  residual, emits this report.
