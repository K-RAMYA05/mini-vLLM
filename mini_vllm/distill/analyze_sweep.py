"""Aggregate the results of a draft-depth sweep.

Reads eval_<D>layer.json files from a depth sweep run, then
emits a markdown report with:

  - Per-depth, per-domain α (acceptance rate).
  - Empirical tokens-per-step (from the JSON).
  - Theoretical tokens-per-step from the Leviathan et al. (2022) formula
        E[tokens/step] = (1 - α^(γ+1)) / (1 - α)
    which assumes per-position acceptance is i.i.d. Bernoulli(α).
  - The residual (empirical - theoretical). Positive residual = acceptance
    is positively autocorrelated; negative = autocorrelated rejections.
  - An "adjusted speedup" estimate that subtracts the draft's amortized
    forward cost as a fraction of the target's: speedup ≈ tokens_per_step
    / (1 + c · γ / accepted_per_step), with c estimated as
    layers_draft / layers_target. This gives a single number that's
    actually comparable across depths.

Usage:
    python -m mini_vllm.distill.analyze_sweep \
        --sweep-root /scratch1/.../sweep_depth \
        --gamma 4 \
        --report /scratch1/.../sweep_depth/report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Tuple


def leviathan_tokens_per_step(alpha: float, gamma: int) -> float:
    """E[tokens accepted per spec step], assuming i.i.d. acceptance."""
    if alpha >= 1.0:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def _depth_from_path(path: str) -> int:
    m = re.search(r"eval_(\d+)layer\.json$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse depth from {path}")
    return int(m.group(1))


def collect(sweep_root: str) -> List[Tuple[int, dict]]:
    """Return list of (depth, parsed_json) tuples sorted by depth."""
    paths = sorted(
        glob.glob(os.path.join(sweep_root, "eval_*layer.json")),
        key=_depth_from_path,
    )
    if not paths:
        raise FileNotFoundError(f"No eval_*layer.json under {sweep_root}")
    out = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        out.append((_depth_from_path(p), data))
    return out


def estimate_draft_cost_ratio(depth: int, target_layers: int = 32) -> float:
    """Crude proxy: forward cost ~ proportional to layer count. Llama-3.1-8B
    has 32 layers; the embedding + LM head are shared scalars at this size."""
    return depth / target_layers


def render_markdown(
    rows: List[dict],
    gamma: int,
    target_layers: int = 32,
) -> str:
    out = []
    out.append(f"# Draft-depth sweep — γ = {gamma}\n")
    out.append("Empirical α and tokens/step come from `eval_acceptance` JSON. "
               "Theoretical tokens/step uses the Leviathan formula `(1 − α^(γ+1)) / (1 − α)`. "
               "The residual is `empirical − theoretical`; large |residual| means "
               "acceptance is not i.i.d. across positions.\n")
    out.append("Adjusted speedup factors in draft-forward cost as `c = depth / "
               f"{target_layers}` (layer-count proxy). It is the per-step throughput "
               "ratio vs. plain target-only decoding.\n")
    out.append("| depth | domain | α | tok/step (emp.) | tok/step (Leviathan) | residual | adj. speedup |")
    out.append("|---:|:--|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {r['depth']} | {r['domain']} | {r['alpha']:.3f} | "
            f"{r['emp']:.3f} | {r['theory']:.3f} | {r['residual']:+.3f} | "
            f"{r['adj_speedup']:.2f}× |"
        )
    out.append("")
    out.append("## How to read this")
    out.append("- A higher α at a deeper draft is **expected** — at the cost of a higher draft-forward share.")
    out.append("- The Pareto frontier is the depth that maximizes adjusted speedup, not raw α.")
    out.append("- Residual close to 0 (within ±0.05) validates the i.i.d. assumption used in spec-decode papers; ")
    out.append("  large residuals are worth investigating — they're a real result either way.")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", required=True)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--target-layers", type=int, default=32,
                   help="Number of layers in the target model (cost proxy).")
    p.add_argument("--report", default=None,
                   help="Output markdown path. If unset, print to stdout.")
    args = p.parse_args()

    runs = collect(args.sweep_root)
    rows: List[dict] = []
    for depth, data in runs:
        c = estimate_draft_cost_ratio(depth, args.target_layers)
        for domain, r in data.get("domains", {}).items():
            alpha = float(r["alpha"])
            emp = float(r["tokens_per_step"])
            theory = leviathan_tokens_per_step(alpha, args.gamma)
            # adjusted speedup = emp_tok_per_step / (1 + c * gamma_amortized)
            # Approximate amortized γ as gamma (we always pay gamma draft-forwards
            # per spec step regardless of acceptance).
            adj = emp / (1.0 + c * args.gamma / max(emp, 1e-6))
            rows.append({
                "depth": depth,
                "domain": domain,
                "alpha": alpha,
                "emp": emp,
                "theory": theory,
                "residual": emp - theory,
                "adj_speedup": adj,
            })

    md = render_markdown(rows, args.gamma, target_layers=args.target_layers)
    if args.report:
        with open(args.report, "w") as f:
            f.write(md + "\n")
        print(f"Wrote {args.report}")
    else:
        print(md)


if __name__ == "__main__":
    main()
