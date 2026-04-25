#!/bin/bash
#SBATCH --job-name=distill-draft
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/distill_%j.out
#SBATCH --error=logs/distill_%j.err

# Edit --account and load your env setup first.
# This is a template — Discovery's actual partition/account names may differ.

module purge
module load gcc/11.3.0 cuda/12.1

# Activate your env — adjust path.
source ~/envs/mini_vllm/bin/activate

cd "$SLURM_SUBMIT_DIR"

bash scripts/run_distillation.sh
