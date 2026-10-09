#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

# Use physical GPU 2 by default. Override it when needed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

exec uv run python -m hw4.train \
  --task format_copy \
  --algo grpo \
  --output_dir runs/local_format_copy_grpo \
  --steps 51 \
  --batch_size 8 \
  --group_size 6 \
  --min_new_tokens 1 \
  --max_new_tokens 24 \
  --lr 3e-5 \
  --ppo_epochs 2 \
  --minibatch_size 8 \
  --grad_accum_steps 6 \
  --clip_eps 0.2 \
  --kl_coef 0.05 \
  --max_grad_norm 0.5 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name format_copy_grpo \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 6 \
  --eval_interval 50 \
  --save_interval 50 \
  --warmup_steps 10
