#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

# Use physical GPU 2 by default. Override it when needed
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

exec uv run python -m hw4.train \
  --task math_hard \
  --algo reinforce \
  --output_dir runs/local_math_hard_reinforce \
  --steps 201 \
  --batch_size 8 \
  --group_size 8 \
  --min_new_tokens 8 \
  --max_new_tokens 512 \
  --max_prompt_tokens 512 \
  --temperature 0.8 \
  --top_p 0.95 \
  --lr 3e-5 \
  --minibatch_size 8 \
  --grad_accum_steps 8 \
  --max_grad_norm 0.5 \
  --kl_coef 0.05 \
  --wandb_enabled \
  --wandb_project llm-rl-hw4 \
  --wandb_name math_hard_reinforce \
  --sample_markdown_log_interval 1 \
  --sample_log_interval 10 \
  --sample_log_n 8 \
  --cuda_empty_cache_interval 50 \
  --eval_interval 100 \
  --save_interval 100
