#!/usr/bin/env bash
# GRPO training for cd_descendants task (OOM-safe settings)
# Fixes from original command:
#   - per_device_train_batch_size 2 -> 1  (halves training activation memory)
#   - max_completion_length 8192 -> 4096  (halves KV cache and generation memory)
set -euo pipefail

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "[run] Using $N_GPUS GPUs"

torchrun --standalone --nproc_per_node="$N_GPUS" \
  experiments/grpo.py \
  --mode train \
  --task cd_descendants \
  --model_id Qwen/Qwen3-4B-Thinking-2507 \
  --cd-train-csv experiments/checkpoints/cd_descendants_1024/01_stage_1_named_obs50_int10/train_mixed.csv \
  --cd-test-csv experiments/prompts/cd_descendants/sachs/splits/stage_1_named_obs50_int10_eval.csv \
  --output_dir experiments/checkpoints/cd_descendants_8192/01_stage_1_named_obs50_int10/grpo \
  --no-use-vllm \
  --max_completion_length 8192 \
  --num_generations 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-6 \
  --cd-format-reward-scale 1.0 \
  --cd-partial-format-reward-scale 1.0 \
  --cd-graph-reward-scale 1.0 \
  --length_penalty_coef 0.00002 \
  --length_penalty_max_abs 0 \
  --save_steps 20 \
  --logging_steps 1 \
  --report_to none
