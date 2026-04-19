#!/bin/bash
#SBATCH --job-name=grpo_verl_1gpu_lora_debug
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdeb-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=/u/chenyuen0103/ENCO/experiments/logs/grpo_verl_1gpu_lora_debug_%j.out
#SBATCH --error=/u/chenyuen0103/ENCO/experiments/logs/grpo_verl_1gpu_lora_debug_%j.err

set -euo pipefail

mkdir -p /u/chenyuen0103/ENCO/experiments/logs

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

cd /u/chenyuen0103/ENCO
export PYTHONPATH=/u/chenyuen0103/ENCO/verl:/u/chenyuen0103/ENCO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VERL_PYTHON=${VERL_PYTHON:-/u/chenyuen0103/anaconda3/envs/verl/bin/python}

echo "[$(timestamp)] Launching 1-GPU VERL LoRA debug run"
echo "[$(timestamp)] Using streamed logs and reduced-memory training settings"

conda run --no-capture-output -n verl bash -c 'set -euo pipefail
cd /u/chenyuen0103/ENCO/verl
export PYTHONPATH=/u/chenyuen0103/ENCO/verl:/u/chenyuen0103/ENCO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export VERL_PYTHON=${VERL_PYTHON:-/u/chenyuen0103/anaconda3/envs/verl/bin/python}
echo "[launcher] python=$VERL_PYTHON"
echo "[launcher] starting verl.trainer.main_ppo"
"$VERL_PYTHON" -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=/u/chenyuen0103/ENCO/experiments/data/verl_cd_mix_train.parquet \
  data.val_files=/u/chenyuen0103/ENCO/experiments/data/verl_cd_mix_eval.parquet \
  data.train_batch_size=1 \
  data.train_max_samples=64 \
  data.val_max_samples=16 \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=8 \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path=/work/hdd/bdeb/chenyuen0103/.cache/hub/models--unsloth--qwen3-4b-thinking-2507/snapshots/fb56efbd0f60c5bcd531901fa5fe3ee3157b5135 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.lora.merge=True \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.max_model_len=2560 \
  actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  reward_model.enable=False \
  custom_reward_function.path=/u/chenyuen0103/ENCO/experiments/verifier_verl.py \
  custom_reward_function.name=compute_score \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +critic.model.override_config.attn_implementation=sdpa \
  trainer.critic_warmup=0 \
  "trainer.logger=[\"console\"]" \
  trainer.project_name=enco_verl \
  trainer.experiment_name=grpo_qwen3_4b_lora_1gpu_debug \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.resume_mode=disable \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1'

echo "[$(timestamp)] 1-GPU VERL LoRA debug run finished"
