#!/bin/bash
#SBATCH --job-name=grpo_verl_sanity
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdeb-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --output=/u/chenyuen0103/ENCO/experiments/logs/grpo_verl_sanity_%j.out
#SBATCH --error=/u/chenyuen0103/ENCO/experiments/logs/grpo_verl_sanity_%j.err

mkdir -p /u/chenyuen0103/ENCO/experiments/logs

# Ensure model symlink exists (recreate if /tmp was cleared)
mkdir -p /tmp/chenyuen0103
ln -sfn /work/hdd/bdeb/chenyuen0103/.cache/hub/models--unsloth--qwen3-4b-thinking-2507/snapshots/fb56efbd0f60c5bcd531901fa5fe3ee3157b5135 /tmp/chenyuen0103/qwen3-4b-thinking

cd /u/chenyuen0103/ENCO
export PYTHONPATH=/u/chenyuen0103/ENCO:$PYTHONPATH

conda run -n verl bash -c 'export PYTHONPATH=/u/chenyuen0103/ENCO:$PYTHONPATH && python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=/u/chenyuen0103/ENCO/experiments/data/verl_cd_mix_train.parquet \
  data.val_files=/u/chenyuen0103/ENCO/experiments/data/verl_cd_mix_eval.parquet \
  data.train_batch_size=8 \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=/tmp/chenyuen0103/qwen3-4b-thinking \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.enable=False \
  custom_reward_function.path=/u/chenyuen0103/ENCO/experiments/verifier_verl.py \
  custom_reward_function.name=compute_score \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  +critic.model.override_config.attn_implementation=sdpa \
  trainer.critic_warmup=0 \
  "trainer.logger=[\"console\",\"wandb\"]" \
  trainer.project_name=enco_verl \
  trainer.experiment_name=grpo_mix_qwen_base_v1_verl_sanity \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.total_epochs=1'
