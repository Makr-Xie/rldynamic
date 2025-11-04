#!/bin/bash

# Run GRPO with custom trainer injection
# This uses monkey patching to inject our custom trainer into verl

set -x
unset ROCR_VISIBLE_DEVICES

# Reward function path
REWARD_FUNCTION_PATH="/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/utils/rewards.py"
MODEL_PATH="/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/models/Qwen3-4B"
HOME="/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/verl"

# Run with our custom main_ppo that patches verl then runs the original main_ppo
python3 -m custom_training.custom_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/math500/train.parquet \
    data.val_files=$HOME/data/math500/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH}\
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_math500' \
    trainer.experiment_name='qwen3_4b_math500' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    custom_reward_function.name="compute_score" \
    $@