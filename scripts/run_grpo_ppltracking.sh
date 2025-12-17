#!/bin/bash

# Run GRPO with PPL tracking during validation
# Uses PPLTrackingTrainer to record token-level log probs and rewards
# Uses VERL's built-in verifiable reward (math_reward) - only checks answer correctness

set -x
unset ROCR_VISIBLE_DEVICES

# WandB setup - set your API key here or export it before running
export WANDB_API_KEY="85e762b6f29dc3dc8c85ac1ea00122767ff75e39"

# Option 2: Use offline mode if no internet
# export WANDB_MODE="offline"

# Option 3: Login interactively (uncomment to use)
# wandb login

# WandB entity (your username or team name) - optional
# export WANDB_ENTITY="your-entity"

MODEL_PATH="/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/models/Qwen3-4B"
VERL_HOME="/home/qian.niu/Takoai/Medical_Reasoning/Mark/rldynamics/verl"

python3 -m custom_training.ppo_trainer_ppl \
    algorithm.adv_estimator=grpo \
    data.train_files=$VERL_HOME/data/math500/train.parquet \
    data.val_files=$VERL_HOME/data/math500/test.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_ppl_tracking' \
    trainer.experiment_name='qwen3_4b_math500_ppl' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    $@
