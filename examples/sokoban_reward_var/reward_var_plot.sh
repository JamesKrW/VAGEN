#!/bin/bash

set -x

PROJECT_NAME="verl_vagen"
EXPERIMENT_NAME="reward_var_plot-200"

BASEDIR=$(pwd)
SCRIPTDIR=$(dirname "$0")
EXPERIMENT_DIR=${BASEDIR}/exps/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_CHECKPOINT_DIR=${EXPERIMENT_DIR}/verl_checkpoints
DATASET_TRAIN=${SCRIPTDIR}/train_sokoban_vision.yaml
DATASET_VAL=${SCRIPTDIR}/val_sokoban_vision.yaml
agent_loop_config_path=${BASEDIR}/vagen/configs/agent_no_concat.yaml
REF_MODEL_PATH=/projects/p32476/projects/viewsuite/VAGEN/exps/verl_vagen/grpo_qwen25vl3b/verl_checkpoints/global_step_200/actor/huggingface
mkdir -p ${EXPERIMENT_DIR}

# Configuration for reward variance visualization
# Set rollout_validation_n to control how many rollouts per sample for variance computation
# Higher values give more accurate variance estimates but take longer
ROLLOUT_VALIDATION_N=32

# Resume from a checkpoint path if needed (set to empty string to start fresh)
# Example: RESUME_PATH=${EXPERIMENT_DIR}/verl_checkpoints/global_step_100
RESUME_PATH=""

PYTHONUNBUFFERED=1 python3 -m vagen.main_ppo_reward_var \
    --config-path=${BASEDIR}/vagen/configs \
    --config-name='vagen_multiturn' \
    data.train_files=${DATASET_TRAIN} \
    data.val_files=${DATASET_VAL} \
    data.train_batch_size=128 \
    algorithm.adv_estimator=no_concat_gae_first \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.disable_log_stats=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${EXPERIMENT_DIR} \
    trainer.log_val_generations=0 \
    +trainer.skip_special_tokens_in_validation=False \
    +trainer.concat_multi_turn=False \
    +trainer.visualize_reward_var_entropy=True \
    +trainer.rollout_validation_n=${ROLLOUT_VALIDATION_N} \
    data.max_prompt_length=1000 \
    data.max_response_length=800 \
    critic.enable=False \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable 2>&1 | \
    tee ${EXPERIMENT_DIR}/${PROJECT_NAME}_${EXPERIMENT_NAME}.log

echo ""
echo "=============================================="
echo "Reward Variance Visualization Complete!"
echo "=============================================="
echo "Results saved to: ${EXPERIMENT_DIR}"
echo "  - Scatter plot: entropy_reward_var_plot_step*.png"
echo "  - Data file: entropy_reward_var_pairs_step*.json"
echo "=============================================="
