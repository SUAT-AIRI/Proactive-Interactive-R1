#!/bin/bash

# Interactive-R1 style training with verl-tool
# This script demonstrates how to train an LLM to use interactive capabilities

set -x

# 2️⃣ 告诉 Ray 用这个目录当临时缓存目录
export RAY_TMPDIR="/data2/chenxin/ray_tmp"

# 对于deepseek distill model要把add_bos_token设为False
model_name="/data1/HF-Models/Qwen/Qwen2.5-0.5B-Instruct"
train_data="/home/chenxin/verl-interactive/verl-tool/data/gsm8k_multi_inter/train.parquet"
val_data="/home/chenxin/verl-interactive/verl-tool/data/gsm8k_multi_inter/test.parquet"
interaction_config_path="/home/chenxin/verl-interactive/verl-tool/verl/examples/interactive-LLM/interaction_config/gsm8k_interaction_config.yaml"

checkpoint_dir="/data2/chenxin/qwen2.5-0.5b_function_rm-gsm8k-sgl-multi-w-interaction"
mkdir -p $checkpoint_dir


rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=6
n_nodes=1
n=6
total_epochs=3
total_training_steps=1005
batch_size=128
ppo_mini_batch_size=64
max_prompt_length=1024
max_action_length=2048
max_response_length=4096
max_obs_length=1024
temperature=1.0
top_p=1.0
enable_agent=False # enable agent for tool use
strategy="fsdp"

max_turns=5
kl_loss_coef=0
kl_coef=0
entropy_coeff=0
kl_loss_type=low_var_kl
lr=1e-6

ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=8
tensor_model_parallel_size=1
gpu_memory_utilization=0.6 # higher gpu_memory_utilization will likely cause the vllm to OOM and get stuck, so set it to a lower value like 0.4 or 0.5
do_offload=False # control actor's fsdp.[param|optimizer]_offload and actor_rollout_ref.rollout.fsdp.[param|optimizer]_offload; if gpu_memory_utilization is set to > 0.6, then do_offload should be set to True otherwise it will cause OOM
use_dynamic_bsz=True # faster
ulysses_sequence_parallel_size=1 # set to 1 for normal verl behavior, otherwise it will cause OOM
fsdp_size=-1
additional_eos_token_ids=[151643] # <|im_end|> token id
mask_observations=True # mask observations for kl loss and gradient descent
enable_mtrl=False # enable multi-turn training
model_pretty_name=$(echo $model_name | tr '/' '_' | tr '[:upper:]' '[:lower:]')
run_name_postfix="debug"
if [ "$enable_agent" = "True" ]; then
    run_name="${strategy}-agent-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
else
    run_name="${strategy}-${model_pretty_name}-${rl_alg}-n${n}-b${batch_size}-${ppo_mini_batch_size}-t${temperature}-lr${lr}${run_name_postfix}"
fi
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=INFO
# export VLLM_USE_V1=1
rollout_mode='sync'


NCCL_P2P_LEVEL=NVL FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    algorithm.use_kl_in_reward=False \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric='reward' \
    +algorithm.filter_groups.max_num_gen_batches=0 \
    data.train_files=$train_data \
    data.val_files=$val_data \
    data.train_batch_size=$batch_size \
    data.val_batch_size=2048 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.truncation='right' \
    data.shuffle=True \
    reward_model.launch_reward_fn_async=True \
    +reward_model.overlong_buffer.enable=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.strategy=$strategy \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8196 \
    actor_rollout_ref.agent.enable_agent=$enable_agent \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=$interaction_config_path \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    critic.optim.lr=1e-5 \
    critic.strategy=$strategy \
    critic.model.path=$model_name \
    critic.model.fsdp_config.fsdp_size=$fsdp_size \
    critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    +critic.model.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +critic.model.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +critic.model.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console','swanlab'] \
    trainer.experiment_name=$run_name \
    trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.max_actor_ckpt_to_keep=1 \
    +trainer.max_critic_ckpt_to_keep=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=10 \
    trainer.total_epochs=$total_epochs \
    trainer.default_local_dir=$checkpoint_dir $@
    

    # trainer.total_training_steps=$total_training_steps \
