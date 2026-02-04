# run on 8xA100
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

export RAY_TMPDIR="/data2/chenxin/ray_tmp"

PROJECT_DIR="$(pwd)"
# CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# model_name="/data1/HF-Models/Qwen/Qwen2.5-0.5B-Instruct"
model_name="/data1/HF-Models/Qwen/Qwen2.5-Math-7B"
train_data="/home/chenxin/verl-interactive/verl-tool_old/data/collabllm_multi/collabllm-multiturn-math-hard-large/train.parquet"
val_data="/home/chenxin/verl-interactive/verl-tool_old/data/collabllm_multi/collabllm-multiturn-math-hard-large/test.parquet"
interaction_config_path="/home/chenxin/verl-interactive/verl-tool_old/verl/examples/interactive-LLM/interaction_config/user_simulator_interaction_config.yaml"
rollout_data_dir="/home/chenxin/verl-interactive/verl-tool_old/verl/verl_step_records"
run_name="qwen2.5-7b_function_rm-math-chat-sgl-multi-w-interaction"

checkpoint_dir="/data2/chenxin/qwen2.5-7b_function_rm-math-chat-sgl-multi-w-interaction"
mkdir -p $checkpoint_dir

SIMULATOR_API_URL="http://10.10.128.132:8725/v1/"
SIMULATOR_API_KEY="sk-1vfcAowGgLnuRcph74Bc349932Dc40C0Af9eDc007eD0785e"
SIMULATOR_NAME="Llama-3.1-8B-Instruct"
export SIMULATOR_URL=$SIMULATOR_API_URL
export SIMULATOR_API_KEY=$SIMULATOR_API_KEY
export SIMULATOR_NAME=$SIMULATOR_NAME


rl_alg=grpo # gae(ppo) or grpo, if grpo, then better set n>1 otherwise the group norm can not be effective
n_gpus_per_node=8
n_nodes=1
n=8
total_epochs=5
batch_size=256
ppo_mini_batch_size=128
ppo_micro_batch_size_per_gpu=2
log_prob_micro_batch_size_per_gpu=8
kl_loss_coef=0
entropy_coeff=0
lr=1e-6
strategy="fsdp"
max_prompt_length=1024
max_response_length=3065
temperature=1.0
top_p=1.0
max_turns=5
tensor_model_parallel_size=1
gpu_memory_utilization=0.6
do_offload=False
ulysses_sequence_parallel_size=1
use_dynamic_bsz=True # faster
fsdp_size=-1


NCCL_P2P_LEVEL=NVL FORCE_TORCHRUN=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$rl_alg \
    algorithm.use_kl_in_reward=False \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric='score' \
    +algorithm.filter_groups.max_num_gen_batches=0 \
    data.train_batch_size=$batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.shuffle=True \
    data.return_raw_chat=True \
    reward_model.launch_reward_fn_async=True \
    +reward_model.overlong_buffer.enable=False \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
    actor_rollout_ref.actor.strategy=$strategy \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=$n \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8196 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='math_chat_async_rl' \
    trainer.experiment_name='qwen2.5-7b_function_rm-math-chat-sgl-multi-w-interaction' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    +trainer.remove_previous_ckpt_in_save=True \
    trainer.save_freq=10 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    data.train_files=$train_data \
    data.val_files=$val_data \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=$interaction_config_path \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.rollout_data_dir=$rollout_data_dir/$run_name \
    trainer.validation_data_dir=$rollout_data_dir/$run_name/validation \
    trainer.total_epochs=$total_epochs $@


