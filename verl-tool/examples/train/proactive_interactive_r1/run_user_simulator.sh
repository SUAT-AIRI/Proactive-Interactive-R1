CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --served-model-name Llama-3.1-8B-Instruct \
    --max-model-len 8192 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.98 \
    --trust-remote-code \
    --port 8725