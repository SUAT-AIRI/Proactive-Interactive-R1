# For MMLU interactive generation & Eval
python -m generalization_eval.factual_knowledge.run_mmlu_interactive_generation \
    --data_dir /home/chenxin/verl-interactive/datasets/mmlu/data \
    --model_url "http://10.10.128.132:1136" \
    --model_name Proactive-Interactive-R1-Math-7B \
    --user_simulator_url http://10.10.128.132:8725/v1/ \
    --user_simulator_name Llama-3.1-8B-Instruct \
    --save_dir results/mmlu/eval_results/Proactive-Interactive-R1-Math-7B_Llama-3.1-8B-Instruct_interactive_generation \
    --eval_tokenizer_path Proactive-Interactive-R1/Proactive-Interactive-R1-Math-7B \
    --max_workers 16

# For MMLU-Pro generation & Eval
python -m generalization_eval.factual_knowledge.run_mmlu_pro_interactive_generation \
    --data_dir /home/chenxin/verl-interactive/datasets/TIGER-Lab/MMLU-Pro \
    --output_dir /home/chenxin/verl-interactive/results/MMLU_Pro/eval_results/Proactive-Interactive-R1-Math-7B_Llama-3.1-8B-Instruct_interactive_result \
    --model_url "http://10.10.128.132:1136" \
    --model_name Proactive-Interactive-R1-Math-7B \
    --user_simulator_url http://10.10.128.132:8725/v1/ \
    --user_simulator_name Llama-3.1-8B-Instruct \
    --eval_tokenizer_path Proactive-Interactive-R1/Proactive-Interactive-R1-Math-7B \
    --num_workers 16
