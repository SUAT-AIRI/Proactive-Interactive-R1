# For Generate
python -m generalization_eval.miss_premise_testing.run_mip_interactive_generation \
    --input_file "/home/chenxin/verl-interactive/datasets/gsm8k.json" \
    --model_url "http://x.x.x.x:xxxx" \
    --model_name Proactive-Interactive-R1-Math-7B \
    --reasoning_model \
    --user_simulator_url "http://x.x.x.x:xxxx/v1/" \
    --question_key "insufficient_question" \
    --user_simulator_name Llama-3.1-8B-Instruct \
    --api_key sk-1vfcAowGgLnuRcph74Bc349932Dc40C0Af9eDc007eD0785e \
    --max_workers 16

# For Evaluate
LLM_JUDGEMENT_API_BASE=https://api.ai-gaochao.cn/v1/ LLM_JUDGEMENT_API_KEY="sk-xxxxxxxxx" LLM_JUDGEMENT_MODEL="gpt-4o-mini" python -m generalization_eval.miss_premise_testing.math_answer_evaluator \
    --dataset_path /home/chenxin/verl-interactive/gpt-4o-mini_math_insufficient_question_direct_generation_result.json \
    --tokenizer_path Proactive-Interactive-R1/Proactive-Interactive-R1-Math-7B \
    --reasoning_model \
    --use_llm_judge