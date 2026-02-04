
# For SquAD interactive generation
python -m generalization_eval.retrieval_question_answering.run_squad_interactive_generation \
    --data_dir /home/chenxin/verl-interactive/datasets/rajpurkar/squad \
    --model_url "http://x.x.x.x:xxxx" \
    --model_name Proactive-Interactive-R1-Math-7B \
    --reasoning_model \
    --user_simulator_url "http://x.x.x.x:xxxx/v1/" \
    --user_simulator_name Llama-3.1-8B-Instruct \
    --output_dir /home/chenxin/verl-interactive/results/squad/Proactive-Interactive-R1-Math-7B_Llama-3.1-8B-Instruct_interactive_generation \
    --num_workers 16

# For RetrivQA interactive generation
python -m generalization_eval.retrieval_question_answering.run_retriaqa_interactive_generation \
    --data_dir /home/chenxin/verl-interactive/datasets/mandarjoshi/trivia_qa \
    --model_url "http://x.x.x.x:xxxx" \
    --model_name Proactive-Interactive-R1-Math-7B \
    --user_simulator_url "http://x.x.x.x:xxxx/v1/" \
    --user_simulator_name Llama-3.1-8B-Instruct \
    --output_dir /home/chenxin/verl-interactive/results/retriaqa/Proactive-Interactive-R1-Math-7B_Llama-3.1-8B-Instruct_interactive_generation \
    --num_workers 32


# For Eval
LLM_JUDGEMENT_API_BASE=https://api.ai-gaochao.cn/v1/ LLM_JUDGEMENT_API_KEY="sk-xxxxxxxxx" LLM_JUDGEMENT_MODEL="gpt-4o-mini" python -m generalization_eval.retrieval_question_answering.retriqa_evaluator \
    --input_file /home/chenxin/verl-interactive/results/squad/Proactive-Interactive-R1-Math-7B_Llama-3.1-8B-Instruct_interactive_generation/squad_results_Proactive-Interactive-R1-Math-7B.jsonl \
    --tokenizer_path Proactive-Interactive-R1/Proactive-Interactive-R1-Math-7B \
    --use_llm_judge