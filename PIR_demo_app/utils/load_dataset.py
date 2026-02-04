import json
import re

def load_dataset(dataset_path):
    dataset = []
    if "generation_result" in dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
            
    if "gsm8k" in dataset_path:
       with open(dataset_path, "r", encoding="utf-8") as f:
            gsm8k_dataset = json.load(f)
            for item in gsm8k_dataset:
                answer = item.get("answer").split("####")[1].strip()
                # some numbers are in the `x,xxx` format, and we want to remove the comma cite: https://github.com/Blue-Raincoat/SelectIT/blob/main/eval/eval/gsm/run_eval.py
                answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
                assert float(answer), f"answer is not a valid number: {answer}"
                dataset.append({
                    "original": item,
                    "question": item["insufficient_question"].strip(),
                    "user_intent":  item['question'].strip(),
                    "solution": item['answer'].strip()
                })

    elif "collabllm-multiturn-math-hard-large" in dataset_path:
        from datasets import load_dataset as load_hf_dataset
        math_chat_dataset = load_hf_dataset("parquet", data_files=dataset_path, split='train').to_list()
        # print(f"{math_chat_dataset[0]}")
        for item in math_chat_dataset:
            question = item['prompt'][1]['content'].strip()
            answer = item['reward_model']['ground_truth']['target'].strip()
            user_intent = user_intent = item['extra_info']['single_turn_prompt_raw'].strip()
            # from real_interaction_app.utils.math_pir_evaluator import MathAnswerEvaluator

            dataset.append({
                "original": item,
                "question": question,
                "user_intent": user_intent,
                "answer": answer,
                "solution": answer,
            })

    elif "math" in dataset_path:
       with open(dataset_path, "r", encoding="utf-8") as f:
            gsm8k_dataset = json.load(f)
            for item in gsm8k_dataset:
                dataset.append({
                    "original": item,
                    "question": item['insufficient_question'].strip(),
                    "user_intent":  item['question'].strip(),
                    "answer": item['answer'].strip(),
                    "solution": item['answer'].strip(),
                })
                
    else:
        raise ValueError("Unsupported dataset path")
    return dataset

if __name__ == "__main__":
    dataset_path = "/data/home/chenxin/verl_interactive/data/collabllm/collabllm-multiturn-math-hard-large/test.parquet"
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} samples from {dataset_path}")
    print(dataset[0])