import json
import re

def load_dataset(dataset_path):
    dataset = []
    if "generation_result" in dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        return dataset
            
    if "math500" in dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            # 遍历每一行
            for line in f:
                # 解析 JSON 数据
                data = json.loads(line)
                dataset.append({
                    "original": data,
                    "question": data.get("problem"),
                    "answer": data.get("answer"),
                    "solution": data.get("solution")
                })

    elif "aime_2024" in dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            aime_2024_dataset = json.load(f)
            for item in aime_2024_dataset:
                dataset.append({
                    "original": item,
                    "question": item.get("problem"),
                    "answer": item.get("answer"),
                    "solution": item.get("solution")
                })

            
    elif "mip_gsm8k" in dataset_path:
       with open(dataset_path, "r", encoding="utf-8") as f:
            gsm8k_dataset = json.load(f)
            for item in gsm8k_dataset:
                dataset.append({
                    "original": item,
                    "question": item.get("insufficient_question"),
                    "answer": item.get("answer").split("####")[1].strip(),
                    "solution": item.get("answer"),
                    "insufficient_info": item.get("insufficient_info",""),
                })
    
    elif "mip_math" in dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            math_dataset = json.load(f)
            for item in math_dataset:
                dataset.append({
                    "original": item,
                    "question": item.get("insufficient_question"),
                    "answer": item.get("answer"),
                    "solution": item.get("solution"),
                    "insufficient_info": item.get("insufficient_info","")
                })

    elif "gsm8k" in dataset_path:
       with open(dataset_path, "r", encoding="utf-8") as f:
            gsm8k_dataset = json.load(f)
            for item in gsm8k_dataset:
                answer = item.get("answer").split("####")[1].strip()
                # some numbers are in the `x,xxx` format, and we want to remove the comma cite: https://github.com/Blue-Raincoat/SelectIT/blob/main/eval/eval/gsm/run_eval.py
                answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
                assert float(answer), f"answer is not a valid number: {answer}"
                dataset.append({
                    "original": item,
                    "question": item.get("question"),
                    "answer": answer,
                    "solution": item.get("answer"),
                    "insufficient_question": item.get("insufficient_question",""),
                    "insufficient_info": item.get("insufficient_info",""),
                })

    elif "math" in dataset_path:
       with open(dataset_path, "r", encoding="utf-8") as f:
            gsm8k_dataset = json.load(f)
            for item in gsm8k_dataset:
                dataset.append({
                    "original": item,
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "solution": item.get("answer"),
                    "insufficient_question": item.get("insufficient_question",""),
                })

                
    else:
        raise ValueError("Unsupported dataset path")
    return dataset
