import os
import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat, DocumentSchema, FunctionToolDefinition
from ai21.models.chat import ToolDefinition, ToolParameters
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from ..utils.model_client import ModelClient
from ..utils.simulator_client import UserSimulatorClient
from transformers import AutoTokenizer

API_KEY = ""
system_prompt = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, present the final answer after </think>."
    )
random.seed(12345)

# Add lock for thread-safe file operations
file_lock = threading.Lock()

def get_client():
    client = OpenAI(api_key=API_KEY, base_url=args.url)
    return client

def interactive_call_api(args, instruction, inputs, max_retries=5):
    user_simulator_client = OpenAI(api_key=args.api_key, base_url=args.user_simulator_url)
    user_simulator = UserSimulatorClient(client=user_simulator_client,
                                                 model_name=args.user_simulator_name,
                                                 task_name="factual knowledge", 
                                                 user_intent=inputs.strip())
    pir_model = ModelClient(
        model_path=args.model_name,
        base_url=args.model_url,
        stop_tokens=["</asking>","<｜end▁of▁sentence｜>"],
        reasoning_model=args.reasoning_model
    )
    # print(instruction)
    # print(inputs)
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction + inputs},
            {"role": "assistant", "content": pir_model.completion},
        ]
        model_response = pir_model.chat(messages=messages)
        # print("===============Initial model response===============")
        # print(model_response)
        # 如果模型要求继续追问
        while pir_model.stop_reason == "</asking>":
            ask_content = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL).group(1).strip()

            # print(f"user_simulator.system_prompt: {user_simulator.system_prompt}")
            user_response = user_simulator.chat(user_message=ask_content)

            pir_model.completion += f"\n<response>{user_response}</response>"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction + inputs},
                {"role": "assistant", "content": pir_model.completion},
            ]
            model_response = pir_model.chat(messages=messages)

        final_response = pir_model.completion
        if "</think>" in final_response:
            output = final_response.split("</think>")[-1]
            # output = final_response
        else:
            print("No thinking end process found.")
            output = final_response
        return output, final_response
    except Exception as e:
        print("error", e)
        return None, None

def load_mmlu_pro(data_dir):
    dataset = load_dataset(data_dir)
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res

def format_example(question, options, cot_content=""):
    if cot_content == "":
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = "Question: {}\nOptions: ".format(question)
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}. {}\n".format(choice_map[i], opt)
    if cot_content == "":
        example += "Answer: "
    else:
        example += "Answer: " + cot_content + "\n\n"
    return example

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        # print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

def _compute_token_length(tokenizer,response):
        """Computes token length using the tokenizer."""
        token_count = tokenizer.encode(response, return_tensors='pt')
        return token_count.shape[1]

def single_request(args, single_question, cot_examples_dict):
    """Modified: removed exist_result parameter, check moved outside"""
    category = single_question["category"]
    cot_examples = cot_examples_dict[category]
    question = single_question["question"]
    options = single_question["options"]
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(category)
    # for each in cot_examples:
    #     prompt += format_example(each["question"], each["options"], each["cot_content"])
    input_text = format_example(question, options)
    try:
        output, final_response = interactive_call_api(args, prompt, input_text)
        output = output.replace('**', '')
    except Exception as e:
        print("error", e)
        return None, "Error: " + str(e), "Error: " + str(e)
    pred = extract_answer(output)
    return pred, output, final_response

def update_result(output_res_path):
    category_record = {}
    res = []
    success = False
    while not success:
        try:
            if os.path.exists(output_res_path):
                with open(output_res_path, "r") as fi:
                    res = json.load(fi)
                    for each in res:
                        category = each["category"]
                        if category not in category_record:
                            category_record[category] = {"corr": 0.0, "wrong": 0.0}
                        if not each["pred"]:
                            x = random.randint(0, len(each["options"]) - 1)
                            if x == each["answer_index"]:
                                category_record[category]["corr"] += 1
                            else:
                                category_record[category]["wrong"] += 1
                        elif each["pred"] == each["answer"]:
                            category_record[category]["corr"] += 1
                        else:
                            category_record[category]["wrong"] += 1
            success = True
        except Exception as e:
            print("Error", e, "sleep 2 seconds")
            time.sleep(2)
    return res, category_record

def merge_result(res, curr):
    merged = False
    for i, single in enumerate(res):
        if single["question_id"] == curr["question_id"] and single["question"] == curr["question"]:
            res[i] = curr
            merged = True
    if not merged:
        res.append(curr)
    return res

def process_single_question(args_tuple):
    """New function for parallel processing"""
    args, each, dev_df, output_res_path, subject = args_tuple
    label = each["answer"]

    # Check existence inside thread
    res, category_record = update_result(output_res_path)
    q_id = each["question_id"]
    # for existing in res:
    #     if q_id == existing["question_id"] and each["question"] == existing["question"]:
    #         return None  # Already processed

    pred, output, final_response = single_request(args, each, dev_df)

    with file_lock:  # Thread-safe file operations
        res, category_record = update_result(output_res_path)
        if subject not in category_record:
            category_record[subject] = {"corr": 0.0, "wrong": 0.0}

        each["pred"] = pred
        each["model_output"] = output
        each["final_response"] = final_response
        each["token_length"] = _compute_token_length(AutoTokenizer.from_pretrained(args.eval_tokenizer_path), final_response)

        if pred is not None:
            if pred == label:
                category_record[subject]["corr"] += 1
            else:
                category_record[subject]["wrong"] += 1
        else:
            category_record[subject]["wrong"] += 1

        res = merge_result(res, each)
        save_res(res, output_res_path)
        save_summary(category_record, os.path.join(args.output_dir, subject + "_summary.json"))

    return each

def evaluate(subjects):
    test_df, dev_df = load_mmlu_pro(args.data_dir)
    if not subjects:
        subjects = list(test_df.keys())
    print("assigned subjects", subjects)

    for subject in subjects:
        test_data = test_df[subject]
        output_res_path = os.path.join(args.output_dir, subject + "_result.json")

        # Prepare arguments for parallel processing
        process_args = [
            (args, each, dev_df, output_res_path, subject)
            for each in test_data
        ]

        # Process X queries at a time
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(
                executor.map(process_single_question, process_args),
                total=len(test_data),
                desc=f"Processing {subject}"
            ))

def save_res(res, output_res_path):
    temp = []
    exist_q_id = []
    for each in res:
        if each["question_id"] not in exist_q_id:
            exist_q_id.append(each["question_id"])
            temp.append(each)
    res = temp
    with open(output_res_path, "w") as fo:
        fo.write(json.dumps(res))

def save_summary(category_record, output_summary_path):
    total_corr = 0.0
    total_wrong = 0.0
    for k, v in category_record.items():
        if k == "total":
            continue
        cat_acc = v["corr"] / (v["corr"] + v["wrong"]) if (v["corr"] + v["wrong"]) > 0 else 0
        category_record[k]["acc"] = cat_acc
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    acc = total_corr / (total_corr + total_wrong) if (total_corr + total_wrong) > 0 else 0
    category_record["total"] = {"corr": total_corr, "wrong": total_wrong, "acc": acc}
    with open(output_summary_path, "w") as fo:
        fo.write(json.dumps(category_record))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="/home/chenxin/verl-interactive/datasets/TIGER-Lab/MMLU-Pro")
    parser.add_argument("--output_dir", "-o", type=str, default="/home/chenxin/verl-interactive/generalization_eval/MMLU-Pro-main/eval_results/test")
    parser.add_argument("--assigned_subjects", "-a", type=str, default="all")
    parser.add_argument("--model_url", "-u", type=str, default="http://localhost:6092")
    parser.add_argument("--model_name", "-m", type=str, default="Proactive-Interactive-R1-Math-7B")
    parser.add_argument("--reasoning_model", required=True,action="store_true",
                        help="Whether to use the reasoning model setup")
    parser.add_argument("--user_simulator_url", type=str, default="https://api.ai-gaochao.cn")
    parser.add_argument("--user_simulator_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_key", type=str, default="sk-xxxxx")
    parser.add_argument("--eval_tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer for evaluating token lengths")
    parser.add_argument("--num_workers", "-n", type=int, default=1,
                       help="Number of concurrent queries")
    assigned_subjects = []
    args = parser.parse_args()
    if args.assigned_subjects == "all":
        assigned_subjects = []
    else:
        assigned_subjects = args.assigned_subjects.split(",")
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(assigned_subjects)
    # static result
    import json
    import os
    import glob
    import argparse
    import pandas as pd

    def analyze_results(output_dir, save_path=None):
        pattern = os.path.join(output_dir, "*_result.json")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No result files found in directory {output_dir}.")
            return

        results = []
        total_correct = 0
        total_count = 0

        print(f"Analyzing {len(files)} subject files...\n")

        for file_path in files:
            subject_name = os.path.basename(file_path).replace("_result.json", "")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Failed to read file {file_path}: {e}")
                continue

            subject_total = len(data)
            subject_correct = 0
            
            for item in data:
                pred = item.get("pred")
                label = item.get("answer")
                if pred and pred == label:
                    subject_correct += 1
            
            if subject_total > 0:
                acc = subject_correct / subject_total
            else:
                acc = 0.0

            results.append({
                "Subject": subject_name,
                "Correct": subject_correct,
                "Total": subject_total,
                "Accuracy": acc,
                "avg_token_length": sum(item.get("token_length", 0) for item in data) / subject_total if subject_total > 0 else 0.0
            })

            total_correct += subject_correct
            total_count += subject_total

        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values(by="Subject")
        
        # Calculate overall accuracy
        if total_count > 0:
            overall_acc = total_correct / total_count
        else:
            overall_acc = 0.0

        # Print results
        print("="*50)
        print("MMLU-Pro Detailed Evaluation Results")
        print("="*50)
        df_display = df.copy()
        df_display["Accuracy"] = df_display["Accuracy"].apply(lambda x: f"{x:.2%}")
        print(df_display.to_markdown(index=False)) 
        print("\n")
        print("="*50)
        print(f"Overall Accuracy: {overall_acc:.2%} ({total_correct}/{total_count})")
        print("="*50)
        print("\n")
        print(f"Weighted Average Token Length: {sum(item['avg_token_length'] * item['Total'] for item in results) / total_count if total_count > 0 else 0.0:.2f}")
        print("\n")

        # Save results to file
        if save_path is None:
            save_path = os.path.join(output_dir, "evaluation_summary")
    
        # Save as JSON
        summary = {
            "subjects": results,
            "overall": {
                "total_correct": total_correct,
                "total_count": total_count,
                "accuracy": overall_acc,
                "weighted_avg_token_length": sum(item["avg_token_length"] * item["Total"] for item in results) / total_count if total_count > 0 else 0.0
            }
        }
        with open(f"{save_path}.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {save_path}.json")

        return summary

    # Usage
    result = analyze_results(
        args.output_dir
    )