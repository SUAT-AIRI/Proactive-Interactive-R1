import argparse
import os
import torch
import numpy as np
import pandas as pd
import time
import json
from tqdm import tqdm
import time
from .mmlu_categories import subcategories, categories
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
import re
from ..utils.model_client import ModelClient
from ..utils.simulator_client import UserSimulatorClient
from openai import OpenAI
from transformers import AutoTokenizer

choices = ["A", "B", "C", "D"]
system_prompt = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, present the final answer after </think>."
    )

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

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def interactive_call_api(args, prompt, question, max_retries=5):
    # print("Prompt to the model:\n", prompt)
    pir_model = ModelClient(
        model_path=args.model_name,
        base_url=args.model_url,
        stop_tokens=["</asking>","<｜end▁of▁sentence｜>"],
        reasoning_model=args.reasoning_model
    )
    user_simulator_client = OpenAI(api_key=args.api_key, base_url=args.user_simulator_url)
    user_simulator = UserSimulatorClient(client=user_simulator_client,
                                                 model_name=args.user_simulator_name,
                                                 task_name="factual knowledge", 
                                                 user_intent=question)
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": pir_model.completion},
        ]
        model_response = pir_model.chat(messages=messages)
        
        while pir_model.stop_reason == "</asking>":
            ask_content = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL).group(1).strip()
        
            user_response = user_simulator.chat(user_message=ask_content)
        
            pir_model.completion += f"\n<response>{user_response}</response>"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
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
        return output,final_response
    except Exception as e:
        print("error", e)
        return None,None


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}. Think step by" \
             " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n" \
        .format(subject)
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def process_single_prompt(prompt, question, args):
    try:
        # 调用 API
        output, final_response = interactive_call_api(args, prompt, question)
        output = output.replace('**', '')
        
        # 提取答案
        pred = extract_answer(output)
        return pred, output, final_response
        
    except Exception as e:
        print(f"Error processing prompt: {e}")
        # 发生错误时，返回空值或特定的错误标记，以免打断整个进程
        return None,"Error processing prompt: {e}", "Error processing prompt: {e}"

def _compute_token_length(tokenizer,response):
        """Computes token length using the tokenizer."""
        token_count = tokenizer.encode(response, return_tensors='pt')
        return token_count.shape[1]


def eval_vllm(args, subject, dev_df, test_df):
    prompts = []
    
    questions = []
    # print("Test_df")
    # print(test_df)
    for i in range(0, test_df.shape[0]):
        questions.append(test_df.iloc[i,0])
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if prompt[-1] in ["\n", " "]:
            prompt += " Let's think step by step."
        else:
            prompt += " Let's think step by step."
                
        prompts.append(prompt)

    assert len(prompts) == len(questions), "Length of prompts and questions must be the same."
    outputs = []
    from functools import partial
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 使用 partial 将 args 固定，因为 map 只接受一个可变参数
        func = partial(process_single_prompt, args=args)

        results_generator = list(tqdm(
            executor.map(func, prompts, questions), 
            total=len(prompts), 
            desc="Evaluating {}".format(subject)
        ))

    # results_generator = []
    # for prompt, question in tqdm(zip(prompts, questions), desc="Evaluating {}".format(subject)):
    #     pred, response = process_single_prompt(prompt, question, args)
    #     results_generator.append((pred, response))

    
    cors = []
    outputs = []
    final_responses = []
    ground_truths = test_df.iloc[:, -1].values
    # print(responses)
    for idx, (pred, output, final_response) in enumerate(results_generator):
        if pred is None:
            cors.append(False)
            outputs.append(output)
            final_responses.append(final_response)
        else:
            prediction = pred.strip()
            # 使用 idx 获取对应的真实标签
            ground_truth = ground_truths[idx] 
            cors.append(prediction == ground_truth)
            outputs.append(output)
            final_responses.append(final_response)
            # print(prediction, ground_truth)

    acc = np.mean(cors)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, outputs, final_responses


def main(args):
    print(f"args: {args}")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if args.subjects:
        assert all(subj in subjects for subj in args.subjects), f"Some of the subjects you specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    tokenizer = AutoTokenizer.from_pretrained(args.eval_tokenizer_path)
    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        if args.n_instances and args.n_instances < test_df.shape[0]:
            test_df = test_df.sample(args.n_instances, random_state=42)


        cors, acc, outputs, final_responses = eval_vllm(args, subject, dev_df, test_df)
            
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["correct"] = cors
        test_df["model_output"] = outputs
        test_df["final_response"] = final_responses
        test_df['token_length'] = [_compute_token_length(tokenizer, resp) for resp in final_responses]
        # test_df["model_response"] = response
        test_df.to_csv(
            os.path.join(
                args.save_dir, "{}.csv".format(subject)
            ),
            index=None,
        )
    # print(subcat_cors)
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        subcat_avg_token_length = np.mean([_compute_token_length(tokenizer, resp) for sublist in subcat_cors[subcat] for resp in sublist])
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        print("Average token length {:.3f} - {}".format(subcat_avg_token_length, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        cat_avg_token_length = np.mean([_compute_token_length(tokenizer, resp) for sublist in cat_cors[cat] for resp in sublist])
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        print("Average token length {:.3f} - {}".format(cat_avg_token_length, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))
    weighted_avg_token_length = np.mean([_compute_token_length(tokenizer, resp) for sublist in all_cors for resp in sublist])
    print("Average token length: {:.3f}".format(weighted_avg_token_length))

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "average_acc": weighted_acc,
                "weighted_avg_token_length": weighted_avg_token_length,
                "subcat_acc": {
                    subcat: np.mean(np.concatenate(subcat_cors[subcat]))
                    for subcat in subcat_cors
                },
                "cat_acc": {
                    cat: np.mean(np.concatenate(cat_cors[cat]))
                    for cat in cat_cors
                },
                "subcat_avg_token_length": {
                    subcat: np.mean([_compute_token_length(tokenizer, resp) for sublist in subcat_cors[subcat] for resp in sublist])
                    for subcat in subcat_cors
                },
                "cat_avg_token_length": {
                    cat: np.mean([_compute_token_length(tokenizer, resp) for sublist in cat_cors[cat] for resp in sublist])
                    for cat in cat_cors
                },
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain",type=int,default=0)
    parser.add_argument("--data_dir",type=str,default="/home/chenxin/verl-interactive/lighteval/dataset/cais/mmlu/data")
    parser.add_argument("--save_dir",type=str,default="/home/chenxin/verl-interactive/generalization_eval/mmlu/test_eval_results")
    parser.add_argument("--subjects",nargs="*",help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated")
    parser.add_argument("--n_instances",type=int,help="if specified, a maximum of n_instances per subject will be used for the evaluation.")
    parser.add_argument("--model_name",type=str,default="Proactive-Interactive-R1-Math-7B")
    parser.add_argument("--model_url",type=str,default="http://localhost:7015")
    parser.add_argument("--reasoning_model", required=True,action="store_true",
                        help="Whether to use the reasoning model setup")
    parser.add_argument("--user_simulator_name",type=str,default="gpt-4o-mini")
    parser.add_argument("--user_simulator_url",type=str,default="https://api.ai-gaochao.cn")
    parser.add_argument("--api_key",type=str,default="sk-xxxxxx")
    parser.add_argument("--max_workers",type=int,default=10,help="number of parallel workers for evaluation")
    parser.add_argument("--eval_tokenizer_path", type=str, default="gpt2", help="Path to the tokenizer model")
    args = parser.parse_args()
    main(args)
