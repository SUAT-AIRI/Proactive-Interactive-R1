import string
import re
import collections
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
from ..utils.model_client import ModelClient
from ..utils.simulator_client import UserSimulatorClient
from ..utils.extract_json_reliable import extract_json
import litellm
from openai import OpenAI

system_prompt = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, present the final answer after </think>."
    )

def load_squad_dataset(args):
    """
    加载 squad rc.wikipedia 的 validation 集。
    """
    print("正在加载数据集...")

    dataset = load_dataset(args.data_dir, split="validation")
    return dataset


def query_llm(prompt, item, args):
    """
    调用 LLM 接口，使用 argparse 传入的参数。
    """
    pir_model = ModelClient(
        model_path=args.model_name,
        base_url=args.model_url,
        stop_tokens=["</asking>","<｜end▁of▁sentence｜>"],
        reasoning_model=True
    )
    user_simulator_client = OpenAI(api_key=args.api_key, base_url=args.user_simulator_url)
    user_simulator = UserSimulatorClient(client=user_simulator_client,
                                                model_name=args.user_simulator_name,
                                                task_name="retrieval question answering", 
                                                context_content = item['context'].strip(),
                                                user_intent=item['question'].strip())
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": pir_model.completion},
        ]
        model_response = pir_model.chat(messages=messages)
        # print("Initial Response:\n", model_response)
        while pir_model.stop_reason == "</asking>":
            ask_content = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL).group(1).strip()

            # print("Model Asks:\n", ask_content)
            user_response = user_simulator.chat(user_message=ask_content)
            # print("User Simulator Response:\n", user_response)
            pir_model.completion += f"\n<response>{user_response}</response>"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": pir_model.completion},
            ]
            model_response = pir_model.chat(messages=messages)
            # print("Updated Model Response:\n", model_response)

        final_response = pir_model.completion
        # print("Final Response:\n", final_response)
        if "</think>" in final_response:
            output = final_response.split("</think>")[-1]
            # output = final_response
        else:
            print("No thinking end process found.")
            output = final_response
        return output,final_response
    except Exception as e:
        print("Main Entrance Error", e)
        return None,None

def process_single_item(item_data):
    """
    处理单条数据。
    item_data 是一个元组: (index, item, args)
    """
    i, item, args = item_data
    question = item['question']
    
    # 获取所有可能的标准答案列表
    ground_truths = item['answers']['text'] 
    
    # --- 构建 Zero-shot Prompt ---
    prompt = f"Answer following question, think step by" \
            f" step and then output the answer in the format of \"The answer is (X)\" at the end.\n\nQuestion: {question}"
    
    # --- 调用模型 ---
    prediction, final_response = query_llm(prompt, item, args)
    
    # 构建结果字典
    result = {
        "index": i,
        "question": question,
        "prompt": prompt,
        "prediction": prediction,
        "final_response": final_response,
        "ground_truths": ground_truths,
    }


    return result

def generate_suqad(args):
    print(args)
    # 1. 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    output_file = os.path.join(args.output_dir, f"squad_results_{args.model_name}.jsonl")
    print(f"结果将保存至: {output_file}")

    # 2. 加载数据
    dataset = load_squad_dataset(args)

    print(f"开始生成，共 {len(dataset)} 个样本，并发数: {args.num_workers}...")

    results = []
    
    # 3. 准备任务列表
    tasks = [(i, item, args) for i, item in enumerate(dataset)]

    # 4. 并发执行
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # 使用 tqdm 显示进度
        futures = {executor.submit(process_single_item, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task failed: {e}")


    print("\n" + "="*30)
    print(f"Final Results ({len(results)} samples processed):")
    print("="*30)

    # 5. 保存详细结果
    results.sort(key=lambda x: x['index']) # 按索引排序
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/chenxin/verl-interactive/generalization_eval/squad/dataset")
    parser.add_argument("--output_dir", "-o", type=str, default="/home/chenxin/verl-interactive/generalization_eval/squad/eval_results/fuck_test")
    parser.add_argument("--model_name",type=str,default="Proactive-Interactive-R1-Math-7B")
    parser.add_argument("--model_url",type=str,default="http://localhost:7015")
    parser.add_argument("--user_simulator_name",type=str,default="gpt-4o-mini")
    parser.add_argument("--user_simulator_url",type=str,default="https://api.ai-gaochao.cn")
    parser.add_argument("--api_key", type=str, default="sk-xxxxx")
    parser.add_argument("--reasoning_model", required=True,action="store_true",
                        help="Whether to use the reasoning model setup")
    parser.add_argument("--num_workers", "-n", type=int, default=32,
                       help="Number of concurrent queries")
    
    args = parser.parse_args()
    
    generate_suqad(args)