import argparse
import re
import json
from openai import OpenAI
from ..utils.model_client import ModelClient
from ..utils.simulator_client import UserSimulatorClient
import os
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from .load_dataset import load_dataset

system_prompt = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, present the final answer after </think>."
    )

user_trigger_prompt = ''' Please reason step by step, and put your final answer within \\boxed{}.'''
 # 定义一个函数用于保存数据
def save_partial_output(dataset, output_file_path, processed_count):
    with open(output_file_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"已处理并保存 {processed_count} 条数据至 {output_file_path}")


def batch_run(args):
    print(args)
    # 单条数据处理函数
    def process_item(idx, item, args):
        try:
            if item.get("output"):
                return idx, item, None  # 跳过已处理

            # 每个线程独立创建自己的 user_simulator 保证线程安全
            user_simulator_client = OpenAI(api_key=args.api_key, base_url=args.user_simulator_url)
            user_simulator = UserSimulatorClient(client=user_simulator_client,
                                                 model_name=args.user_simulator_name,
                                                 task_name="question answering", 
                                                 user_intent=item['question'])

            model_client = ModelClient(
                model_path=args.model_name,
                base_url=args.model_url,
                stop_tokens=["</asking>", "<｜end▁of▁sentence｜>"],
                reasoning_model=args.reasoning_model,
            )

            question = item[args.question_key]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question + user_trigger_prompt},
                {"role": "assistant", "content": model_client.completion},
            ]
            model_response = model_client.chat(messages=messages)

            # 如果模型要求继续追问
            while model_client.stop_reason == "</asking>":
                ask_content = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL).group(1).strip()
                user_response = user_simulator.chat(user_message=ask_content)
                model_client.completion += f"\n<response>{user_response}</response>"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question + user_trigger_prompt},
                    {"role": "assistant", "content": model_client.completion},
                ]
                model_response = model_client.chat(messages=messages)

            item["output"] = model_client.completion
            return idx, item, None

        except Exception as e:
            return idx, None, str(e)

    # === 批处理逻辑 ===
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    file_name = args.input_file.split("/")[-1].split(".json")[0]

    generation_output_file_path = os.path.join(
        args.output_path, f"{args.model_name}_{args.user_simulator_name}_{file_name}_{args.question_key}_interactive_generation_result.json"
    )

    evaluation_output_file_path = os.path.join(
        args.output_path, f"{args.model_name}_{args.user_simulator_name}_{file_name}_{args.question_key}_interactive_eval_result.json"
    )

    # 加载数据
    dataset = load_dataset(args.input_file)

    processed_count = 0
    error_count = 0
    lock = threading.Lock()  # 保证写文件时线程安全

    # 多线程处理
    with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        futures = {
            executor.submit(
                process_item,
                idx,
                item,
                args
            ): idx
            for idx, item in enumerate(dataset)
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Data"):
            idx = futures[future]
            try:
                i, processed_item, error = future.result()
                with lock:
                    if error:
                        error_count += 1
                        print(f"Error processing item {i}: {error}")
                    else:
                        dataset[i] = processed_item
                        processed_count += 1
                        # 每处理 200 条保存一次
                        if processed_count % 200 == 0:
                            save_partial_output(dataset[:processed_count], generation_output_file_path, processed_count)
            except Exception as e:
                with lock:
                    error_count += 1
                    print(f"Unexpected error at item {idx}: {e}")

    # 保存全部
    save_partial_output(dataset, generation_output_file_path, processed_count)
    print(f"处理完成，共处理 {processed_count} 条数据，发生错误 {error_count} 次。")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Generation with User Simulator")
    parser.add_argument("--input_file", type=str, required=False, help="训练数据集路径", default="/home/chenxin/proactive_interactive_r1/datasets/Math/math500_test.jsonl")
    parser.add_argument("--model_url", type=str, required=False, help="模型URL", default="http://localhost:8729/v1/chat/completions")
    parser.add_argument("--model_name", type=str, required=False, help="模型名称", default="distill_r1_coing_neo_cleaned_uncertainty_threshold_40_sft_conversation_train_dataset")
    parser.add_argument("--user_simulator_url", type=str, required=False, help="用户模拟器URL", default="https://api.ai-gaochao.cn/v1/")
    parser.add_argument("--reasoning_model", action="store_true", required=True, help="是否启用推理模型")
    parser.add_argument("--question_key", type=str, required=False, help="问题字段名称", default="question")
    parser.add_argument("--api_key", type=str, required=False, help="用户模拟器API Key", default="sk-xxxxxxx")
    parser.add_argument("--user_simulator_name", type=str, required=False, help="用户模拟器名称", default="gpt-4o")
    parser.add_argument("--output_path", type=str, required=False, help="输出目录", default="results")
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    args = parser.parse_args()
    batch_run(args)