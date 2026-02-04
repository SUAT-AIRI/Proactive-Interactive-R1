import argparse
import re
import json
from openai import OpenAI
from utils.model_client import ModelClient
import os
from tqdm import tqdm
from utils.load_dataset import load_dataset
from utils.math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd

system_prompt = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, present the final answer after </think>."
    )

user_trigger_prompt = ''' Please reason step by step, and put your final answer within \\boxed{}.'''

# Define a function to save data
def save_partial_output(dataset, output_file_path, processed_count):
    with open(output_file_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    print(f"Processed and saved {processed_count} items to {output_file_path}")

def eval_output(model_completion, solution):
    
    def verify_answer(output_content, item):
        """Verify answer"""
        try:
            if item.get("answer"):
                gold_parsed = item['answer']
            else:
                gold_parsed = extract_answer(solution)

            answer_parsed = extract_answer(output_content)
            print(f"Extracted Answer: {answer_parsed}, Gold: {gold_parsed}")
            
            try:
                reward = grade_answer_mathd(answer_parsed, gold_parsed) or grade_answer_sympy(answer_parsed, gold_parsed)
            except Exception as e:
                print(f"Verification failed: {e}")
                reward = False
            
            return reward, answer_parsed, gold_parsed
        except Exception as e:
            print(f"Answer extraction failed: {e}")
            return None, None, None
    if "</think>" in model_completion:
        output_content = model_completion.split('</think>')[-1].strip()
    else:
        output_content = model_completion.strip()

    reward, answer_parsed, gold_parsed = verify_answer(model_completion, solution) if model_completion else (None, None, None)

    result = {
        "reward": reward,
        "answer": answer_parsed,
        "gold": gold_parsed,
        "output": output_content,
    }

    return result

def batch_run(args):
    print(args)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    file_name = args.input_file.split("/")[-1].split(".json")[0]

    # Modify the file name to mark it as human_interactive
    generation_output_file_path = os.path.join(
        args.output_dir, f"{args.model_name}_human_interactive_{file_name}_{args.question_key}_generation_result.json"
    )

    # evaluation_output_file_path = os.path.join(
    #     args.output_path, f"{args.model_name}_human_interactive_{file_name}_{args.question_key}_eval_result.json"
    # )

    # Load data
    dataset = load_dataset(args.input_file)[0:1]
    processed_count = 0
    error_count = 0
    
    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Data"):
        try:
            if item.get("output"):
                processed_count += 1
                continue  # 跳过已处理

            print(f"\n{'='*20} Now processing the {idx+1} data {'='*20}")
            question = item['question']
            intend_question = item['user_intent']
            solution = item['solution']
            print(f"[Your Intend Question (Hidden to Assistant)]: {intend_question}")
            print(f"[Your Question To Assistant]: {question}")
            print(f"[Your Gold Solution]: {solution}\n")

            # Initialize model client
            model_client = ModelClient(
                model_path=args.model_name,
                base_url=args.model_url,
                stop_tokens=["</asking>", "<｜end▁of▁sentence｜>"],
                reasoning_model=True,
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": model_client.completion},
            ]
            
            print(">>> Model Are Reasoning...")
            model_response = model_client.chat(messages=messages)
            # print(model_response)

            # If the model requests to continue asking
            while model_client.stop_reason == "</asking>":
                # Extract the question the model wants to ask
                ask_match = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL)
                if ask_match:
                    ask_content = ask_match.group(1).strip()
                    print(f"\n[Model Reasoning]: {model_response.split(f'<asking>{ask_content}</asking>')[0].strip()}")
                    print(f"[Model Asking]: {ask_content}")
                    
                    # Wait for real user input
                    user_response = input("[User Response] (Press Enter to send): ").strip()
                    print("-" * 50)
                    # ===============================                    

                    model_client.completion += f"\n<response>{user_response}</response>"
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question + user_trigger_prompt},
                        {"role": "assistant", "content": model_client.completion},
                    ]
                    print(">>> Model Received Response, Continuing Reasoning...")
                    model_response = model_client.chat(messages=messages)
                    # print(f"[Model Again Response]: {model_response}\n")
                else:
                    # 如果匹配不到 asking 标签但停止原因是 asking，可能是格式错误，强制跳出
                    print("Warning: Detected asking stop reason but regex failed.")
                    break

            item["output"] = model_client.completion
            eval_result = eval_output(model_client.completion, item)
            processed_count += 1
            
            # Print final output preview
            print(f"\n[Model Final Output]:\n{model_client.completion.split('</think>')[-1].strip()}")

            print(f"[Evaluation Result]: Reward: {eval_result['reward']}, Model Answer: {eval_result['answer']}, Gold Answer: {eval_result['gold']}")

            # Save after processing each item to prevent data loss during manual input
            save_partial_output(dataset[:processed_count], generation_output_file_path, processed_count)

        except KeyboardInterrupt:
            print("\nUser manually interrupted the program. Saving processed data...")
            save_partial_output(dataset, generation_output_file_path, processed_count)
            exit()
        except Exception as e:
            error_count += 1
            print(f"Error processing item {idx}: {e}")

    # Save all
    save_partial_output(dataset, generation_output_file_path, processed_count)
    print(f"Processing completed. Total processed: {processed_count}, Errors occurred: {error_count}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Generation with Human Input")
    parser.add_argument("--input_file", type=str, required=False, help="训练数据集路径", default="/data/home/chenxin/verl_interactive/datasets/mip/gsm8k.json")
    parser.add_argument("--model_url", type=str, required=False, help="模型URL", default="http://localhost:1136")
    parser.add_argument("--model_name", type=str, required=False, help="模型名称", default="Proactive-Interactive-R1-Math-7B")
    parser.add_argument("--question_key", type=str, required=False, help="问题字段名称", default="question")
    parser.add_argument("--output_dir", type=str, required=False, help="输出目录", default="results/")
    args = parser.parse_args()
    batch_run(args)