# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from .math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
import os
import json
import numpy as np

def _compute_helpfullness_reward(question, response):
    from .helpfulness_judgement import HelpfullnessMetric
    helpfullness_metric = HelpfullnessMetric()
    helpfulness_reward = helpfullness_metric.score(
        question=question,
        response=response,
    )
    helpfulness_goldens = float(helpfulness_reward)
    return helpfulness_goldens



def _parse_conversation(text):
    # 1. 使用正则表达式根据 user 或 assistant 标签进行分割
    # pattern说明: \n(user|assistant)\n 匹配前后有换行的角色名
    # 使用括号 () 是为了保留分隔符，这样我们知道是谁说的话
    parts = re.split(r'\n(user|assistant)\n', text)
    
    # parts[0] 通常是开头没有标记的文本（如果有的话，通常默认为 assistant 的开场白）
    # 之后的元素是成对出现的：[角色, 内容, 角色, 内容...]
    
    conversation = []
    
    # 处理第一段（如果不是空的）
    if parts[0].strip():
        conversation.append({
            "role": "assistant (start)", # 开头通常默认是助手
            "content": parts[0].strip()
        })
    
    # 处理后续的对话（从索引1开始，步长为2）
    for i in range(1, len(parts), 2):
        role = parts[i].strip()      # 获取角色 (user 或 assistant)
        content = parts[i+1].strip() # 获取对应的内容
        
        conversation.append({
            "role": role,
            "content": content
        })
        
    return conversation


def compute_score(solution_str, ground_truth, extra_info):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_answer(solution_str)
    # print(f"Solution string: {solution_str}")
    # print(f"Ground truth: {ground_truth}")
    # print(f"extra_info: {extra_info}")
    # print(f"Extracted answer: {answer}")
    if answer is None:
        return 0.0
    # open_count = len(extra_info['llm_contents'])
    target_answer = extract_answer(ground_truth.get('target', ground_truth) if isinstance(ground_truth, dict) else ground_truth)
    target_answer_is_correct = grade_answer_mathd(answer, target_answer) or grade_answer_sympy(answer, target_answer)

    if target_answer_is_correct:

        open_count = solution_str.count("\nassistant\n")  # Count the number of times the model responded
        qa_pairs = _parse_conversation(solution_str)
        llm_contents = [chat['content'] for chat in qa_pairs if 'assistant' in chat['role']]

        eff_reward = (5 - open_count) / (5 - 1)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        single_turn_prompt = extra_info.get('single_turn_prompt_raw', None)
        insufficient_question = extra_info.get('insufficient_question', None)
        helpful_reward = []
    
        
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(_compute_helpfullness_reward, question=insufficient_question, response=q)
                for q in llm_contents
            ]
            for f in as_completed(futures):
                helpful_reward.append(f.result())

        # For test
        record = {
        "insufficient_question":insufficient_question,
        "single_turn_prompt": single_turn_prompt,
        "model_solution": solution_str,
        "open_count": open_count,
        "ground_truth": ground_truth,
        "answer": answer,
        "target_answer": target_answer,
        "target_answer_is_correct": target_answer_is_correct,
        "llm_contents": llm_contents,
        "helpful_reward": helpful_reward,
        "eff_reward": eff_reward,
        "final_reward": 0.5 + 0.5* np.mean(helpful_reward) + 0.5*eff_reward,
        }
        try:
            os.makedirs(os.path.dirname("/home/chenxin/verl-interactive/compute_score_log/compute_score_log_LLM_GRPO.jsonl"), exist_ok=True) if os.path.dirname("/home/chenxin/verl-interactive/compute_score_log/compute_score_log_LLM_GRPO.jsonl") else None
            with open("/home/chenxin/verl-interactive/compute_score_log/compute_score_log_LLM_GRPO.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[Logging Error] Could not write log: {e}")

        return 0.5 + 0.5* np.mean(helpful_reward) + 0.5*eff_reward
            
    else:
        return 0.0

