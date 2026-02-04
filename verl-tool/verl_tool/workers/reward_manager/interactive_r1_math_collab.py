"""
Interactive-R1 style Math Exact Match Reward Manager
"""
import torch
import random
import regex as re
import json
import time
import os
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple
from verl import DataProto
from verl.workers.reward_manager.registry import register
from .reward_score import _default_compute_score
from collections import defaultdict
from pathlib import Path
# from math_verify import LatexExtractionConfig, parse, verify
# from latex2sympy2_extended import NormalizationConfig
from .math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from .metrics.helpfulness import HelpfullnessMetric

# Global locks to ensure thread safety for file I/O and printing
FILE_LOCK = threading.Lock()
f = threading.Lock()

def _count_asking_tags(text):
    opening_tags = text.count("<asking>")
    closing_tags = text.count("</asking>")
    return opening_tags, closing_tags

def _extract_qa_pairs(think_inner):

    if "<asking>" not in think_inner or "<response>" not in think_inner or "</asking>" not in think_inner or "</response>" not in think_inner:
        return False, ""

    pattern_full = r"^(?:\s*\S.*?<asking>\s*\S.*?</asking>\s*<response>\s*\S.*?</response>\s*.*?)*$"

    if re.match(pattern_full, think_inner, re.DOTALL):
        pairs = re.findall(
            r"<asking>\s*(\S.*?)\s*</asking>\s*<response>\s*(\S.*?)\s*</response>",
            think_inner,
            re.DOTALL,
        )

        return True, pairs

    return False, ""


def _compute_helpfullness_reward(question, response, qa_pairs):
    helpfullness_metric = HelpfullnessMetric()
    helpfulness_reward = helpfullness_metric.score(
        question=question,
        response=response,
        qa_pairs=qa_pairs,

    )
    helpfulness_goldens = float(helpfulness_reward)
    return helpfulness_goldens


def compute_score(solution_str, ground_truth, base_score = 0.5):
    """
    The scoring function for Interactive-R1 style exact match (EM).
    """
    if "</think>" in solution_str.split("<｜Assistant｜>")[-1]:
        model_think = solution_str.split("<｜Assistant｜>")[-1].split("</think>")[0]
        model_output = solution_str.split("<｜Assistant｜>")[-1].split("</think>")[1]

    else:
        # model_output = solution_str.split("<｜Assistant｜>")[-1]
        model_output = ""
        model_think = ""
    answer = extract_answer(model_output)
    # answer = _parse_answer(solution_str)
    open_count, close_count = _count_asking_tags(model_think)

    do_print = random.randint(1, 64) == 1

    if do_print:
        with PRINT_LOCK:
            print("--------------------------------")
            # ground truth
            print(f"Golden answers: {ground_truth['target']}")

            # extracted answer from model
            if answer is not None:
                print(f"Extracted answer is not None: {answer}")
            else:
                print("Extracted answer: None!")

            # raw output of the model
            print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0, 0.0, 0.0
    else:
        # Handle both dict and list ground truth formats
        target_answer = extract_answer(ground_truth['target'])
        target_aquestion = ground_truth['insufficient_question']
        single_turn_prompt_raw = ground_truth['single_turn_prompt_raw']
        target_answer_is_correct = grade_answer_mathd(answer, target_answer) or grade_answer_sympy(answer, target_answer)
        judge, qa_pairs = _extract_qa_pairs(model_think)

        if target_answer_is_correct:
            if open_count > 0 or close_count > 0:
                if judge and open_count == close_count:
                    eff_reward = (5 - open_count) / (5 - 1)
                    
                    helpful_reward = []

                    futures = []
                    # Keep the internal thread pool as requested, logic unchanged
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        for i, (current_asking, current_response) in enumerate(qa_pairs):

                            prev_history = qa_pairs[:i]
                            
                            futures.append(
                                executor.submit(
                                    _compute_helpfullness_reward,
                                    question=single_turn_prompt_raw, # 原始题目
                                    response=current_asking,      # 当前要打分的 asking
                                    qa_pairs=prev_history          # 之前的 QA 历史
                                )
                            )
                        # 用索引保持顺序
                        results = [None] * len(qa_pairs)
                        for i, f in enumerate(futures):
                            results[i] = futures[i].result()  # 不用 as_completed
                        helpful_reward = results

                    # return base_score + (base_score * np.mean(helpful_reward) * eff_reward)
                    return base_score, helpful_reward, eff_reward
                else:
                    return base_score, 0.0, 0.0
            else:
                return base_score, 0.0, 0.0
        else:
            return 0.0, 0.0, 0.0


@register("interactive_r1_math_collab")
class InteractiveR1MathRewardManager:
    name = "interactive_r1_math_collab"
    
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, base_score=0.5, run_id=None, **kwargs) -> None:
        if tokenizer is None:
            from transformers import AutoTokenizer
            model_path = "/data1/HF-Models/Qwen/Qwen2.5-7B-Instruct" 
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                tokenizer = None
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.base_score = base_score
        self.step = None
        self.run_id = run_id
        self.reward_max_workers = int(os.getenv('REWARD_MAX_WORKERS', 64))
        
        # 预热
        # get_helpfulness_metric()

    def _setup_record_dir(self, data):
        if not hasattr(self, 'record_dir'):
            base_parent = Path(__file__).parent.parent.parent.parent / "verl_step_records"
            if hasattr(self, 'run_id') and self.run_id:
                self.record_dir = base_parent / self.run_id
            else:
                self.record_dir = base_parent / f"torl-{time.strftime('%Y-%m-%d-%H-%M-%S')}"
            self.record_dir.mkdir(parents=True, exist_ok=True)

        if self.step is None:
            max_step = -1
            if self.record_dir.exists():
                for file in os.listdir(self.record_dir):
                    match = re.search(r"step(?:-val)?-(\d+)\.json", file)
                    if match:
                        step_idx = int(match.group(1))
                        if step_idx > max_step:
                            max_step = step_idx
            self.step = max_step + 1
        
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

    def __call__(self, data: DataProto, return_dict=False):
        save_record = data.meta_info.get('save_record', True)
        self._setup_record_dir(data)

        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # 用于临时存储结果的列表，确保索引对应
        batch_results = [None] * batch_size
        already_print_cnt = 0

        # --- Worker Function ---
        def process_item(i, data_item):
            try:
                prompt_ids = data_item.batch['prompts']
                prompt_length = prompt_ids.shape[-1]
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                sequences_str = self.tokenizer.decode(sequences)

                if 'reward_model' in data_item.non_tensor_batch:
                    ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
                else:
                    ground_truth = data_item.non_tensor_batch.get('ground_truth', 
                                data_item.non_tensor_batch.get('golden_answers', []))
                
                data_source = data_item.non_tensor_batch.get('data_source', 'unknown')

                # 调用计算函数
                # calculated_score = compute_score(
                #     solution_str=sequences_str, 
                #     ground_truth=ground_truth, 
                # )
                
                base_score, helpfulness_score, efficiency_score = compute_score(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth, 
                )
                calculated_score = base_score + (base_score * np.mean(helpfulness_score) * efficiency_score)

                record = {
                    'id': data_item.non_tensor_batch['extra_info']['id'] if 'id' in data_item.non_tensor_batch['extra_info'] else None,
                    'data_source': data_source,
                    "prompt": self.tokenizer.decode(prompt_ids[-valid_prompt_length:], skip_special_tokens=False),
                    "response": self.tokenizer.decode(response_ids[:valid_response_length], skip_special_tokens=False),
                    'ground_truth': ground_truth,
                    'score': calculated_score,
                    "base_score": base_score,
                    'helpfulness_score': helpfulness_score,
                    'efficiency_score': efficiency_score,
                    'tool_interact_info': data[i].non_tensor_batch.get('tool_interact_info', None),
                    'extra_info': data_item.non_tensor_batch.get('extra_info', None),
                }
                
                if "turns_stats" in data_item.non_tensor_batch:
                    record['num_turn'] = data_item.non_tensor_batch["turns_stats"]
                    record['num_valid_action'] = data_item.non_tensor_batch["valid_action_stats"]
                    record['is_done'] = not data_item.non_tensor_batch["active_mask"]

                return i, calculated_score, valid_response_length, record

            except Exception as e:
                print(f"[Batch Error] Item {i}: {e}")
                return i, 0.0, 1, None

        # --- 并行执行 ---
        # 这里的 max_workers 设置为 32 或 64
        with ThreadPoolExecutor(max_workers=64) as executor:
            # 提交任务
            futures = [executor.submit(process_item, i, data[i]) for i in range(batch_size)]
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    i, score, valid_len, record = future.result()
                    batch_results[i] = (score, valid_len, record)
                except Exception as e:
                    print(f"Future Error: {e}")

        # --- 关键：按原始顺序组装结果 (Strict Order Assembly) ---
        to_save_records = []
        for i in range(batch_size):
            res = batch_results[i]
            
            # 如果某个线程彻底失败，填充默认值
            if res is None:
                score, valid_len, record = 0.0, 1, None
            else:
                score, valid_len, record = res
            
            # 1. 填充 Tensor
            idx_pos = max(0, int(valid_len) - 1)
            idx_pos = min(idx_pos, reward_tensor.shape[1] - 1)
            reward_tensor[i, idx_pos] = score
            
            # 2. 填充 extra_info (必须有序！)
            # 修复了 KeyError: 'score'
            reward_extra_info['score'].append(score) 
            
            if score > 0:
                reward_extra_info['correct_response_length'].append(valid_len)
            else:
                reward_extra_info['wrong_response_length'].append(valid_len)
            
            # 3. 收集记录
            to_save_records.append(record)
            
            # 调试打印
            if already_print_cnt < self.num_examine:
                already_print_cnt += 1
                print(f"=== Debug Item {i} ===")
                print(f"Score: {score}")
                print("=" * 30)

        # --- 保存记录 ---
        if save_record:
            filename = f"{self.name}-step-val-{self.step}.json" if self.num_examine == 1 else f"{self.name}-step-{self.step}.json"
            save_path = self.record_dir / filename
            valid_records = [r for r in to_save_records if r is not None]
            try:
                if save_path.exists():
                    with open(save_path, "r") as f:
                        existing = json.load(f)
                    existing.extend(valid_records)
                    with open(save_path, "w") as f:
                        json.dump(existing, f, indent=4)
                else:
                    with open(save_path, "w") as f:
                        json.dump(valid_records, f, indent=4)
            except Exception as e:
                print(f"Save Error: {e}")

        self.step += 1

        # 计算平均长度统计
        c_len = reward_extra_info.get('correct_response_length', [])
        w_len = reward_extra_info.get('wrong_response_length', [])
        mean_c = float(np.mean(c_len)) if c_len else 0.0
        mean_w = float(np.mean(w_len)) if w_len else 0.0
        
        # 广播回列表长度
        reward_extra_info['correct_response_length'] = [mean_c] * batch_size
        reward_extra_info['wrong_response_length'] = [mean_w] * batch_size

        if return_dict: 
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor