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

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # 定义单条数据的处理函数（将在线程中运行）
        def process_sample(i):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            
            # 浅拷贝 extra_info 以防止线程间冲突（虽然 DataProto 切片通常返回独立对象，但保险起见）
            # 注意：如果 extra_info 内部包含 mutable 对象且被共享，这里可能需要深拷贝
            if extra_info:
                extra_info = extra_info.copy()
                
            if num_turns is not None:
                extra_info["num_turns"] = num_turns

            # 调用打分函数
            try:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
            except Exception as e:
                print(f"[RewardManager Error] Index {i}: {e}")
                score = 0.0

            # 返回所有需要汇总或打印的信息
            return {
                "index": i,
                "valid_response_length": valid_response_length,
                "score": score,
                "data_source": data_source,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth
            }

        # --- 并行执行 ---
        batch_size = len(data)
        # 建议线程数设为 batch_size，或者限制在 64 左右，避免过多系统开销
        max_workers = min(batch_size, 128)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map 会保持返回顺序与输入 range(batch_size) 一致
            results = list(executor.map(process_sample, range(batch_size)))

        # --- 汇总结果 ---
        already_print_data_sources = {}
        
        for res in results:
            i = res['index']
            valid_response_length = res['valid_response_length']
            score = res['score']
            data_source = res['data_source']

            # 处理 score 返回值（可能是 dict 或 float）
            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            # 填充 reward tensor
            reward_tensor[i, valid_response_length - 1] = reward

            # 打印调试信息（在主线程打印，避免日志错乱）
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", res['prompt_str'])
                print("[response]", res['response_str'])
                print("[ground_truth]", res['ground_truth'])
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor