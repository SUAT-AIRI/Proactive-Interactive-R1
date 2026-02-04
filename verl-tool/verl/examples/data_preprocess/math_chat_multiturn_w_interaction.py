# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

with open("/home/chenxin/verl-interactive/verl-tool_old/examples/data_preprocess/proactive_prompt.txt", "r") as f:
    system_prompt = f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/home/chenxin/verl-interactive/verl-tool_old/data/collabllm_multi/collabllm-multiturn-math-hard-large")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "collabllm-multiturn-math-hard"
    dataset = datasets.load_dataset(args.local_dir)['train']
    from datasets import Dataset
    train_dataset = Dataset.from_list(dataset.select(range(0, len(dataset) - 1000)))
    test_dataset = Dataset.from_list(dataset.select(range(len(dataset) - 1000, len(dataset))))

    instruction_following = " Please put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            single_turn_prompt_raw = example.pop('single_turn_prompt')
            # ground_truth answer
            single_turn_completion_raw = example.pop('single_turn_completion')
            insufficient_question_raw = example.pop('prompt')[0]['content']
            completion_raw = example.pop('completion')
            conv_id_raw = example.pop('conv_id')
            score_raw = example.pop('score')
            session_raw = example.pop('sessions')
            rewards_raw = example.pop('rewards')

            # question_raw = example.pop("question")

            question = insufficient_question_raw + instruction_following

            # answer_raw = example.pop("answer")
            # solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule", 
                    "ground_truth": {
                        "target": single_turn_completion_raw,
                    }},
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'single_turn_prompt_raw': single_turn_prompt_raw,
                    "insufficient_question": insufficient_question_raw,
                    "data_source": data_source,
                    "interaction_kwargs": {
                        "name": "collabllm-multiturn-math-hard",
                        "query": single_turn_completion_raw,
                        "ground_truth": single_turn_completion_raw,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
