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
"""
Preprocess the MIP dataset to parquet format
"""

import re
import os

import json
from verl.utils.hdfs_io import copy, makedirs
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/chenxin/verl-interactive/verl-tool_old/data/collabllm/collabllm-multiturn-bigcodebench-large')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'collabllm-multiturn-bigcodebench'

    from datasets import load_dataset
    dataset = load_dataset(args.local_dir)['train']
    from datasets import Dataset
    train_dataset = Dataset.from_list(dataset.select(range(0, len(dataset) - 1000)))
    test_dataset = Dataset.from_list(dataset.select(range(len(dataset) - 1000, len(dataset))))

    system_prompt = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    "You can ask as many times as your want. "
    "If you find no further external knowledge needed, present the final answer after </think>."
    )


    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            # profile
            single_turn_prompt_raw = example.pop('single_turn_prompt')
            # ground_truth answer
            single_turn_completion_raw = example.pop('single_turn_completion')
            insufficient_question_raw = example.pop('prompt')[0]['content']
            single_turn_metadata_raw = example.pop('single_turn_metadata')
            completion_raw = example.pop('completion')
            conv_id_raw = example.pop('conv_id')
            score_raw = example.pop('score')
            session_raw = example.pop('sessions')
            rewards_raw = example.pop('rewards')

            if 'You should write self-contained code starting with:' not in single_turn_prompt_raw:
                raise ValueError("Prompt format error!")
            instruction_prompt = single_turn_prompt_raw.split("You should write self-contained code starting with:")[-1].strip()
            instruction_prompt = "\nYou should write self-contained code starting with:\n" + instruction_prompt

            insufficient_question = insufficient_question_raw.strip() + instruction_prompt

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": insufficient_question,
                    }],
                "ability": "mip_math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": single_turn_completion_raw,
                        "insufficient_question":insufficient_question_raw,
                        "single_turn_prompt": single_turn_prompt_raw,
                        "single_turn_metadata":single_turn_metadata_raw,
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'single_turn_prompt_raw': single_turn_prompt_raw,
                    "insufficient_question": insufficient_question_raw,
                    "data_source": data_source,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
