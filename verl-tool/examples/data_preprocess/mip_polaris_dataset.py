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
from datasets import Dataset
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse
import random
from collections import defaultdict
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def split_dataset_balanced(dataset):
    """
    Split the dataset into train and test sets with a balanced distribution of difficulty levels.
    """
    random.seed(2025)
    train_set = []
    test_set = []
    SPLIT_RATIO = 0.7
    # transform difficulty to integer
    for item in dataset:
        if isinstance(item["difficulty"], str) and "/" in item["difficulty"]:
            item["difficulty_level"] = int(item["difficulty"].split("/")[0])
        else:
            item["difficulty_level"] = int(item["difficulty"])


    difficulty_groups = defaultdict(list)
    for item in dataset:
        difficulty_groups[item["difficulty_level"]].append(item)


    # 对每个难度分层采样
    for diff_level in sorted(difficulty_groups.keys(), reverse=True):
        group = difficulty_groups[diff_level]
        random.shuffle(group)
        
        split_idx = int(len(group) * SPLIT_RATIO)
        train_set.extend(group[:split_idx])
        test_set.extend(group[split_idx:])

    return train_set, test_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/chenxin/verl-interactive/verl-tool/data/mip_polaris/')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'mip_polaris'

    with open(args.local_dir+f'{data_source}.json', 'r') as f:
        dataset = json.load(f)
    # dataset = dataset.reverse()
    # print(f"Dataset size: {len(dataset)}")
    train_set, test_set = split_dataset_balanced(dataset)

    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    train_dataset = Dataset.from_list(train_set)
    test_dataset = Dataset.from_list(test_set)

    
    # system_prompt =  (
    # "Answer the given question. "
    # "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    # "If you find you lack some knowledge or clarification is required, you can call a asking engine by <asking> query </asking> and it will return the requested information between <response> and </response>. "
    # "You can ask as many times as your want. "
    # "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations."
    # )
    system_prompt = '''Before providing an answer, you must conduct internal reasoning within <think> and </think> whenever new information is received. If additional knowledge or clarification is required, issue a query inside the reasoning using <asking> and </asking>. The asking engine’s reply will appear within <response> and </response> inside the reasoning. If no further information is needed, present the final answer after </think>, without detailed illustrations.'''


    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            id_raw = example.pop('id')
            question_raw = example.pop('original_question')
            answer_raw = example.pop('ground_truth')
            difficulty_raw = example.pop('difficulty')

            removed_condition_raw = example.pop('removed_condition')
            incomplete_question_raw = example.pop('incomplete_question')

            
            # some numbers are in the `x,xxx` format, and we want to remove the comma cite: https://github.com/Blue-Raincoat/SelectIT/blob/main/eval/eval/gsm/run_eval.py

            insufficient_question = incomplete_question_raw.strip() + " Please reason step by step, and put your final answer within \\boxed{}."

            data = {
                "data_source": data_source,
                "id": id_raw,
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
                        "target": answer_raw,
                        "turn": 1
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'id': id_raw,
                    'difficulty': difficulty_raw,
                    'answer': answer_raw,
                    "question": question_raw,
                    "insufficient_question": incomplete_question_raw,
                    'insufficient_info': [removed_condition_raw]
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
