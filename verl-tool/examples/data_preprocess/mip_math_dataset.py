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
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    with open(args.local_dir, 'r') as f:
        dataset = json.load(f)
    dataset = datasets.load_dataset(data_source, 'main')

    train_dataset = dataset
    # train_dataset = dataset['train']
    # test_dataset = dataset['test']

    system_prompt = '''Before providing an answer, you must conduct internal reasoning within <think> and </think> whenever new information is received. If additional knowledge or clarification is required, issue a query inside the reasoning using <asking> and </asking>. The asking engine’s reply will appear within <response> and </response> inside the reasoning. If no further information is needed, present the final answer after </think>, without detailed illustrations.'''

    # add a row to each data item that represents a unique id
    def make_map_fn():

        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = question_raw.strip() + " Please reason step by step, and put your final answer within \\boxed{}."

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
                    }],
                "ability": "mip_math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "the key of item"
                },
                # "extra_info": {
                #     'split': split,
                #     'index': idx,
                #     'answer': answer_raw,
                #     "question": question_raw,
                # }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)
    # train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
