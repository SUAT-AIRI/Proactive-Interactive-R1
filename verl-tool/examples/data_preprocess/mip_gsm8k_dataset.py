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
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/chenxin/verl-interactive/verl-tool/data/mip_gsm8k/')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'mip_gsm8k'

    with open(args.local_dir+f'{data_source}.json', 'r') as f:
        dataset = json.load(f)
    # dataset = datasets.load_dataset(data_source, 'main')
    train_dataset = Dataset.from_list(dataset)
    test_dataset = Dataset.from_list(dataset)
    # train_dataset = dataset
    # train_dataset = dataset['train']
    # test_dataset = dataset['test']

    
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
            question_raw = example.pop('question')
            answer_raw = example.pop('answer')
            insufficient_info_raw = example.pop('insufficient_info')
            insufficient_question_raw = example.pop('insufficient_question')

            
            # some numbers are in the `x,xxx` format, and we want to remove the comma cite: https://github.com/Blue-Raincoat/SelectIT/blob/main/eval/eval/gsm/run_eval.py
            answer = answer_raw.split("####")[1].strip()
            answer = re.sub(r"(\d),(\d)", r"\1\2", answer)
            assert float(answer), f"answer is not a valid number: {answer}"

            insufficient_question = insufficient_question_raw.strip() + " Please reason step by step, and put your final answer within \\boxed{}."

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
                        "target": answer,
                        "turn": len(insufficient_info_raw)
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                    "insufficient_question": insufficient_question_raw,
                    'insufficient_info': insufficient_info_raw
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
