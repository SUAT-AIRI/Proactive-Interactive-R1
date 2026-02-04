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


def collabllm_multiturn_math_chat_judge_correct(solution_str, ground_truth):
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

    if answer is None:
        return 0.0
    
    target_answer = extract_answer(ground_truth.get('target', ground_truth) if isinstance(ground_truth, dict) else ground_truth)
    target_answer_is_correct = grade_answer_mathd(answer, target_answer) or grade_answer_sympy(answer, target_answer)
    # print(f"Evaluating solution: {solution_str} against ground truth: {ground_truth} => Extracted answer: {answer}, Target answer: {target_answer}, Correct: {target_answer_is_correct}")
    
    if target_answer_is_correct:
        return 0.5
    else:
        return 0.0
    

if __name__ == "__main__":
    # Test the function
    solution = " \\boxed{\\frac{1}{(a + b)^3}}."
    ground_truth = "Target answer: \\frac{1}{(a + b)^3}, "
    score = collabllm_multiturn_math_chat_judge_correct(solution, ground_truth)
    print(f"Score: {score}")

