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

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction

from .simulator_client_collab import LocalUserSimulatorClient
from openai import OpenAI
import numpy as np
import json
import asyncio
import random


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class UserSimulatorInteraction(BaseInteraction):
    """A demo interaction for calculating the reward of gsm8k.

    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """

    def __init__(self, config: dict,**kwargs):
        super().__init__(config)
        self._instance_dict = {}
        self.simulator_url =  kwargs.get('simulator_url', os.getenv('SIMULATOR_URL'))
        self.user_simulator_client = OpenAI(api_key=kwargs.get('simulator_api_key', os.getenv('SIMULATOR_API_KEY')), base_url=self.simulator_url)
        self.simulator_name =kwargs.get('simulator_name', os.getenv('SIMULATOR_NAME'))
        self.user_simulator = LocalUserSimulatorClient(
            client=self.user_simulator_client,
            model_name=self.simulator_name,
            # history_path=self.simulator_history_path
            )
        

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        # print("Messages received for response generation:", messages)
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            item = messages[i]
            if item.get("role") == "assistant":
                content = item.get("content")
                break

        # if content and content.startswith("#### "):
        #     self._instance_dict[instance_id]["response"] = content
        # else:
        #     self._instance_dict[instance_id]["response"] = "#### " + (content or "")
        self._instance_dict[instance_id]["response"] = content
        self._instance_dict[instance_id]["messages"] = messages
        data_source = kwargs.get('name', None)
        reward = await self.calculate_score(instance_id, data_source=data_source)
        print(f"Calculated reward: {reward}")
        if reward >= 0.5:
            user_response = "Your response is correct!"
            should_terminate_sequence = True
            return should_terminate_sequence, user_response, reward, {}
        
        else:
            
            # response = "Your response is incorrect! You need to reflect on your answer and try again."
            
            parsed_asking = self._instance_dict[instance_id]["response"]
            # print(f"kwargs: {kwargs}")
            if kwargs.get('name', None) == 'collabllm-multiturn-math-hard':
                # For collabllm dataset, provide additional context
                task_desc = "question answering"
            elif kwargs.get('name', None) == 'collabllm-multiturn-medium':
                task_desc = "document editing"
            elif kwargs.get('name', None) == 'collabllm-multiturn-bigcodebench':
                task_desc = "code generation"
            else:
                raise ValueError(f"Unknown data_source: {kwargs.get('data_source', None)}")
            single_turn_prompt_raw = kwargs.get('query', None)
            conversation_history = messages[1:]  # Exclude the first system prompt
            trajectory_id = instance_id
            def run_sync_chat():
                # 重新实例化或复用 (注意线程安全)
                try:
                    user_response = self.user_simulator.chat(
                        parsed_asking, 
                        trajectory_id=trajectory_id, 
                        single_turn_prompt=single_turn_prompt_raw, 
                        task_desc=task_desc,
                        conversation_history=conversation_history
                    )
                except Exception as e:
                    logger.error(f"User simulator error for trajectory {trajectory_id}: {e}")
                    fallback_responses = [
                        "I don't have a specific intent right now. Please proceed based on your best judgment.",
                        "I'm not sure about that. You can decide what's best.",
                        "I don't have more information to add. Just carry on.",
                        "I don't really have an answer for that. Can you try to solve it with what you have?",
                        "That's not something I can answer. Please continue with the task.",
                    ]
                    user_response = random.choice(fallback_responses)
                return user_response
            loop = asyncio.get_running_loop()
            user_response = await loop.run_in_executor(None, run_sync_chat)
            if "TERMINATE CHAT" in str(user_response):
                should_terminate_sequence = True
                return should_terminate_sequence, user_response, reward, {}
            # user_response = self.user_simulator.chat(parsed_asking, trajectory_id=trajectory_id, single_turn_prompt=single_turn_prompt_raw, task_desc=task_desc,conversation_history = conversation_history)

            # user_response = "Your response is incorrect! You need to reflect on your answer and try again."
            # print(f"parsed_asking: {parsed_asking}")
            # print(f"User simulator response: {user_response}")
            else:
                should_terminate_sequence = False
                return should_terminate_sequence, user_response, reward, {}

    async def calculate_score(self, instance_id: str, data_source,**kwargs) -> float:
        solution_str = self._instance_dict[instance_id]["response"]
        ground_truth = self._instance_dict[instance_id]["ground_truth"]
        # llm_contents = [msg.get("content") for msg in self._instance_dict[instance_id]["messages"] if msg.get("role") == "assistant"]
        # kwargs['llm_contents'] = llm_contents
        from verl.interactions.collabllm_multiturn_judge_tool import collabllm_multiturn_math_chat_judge_correct
        if data_source == 'collabllm-multiturn-math-hard':
            return collabllm_multiturn_math_chat_judge_correct(
                solution_str=solution_str,
                ground_truth=ground_truth
            )
        elif data_source == 'collabllm-multiturn-medium':
            return collabllm_multiturn_math_chat_judge_correct(
                solution_str=solution_str,
                ground_truth=ground_truth
            )
        elif data_source == 'collabllm-multiturn-bigcodebench':
            return collabllm_multiturn_math_chat_judge_correct(
                solution_str=solution_str,
                ground_truth=ground_truth
            )
        else:
            raise NotImplementedError(f"Data source {data_source} not supported in UserSimulatorInteraction.")
    
    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
