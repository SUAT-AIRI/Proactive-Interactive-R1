"""
User Simulator Tool for verl-tool - Compatible with Interactive-R1 functionality
"""
from .base import BaseTool, register_tool
import regex as re
import requests
from typing import Tuple, Dict, Any, List
import logging
from .simulator_client_collab import UserSimulatorClient
from openai import OpenAI
import os
import time
import json
import random
logger = logging.getLogger(__name__)

@register_tool
class UserSimulatorTool(BaseTool):
    tool_type = "user_simulator_collab"
# https://api.ai-gaochao.cn/v1/. http://localhost:8725/v1/
# gpt-4o-mini Llama-3.1-8B-Instruct
    def __init__(self, num_workers=1, **kwargs):
        super().__init__(num_workers)
        
        self.simulator_url =  kwargs.get('simulator_url', os.getenv('SIMULATOR_URL'))
        self.user_simulator_client = OpenAI(api_key=kwargs.get('simulator_api_key', os.getenv('SIMULATOR_API_KEY')), base_url=self.simulator_url)
        self.simulator_name =kwargs.get('simulator_name', os.getenv('SIMULATOR_NAME'))
        self.user_simulator = UserSimulatorClient(
        client=self.user_simulator_client,
        model_name=self.simulator_name,
        )
        logger.info(f"UserSimulatorTool initialized with URL: {self.simulator_url}, simulator_name: {self.simulator_name}")

    
    def _parse_asking_query(self, action: str) -> str:
        """
        Extract the search query from the action string.
        This is a helper function to parse the <search> tags.
        
        Args:
            action: Raw action string containing search query
            
        Returns:
            Extracted search query
        """
        # Priority logic moved from serve.py: prioritize search tool for <search> tags
        # This implements the original logic: if "</search>" in action and "search_retrieval" in self.tools
        if "</asking>" in action:
            # Extract search query from <asking>query</asking> tags
            search_matches = re.findall(r"<asking>(.*?)</asking>", action, re.DOTALL)

            if len(search_matches) > 0:
                # Use the last search query if multiple are found
                query = search_matches[-1].strip()
                return query, True
        return "", False

    def _parse_answer_tags(self, action: str) -> Tuple[str, bool]:
        """
        Parse the action string to extract answer tags.
        This is a helper function to handle <answer> tags.
        
        Args:
            action: Raw action string containing answer tags
            
        Returns:
            Tuple containing the extracted answer and a validity flag
        """
        # Priority logic moved from serve.py: check for finish condition (</answer> tag)
        # This implements the original logic: if "</answer>" in action
        if "</answer>" in action:
            # Check for <answer> tags (Search-R1 style)
            answer_matches = re.findall(r"<answer>(.*?)</answer>", action, re.DOTALL)
            if len(answer_matches) > 0:
                final_answer = answer_matches[-1].strip()
                return final_answer, True
        return "", False
    
    def _parse_eos_tags(self, action: str) -> Tuple[str, bool]:
        """
        Parse the action string to extract end-of-sequence tags.
        This is a helper function to handle <|end▁of▁sentence|> tags.

        Args:
            action: Raw action string containing end-of-sequence tags

        Returns:
            Tuple containing the extracted end-of-sequence and a validity flag
        """
        # Priority logic moved from serve.py: check for finish condition (</answer> tag)
        # This implements the original logic: if "</answer>" in action
        if "<|end▁of▁sentence|>" in action:
            # Check for <|end▁of▁sentence|> tags (Search-R1 style)
            if "</think>" in action:
                eos_matches = action.split("</think>")[-1].strip()
                return eos_matches, True
        return "", False
    
    def _parse_response_tags(self, action: str) -> Tuple[str, bool]:
        """
        Parse the action string to extract answer tags.
        This is a helper function to handle <answer> tags.
        
        Args:
            action: Raw action string containing answer tags
            
        Returns:
            Tuple containing the extracted answer and a validity flag
        """
        # Priority logic moved from serve.py: check for finish condition (</answer> tag)
        # This implements the original logic: if "</answer>" in action
        if "</response>" in action:
            # Check for <response> tags (Search-R1 style)
            response_matches = re.findall(r"<response>(.*?)</response>", action, re.DOTALL)
            if len(response_matches) > 0:
                final_response = response_matches[-1].strip()
                return final_response, True
        return "", False
    
    def parse_action(self, action: str) -> Dict[str, Any]:
        """
        Parse the action string to extract relevant information for the user simulator.
        
        Args:
            action: Raw action string containing various tags
        Returns:
            Dictionary containing parsed information
        """
        # Check for asking query first
        asking_query, is_asking = self._parse_asking_query(action)
        if is_asking:
            return asking_query, True

        # If no search query found, check for <answer> tags
        answer, is_valid = self._parse_eos_tags(action)
        if is_valid:
            return answer, True
        
        # Default return if no relevant tags found
        return "", False
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Determine the priority of the action based on its content.
        
        Args:
            action: Raw action string
            extra_field: Additional fields that may influence priority
            
        Returns:
            Integer representing the priority level
        """
        # Example priority logic: prioritize asking queries over responses
        if "</asking>" in action:
            _, valid = self.parse_action(action)
            if valid:
                return 100  # High priority for asking actions
            
         # Standard priority check
        _, valid = self.parse_action(action)
        return 0 if valid else -1

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Conduct the action by sending a request to the user simulator service.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string containing asking query
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        # print(f"extra_field: {extra_field}")

        parsed_asking, is_valid = self._parse_asking_query(action)
        env = self.load_env(trajectory_id)

        if not is_valid:
            # try answer tags if no valid asking query found
            # parsed_asking, is_valid = self._parse_answer_tags(action)
            # parsed_asking, is_valid = self._parse_eos_tags(action)
            if is_valid:
                observation = ""
                execution_result = ""
                done = True
                valid = False
            else:
                observation = ""
                execution_result = ""
                done = False
                valid = False
        else:
            # Call the user simulator to get the response
            single_turn_prompt_raw = extra_field.get('single_turn_prompt_raw', None)
            
            # question = extra_field.get('insufficient_question', None)
            if extra_field.get('data_source', None) == 'collabllm-multiturn-math-hard':
                # For collabllm dataset, provide additional context
                task_desc = "question answering"
            elif extra_field.get('data_source', None) == 'collabllm-multiturn-medium':
                task_desc = "document editing"
            elif extra_field.get('data_source', None) == 'collabllm-multiturn-bigcodebench':
                task_desc = "code generation"
            else:
                raise ValueError(f"Unknown data_source: {extra_field.get('data_source', None)}")
            # [{'role': 'user', 'content': 'Hello!'}, 
            #                  {'role': 'assistant', 'content': 'Hi!'}, ...]
            conversation_history = []
            if 'previous_obs' in env:
                for his in env['previous_obs']:
                    assistant = {"role": "user", "content": his['action']}
                    user = {"role": "assistant", "content": his['observation']}
                    conversation_history.append(assistant)
                    conversation_history.append(user)
            else:
                print("No previous_obs found.")

            # print(f"env = {env}\n length of conversation_history = {len(conversation_history)}")
            

            user_response = self.user_simulator.chat(parsed_asking, trajectory_id=trajectory_id, single_turn_prompt=single_turn_prompt_raw, task_desc=task_desc,conversation_history = conversation_history)
            

            # Format observation similar to Search-R1
            observation = f"\n<response>{user_response}</response>"
            execution_result = user_response

            done = False  # Search doesn't end the trajectory
            valid = True
                
        self.update_env(trajectory_id, env, parsed_asking, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)

        return observation, done, valid
