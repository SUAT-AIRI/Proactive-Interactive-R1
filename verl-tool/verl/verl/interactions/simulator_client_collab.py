import json
import time
import os
from typing import List, Dict, Union, Optional, Any
import requests
from .extract_json_reliable import extract_json
import random

def parse_messages(messages, strip_sys_prompt=True):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    if not messages: return ''

    if strip_sys_prompt:
        messages = strip_system_prompt(messages)

    chat = ""
    # Flip roles: user -> assistant, assistant -> user
    for m in messages:
        if m['role'] == 'user':
            chat += f"**AI Collaborator**: {m['content']}\n"
        elif m['role'] == 'assistant':
            chat += f"**USER (You)**: {m['content']}\n"
        
        # chat += f"**{m['role'].capitalize()}**: {m['content']}\n"

    return chat.strip()

def strip_system_prompt(messages):
    '''
    Args:
        messages: List[dict]
            List of dictionaries with keys 'role' and 'content'
            Example: messages = [{'role': 'user', 'content': 'Hello!'}, 
                                 {'role': 'assistant', 'content': 'Hi!'}, ...]
    '''
    return [msg for msg in messages if msg['role'] != 'system']


with open("/home/chenxin/verl-interactive/verl-tool_old/verl/verl/interactions/user_simulator_collab_prompt_for_LLM.txt", 'r') as f:
    USER_SIMULATOR_PROMPT = f.read()

# fallback_responses = [
#             "I don't have a specific intent right now. Please proceed based on your best judgment.",
#             "I'm not sure about that. You can decide what's best.",
#             "I don't have more information to add. Just carry on.",
#             "I don't really have an answer for that. Can you try to solve it with what you have?",
#             "That's not something I can answer. Please continue with the task.",
#         ]

class LocalUserSimulatorClient:
    """
    Local User Simulator Client with auto history file creation.
    """

    def __init__(
        self,
        client: Any,
        model_name: str,
        timeout: int = 60,
        max_model_len: int = 4096,
        history_path: str = "simulator_history/"
    ):
        self.client = client
        self.model_name = model_name
        self.timeout = timeout
        self.max_model_len = max_model_len

        self.simulator_history_path = history_path
        os.makedirs(self.simulator_history_path, exist_ok=True)


    def _handle_error(self, error: Exception, response: Optional[requests.Response] = None) -> str:
        """Unified error handling."""
        if response is not None:
            return f"HTTP Error {response.status_code}: {response.text}"
        if isinstance(error, requests.exceptions.RequestException):
            return f"Request failed: {str(error)}"
        return f"Unexpected error: {str(error)}"

    def _post_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        retry_wait: int = 10
    ) -> Union[Any, str]:
        """Send chat completion request to model with retry."""
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_model_len,
                    timeout=self.timeout
                )
                if isinstance(resp, str):
                    raise ValueError(f"Model error: {resp}")

                full_response = resp.choices[0].message.content
                try:
                    if isinstance(full_response, str):
                        full_response = extract_json(full_response)
                except Exception as e:
                    last_error = self._handle_error(e)
                    print(f"[UserSimulator] Error extracting JSON: {e}")
                    continue

                if isinstance(full_response, dict):
                    keys = full_response.keys()
                    if {'current_answer', 'thought', 'response'}.issubset(keys):
                        response = full_response.pop('response')
                    else:
                        print(f"[UserSimulator] Keys {keys} do not match expected keys in full_response: {full_response}. Retrying...")
                else:
                    print(f"[UserSimulator] Extracted response from full_response: {full_response} is not a dict. Retrying...")
                
                
                return response
            
            except Exception as e:
                last_error = self._handle_error(e)
                print(f"[Attempt {attempt}] Failed: {last_error}")
                if attempt < max_retries:
                    print(f"Retrying in {retry_wait} seconds...")
                    time.sleep(retry_wait)

        
        raise ValueError(f"Model call failed after {max_retries} attempts. Last error: {last_error}")


    def chat(self, user_message: str, trajectory_id: str, single_turn_prompt: Optional[str] = None, task_desc: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = []) -> str:
        """
        Send message, get response, automatically load/save conversation history.
        """

        system_prompt = USER_SIMULATOR_PROMPT.format(
                task_desc = task_desc,
                single_turn_prompt = single_turn_prompt,
                chat_history = parse_messages(conversation_history),
        )
       
        message_to_send = []
        message_to_send.append({"role": "system", "content": system_prompt})
        message_to_send.append({"role": "user", "content": user_message})

        # Call model
        resp = self._post_with_retry(message_to_send)

        return resp


if __name__ == "__main__":
    from openai import OpenAI

    model_name = "gpt-4o-mini"
    # model_name = "Llama-3.1-8B-Instruct"
    base_url = "https://api.ai-gaochao.cn/v1/"
    # base_url = "http://10.10.128.132:8725/v1/"
    client = OpenAI(api_key="sk-EAR9W8gMRCM2JIMJ2e5b8b095c494718B5CeC1C17004577d", base_url=base_url)

    simulator = LocalUserSimulatorClient(
        client=client,
        model_name=model_name,
        history_path="/home/chenxin/verl-interactive/simulator_history/"
    )
    # Hey, can you help me with this math problem? I need to rationalize a denominator, but I'm not sure where to start.
    trajectory_id = "test_trajectory_013"
    # user_message = "Sure! Rationalizing denominators is a key algebra technique. Do you have a specific problem in mind, or would you like me to go over a general example with you?"
    user_message = "Would you like me to explain how to rationalize the denominator of the expression \\(\\frac{7}{\\sqrt{5} - 2}\\)? The process involves multiplying by the conjugate, and I can guide you step by step"
    single_turn_prompt = "Rationalize the denominator of $\\frac{5}{2+\\sqrt{6}}$. The answer can be written as $\\frac{A\\sqrt{B}+C}{D}$, where $A$, $B$, $C$, and $D$ are integers, $D$ is positive, and $B$ is not divisible by the square of any prime. If the greatest common divisor of $A$, $C$, and $D$ is 1, find $A+B+C+D$."
    task_desc = "question answering"


    trajectory_id = "test_trajectory_014"
    user_message = "Can you explain what you're looking for regarding the point (a, b) where the mouse starts getting farther from the cheese? Are you asking about how to determine this point, or are you trying to understand the reasoning behind it?"
    single_turn_prompt = "Suppose $a,$ $b,$ and $c$ are real numbers such that\n\\[\\frac{ac}{a + b} + \\frac{ba}{b + c} + \\frac{cb}{c + a} = -9\\]and\n\\[\\frac{bc}{a + b} + \\frac{ca}{b + c} + \\frac{ab}{c + a} = 10.\\]Compute the value of\n\\[\\frac{b}{a + b} + \\frac{c}{b + c} + \\frac{a}{c + a}.\\]"
    task_desc = "question answering"

    response = simulator.chat(
        user_message=user_message,
        trajectory_id=trajectory_id,
        single_turn_prompt=single_turn_prompt,
        task_desc=task_desc
    )

    print("Simulator Response:")
    print(response)