import time
from typing import List, Dict, Any

import requests
import re


class ModelClient:
    """Local model client for chat completions with retry mechanism."""
    def __init__(
        self,
        model_path,
        base_url="http://localhost:8716/",
        stop_tokens=["<｜end▁of▁sentence｜>"],
        reasoning_model=True,
        api_key="none",
        temperature=0.6,
        top_p=0.95,
        timeout=120,
    ):
        self.api_key = api_key
        self.model_path = model_path
        self.base_url = base_url
        self.chat_completions_url = base_url + "/v1/chat/completions"
        self.tokenizer_url = base_url + "/tokenize"
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.stop_reason = None
        if reasoning_model == True:
            self.completion = "<think>\n"
        else:
            self.completion = ""
        self.max_model_len = 4096
        self.generation_len = 0
        self.stop_tokens = stop_tokens

    def _get_headers(self, with_auth: bool = False) -> Dict[str, str]:
        """构建请求头。"""
        headers = {"Content-Type": "application/json"}
        if with_auth:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _token_count(self, messages: List[Dict[str, Any]]) -> int:
        """计算消息的 token 数量。"""
        result = self._post_with_retry(
            url=self.tokenizer_url,
            headers=self._get_headers(),
            payload={"messages": messages},
        )
        if isinstance(result, dict):
            return result.get("count", 0)
        return 0

    def _build_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """构建聊天请求的 payload。"""
        return {
            "model": self.model_path,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": self.stop_tokens,
            "max_tokens": self.generation_len,
            "add_generation_prompt": False,
            "continue_final_message": True,
            "include_stop_str_in_output": True,
            "echo": False,
        }

    def _post_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        max_retries: int = 3,
        retry_wait: int = 10,
    ) -> Dict[str, Any] | str:
        """带重试机制的 POST 请求，成功返回 dict，失败返回错误信息字符串。"""
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    url=url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return response.json()
                
                last_error = f"HTTP Error {response.status_code}: {response.text}"
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"

            print(f"[Attempt {attempt}/{max_retries}] 调用模型失败: {last_error}")

            if attempt < max_retries:
                print(f"将在 {retry_wait} 秒后重试...")
                time.sleep(retry_wait)

        return last_error

    def _parse_response(self, result: Dict[str, Any]) -> str:
        """解析模型响应，更新内部状态并返回内容。"""
        choice = result["choices"][0]
        
        self.stop_reason = choice.get("stop_reason")
        content = choice["message"]["content"]
        self.completion += content
        # self.max_model_len -= result["usage"]["total_tokens"]
        # self.max_model_len = max(1, self.max_model_len)
        
        return content

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """发起一次聊天请求。"""
        token_count = self._token_count(messages)
        if token_count:
            self.generation_len = self.max_model_len - token_count
        else:
            raise ValueError("Failed to calculate token count.")

        print(f"Recent residue completion_tokens: {self.generation_len}")

        if self.generation_len <= 0:
            raise ValueError("Out of tokens for this session.")

        result = self._post_with_retry(
            url=self.chat_completions_url,
            headers=self._get_headers(with_auth=True),
            payload=self._build_payload(messages),
        )

        if isinstance(result, dict):
            return self._parse_response(result)
        
        raise ValueError(result)



if __name__ == "__main__":
    
    system_prompt = '''Before providing an answer, you must conduct internal reasoning within <think> and </think> whenever new information is received. If additional knowledge or clarification is required, issue a query inside the reasoning using <asking> and </asking>. The asking engine’s reply will appear within <response> and </response> inside the reasoning. If no further information is needed, present the final answer after </think>, without detailed illustrations.'''        
    model_path = "Proactive-Interactive-R1-Math-7B"
    user_intent = "James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?"
    question = "James decides to run 3 sprints 3 times a week. How many total meters does he run a week?"


    base_url = "http://xx.xx.xxx.xxx:1136"
    model_client = ModelClient(model_path=model_path,
                               base_url=base_url,
                               stop_tokens=["</asking>", "<｜end▁of▁sentence｜>"],
                               reasoning_model=True,
                               api_key="sk-xxxxx")
    from generalization_eval.utils.simulator_client import UserSimulatorClient
    from openai import OpenAI
    user_simulator_name = "Llama-3.1-8B-Instruct"
    user_simulator_url = "http://xx.xx.xxx.xxx:8725/v1"
    user_simulator_client = OpenAI(api_key="none", base_url=user_simulator_url)
    user_simulator = UserSimulatorClient(client=user_simulator_client, model_name=user_simulator_name,task_name="question answering",user_intent = user_intent)


    messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": model_client.completion}
            ]
    model_response = model_client.chat(messages=messages)
    print("full completion  :\n", model_client.completion)

    while model_client.stop_reason == "</asking>":
        ask_content = re.search(r"<asking>(.*?)</asking>", model_response, re.DOTALL).group(1).strip()
        user_response = user_simulator.chat(ask_content)
        model_client.completion += "\n<response>" + user_response + "</response>"
        messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": model_client.completion}
                ]
        model_response = model_client.chat(messages=messages)
        print(1)
    print("full completion  :\n", model_client.completion)
    print("conversation history:\n", user_simulator.conversation_history)
    print("system prompt:\n", user_simulator.system_prompt)





