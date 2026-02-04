import time
import random
from typing import List, Dict, Optional, Any
from .extract_json_reliable import extract_json


FALLBACK_RESPONSES = [
    "I don't have a specific intent right now. Please proceed based on your best judgment.",
    "I'm not sure about that. You can decide what's best.",
    "I don't have more information to add. Just carry on.",
    "I don't really have an answer for that. Can you try to solve it with what you have?",
    "That's not something I can answer. Please continue with the task.",
]


def parse_messages(messages: List[Dict[str, str]], strip_sys_prompt: bool = True) -> str:
    """将消息列表转换为格式化的对话字符串。"""
    if not messages:
        return ''

    if strip_sys_prompt:
        messages = [msg for msg in messages if msg['role'] != 'system']

    role_map = {
        'user': '**AI Collaborator**',
        'assistant': '**USER (You)**'
    }
    
    lines = [
        f"{role_map.get(m['role'], '')}: {m['content']}"
        for m in messages
        if m['role'] in role_map
    ]
    
    return '\n'.join(lines)


class UserSimulatorClient:
    """Local User Simulator Client for collaborative task simulation."""

    def __init__(
        self,
        client: Any,
        model_name: str,
        task_name: str,
        user_intent: str,
        context_content: str = None,
        timeout: int = 60,
        max_model_len: int = 1024,
        max_retries: int = 3,
        retry_wait: int = 10,
        fallback_responses: Optional[List[str]] = None,
    ):
        self.client = client
        self.model_name = model_name
        self.timeout = timeout
        self.max_model_len = max_model_len
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.fallback_responses = fallback_responses or FALLBACK_RESPONSES
        if task_name == "factual knowledge":
            from .prompt import USER_SIMULATOR_FK_PROMPT as user_simulator_prompt
            self.user_simulator_prompt = user_simulator_prompt
            self.task_name = task_name
            self.task_desc = "factual knowledge"
        elif task_name == "retrieval question answering":
            from .prompt import USER_SIMULATOR_QA_PROMPT as user_simulator_prompt
            self.user_simulator_prompt = user_simulator_prompt
            self.task_name = task_name
            self.task_desc = "question answering"
            self.context_content = context_content
        elif task_name == "question answering":
            from .prompt import USER_SIMULATOR_COLLAB_PROMPT as user_simulator_prompt
            self.user_simulator_prompt = user_simulator_prompt
            self.task_name = task_name
            self.task_desc = "question answering"
        else:
            raise ValueError("task_name cannot be determined.")
        self.user_intent = user_intent
        # add for external
        self.conversation_history: List[Dict[str, Any]] = []
        self.system_prompt: str = ""

    def _get_fallback_response(self) -> str:
        """返回随机的 fallback 响应。"""
        return random.choice(self.fallback_responses)

    def _build_system_prompt(
        self,
    ) -> str:
        """构建系统提示词。"""
        if self.task_name == "retrieval question answering":
            return self.user_simulator_prompt.format(
                task_desc=self.task_desc,
                context_content=self.context_content.strip(),
                single_turn_prompt=self.user_intent,
                chat_history=parse_messages(self.conversation_history),
            )
        else:
            return self.user_simulator_prompt.format(
                task_desc=self.task_desc,
                single_turn_prompt=self.user_intent,
                chat_history=parse_messages(self.conversation_history),
            )

    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """调用模型并返回原始响应内容。"""
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_model_len,
            timeout=self.timeout,
        )
        
        if isinstance(resp, str):
            raise ValueError(f"Model returned error: {resp}")
        
        return resp.choices[0].message.content

    def _parse_response(self, content: str) -> str:
        """解析模型响应，提取 JSON 中的 response 字段。"""
        parsed = extract_json(content)
        
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict, got {type(parsed).__name__}: {parsed}")
        
        required_keys = {'current_answer', 'thought', 'response'}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError(f"Missing keys. Expected {required_keys}, got {parsed.keys()}")
        
        return parsed['response']

    def _post_with_retry(self, messages: List[Dict[str, str]]) -> str:
        """带重试机制的模型调用，失败后返回 fallback 响应。"""
        last_error = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                content = self._call_model(messages)
                return self._parse_response(content)
            except Exception as e:
                last_error = e
                print(f"[UserSimulator] Attempt {attempt}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries:
                    print(f"[UserSimulator] Retrying in {self.retry_wait} seconds...")
                    time.sleep(self.retry_wait)

        # 所有重试都失败，返回 fallback 响应
        fallback = self._get_fallback_response()
        print(
            f"[UserSimulator] All {self.max_retries} attempts failed. "
            f"Last error: {last_error}. Returning fallback response."
        )
        return fallback

    def chat(
        self,
        user_message: str,
    ) -> str:
        """发送消息并获取模拟用户的响应。"""
        # if selfconversation_history is None:
        #     conversation_history = []

        self.system_prompt = self._build_system_prompt()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        post_result = self._post_with_retry(messages)

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": post_result})

        return post_result

if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI(
        api_key="sk-xxx",
        base_url="http://10.10.128.132:8725/v1/",
    )

    simulator = UserSimulatorClient(
        client=client,
        model_name="Llama-3.1-8B-Instruct",
    )

    response = simulator.chat(
        user_message="Hi! I've drafted an exciting update...",
        task_desc="question answering",
        user_intent="Please write an article...",
    )


    print("Simulator Response:")
    print(response)