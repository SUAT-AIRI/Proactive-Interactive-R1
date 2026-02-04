import logging
from typing import Any, Dict, List, Optional

import litellm

from .utils.extract_json_reliable import extract_json
from typing import Tuple
# litellm._turn_on_debug()
# logger = logging.getLogger(__name__)
import os
# os.environ["OPENAI_API_KEY"] = "sk-EAR9W8gMRCM2JIMJ2e5b8b095c494718B5CeC1C17004577d"
# os.environ["LITELLM_API_BASE"] = "https://api.ai-gaochao.cn/v1/"


# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
# with open("/home/chenxin/verl-interactive/verl-tool_old/verl_tool/workers/reward_manager/helpfulness_reward_prompt_his.txt", "r", encoding="utf-8") as f:
#     HELPFULNESS_REWARD_PROMPT = f.read()
from .prompt import HELPFULNESS_REWARD_PROMPT

# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #

def _parase_history(history: Optional[List[Tuple[str, str]]]) -> str:
    if history is None:
        return ""
    history_str = ""
    for q,a in history:
        history_str += f"Assistant: {q}\nUser: {a}\n"
    return history_str

class HelpfullnessMetric():
    """
    Uses an LLM judge to produce an interactivity score in [0, 1].
    """

    def __init__(self, num_retries: int = 5, retry_after: int = 30, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Default to a deterministic model unless overridden.
        self.llm_kwargs: Dict[str, Any] = {
            "temperature": 0.0,
            "model": os.getenv('JUDGEMENT_MODEL', 'gpt-4o-mini'),
            "api_base": os.getenv('JUDGEMENT_API_BASE', "https://api.ai-gaochao.cn/v1"),
            **llm_kwargs,
        }

    # --------------------------------------------------------------------- #
    def score(
        self,
        question,
        response,
        qa_pairs: Optional[List[Tuple[str, str]]] = None,
        ) -> Dict[str, float]:
        """
        `prompt`, `groundtruth`, and `response` are unused here;
        the full conversation in `messages` is what matters.
        """

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        chat_history = _parase_history(qa_pairs)

        eval_prompt = HELPFULNESS_REWARD_PROMPT.format(question=question, response=response,chat_history=chat_history)
        # print("Helpfullness evaluation prompt:\n", eval_prompt)
        # print("Accuracy evaluator prompt:\n%s", eval_prompt)
        helpfulness = 0
        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs, messages=[{"role": "user", "content": eval_prompt}], num_retries=1
                ).choices[0].message.content
                # print(full_response)
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                print(f"[retry={i + 1}] Error during LLM call: {e}")
                continue
            # print("Full response from LLM judge:\n", full_response)
            # ------------------------------------------------------------------ #
            # 4) Parse JSON                                                      #
            # ------------------------------------------------------------------ #
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                print(f"Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'helpfulness'}.issubset(keys):
                    helpfulness = full_response.pop('helpfulness')
                    break
                else:
                    print(f"Keys {keys} do not match expected keys. Retrying...")
                    continue
        return helpfulness
    


if __name__ == "__main__":
    # 1. 初始化评估器
    # 注意：这里会使用你环境变量里设置的 key 和 base_url
    metric = HelpfullnessMetric()

    # ------------------------------------------------------------------ #
    # 模拟场景：一道微积分题目
    # ------------------------------------------------------------------ #
    print(">>> Test Case 1: Calculus Problem (Contextual Guidance)")

    # 1. 原始问题 (Root Question)
    root_question = "Find the integral of f(x) = x * e^x using integration by parts."

    # 2. 历史对话 (History QA Pairs)
    # 模拟之前的交互：模型提问 -> 模型/用户确认 -> 模型提问 -> ...
    # 这里模拟的是模型之前的思考步骤被拆解成了 QA 对
    qa_pairs = [
        ("What is the formula for integration by parts?", 
         "The formula is ∫ u dv = uv - ∫ v du."),
         
        ("How should we choose u and dv in this case?", 
         "A common rule is LIATE. Here, Algebraic (x) comes before Exponential (e^x). So let u = x and dv = e^x dx.")
    ]
    # qa_pairs = [
    #     ("What is the formula for integration by parts?", 
    #      "The formula is ∫ u dv = uv - ∫ v du."),
    # ]


    # 4. 执行打分
    print(f"Original Question: {root_question}")
    print(f"History Depth: {len(qa_pairs)} turns")
    print("... Calling LLM Judge ...")

    try:
        for i, (current_asking, current_response) in enumerate(qa_pairs):
            # 1. 提取历史：当前索引 i 之前的所有对
            # 当 i=0 时，history 为空 [] -> 正确
            # 当 i=1 时，history 为 [(q0, a0)] -> 正确
            prev_history = qa_pairs[:i]
            score = metric.score(
                question=root_question,
                response=current_asking,
                qa_pairs=prev_history
            )
        print(f"\n>>> 🏆 Helpfulness Score: {score}")
        print(f">>> (Expect High score as it guides the next logical step)\n")
    except Exception as e:
        print(f"\n>>> ❌ Execution Failed: {e}\n")