import logging
from typing import Any, Dict, List, Optional

import litellm

from .metrics.utils.extract_json_reliable import extract_json
# litellm._turn_on_debug()
# logger = logging.getLogger(__name__)
import os
os.environ["OPENAI_API_KEY"] = "sk-EAR9W8gMRCM2JIMJ2e5b8b095c494718B5CeC1C17004577d"
os.environ["LITELLM_API_BASE"] = "https://api.ai-gaochao.cn/v1/"

# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
with open("/home/chenxin/verl-interactive/verl-tool/verl_tool/workers/reward_manager/helpfulness_reward_prompt.txt", "r", encoding="utf-8") as f:
    HELPFULNESS_JUDGEMENT_PROMPT = f.read()

# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
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
            "model": "gpt-4o-mini",
            "api_base": "https://api.ai-gaochao.cn/v1",
            **llm_kwargs,
        }

    # --------------------------------------------------------------------- #
    def score(
        self,
        question,
        response,
        history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, float]:
        """
        `prompt`, `groundtruth`, and `response` are unused here;
        the full conversation in `messages` is what matters.
        """

        # ------------------------------------------------------------------ #
        # 1) Build chat history string                                       #
        # ------------------------------------------------------------------ #
        # chat_history = "\n".join(
        #     f"{m['role'].capitalize()}: {m['content']}" for m in messages
        # )

        eval_prompt = HELPFULNESS_JUDGEMENT_PROMPT.format(question=question, response=response)

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
    # Example usage
    metric = HelpfullnessMetric()
    score = metric.score(
        question="How to improve code quality?",
        response="Which aspects of code quality are you interested in improving?",
    )
    print("Helpfullness score:", score)