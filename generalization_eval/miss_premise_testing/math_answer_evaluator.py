import json
from tqdm import tqdm
from ..utils.math_utils import extract_answer, grade_answer_sympy, grade_answer_mathd
import litellm
from ..utils.extract_json_reliable import extract_json
from ..utils.prompt import ACC_PROMPT
import os
from transformers import AutoTokenizer
llm_judgement_api_base = os.getenv("LLM_JUDGEMENT_API_BASE", "https://api.ai-gaochao.cn/v1/")
llm_judgement_api_key = os.getenv("LLM_JUDGEMENT_API_KEY", "sk-xxxxx")
llm_judgement_model = os.getenv("LLM_JUDGEMENT_MODEL", "gpt-4o-mini")


class MathAnswerEvaluator:
    def __init__(self, dataset_path, tokenizer_path, reasoning_model=False,judge_by_llm=False):
        """
        Initializes the MathAnswerEvaluator with the dataset file path.
        
        Args:
            dataset_path (str): Path to the dataset JSON file.
        """
        self.dataset_path = dataset_path
        self.dataset_name = os.path.basename(self.dataset_path).replace(".json", "")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.reasoning_model = reasoning_model
        self.judge_by_llm = judge_by_llm
        self.dataset = self._load_dataset()
        self.rewards = []
        self.token_lengths = []  # 新增：存储token长度

    def _load_dataset(self):
        """
        Loads the dataset from the given JSON file.

        Returns:
            list: A list of dataset entries.
        """
        with open(self.dataset_path, "r") as f:
            dataset = json.load(f)
        dataset = [item for item in dataset if 'output' in item]
        return dataset
    
    def _judge_by_llm(self, question, answer, output):
        """
        Uses an LLM to judge the correctness of the answer.

        Args:
            question (str): The question string.
            answer (str): The correct answer string.
            output (str): The LLM's output string.

        Returns:
            float: Reward score (1.0 for correct, 0.0 for incorrect).
        """
        prompt = ACC_PROMPT.format(
            single_turn_prompt=question,
            groundtruth=answer,
            completion=output
        )
        try:
            full_response = litellm.completion(
                base_url=llm_judgement_api_base,
                api_key=llm_judgement_api_key,
                model=llm_judgement_model,
                messages=[{"role": "user", "content": prompt}],
                retry_delay=5,
                num_retries=3,
            ).choices[0].message.content
            
            if isinstance(full_response, str):
                full_response = extract_json(full_response)

            accuracy = 0.0
            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'accuracy'}.issubset(keys):
                    accuracy = float(full_response.pop('accuracy'))
                elif 'accuracy' in keys:
                    accuracy = float(full_response['accuracy'])
                else:
                    print(f"Keys {keys} do not match expected keys.")
            return accuracy

        except Exception as e:
            print(f"Error in LLM judgment: {e}")
            return 0.0
        

    def _judge_by_rule(self, parsed_answer, output):
        """
        Computes a binary reward by verifying the gold and answer expressions.

        Args:
            answer (str): The correct answer string.
            output (str): The LLM's output string.
        Returns:
            float or None: Binary reward (1.0 or 0.0) if verifiable, or None if verification fails.
        """
        # parsed_answer = extract_answer(answer)
        parsed_output = extract_answer(output)
        if parsed_output is None or parsed_output == "" or parsed_output == "no final answer found":
            print(f"No valid output extracted, returning 0.0")
            return 0.0
        # print(f"Parsed answer: {parsed_answer}, Parsed output: {parsed_output}")
        try:
            result = grade_answer_mathd(parsed_output, parsed_answer) or grade_answer_sympy(parsed_output, parsed_answer)
            # print(f"Verification result: {result}")
            return 1.0 if result else 0.0
        except Exception as e:
            print(f"Verification failed: {e}, answer: {parsed_answer}, output: {output}")
        return 0.0
    
    def _compute_token_length(self, response):
        """Computes token length using the tokenizer."""
        token_count = self.tokenizer.encode(response, return_tensors='pt')
        return token_count.shape[1]

    def _process_single_item(self, index, item):
        """
        Helper function to process a single dataset item.
        Returns tuple: (index, reward, token_length)
        """
        try:
            con = item['output']
            question = item['question']
            parsed_answer = item['answer']
            token_length = self._compute_token_length(con)  # 计算token长度

            # Handle thinking tokens if present
            if self.reasoning_model:
                if "</think>" in con:
                    output = con.split("</think>")[-1]
                else:
                    output = "no final answer found"
            else:
                output = con
            # print(f"Output: {output}")
            reward = self._judge_by_rule(parsed_answer, output)
            # print(f"Rule-based reward: {reward}")
            if reward == 1.0:
                return index, reward, token_length
            else:
                if not self.judge_by_llm:
                    return index, reward, token_length
                reward = self._judge_by_llm(question, parsed_answer, output)
                return index, reward, token_length

        except Exception as e:
            print(f"Exception processing item {index}: {e}")
            return index, None, None

    def evaluate(self, max_workers=128):
        """
        Evaluates the dataset using multi-threading.
        
        Args:
            max_workers (int): Number of concurrent threads.
        """
        import concurrent.futures
        print(f"Starting evaluation with {max_workers} threads...")
        self.rewards = [None] * len(self.dataset)
        self.token_lengths = [None] * len(self.dataset)  # 初始化token长度列表
        
        # Use ThreadPoolExecutor for I/O bound tasks (API calls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self._process_single_item, i, item): i 
                for i, item in enumerate(self.dataset)
            }
            
            valid_count = 0
            total_score = 0.0
            total_token_length = 0
            valid_token_count = 0

            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(self.dataset), desc="Evaluating"):
                index, reward, token_length = future.result()
                
                # Store reward and token_length in the correct position
                self.rewards[index] = reward
                self.token_lengths[index] = token_length
                
                if reward is not None:
                    valid_count += 1
                    total_score += reward
                
                if token_length is not None:
                    valid_token_count += 1
                    total_token_length += token_length

        # Calculate accuracy and average token length
        accuracy = total_score / valid_count if valid_count > 0 else 0.0
        avg_token_length = total_token_length / valid_token_count if valid_token_count > 0 else 0.0

        print(f"Evaluation completed. Accuracy: {accuracy:.4f} over {valid_count} valid examples.")
        print(f"Average token length: {avg_token_length:.2f} over {valid_token_count} examples.")

        return {
            "accuracy": accuracy,
            "avg_token_length": avg_token_length,
            "total_token_length": total_token_length,
            "rewards": self.rewards,
            "token_lengths": self.token_lengths
        }

    def save_results(self, output_path):
        """
        Saves the evaluation results (including accuracy) to a JSON file.

        Args:
            output_path (str): Path to save the results JSON file.
        """
        results = self.evaluate()
        print(f"Evaluation results: {results}")
        print(f"Saving results to {output_path}...")
        output_path = os.path.join(os.path.dirname(output_path), f"{self.dataset_name}_eval_result.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_path}")


# Example usage:
if __name__ == "__main__":
    # evaluator = MathAnswerEvaluator(
    #     "/home/chenxin/verl-interactive/gpt-4o-mini_math_insufficient_question_direct_generation_result.json",
    #     "/data1/HF-Models/meta-llama/Llama-3.1-8B-Instruct"  # 添加tokenizer路径
    # )
    # evaluator.save_results("results/")
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate math answers using rule-based and LLM-based methods.")
    parser.add_argument("--dataset_path",type=str,required=True,help="Path to the dataset JSON file.")
    parser.add_argument("--tokenizer_path",type=str,required=True,help="Path to the tokenizer (HuggingFace model path).")
    parser.add_argument("--use_llm_judge",action="store_true",default=True,help="Whether to use LLM for judgment. Default: True")
    parser.add_argument("--reasoning_model",action="store_true",help="Whether reasoning model")
    parser.add_argument("--output_path",type=str,default="results/",help="Path to save the evaluation results. Default: results/")
    parser.add_argument("--max_workers",type=int,default=64,help="Number of concurrent threads for evaluation. Default: 128")
    args = parser.parse_args()
    print(f"Arguments: {args}")
    evaluator = MathAnswerEvaluator(
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        reasoning_model=args.reasoning_model,
        judge_by_llm=args.use_llm_judge
    )
    evaluator.save_results(args.output_path)