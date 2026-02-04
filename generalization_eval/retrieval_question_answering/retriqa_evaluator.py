import re
import string
import json
import os
import litellm
import argparse
from tqdm import tqdm
# from datasets import load_dataset # 如果不读取 Huggingface 数据集，可以注释掉
from ..utils.extract_json_reliable import extract_json
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..utils.prompt import EXTRACT_MATCH_PROMPT
from transformers import AutoTokenizer
llm_judgement_api_base = os.getenv("LLM_JUDGEMENT_API_BASE", "https://api.ai-gaochao.cn/v1/")
llm_judgement_api_key = os.getenv("LLM_JUDGEMENT_API_KEY", "sk-xxxxx")
llm_judgement_model = os.getenv("LLM_JUDGEMENT_MODEL", "gpt-4o-mini")

class TriviaQAEvaluator:
    def __init__(self, 
                 dataset_path,
                 tokenizer_path):
        """
        Initialize the TriviaQA Evaluator.
        """
        self.extract_match_prompt = EXTRACT_MATCH_PROMPT
        
        # 加载数据
        if dataset_path:
            self._load_data(dataset_path)
        else:
            self.dataset = []
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load the prompt template
        # 注意：请确保这个路径是存在的，或者将其改为相对路径

    def _load_data(self, dataset_path):
        """
        Load dataset from a JSONL file.
        """
        print(f"Loading data from {dataset_path}...")
        data = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            self.dataset = data
            print(f"Successfully loaded {len(self.dataset)} samples.")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.dataset = []

    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def handle_punc(text):
            exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text):
            return text.lower()

        def replace_underscore(text):
            return text.replace('_', ' ')

        return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

    def harness_triviaqa_normalizer(self, text: str) -> str:
        """Normalize text for TriviaQA evaluation."""
        if not text: return ""
        light_eval_extract = text.lower().translate(str.maketrans("", "", string.punctuation))
        official_repo_normalized = self.normalize_answer(light_eval_extract)
        return official_repo_normalized

    def extract_answer_from_prediction(self, prediction: str) -> str:
        """Extracts the potential answer part from a long model generation."""
        if not prediction: return ""
        extract_answer_text = ""
        if "answer is" in prediction:
            extract_answer_text = prediction.split("answer is")[-1].strip()
        elif "**" in prediction:
            matches = re.findall(r'\*\*(.*?)\*\*', prediction)
            if matches:
                extract_answer_text = matches[-1]
        elif "is the correct answer" in prediction:
            temp = prediction.split("is the correct answer")[0]
            extract_answer_text = temp.split(",")[-1].split("\n")[-1]
        else:
            extract_answer_text = prediction.strip()

        extract_answer_text = extract_answer_text.strip("()").strip()
        return extract_answer_text

    def metric_max_over_ground_truths(self, nor_answer: str, ground_truths: list) -> float:
        """Checks if the normalized answer matches any of the ground truth aliases."""
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            normalized_gt = self.normalize_answer(ground_truth)
            if nor_answer == normalized_gt:
                score = 1.0
            else:
                score = 0.0
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths) if scores_for_ground_truths else 0.0


    def judge_by_llm(self, question: str, groundtruths: list, output: str) -> float:
        """Uses an LLM to judge the correctness of the answer."""
        prompt = self.extract_match_prompt.format(
            single_turn_prompt=question,
            groundtruths=groundtruths,
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
                # 假设 extract_json 是你自己定义的函数，如果报错请检查该函数
                try:
                    full_response = extract_json(full_response)
                except:
                    pass # 如果提取失败，保持原样或处理

            exact_match = 0.0
            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'thought', 'exact_match'}.issubset(keys):
                    exact_match = float(full_response.pop('exact_match'))
                elif 'exact_match' in keys:
                        exact_match = float(full_response['exact_match'])
                else:
                    print(f"Keys {keys} do not match expected keys.")
            return exact_match

        except Exception as e:
            print(f"Error in LLM judgment: {e}")
            return 0.0


    def _process_single_sample(self, sample, use_llm_judge):
        """
        处理单个样本的辅助函数，用于多线程调用。
        """
        # 兼容不同的数据格式
        question = sample.get('question', '')
        if not question and 'prompt' in sample:
            question = sample['prompt']
        
        ground_truths = sample.get('ground_truths', [])
        prediction = sample.get('prediction', '')
        final_response = sample.get('final_response', '')
        token_length = self._compute_token_length(final_response)  # 计算token长度
        
        # 1. 规则匹配 (Rule-based Evaluation)
        extracted_text = self.extract_answer_from_prediction(prediction)
        normalized_answer = self.harness_triviaqa_normalizer(extracted_text)
        score = self.metric_max_over_ground_truths(normalized_answer, ground_truths)
        
        # 2. LLM 裁判 (LLM Judge)
        # 只有当规则匹配失败且开启了LLM裁判时才调用
        if score < 1.0 and use_llm_judge:
            score = self.judge_by_llm(question, ground_truths, prediction)

        return {
            "question": question,
            "ground_truths": ground_truths,
            "prediction": prediction,
            "token_length": token_length,
            "score": score
        }
    
    def _compute_token_length(self, response):
        """Computes token length using the tokenizer."""
        token_count = self.tokenizer.encode(response, return_tensors='pt')
        return token_count.shape[1]

    def run_evaluation(self, output_dir: str, use_llm_judge: bool = False, max_workers: int = 10):
        """
        Run evaluation on the loaded dataset using multi-threading.
        
        Args:
            output_dir: 输出目录
            use_llm_judge: 是否使用 LLM 进行裁判
            max_workers: 线程池的最大线程数 (并发请求数)
        """
        if not self.dataset:
            print("No dataset loaded. Exiting evaluation.")
            return 0.0

        em_scores = []
        results = []

        print(f"Starting evaluation on {len(self.dataset)} samples with {max_workers} threads...")
        
        # 使用 ThreadPoolExecutor 进行并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [
                executor.submit(self._process_single_sample, sample, use_llm_judge) 
                for sample in self.dataset
            ]
            
            # 使用 tqdm 显示进度，as_completed 会在任务完成时 yield
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
                try:
                    result = future.result()
                    results.append(result)
                    em_scores.append(result['score'])
                except Exception as e:
                    print(f"An error occurred during sample processing: {e}")

        total_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
        print(f"Evaluation finished. Average EM: {total_em:.4f}")

        avg_token_length = sum(r['token_length'] for r in results) / len(results) if results else 0
        print(f"Average Token Length of Responses: {avg_token_length:.2f}")

        # 保存结果
        print(f"Saving results to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        summary_path = os.path.join(output_dir, "summary.json")
        details_path = os.path.join(output_dir, "details.json")

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({"average_em": total_em, "num_samples": len(em_scores), "average_token_length": avg_token_length}, f, indent=4)
        
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        return total_em

# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriviaQA Evaluation Script")
    parser.add_argument("--input_file", "-i",type=str,required=True,help="输入文件路径 (JSONL格式)")
    parser.add_argument("--tokenizer_path",type=str,required=True,help="Path to the tokenizer (HuggingFace model path).")
    parser.add_argument("--output_dir", "-o",type=str,default=None,help="输出目录路径，默认为输入文件所在目录")
    parser.add_argument("--use_llm_judge",action="store_true",default=True,help="是否使用 LLM 进行裁判")
    parser.add_argument("--max_workers",type=int,default=32,help="最大并行工作线程数，默认为32")
    args = parser.parse_args()
    # 如果没有指定输出目录，则使用输入文件所在目录
    output_directory = args.output_dir if args.output_dir else os.path.dirname(args.input_file)
    print(f"Input File: {args.input_file}")
    print(f"Output Directory: {output_directory}")
    print(f"Use LLM Judge: {args.use_llm_judge}")
    print(f"Max Workers: {args.max_workers}")

    # 初始化并加载数据
    evaluator = TriviaQAEvaluator(
        dataset_path=args.input_file,
        tokenizer_path=args.tokenizer_path
    )
    
    # 运行评估
    evaluator.run_evaluation(
        output_dir=output_directory, 

        use_llm_judge=args.use_llm_judge,
        max_workers=args.max_workers,
    )

    # # 1. 定义输入文件路径
    # input_file_path = "/home/chenxin/verl-interactive/generalization_eval/RetriaQA/eval_results/Proactive-Interactive-R1-Math-7B_gpt-4o-mini_interactive_generation/triviaqa_results_Proactive-Interactive-R1-Math-7B.jsonl"
    
    # # 2. 自动获取该文件所在的目录作为输出目录
    # # os.path.dirname 会去掉文件名，只保留目录路径
    # output_directory = os.path.dirname(input_file_path)
    
    # print(f"Input File: {input_file_path}")
    # print(f"Output Directory: {output_directory}")

    # # # 3. 初始化并加载数据
    # evaluator = TriviaQAEvaluator(dataset_path=input_file_path)
    
    # # 4. 运行评估
    # # 注意：use_llm_judge=True 会消耗 API token，测试时可以先设为 False
    # evaluator.run_evaluation(
    #     output_dir=output_directory, 
    #     use_llm_judge=True,
    #     max_workers=32
    # )