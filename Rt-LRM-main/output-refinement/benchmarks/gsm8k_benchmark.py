#!/usr/bin/env python3
"""
GSM8K 基准评测：验证 LCO 方法在数学推理任务上的有效性保持
完整实现 LCO：包含自思考约束生成和进化采样两部分
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
import re
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from filtering import query, is_vllm_model
from reward_hacking.utils import process_response, get_crossover_prompt, get_mutate_prompt

# GSM8K 示例数据（8-shot Chain-of-Thought prompting）
GSM8K_FEW_SHOT_EXAMPLES = """The following are math problems with step-by-step solutions.

Question: James has 3 boxes of cookies. Each box has 8 cookies. He eats 5 cookies. How many cookies does he have left?
Answer: Let's think step by step. James has 3 boxes of cookies. Each box has 8 cookies. So he has 3 * 8 = 24 cookies. He eats 5 cookies. So he has 24 - 5 = 19 cookies left. The answer is 19.

Question: A train travels at 60 miles per hour. How far will it travel in 2.5 hours?
Answer: Let's think step by step. The train travels at 60 miles per hour. In 2.5 hours, it will travel 60 * 2.5 = 150 miles. The answer is 150.

Question: Sarah has $25. She buys 3 books that cost $4 each and a pen that costs $2. How much money does she have left?
Answer: Let's think step by step. The 3 books cost 3 * $4 = $12. The pen costs $2. Total cost is $12 + $2 = $14. Sarah had $25. So she has $25 - $14 = $11 left. The answer is 11.

Question: A rectangle has length 12 cm and width 8 cm. What is its perimeter?
Answer: Let's think step by step. The perimeter of a rectangle is 2 * (length + width). So perimeter = 2 * (12 + 8) = 2 * 20 = 40 cm. The answer is 40.

Question: Tom has 48 marbles. He gives 1/4 of them to his friend and 1/3 of the remaining to his sister. How many marbles does he have left?
Answer: Let's think step by step. Tom gives away 1/4 of 48 = 12 marbles. Remaining: 48 - 12 = 36 marbles. He gives 1/3 of 36 = 12 marbles to his sister. Remaining: 36 - 12 = 24 marbles. The answer is 24.

Question: A bakery makes 240 cupcakes per day. They sell 3/4 of them and give 1/5 of the remaining to staff. How many cupcakes are left at the end of the day?
Answer: Let's think step by step. They sell 3/4 of 240 = 180 cupcakes. Remaining: 240 - 180 = 60 cupcakes. They give 1/5 of 60 = 12 cupcakes to staff. Remaining: 60 - 12 = 48 cupcakes. The answer is 48.

Question: Mike is 8 years older than his sister. In 5 years, Mike will be twice as old as his sister. How old is Mike now?
Answer: Let's think step by step. Let sister's current age be x. Mike's current age is x + 8. In 5 years: Mike = x + 8 + 5 = x + 13, Sister = x + 5. We know Mike will be twice as old: x + 13 = 2(x + 5). Solving: x + 13 = 2x + 10, so x = 3. Mike's age is 3 + 8 = 11. The answer is 11.

Question: A store sells apples for $0.50 each and oranges for $0.75 each. If a customer buys 8 apples and 6 oranges with a $20 bill, how much change should they receive?
Answer: Let's think step by step. Cost of apples: 8 * $0.50 = $4.00. Cost of oranges: 6 * $0.75 = $4.50. Total cost: $4.00 + $4.50 = $8.50. Change from $20: $20 - $8.50 = $11.50. The answer is 11.50.
"""

# 用于数学问题的 LCO 约束生成提示
CONSTRAINT_SYSTEM_PROMPT = """You are an expert at solving math word problems. Your task is to analyze the problem and generate specific constraints/strategies that will help solve it correctly while avoiding common mistakes."""

CONSTRAINT_USER_PROMPT = """Please analyze what should be paid attention to when solving this type of math problem to avoid making mistakes:
Problem type: {topic}

Please provide 3-5 specific constraints or strategies that would help solve such problems correctly. Be concise and practical."""

# 数学答案评判提示
MATH_JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating math problem solutions. Your task is to select the best solution based on correctness, clarity, and completeness."""

def get_math_judge_prompt(question, responses):
    """生成数学答案评判prompt"""
    prompt = f"""Question: {question}

I have {len(responses)} candidate solutions:

"""
    for i, response in enumerate(responses):
        prompt += f"Solution {i+1}:\n{response}\n\n"
    
    prompt += """Which solution is the most correct, clear, and complete? Consider:
1. Correctness: Is the final answer accurate?
2. Reasoning: Is the step-by-step logic sound?
3. Clarity: Is the explanation easy to follow?

Respond with ONLY the number of the best solution (1, 2, 3, etc.)."""
    return prompt


def generate_gsm8k_prompt(question):
    """生成 GSM8K 评测 prompt"""
    prompt = GSM8K_FEW_SHOT_EXAMPLES + "\n"
    prompt += f"Question: {question}\n"
    prompt += "Answer: Let's think step by step."
    return prompt


def extract_final_answer(response):
    """
    从模型输出中提取最终答案
    寻找 "The answer is X" 或最后一行的数字
    """
    # 首先尝试 process_response
    cleaned = process_response(response)
    
    # 策略 1: 寻找 "The answer is X" 格式
    patterns = [
        r"the answer is\s*[:\s]*\$?([\d,]+\.?\d*)",
        r"answer[:\s]+\$?([\d,]+\.?\d*)",
        r"=\s*\$?([\d,]+\.?\d*)\s*$",  # 行尾等号后的数字
    ]
    
    response_lower = response.lower()
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            # 移除逗号并转换为数字
            num_str = match.group(1).replace(',', '')
            try:
                if '.' in num_str:
                    return float(num_str)
                else:
                    return int(num_str)
            except:
                pass
    
    # 策略 2: 查找所有数字，取最后一个
    numbers = re.findall(r'\$?([\d,]+\.?\d*)', response)
    if numbers:
        # 尝试返回最后一个看起来合理的数字
        for num_str in reversed(numbers):
            try:
                num_str = num_str.replace(',', '')
                if '.' in num_str:
                    num = float(num_str)
                else:
                    num = int(num_str)
                # 过滤掉明显不合理的（太大的数字）
                if 0 < num < 1000000:
                    return num
            except:
                continue
    
    return None


def compare_answers(predicted, correct, tolerance=0.01):
    """比较预测答案和正确答案，允许小数误差"""
    if predicted is None:
        return False
    
    # 转换为浮点数比较
    try:
        pred_float = float(predicted)
        corr_float = float(correct)
        return abs(pred_float - corr_float) < tolerance
    except:
        # 如果转换失败，进行字符串比较
        return str(predicted).strip() == str(correct).strip()


def generate_constraints(args, topic, model_config):
    """自思考约束生成：第一阶段"""
    constraint_prompt = CONSTRAINT_USER_PROMPT.format(topic=topic)
    constraints = query(
        model_name=args.agent_model,
        user_prompt=constraint_prompt,
        system_prompt=CONSTRAINT_SYSTEM_PROMPT,
        max_tokens=200,
        **model_config
    )
    return constraints


def generate_multiple_responses(args, full_prompt, system_prompt, n=3, model_config=None):
    """生成多个候选回复（种群初始化）"""
    if model_config is None:
        model_config = {}
    
    responses = []
    
    # 检测是否使用 VLLM
    if is_vllm_model(args.agent_model):
        # VLLM 模式：串行生成
        for i in range(n):
            try:
                response = query(
                    model_name=args.agent_model,
                    user_prompt=full_prompt,
                    system_prompt=system_prompt,
                    max_tokens=512,
                    temperature=0.7,  # 使用一定随机性增加多样性
                    **model_config
                )
                responses.append(response)
            except Exception as e:
                print(f"Error generating response {i+1}/{n}: {e}")
                responses.append(None)
    else:
        # API 模式：使用线程池并行生成
        with ThreadPoolExecutor(max_workers=min(n, 5)) as executor:
            futures = [
                executor.submit(
                    query,
                    args.agent_model,
                    full_prompt,
                    system_prompt,
                    512,
                    0.7,
                    [],
                    **model_config
                )
                for _ in range(n)
            ]
            for future in as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as e:
                    print(f"Error generating response: {e}")
                    responses.append(None)
    
    # 过滤 None
    return [r for r in responses if r is not None]


def score_responses(question, responses, correct_answer, args, model_config):
    """评分响应：基于答案正确性"""
    scored = []
    for response in responses:
        predicted = extract_final_answer(response)
        is_correct = compare_answers(predicted, correct_answer)
        # 正确得1分，错误得0分
        score = 1.0 if is_correct else 0.0
        scored.append((response, score))
    return scored


def crossover(scored_responses, args, model_config, num_crossovers=2):
    """交叉操作：结合两个父代回复的优点"""
    # 取分数最高的前 K 个
    K = min(3, len(scored_responses))
    top_k = [resp for resp, score in scored_responses[:K]]
    
    if len(top_k) < 2:
        return []
    
    children = []
    
    # 检测是否使用 VLLM
    if is_vllm_model(args.agent_model):
        for i in range(num_crossovers):
            try:
                parent1, parent2 = random.sample(top_k, 2)
                crossover_prompt = get_crossover_prompt(
                    "math problem solving", 
                    parent1, 
                    parent2
                )
                child = query(
                    model_name=args.agent_model,
                    user_prompt=crossover_prompt,
                    system_prompt="You are a helpful assistant. Create a better solution by combining the best aspects of both solutions.",
                    max_tokens=512,
                    temperature=0.7,
                    **model_config
                )
                children.append(child)
            except Exception as e:
                print(f"Error during crossover: {e}")
    else:
        # API 模式：并行
        def crossover_single():
            try:
                parent1, parent2 = random.sample(top_k, 2)
                crossover_prompt = get_crossover_prompt(
                    "math problem solving", 
                    parent1, 
                    parent2
                )
                child = query(
                    model_name=args.agent_model,
                    user_prompt=crossover_prompt,
                    system_prompt="You are a helpful assistant. Create a better solution by combining the best aspects of both solutions.",
                    max_tokens=512,
                    temperature=0.7,
                    **model_config
                )
                return child
            except Exception as e:
                print(f"Error during crossover: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=min(num_crossovers, 3)) as executor:
            futures = [executor.submit(crossover_single) for _ in range(num_crossovers)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    children.append(result)
    
    return children


def mutate(scored_responses, args, model_config):
    """变异操作：对较差的回复进行改进"""
    # 取分数最低的 K 个
    K = min(2, len(scored_responses))
    bottom_k = [resp for resp, score in scored_responses[-K:]]
    
    children = []
    
    # 检测是否使用 VLLM
    if is_vllm_model(args.agent_model):
        for response in bottom_k:
            try:
                mutate_prompt = get_mutate_prompt("math problem solving", response)
                child = query(
                    model_name=args.agent_model,
                    user_prompt=mutate_prompt,
                    system_prompt="You are a helpful assistant. Improve the solution to make it clearer and more accurate.",
                    max_tokens=512,
                    temperature=0.7,
                    **model_config
                )
                children.append(child)
            except Exception as e:
                print(f"Error during mutation: {e}")
    else:
        # API 模式：并行
        def mutate_single(response):
            try:
                mutate_prompt = get_mutate_prompt("math problem solving", response)
                child = query(
                    model_name=args.agent_model,
                    user_prompt=mutate_prompt,
                    system_prompt="You are a helpful assistant. Improve the solution to make it clearer and more accurate.",
                    max_tokens=512,
                    temperature=0.7,
                    **model_config
                )
                return child
            except Exception as e:
                print(f"Error during mutation: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=min(len(bottom_k), 3)) as executor:
            futures = [executor.submit(mutate_single, resp) for resp in bottom_k]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    children.append(result)
    
    return children


def vote_for_best_response(question, responses, args, model_config):
    """投票选择最佳回复"""
    if len(responses) == 0:
        return None
    if len(responses) == 1:
        return responses[0]
    
    # 构建候选文本
    candidate_texts = "\n\n".join([
        f"[{i}] {resp[:500]}..." if len(resp) > 500 else f"[{i}] {resp}"
        for i, resp in enumerate(responses)
    ])
    
    judge_prompt = get_math_judge_prompt(question, responses)
    
    try:
        result = query(
            model_name=args.agent_model,
            user_prompt=judge_prompt,
            system_prompt=MATH_JUDGE_SYSTEM_PROMPT,
            max_tokens=50,
            temperature=0.0,
            **model_config
        )
        
        # 解析投票结果
        match = re.search(r"(?:\[\(`\"']\s*)?(\d+)(?:\s*\[\)`\"'])?", result)
        if match:
            idx = int(match.group(1)) - 1  # 转换为0-based索引
            if 0 <= idx < len(responses):
                return responses[idx]
        
        # 如果解析失败，返回第一个
        return responses[0]
    except Exception as e:
        print(f"Error during voting: {e}")
        return responses[0]


def lco_optimize(question, correct_answer, args, model_config):
    """
    完整 LCO 优化流程
    1. 自思考约束生成
    2. 生成多个候选回复（种群初始化）
    3. 进化采样（交叉和变异）
    4. 投票选择最佳回复
    """
    # 1. 自思考约束生成
    constraints = generate_constraints(args, "math problem solving", model_config)
    
    # 2. 构建带约束的 prompt
    base_prompt = generate_gsm8k_prompt(question)
    full_prompt = f"{constraints}\n\n###\n\n{base_prompt}"
    
    # 3. 生成多个候选回复（种群初始化）
    population_size = getattr(args, 'lco_population_size', 3)
    responses = generate_multiple_responses(
        args, 
        full_prompt, 
        "You are a helpful assistant. Solve the math problem step by step.",
        n=population_size,
        model_config=model_config
    )
    
    if len(responses) == 0:
        # 如果生成失败，使用基础方法
        response = query(
            model_name=args.agent_model,
            user_prompt=base_prompt,
            system_prompt="You are a helpful assistant. Solve the math problem step by step.",
            max_tokens=512,
            temperature=0.0,
            **model_config
        )
        return response, constraints
    
    # 4. 评分响应
    scored_responses = score_responses(question, responses, correct_answer, args, model_config)
    
    # 5. 进化操作（交叉和变异）
    children = []
    
    # 交叉
    if len(scored_responses) >= 2:
        crossover_children = crossover(scored_responses, args, model_config, num_crossovers=2)
        children.extend(crossover_children)
    
    # 变异
    mutation_children = mutate(scored_responses, args, model_config)
    children.extend(mutation_children)
    
    # 6. 合并所有候选
    all_responses = responses + children
    
    # 7. 投票选择最佳回复
    best_response = vote_for_best_response(question, all_responses, args, model_config)
    
    return best_response, constraints


def evaluate_single_question(q_data, args, model_config):
    """评估单个数学问题"""
    question = q_data['question']
    correct_answer = q_data['answer']
    
    try:
        # 使用 LCO 方法（完整版）
        if args.method == 'LCO':
            response, constraints = lco_optimize(question, correct_answer, args, model_config)
        elif args.method == 'self_reflection':
            # 仅约束生成，无进化采样
            constraints = generate_constraints(args, "math problem solving", model_config)
            prompt = generate_gsm8k_prompt(question)
            full_prompt = f"{constraints}\n\n###\n\n{prompt}"
            response = query(
                model_name=args.agent_model,
                user_prompt=full_prompt,
                system_prompt="You are a helpful assistant. Solve the math problem step by step.",
                max_tokens=512,
                temperature=0.0,
                **model_config
            )
        else:
            # 基础方法
            prompt = generate_gsm8k_prompt(question)
            response = query(
                model_name=args.agent_model,
                user_prompt=prompt,
                system_prompt="You are a helpful assistant. Solve the math problem step by step.",
                max_tokens=512,
                temperature=0.0,
                **model_config
            )
        
        # 提取答案
        predicted = extract_final_answer(response)
        
        # 比较答案
        is_correct = compare_answers(predicted, correct_answer)
        
        return {
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': predicted,
            'is_correct': is_correct,
            'raw_response': response
        }
        
    except Exception as e:
        print(f"Error evaluating question: {e}")
        return {
            'question': question,
            'correct_answer': correct_answer,
            'predicted_answer': None,
            'is_correct': False,
            'raw_response': str(e)
        }


def load_gsm8k_data(data_dir=None, num_samples=50, seed=42):
    """
    加载 GSM8K 数据集 (使用 HuggingFace datasets)
    如果可用则使用真实数据，否则使用示例数据
    """
    try:
        from datasets import load_dataset
        
        print(f"Loading GSM8K dataset from HuggingFace (sample {num_samples} examples)...")
        
        # 加载 GSM8K 数据集 (使用 test 分割)
        dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        
        # 转换为列表
        all_data = []
        for item in dataset:
            # GSM8K 格式: question, answer (包含推理过程和最终答案)
            # answer 格式: "... The answer is 42."
            answer_text = item["answer"]
            
            # 提取最终答案（最后一个数字）
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', answer_text)
            if numbers:
                final_answer = numbers[-1].replace(',', '')
            else:
                final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else "0"
            
            all_data.append({
                "question": item["question"],
                "answer": final_answer
            })
        
        # 随机抽样
        import random
        random.seed(seed)
        sampled_data = random.sample(all_data, min(num_samples, len(all_data)))
        
        print(f"Successfully loaded {len(sampled_data)} real GSM8K examples")
        return sampled_data
        
    except Exception as e:
        print(f"Warning: Could not load real GSM8K data: {e}")
        print("Falling back to sample data...")
        
        # 示例 GSM8K 数据（备用）
        sample_data = [
            {
                "question": "John has 15 apples. He gives 6 to his friend and buys 9 more from the store. How many apples does John have now?",
                "answer": "18"
            },
            {
                "question": "A car travels 240 miles using 8 gallons of gas. How many miles per gallon does the car get?",
                "answer": "30"
            },
            {
                "question": "Lisa has $50. She spends $18 on a book and $12 on lunch. How much money does she have left?",
                "answer": "20"
            },
            {
                "question": "A classroom has 24 students. If they form groups of 4, how many groups will there be?",
                "answer": "6"
            },
            {
                "question": "Tom walks 3 miles every day for a week. How many miles does he walk in total?",
                "answer": "21"
            }
        ]
        
        # 循环复制以获得更多样本
        data = []
        while len(data) < num_samples:
            data.extend(sample_data)
        
        return data[:num_samples]


def run_gsm8k_evaluation(args):
    """运行 GSM8K 评测"""
    print("=" * 70)
    print("GSM8K 基准评测")
    print("=" * 70)
    print(f"模型: {args.agent_model}")
    print(f"方法: {args.method}")
    if args.method == 'LCO':
        print(f"LCO 种群大小: {args.lco_population_size}")
    print(f"样本数: {args.num_samples}")
    print("-" * 70)
    
    # 加载数据（使用 seed 保证可复现）
    data = load_gsm8k_data(args.data_dir, args.num_samples, args.seed)
    
    # 准备模型配置
    model_config = {}
    if is_vllm_model(args.agent_model):
        model_config = {
            "tensor_parallel_size": args.vllm_tp_size,
            "gpu_memory_utilization": args.vllm_gpu_util
        }
    
    # 运行评测
    results = []
    
    if args.max_workers > 1 and not is_vllm_model(args.agent_model):
        # API 模式：并发
        print(f"使用 {args.max_workers} 个并发 workers...")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(evaluate_single_question, q_data, args, model_config)
                for q_data in data
            ]
            for future in tqdm(as_completed(futures), total=len(data), desc="Evaluating GSM8K"):
                results.append(future.result())
    else:
        # VLLM 模式：串行
        print("使用串行模式...")
        for q_data in tqdm(data, desc="Evaluating GSM8K"):
            result = evaluate_single_question(q_data, args, model_config)
            results.append(result)
    
    # 计算指标
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    # 输出结果
    print("\n" + "=" * 70)
    print("GSM8K 评测结果")
    print("=" * 70)
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy * 100:.2f}%")
    
    # 显示一些错误示例
    incorrect_examples = [r for r in results if not r['is_correct']][:3]
    if incorrect_examples:
        print("\n错误示例:")
        for i, ex in enumerate(incorrect_examples, 1):
            print(f"\n{i}. 问题: {ex['question'][:80]}...")
            print(f"   正确答案: {ex['correct_answer']}")
            print(f"   预测答案: {ex['predicted_answer']}")
    
    print("=" * 70)
    
    # 保存结果
    output_data = {
        "benchmark": "GSM8K",
        "model": args.agent_model,
        "method": args.method,
        "num_samples": total,
        "accuracy": accuracy,
        "correct": correct,
        "details": results
    }
    
    output_file = f"benchmark_results/gsm8k_{args.method}_{args.agent_model.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {output_file}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K 基准评测")
    parser.add_argument("--agent_model", type=str, required=True, help="Agent模型")
    parser.add_argument("--method", type=str, default="base", help="方法 (base, LCO, self_reflection)")
    parser.add_argument("--data_dir", type=str, default=None, help="GSM8K数据目录")
    parser.add_argument("--num_samples", type=int, default=50, help="评测样本数 (默认50)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (用于数据抽样)")
    parser.add_argument("--max_workers", type=int, default=1, help="并发数 (API模式)")
    parser.add_argument("--vllm_tp_size", type=int, default=1, help="VLLM tensor parallel")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.85, help="VLLM GPU利用率")
    parser.add_argument("--lco_population_size", type=int, default=3, help="LCO种群大小 (默认3)")
    
    args = parser.parse_args()
    
    run_gsm8k_evaluation(args)
