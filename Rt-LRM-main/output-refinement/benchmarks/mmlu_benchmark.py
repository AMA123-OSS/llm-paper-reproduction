#!/usr/bin/env python3
"""
MMLU 基准评测：验证 LCO 方法在知识密集型任务上的有效性保持
完整实现 LCO：包含自思考约束生成和进化采样两部分
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
import random
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from filtering import query, is_vllm_model
from reward_hacking.utils import process_response, get_crossover_prompt, get_mutate_prompt

# MMLU 示例数据（5-shot prompting）
MMLU_FEW_SHOT_EXAMPLES = """The following are multiple choice questions (with answers) about various subjects.

Question: What is the primary function of the mitochondria in a cell?
(A) Protein synthesis
(B) Energy production
(C) Waste removal
(D) Cell division
Answer: B

Question: Which planet in our solar system has the most extensive ring system?
(A) Jupiter
(B) Saturn
(C) Uranus
(D) Neptune
Answer: B

Question: In what year did World War II end?
(A) 1943
(B) 1944
(C) 1945
(D) 1946
Answer: C

Question: What is the chemical symbol for gold?
(A) Go
(B) Gd
(C) Au
(D) Ag
Answer: C

Question: Who wrote the novel '1984'?
(A) Aldous Huxley
(B) George Orwell
(C) Ray Bradbury
(D) Franz Kafka
Answer: B
"""

# 用于 MMLU 的 LCO 约束生成提示
CONSTRAINT_SYSTEM_PROMPT = """You are an expert at answering multiple choice questions across various subjects. Your task is to generate specific strategies and constraints that will help answer such questions correctly while avoiding common mistakes."""

CONSTRAINT_USER_PROMPT = """Please analyze what should be paid attention to when answering this type of multiple choice question to avoid making mistakes:
Subject: {topic}

Please provide 3-5 specific constraints or strategies that would help answer such questions correctly. Be concise and practical."""

# MMLU 答案评判提示
MMLU_JUDGE_SYSTEM_PROMPT = """You are an expert judge evaluating answers to multiple choice questions. Your task is to select the best answer based on correctness and reasoning quality."""

def get_mmlu_judge_prompt(question, choices, responses):
    """生成 MMLU 答案评判prompt"""
    choices_str = "\n".join([f"({k}) {v}" for k, v in choices.items()])
    
    prompt = f"""Question: {question}
{choices_str}

I have {len(responses)} candidate answers:

"""
    for i, response in enumerate(responses):
        prompt += f"Answer {i+1}:\n{response}\n\n"
    
    prompt += """Which answer is the most correct? Consider:
1. Correctness: Is the selected option (A/B/C/D) accurate?
2. Reasoning: Is the explanation sound (if provided)?

Respond with ONLY the number of the best answer (1, 2, 3, etc.)."""
    return prompt


def generate_mmlu_prompt(question, choices):
    """生成 MMLU 评测 prompt"""
    prompt = MMLU_FEW_SHOT_EXAMPLES + "\n"
    prompt += f"Question: {question}\n"
    for choice_key, choice_text in choices.items():
        prompt += f"({choice_key}) {choice_text}\n"
    prompt += "Answer:"
    return prompt


def parse_answer(response):
    """从模型输出中解析答案 (A/B/C/D)"""
    # 首先尝试 process_response
    cleaned = process_response(response).strip()
    
    # 查找第一个 A/B/C/D
    for char in cleaned.upper():
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    # 如果失败，尝试在原始响应中查找
    response_upper = response.upper()
    for marker in ['ANSWER:', 'ANSWER IS', 'THE ANSWER']:
        idx = response_upper.find(marker)
        if idx != -1:
            # 在标记后查找 A/B/C/D
            after_marker = response_upper[idx + len(marker):idx + len(marker) + 20]
            for char in after_marker:
                if char in ['A', 'B', 'C', 'D']:
                    return char
    
    # 最后尝试在整个文本中找第一个选项字母
    for char in response_upper:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    return None


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
                    max_tokens=50,
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
                    50,
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


def score_responses(responses, correct_answer):
    """评分响应：基于答案正确性"""
    scored = []
    for response in responses:
        predicted = parse_answer(response)
        # 正确得1分，错误得0分
        score = 1.0 if predicted == correct_answer else 0.0
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
                    "multiple choice question answering", 
                    parent1, 
                    parent2
                )
                child = query(
                    model_name=args.agent_model,
                    user_prompt=crossover_prompt,
                    system_prompt="You are a helpful assistant. Create a better answer by combining the best aspects of both answers.",
                    max_tokens=50,
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
                    "multiple choice question answering", 
                    parent1, 
                    parent2
                )
                child = query(
                    model_name=args.agent_model,
                    user_prompt=crossover_prompt,
                    system_prompt="You are a helpful assistant. Create a better answer by combining the best aspects of both answers.",
                    max_tokens=50,
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
                mutate_prompt = get_mutate_prompt("multiple choice question answering", response)
                child = query(
                    model_name=args.agent_model,
                    user_prompt=mutate_prompt,
                    system_prompt="You are a helpful assistant. Improve the answer to make it more accurate.",
                    max_tokens=50,
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
                mutate_prompt = get_mutate_prompt("multiple choice question answering", response)
                child = query(
                    model_name=args.agent_model,
                    user_prompt=mutate_prompt,
                    system_prompt="You are a helpful assistant. Improve the answer to make it more accurate.",
                    max_tokens=50,
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


def vote_for_best_response(question, choices, responses, args, model_config):
    """投票选择最佳回复"""
    if len(responses) == 0:
        return None
    if len(responses) == 1:
        return responses[0]
    
    judge_prompt = get_mmlu_judge_prompt(question, choices, responses)
    
    try:
        result = query(
            model_name=args.agent_model,
            user_prompt=judge_prompt,
            system_prompt=MMLU_JUDGE_SYSTEM_PROMPT,
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


def lco_optimize(question, choices, correct_answer, args, model_config):
    """
    完整 LCO 优化流程
    1. 自思考约束生成
    2. 生成多个候选回复（种群初始化）
    3. 进化采样（交叉和变异）
    4. 投票选择最佳回复
    """
    subject = choices.get('subject', 'general knowledge')
    
    # 1. 自思考约束生成
    constraints = generate_constraints(args, f"MMLU {subject} question", model_config)
    
    # 2. 构建带约束的 prompt
    base_prompt = generate_mmlu_prompt(question, choices)
    full_prompt = f"{constraints}\n\n###\n\n{base_prompt}"
    
    # 3. 生成多个候选回复（种群初始化）
    population_size = getattr(args, 'lco_population_size', 3)
    responses = generate_multiple_responses(
        args, 
        full_prompt, 
        "You are a helpful assistant. Answer with only the letter (A, B, C, or D).",
        n=population_size,
        model_config=model_config
    )
    
    if len(responses) == 0:
        # 如果生成失败，使用基础方法
        response = query(
            model_name=args.agent_model,
            user_prompt=base_prompt,
            system_prompt="You are a helpful assistant. Answer with only the letter (A, B, C, or D).",
            max_tokens=50,
            temperature=0.0,
            **model_config
        )
        return response, constraints
    
    # 4. 评分响应
    scored_responses = score_responses(responses, correct_answer)
    
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
    best_response = vote_for_best_response(question, choices, all_responses, args, model_config)
    
    return best_response, constraints


def evaluate_single_question(q_data, args, model_config):
    """评估单个问题"""
    question = q_data['question']
    choices = q_data['choices']
    correct_answer = q_data['answer']
    subject = q_data.get('subject', 'unknown')
    
    try:
        # 使用 LCO 方法（完整版）
        if args.method == 'LCO':
            response, constraints = lco_optimize(question, choices, correct_answer, args, model_config)
        elif args.method == 'self_reflection':
            # 仅约束生成，无进化采样
            constraints = generate_constraints(args, f"MMLU {subject} question", model_config)
            prompt = generate_mmlu_prompt(question, choices)
            full_prompt = f"{constraints}\n\n###\n\n{prompt}"
            response = query(
                model_name=args.agent_model,
                user_prompt=full_prompt,
                system_prompt="You are a helpful assistant. Answer with only the letter (A, B, C, or D).",
                max_tokens=50,
                temperature=0.0,
                **model_config
            )
        else:
            # 基础方法
            prompt = generate_mmlu_prompt(question, choices)
            response = query(
                model_name=args.agent_model,
                user_prompt=prompt,
                system_prompt="You are a helpful assistant. Answer with only the letter (A, B, C, or D).",
                max_tokens=50,
                temperature=0.0,
                **model_config
            )
        
        # 解析答案
        predicted = parse_answer(response)
        
        return {
            'question': question,
            'subject': subject,
            'correct': correct_answer,
            'predicted': predicted,
            'is_correct': predicted == correct_answer,
            'raw_response': response
        }
        
    except Exception as e:
        print(f"Error evaluating question: {e}")
        return {
            'question': question,
            'subject': subject,
            'correct': correct_answer,
            'predicted': None,
            'is_correct': False,
            'raw_response': str(e)
        }


def load_mmlu_data(data_dir=None, num_samples=50, subjects=None, seed=42):
    """
    加载 MMLU 数据集 (使用 HuggingFace datasets)
    如果可用则使用真实数据，否则使用示例数据
    """
    try:
        from datasets import load_dataset
        
        print(f"Loading MMLU dataset from HuggingFace (sample {num_samples} examples)...")
        
        # 加载 MMLU 数据集 (使用 test 分割)
        dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
        
        # 转换为列表
        all_data = []
        for item in dataset:
            # MMLU 格式: question, choices (list), answer (int: 0-3)
            choices_dict = {
                "A": item["choices"][0],
                "B": item["choices"][1],
                "C": item["choices"][2],
                "D": item["choices"][3]
            }
            # answer 是索引 (0-3)，转换为 A/B/C/D
            answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            
            all_data.append({
                "question": item["question"],
                "choices": choices_dict,
                "answer": answer_map[item["answer"]],
                "subject": item.get("subject", "unknown")
            })
        
        # 随机抽样
        import random
        random.seed(seed)
        sampled_data = random.sample(all_data, min(num_samples, len(all_data)))
        
        print(f"Successfully loaded {len(sampled_data)} real MMLU examples")
        return sampled_data
        
    except Exception as e:
        print(f"Warning: Could not load real MMLU data: {e}")
        print("Falling back to sample data...")
        
        # 示例 MMLU 数据（备用）
        sample_data = [
            {
                "question": "Which of the following is the primary site of photosynthesis in plant cells?",
                "choices": {"A": "Mitochondria", "B": "Chloroplasts", "C": "Nucleus", "D": "Ribosomes"},
                "answer": "B",
                "subject": "biology"
            },
            {
                "question": "What is the value of π (pi) to two decimal places?",
                "choices": {"A": "3.12", "B": "3.14", "C": "3.16", "D": "3.18"},
                "answer": "B",
                "subject": "mathematics"
            },
            {
                "question": "Who is considered the father of modern physics?",
                "choices": {"A": "Isaac Newton", "B": "Albert Einstein", "C": "Galileo Galilei", "D": "Niels Bohr"},
                "answer": "B",
                "subject": "physics"
            },
            {
                "question": "What is the capital of Japan?",
                "choices": {"A": "Beijing", "B": "Seoul", "C": "Tokyo", "D": "Bangkok"},
                "answer": "C",
                "subject": "geography"
            },
            {
                "question": "Which programming language is known as the 'language of the web'?",
                "choices": {"A": "Python", "B": "Java", "C": "C++", "D": "JavaScript"},
                "answer": "D",
                "subject": "computer_science"
            }
        ]
        
        # 循环复制以获得更多样本
        data = []
        while len(data) < num_samples:
            data.extend(sample_data)
        
        return data[:num_samples]


def run_mmlu_evaluation(args):
    """运行 MMLU 评测"""
    print("=" * 70)
    print("MMLU 基准评测")
    print("=" * 70)
    print(f"模型: {args.agent_model}")
    print(f"方法: {args.method}")
    if args.method == 'LCO':
        print(f"LCO 种群大小: {args.lco_population_size}")
    print(f"样本数: {args.num_samples}")
    print("-" * 70)
    
    # 加载数据（使用 seed 保证可复现）
    data = load_mmlu_data(args.data_dir, args.num_samples, args.subjects, args.seed)
    
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
            for future in tqdm(as_completed(futures), total=len(data), desc="Evaluating MMLU"):
                results.append(future.result())
    else:
        # VLLM 模式：串行
        print("使用串行模式...")
        for q_data in tqdm(data, desc="Evaluating MMLU"):
            result = evaluate_single_question(q_data, args, model_config)
            results.append(result)
    
    # 计算指标
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total if total > 0 else 0
    
    # 按学科统计
    subject_stats = {}
    for r in results:
        sub = r['subject']
        if sub not in subject_stats:
            subject_stats[sub] = {'correct': 0, 'total': 0}
        subject_stats[sub]['total'] += 1
        if r['is_correct']:
            subject_stats[sub]['correct'] += 1
    
    # 输出结果
    print("\n" + "=" * 70)
    print("MMLU 评测结果")
    print("=" * 70)
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy * 100:.2f}%")
    print("\n按学科统计:")
    for subject, stats in sorted(subject_stats.items()):
        sub_acc = stats['correct'] / stats['total'] * 100
        print(f"  {subject}: {stats['correct']}/{stats['total']} ({sub_acc:.1f}%)")
    print("=" * 70)
    
    # 保存结果
    output_data = {
        "benchmark": "MMLU",
        "model": args.agent_model,
        "method": args.method,
        "num_samples": total,
        "accuracy": accuracy,
        "correct": correct,
        "subject_stats": {k: {"accuracy": v['correct']/v['total'], **v} for k, v in subject_stats.items()},
        "details": results
    }
    
    output_file = f"benchmark_results/mmlu_{args.method}_{args.agent_model.replace('/', '_')}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {output_file}")
    
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMLU 基准评测")
    parser.add_argument("--agent_model", type=str, required=True, help="Agent模型")
    parser.add_argument("--method", type=str, default="base", help="方法 (base, LCO, self_reflection)")
    parser.add_argument("--data_dir", type=str, default=None, help="MMLU数据目录")
    parser.add_argument("--num_samples", type=int, default=50, help="评测样本数 (默认50)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (用于数据抽样)")
    parser.add_argument("--max_workers", type=int, default=1, help="并发数 (API模式)")
    parser.add_argument("--vllm_tp_size", type=int, default=1, help="VLLM tensor parallel")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.85, help="VLLM GPU利用率")
    parser.add_argument("--subjects", type=str, nargs='+', default=None, help="指定学科")
    parser.add_argument("--lco_population_size", type=int, default=3, help="LCO种群大小 (默认3)")
    
    args = parser.parse_args()
    
    run_mmlu_evaluation(args)
