#!/usr/bin/env python3
"""
统一运行 MMLU 和 GSM8K 基准评测
对比不同方法 (base vs LCO) 在知识推理任务上的性能保持
"""

import os
import sys
import json
import argparse
from tabulate import tabulate


def run_single_benchmark(benchmark, method, args):
    """运行单个基准评测"""
    cmd = f"python benchmarks/{benchmark}_benchmark.py " \
          f"--agent_model {args.agent_model} " \
          f"--method {method} " \
          f"--num_samples {args.num_samples} " \
          f"--max_workers {args.max_workers}"
    
    if args.vllm_tp_size > 1:
        cmd += f" --vllm_tp_size {args.vllm_tp_size}"
    
    print(f"\n{'='*70}")
    print(f"Running {benchmark.upper()} with method={method}")
    print(f"{'='*70}")
    
    ret = os.system(cmd)
    return ret == 0


def load_benchmark_results(benchmark, method, model_name):
    """加载评测结果"""
    result_file = f"benchmark_results/{benchmark}_{method}_{model_name.replace('/', '_')}.json"
    if os.path.exists(result_file):
        with open(result_file, 'r') as f:
            return json.load(f)
    return None


def compare_methods(benchmark, model_name):
    """对比不同方法的结果"""
    methods = ['base', 'LCO', 'self_reflection', 'self_defense', 'goal_priority']
    results = {}
    
    for method in methods:
        data = load_benchmark_results(benchmark, method, model_name)
        if data:
            results[method] = data
    
    if not results:
        print(f"No results found for {benchmark}")
        return
    
    print(f"\n{'='*70}")
    print(f"{benchmark.upper()} 对比结果")
    print(f"{'='*70}")
    
    # 准备表格
    headers = ["Method", "Accuracy", "Correct/Total", "Performance Retention"]
    rows = []
    
    base_accuracy = None
    if 'base' in results:
        base_accuracy = results['base']['accuracy']
    
    for method, data in sorted(results.items()):
        accuracy = data['accuracy']
        correct = data['correct']
        total = data['num_samples']
        
        if base_accuracy and base_accuracy > 0:
            retention = (accuracy / base_accuracy) * 100
            retention_str = f"{retention:.1f}%"
        else:
            retention_str = "N/A"
        
        rows.append([
            method,
            f"{accuracy*100:.2f}%",
            f"{correct}/{total}",
            retention_str
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    # 关键结论
    if 'base' in results and 'LCO' in results:
        base_acc = results['base']['accuracy']
        lco_acc = results['LCO']['accuracy']
        retention = (lco_acc / base_acc) * 100 if base_acc > 0 else 0
        
        print(f"\n关键发现:")
        print(f"  - Base 准确率: {base_acc*100:.2f}%")
        print(f"  - LCO 准确率: {lco_acc*100:.2f}%")
        print(f"  - 性能保持: {retention:.1f}%")
        
        if retention >= 95:
            print(f"  - 结论: LCO 几乎完全保持了知识推理能力")
        elif retention >= 85:
            print(f"  - 结论: LCO 保持了大部分知识推理能力")
        else:
            print(f"  - 结论: LCO 对知识推理能力有较大影响")
    
    print(f"{'='*70}\n")


def main(args):
    """主函数"""
    print("=" * 70)
    print("基准评测套件 (MMLU & GSM8K)")
    print("=" * 70)
    print(f"模型: {args.agent_model}")
    print(f"样本数: {args.num_samples}")
    print(f"方法: {args.methods}")
    print(f"基准: {args.benchmarks}")
    print("=" * 70)
    
    benchmarks = args.benchmarks if isinstance(args.benchmarks, list) else [args.benchmarks]
    methods = args.methods if isinstance(args.methods, list) else [args.methods]
    
    # 运行所有评测
    success_count = 0
    total_count = len(benchmarks) * len(methods)
    
    for benchmark in benchmarks:
        if benchmark not in ['mmlu', 'gsm8k']:
            print(f"Unknown benchmark: {benchmark}")
            continue
        
        for method in methods:
            success = run_single_benchmark(benchmark, method, args)
            if success:
                success_count += 1
    
    print(f"\n{'='*70}")
    print(f"评测完成: {success_count}/{total_count} 成功")
    print(f"{'='*70}")
    
    # 对比结果
    if args.compare:
        for benchmark in benchmarks:
            compare_methods(benchmark, args.agent_model)
    
    # 生成对比报告
    generate_report(args.agent_model, benchmarks, methods)


def generate_report(model_name, benchmarks, methods):
    """生成综合评测报告"""
    report_file = f"benchmark_results/report_{model_name.replace('/', '_')}.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("基准评测综合报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"评测时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for benchmark in benchmarks:
            f.write(f"\n{benchmark.upper()} 结果:\n")
            f.write("-" * 70 + "\n")
            
            base_result = load_benchmark_results(benchmark, 'base', model_name)
            lco_result = load_benchmark_results(benchmark, 'LCO', model_name)
            
            if base_result and lco_result:
                base_acc = base_result['accuracy'] * 100
                lco_acc = lco_result['accuracy'] * 100
                retention = (lco_acc / base_acc) * 100 if base_acc > 0 else 0
                
                f.write(f"  Base 准确率: {base_acc:.2f}%\n")
                f.write(f"  LCO 准确率: {lco_acc:.2f}%\n")
                f.write(f"  性能保持: {retention:.1f}%\n")
                
                if retention >= 95:
                    f.write(f"  评价: ✓ LCO 几乎完全保持了{benchmark.upper()}能力\n")
                elif retention >= 85:
                    f.write(f"  评价: △ LCO 保持了大部分{benchmark.upper()}能力\n")
                else:
                    f.write(f"  评价: ✗ LCO 对{benchmark.upper()}能力有较大影响\n")
            else:
                f.write(f"  数据不完整\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"综合报告已保存: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一运行基准评测")
    parser.add_argument("--agent_model", type=str, required=True, help="Agent模型")
    parser.add_argument("--benchmarks", type=str, nargs='+', default=['mmlu', 'gsm8k'],
                       choices=['mmlu', 'gsm8k'], help="要运行的基准")
    parser.add_argument("--methods", type=str, nargs='+', default=['base', 'LCO'],
                       help="要测试的方法")
    parser.add_argument("--num_samples", type=int, default=100, help="每个基准的样本数")
    parser.add_argument("--max_workers", type=int, default=1, help="并发数")
    parser.add_argument("--vllm_tp_size", type=int, default=1, help="VLLM tensor parallel")
    parser.add_argument("--compare", action="store_true", default=True,
                       help="自动对比结果")
    
    args = parser.parse_args()
    
    main(args)
