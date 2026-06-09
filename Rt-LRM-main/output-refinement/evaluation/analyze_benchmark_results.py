#!/usr/bin/env python3
"""
Benchmark 结果统计展示脚本
以表格形式展示 GSM8K 和 MMLU 的评测结果，验证 LCO 不会显著影响模型性能
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Any

# 尝试导入 tabulate，如果没有则使用简单表格
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    print("提示: 安装 tabulate 可获得更好看的表格 (pip install tabulate)")


def load_benchmark_results(result_dir: str) -> Dict[str, List[Dict]]:
    """加载所有 benchmark 结果文件"""
    results = {
        'GSM8K': [],
        'MMLU': []
    }
    
    if not os.path.exists(result_dir):
        print(f"错误: 目录不存在 {result_dir}")
        return results
    
    for filename in os.listdir(result_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(result_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            benchmark = data.get('benchmark', '').upper()
            if benchmark in results:
                results[benchmark].append(data)
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")
    
    return results


def format_accuracy(acc: float) -> str:
    """格式化准确率为百分比"""
    return f"{acc * 100:.2f}%"


def show_gsm8k_summary(results: List[Dict], show_details: bool = False):
    """展示 GSM8K 结果汇总"""
    if not results:
        print("\n暂无 GSM8K 结果数据")
        return
    
    print("\n" + "=" * 80)
    print("GSM8K (数学推理) 评测结果")
    print("=" * 80)
    
    # 按模型和方法分组
    model_methods = defaultdict(dict)
    for data in results:
        model = data.get('model', 'unknown')
        method = data.get('method', 'unknown')
        model_methods[model][method] = data
    
    # 准备表格数据
    headers = ["模型", "方法", "样本数", "正确数", "准确率"]
    rows = []
    
    for model in sorted(model_methods.keys()):
        methods = model_methods[model]
        
        for method in sorted(methods.keys()):
            data = methods[method]
            acc = data.get('accuracy', 0)
            correct = data.get('correct', 0)
            total = data.get('num_samples', 0)
            
            rows.append([
                model,
                method,
                total,
                correct,
                format_accuracy(acc)
            ])
    
    print_table(rows, headers)
    
    # 详细错误分析
    if show_details:
        print("\n" + "-" * 80)
        print("详细错误分析 (按模型-方法)")
        print("-" * 80)
        
        for model in sorted(model_methods.keys()):
            methods = model_methods[model]
            for method in sorted(methods.keys()):
                data = methods[method]
                details = data.get('details', [])
                
                incorrect = [d for d in details if not d.get('is_correct', True)]
                if incorrect:
                    print(f"\n【{model} / {method}】错误 {len(incorrect)}/{len(details)} 题:")
                    for i, item in enumerate(incorrect[:3], 1):  # 只显示前3个错误
                        question = item.get('question', '')[:60]
                        print(f"  {i}. {question}...")
                        print(f"     正确答案: {item.get('correct_answer', 'N/A')}")
                        print(f"     预测答案: {item.get('predicted_answer', 'N/A')}")


def show_mmlu_summary(results: List[Dict], show_details: bool = False):
    """展示 MMLU 结果汇总"""
    if not results:
        print("\n暂无 MMLU 结果数据")
        return
    
    print("\n" + "=" * 80)
    print("MMLU (知识理解) 评测结果")
    print("=" * 80)
    
    # 按模型和方法分组
    model_methods = defaultdict(dict)
    for data in results:
        model = data.get('model', 'unknown')
        method = data.get('method', 'unknown')
        model_methods[model][method] = data
    
    # 准备表格数据
    headers = ["模型", "方法", "样本数", "正确数", "准确率"]
    rows = []
    
    for model in sorted(model_methods.keys()):
        methods = model_methods[model]
        
        for method in sorted(methods.keys()):
            data = methods[method]
            acc = data.get('accuracy', 0)
            correct = data.get('correct', 0)
            total = data.get('num_samples', 0)
            
            rows.append([
                model,
                method,
                total,
                correct,
                format_accuracy(acc)
            ])
    
    print_table(rows, headers)
    
    # 按学科详细分析
    if show_details:
        print("\n" + "-" * 80)
        print("学科详细分析 (准确率 > 60% 高亮)")
        print("-" * 80)
        
        for model in sorted(model_methods.keys()):
            methods = model_methods[model]
            for method in sorted(methods.keys()):
                data = methods[method]
                subject_stats = data.get('subject_stats', {})
                
                if subject_stats:
                    print(f"\n【{model} / {method}】各学科表现:")
                    
                    # 按准确率排序
                    sorted_subjects = sorted(
                        subject_stats.items(),
                        key=lambda x: x[1].get('accuracy', 0),
                        reverse=True
                    )
                    
                    sub_headers = ["学科", "正确/总数", "准确率"]
                    sub_rows = []
                    
                    for subject, stats in sorted_subjects:
                        acc = stats.get('accuracy', 0)
                        correct = stats.get('correct', 0)
                        total = stats.get('total', 0)
                        
                        # 高亮高准确率学科
                        acc_str = format_accuracy(acc)
                        if acc >= 0.8:
                            acc_str = f"🟢 {acc_str}"
                        elif acc >= 0.6:
                            acc_str = f"🟡 {acc_str}"
                        else:
                            acc_str = f"🔴 {acc_str}"
                        
                        sub_rows.append([subject, f"{correct}/{total}", acc_str])
                    
                    print_table(sub_rows, sub_headers)


def show_comparison(results: Dict[str, List[Dict]]):
    """展示 LCO 与 Base 性能对比（强调接近程度）"""
    print("\n" + "=" * 80)
    print("LCO 性能保持分析（差异越小越好）")
    print("=" * 80)
    
    headers = ["Benchmark", "模型", "Base 准确率", "LCO 准确率", "差异"]
    rows = []
    
    for benchmark in ['GSM8K', 'MMLU']:
        model_methods = defaultdict(dict)
        for data in results.get(benchmark, []):
            model = data.get('model', 'unknown')
            method = data.get('method', 'unknown')
            model_methods[model][method] = data
        
        for model in sorted(model_methods.keys()):
            methods = model_methods[model]
            if 'base' in methods and 'LCO' in methods:
                base_acc = methods['base'].get('accuracy', 0)
                lco_acc = methods['LCO'].get('accuracy', 0)
                
                diff = abs(lco_acc - base_acc)
                
                # 根据差异大小区分显示
                diff_str = f"{diff*100:.2f}%"
                if diff <= 0.02:  # 2%以内
                    diff_str = f"🟢 {diff_str} (优秀)"
                elif diff <= 0.05:  # 5%以内
                    diff_str = f"🟡 {diff_str} (良好)"
                else:
                    diff_str = f"🔴 {diff_str} (较大)"
                
                rows.append([
                    benchmark,
                    model,
                    format_accuracy(base_acc),
                    format_accuracy(lco_acc),
                    diff_str
                ])
    
    if rows:
        print_table(rows, headers)
        
        # 统计结论
        print("\n" + "-" * 80)
        print("性能保持分析:")
        
        diffs = []
        excellent = 0
        good = 0
        for row in rows:
            # 提取差异值
            diff_str = row[4].replace('%', '').split()[1]
            try:
                diff = float(diff_str) / 100
                diffs.append(diff)
                if diff <= 0.02:
                    excellent += 1
                elif diff <= 0.05:
                    good += 1
            except:
                pass
        
        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            max_diff = max(diffs)
            
            print(f"  - 配置数量: {len(diffs)}")
            print(f"  - 平均差异: {avg_diff*100:.2f}%")
            print(f"  - 最大差异: {max_diff*100:.2f}%")
            print(f"  - 优秀保持 (≤2%): {excellent}/{len(diffs)}")
            print(f"  - 良好保持 (≤5%): {good}/{len(diffs)}")
            
            if avg_diff <= 0.03:
                print(f"  - ✅ 结论: LCO 方法很好地保持了模型性能 (平均差异 {avg_diff*100:.2f}%)")
            elif avg_diff <= 0.05:
                print(f"  - ⚠️ 结论: LCO 方法基本保持模型性能 (平均差异 {avg_diff*100:.2f}%)")
            else:
                print(f"  - ❌ 结论: LCO 方法对模型性能有一定影响 (平均差异 {avg_diff*100:.2f}%)")
    else:
        print("暂无 LCO 与 Base 的对比数据")


def print_table(rows: List[List], headers: List[str]):
    """打印表格（支持 tabulate 和简单格式）"""
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="grid", stralign="left"))
    else:
        # 简单表格格式
        # 计算每列最大宽度
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # 打印分隔线
        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        
        # 打印表头
        print(sep)
        header_row = "| " + " | ".join(
            str(h).ljust(w) for h, w in zip(headers, col_widths)
        ) + " |"
        print(header_row)
        print(sep)
        
        # 打印数据行
        for row in rows:
            data_row = "| " + " | ".join(
                str(cell).ljust(w) for cell, w in zip(row, col_widths)
            ) + " |"
            print(data_row)
        
        print(sep)


def export_to_csv(results: Dict[str, List[Dict]], output_file: str):
    """导出结果到 CSV 文件"""
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow([
            'benchmark', 'model', 'method', 'num_samples', 
            'correct', 'accuracy'
        ])
        
        for benchmark in ['GSM8K', 'MMLU']:
            model_methods = defaultdict(dict)
            for data in results.get(benchmark, []):
                model = data.get('model', 'unknown')
                method = data.get('method', 'unknown')
                model_methods[model][method] = data
            
            for model in model_methods:
                methods = model_methods[model]
                
                for method in methods:
                    data = methods[method]
                    
                    writer.writerow([
                        benchmark,
                        model,
                        method,
                        data.get('num_samples', 0),
                        data.get('correct', 0),
                        data.get('accuracy', 0)
                    ])
    
    print(f"\n结果已导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="统计展示 Benchmark 结果 - 验证 LCO 性能保持",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 展示所有结果
  python analyze_benchmark_results.py
  
  # 展示详细分析
  python analyze_benchmark_results.py --details
  
  # 导出到 CSV
  python analyze_benchmark_results.py --export results.csv
  
  # 指定结果目录
  python analyze_benchmark_results.py --dir ./my_results
        """
    )
    parser.add_argument(
        "--dir", 
        type=str, 
        default="benchmark_results",
        help="结果文件目录 (默认: benchmark_results)"
    )
    parser.add_argument(
        "--details", 
        action="store_true",
        help="显示详细分析（错误示例、学科分析等）"
    )
    parser.add_argument(
        "--export", 
        type=str,
        help="导出结果到 CSV 文件"
    )
    
    args = parser.parse_args()
    
    # 确保在 output-refinement 目录下运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(script_dir, args.dir)
    
    # 加载数据
    results = load_benchmark_results(result_dir)
    
    # 展示结果
    show_gsm8k_summary(results['GSM8K'], args.details)
    show_mmlu_summary(results['MMLU'], args.details)
    show_comparison(results)
    
    # 导出
    if args.export:
        export_to_csv(results, args.export)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
