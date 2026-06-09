#!/usr/bin/env python3
"""
对比不同方法在良性基准上的表现
"""

import json
import os
import argparse
from tabulate import tabulate


def load_results(result_dir):
    """加载所有结果文件"""
    results = {}
    if not os.path.exists(result_dir):
        return results
    
    for filename in os.listdir(result_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(result_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                method = data.get('method', 'unknown')
                results[method] = data
    return results


def compare_methods(results):
    """对比不同方法"""
    if not results:
        print("没有找到结果文件！")
        return
    
    # 准备表格数据
    headers = ["Method", "Valid Rate (%)", "Avg Toxicity", "Avg Length", "Diversity"]
    rows = []
    
    for method, data in sorted(results.items()):
        metrics = data.get('metrics', {})
        row = [
            method,
            f"{metrics.get('valid_rate', 0) * 100:.1f}",
            f"{metrics.get('avg_toxicity', 0):.4f}",
            f"{metrics.get('avg_length', 0):.1f}",
            f"{metrics.get('avg_diversity', 0):.4f}"
        ]
        rows.append(row)
    
    print("\n" + "=" * 80)
    print("良性基准对比结果")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("\n指标说明:")
    print("- Valid Rate: 有效推文比例 (越高越好)")
    print("- Avg Toxicity: 平均毒性分数 (越低越好)")
    print("- Avg Length: 平均推文长度")
    print("- Diversity: 同话题推文多样性 (越高越好)")
    print("=" * 80)
    
    # 检查 LCO 是否保持性能
    if 'LCO' in results and 'base' in results:
        lco = results['LCO']['metrics']
        base = results['base']['metrics']
        
        print("\nLCO vs Base 对比:")
        print(f"  有效性保持: {lco['valid_rate']/base['valid_rate']*100:.1f}%")
        print(f"  毒性降低: {(base['avg_toxicity'] - lco['avg_toxicity'])/base['avg_toxicity']*100:.1f}%")
        print(f"  多样性保持: {lco['avg_diversity']/base['avg_diversity']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="benign_eval_results", 
                       help="结果文件目录")
    args = parser.parse_args()
    
    results = load_results(args.result_dir)
    compare_methods(results)
