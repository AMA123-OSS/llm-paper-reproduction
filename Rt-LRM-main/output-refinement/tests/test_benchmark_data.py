#!/usr/bin/env python3
"""
测试 MMLU 和 GSM8K 数据加载
"""

import sys
sys.path.insert(0, '.')

print("=" * 70)
print("测试真实数据加载")
print("=" * 70)

# 测试 MMLU
print("\n[1/2] 测试 MMLU 数据加载...")
try:
    from mmlu_benchmark import load_mmlu_data
    mmlu_data = load_mmlu_data(num_samples=50, seed=42)
    print(f"✅ 成功加载 {len(mmlu_data)} 条 MMLU 数据")
    print(f"示例数据:")
    print(f"  问题: {mmlu_data[0]['question'][:80]}...")
    print(f"  选项: {mmlu_data[0]['choices']}")
    print(f"  答案: {mmlu_data[0]['answer']}")
    print(f"  学科: {mmlu_data[0]['subject']}")
except Exception as e:
    print(f"❌ MMLU 加载失败: {e}")

# 测试 GSM8K
print("\n[2/2] 测试 GSM8K 数据加载...")
try:
    from gsm8k_benchmark import load_gsm8k_data
    gsm8k_data = load_gsm8k_data(num_samples=50, seed=42)
    print(f"✅ 成功加载 {len(gsm8k_data)} 条 GSM8K 数据")
    print(f"示例数据:")
    print(f"  问题: {gsm8k_data[0]['question'][:80]}...")
    print(f"  答案: {gsm8k_data[0]['answer']}")
except Exception as e:
    print(f"❌ GSM8K 加载失败: {e}")

print("\n" + "=" * 70)
print("数据加载测试完成")
print("=" * 70)
