#!/bin/bash
# 快速运行 MMLU 和 GSM8K 评测（使用真实数据，各50条）
# 使用示例: ./run_mmlu_gsm8k.sh qwen2.5-7b-instruct

MODEL=${1:-"gpt-3.5-turbo"}
NUM_SAMPLES=${2:-50}  # 默认50条

echo "=================================="
echo "MMLU & GSM8K 基准评测 (真实数据)"
echo "模型: $MODEL"
echo "样本数: $NUM_SAMPLES"
echo "=================================="

# 运行 MMLU
echo ""
echo "[1/2] 运行 MMLU 评测 (真实数据)..."
python benchmarks/mmlu_benchmark.py \
    --agent_model "$MODEL" \
    --method base \
    --num_samples $NUM_SAMPLES \
    --seed 42 \
    --max_workers 1

python benchmarks/mmlu_benchmark.py \
    --agent_model "$MODEL" \
    --method LCO \
    --num_samples $NUM_SAMPLES \
    --seed 42 \
    --max_workers 1

# 运行 GSM8K
echo ""
echo "[2/2] 运行 GSM8K 评测 (真实数据)..."
python benchmarks/gsm8k_benchmark.py \
    --agent_model "$MODEL" \
    --method base \
    --num_samples $NUM_SAMPLES \
    --seed 42 \
    --max_workers 1

python benchmarks/gsm8k_benchmark.py \
    --agent_model "$MODEL" \
    --method LCO \
    --num_samples $NUM_SAMPLES \
    --seed 42 \
    --max_workers 1

# 生成对比报告
echo ""
echo "=================================="
echo "生成对比报告"
echo "=================================="
python benchmarks/run_benchmarks.py \
    --agent_model "$MODEL" \
    --benchmarks mmlu gsm8k \
    --methods base LCO \
    --num_samples $NUM_SAMPLES

echo ""
echo "评测完成！"
echo "结果保存在 benchmark_results/ 目录"
