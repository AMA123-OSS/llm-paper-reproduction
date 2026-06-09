#!/bin/bash
# 快速运行良性基准评测
# 使用示例: ./run_benign_eval.sh qwen2.5-7b-instruct

MODEL=${1:-"gpt-3.5-turbo"}
NUM_TOPICS=${2:-50}

echo "=================================="
echo "良性基准评测"
echo "模型: $MODEL"
echo "话题数: $NUM_TOPICS"
echo "=================================="

# 运行 base 方法
echo ""
echo "[1/2] 评测 Base 方法..."
python benchmarks/benign_evaluation.py \
    --agent_model "$MODEL" \
    --method base \
    --num_topics $NUM_TOPICS

# 运行 LCO 方法
echo ""
echo "[2/2] 评测 LCO 方法..."
python benchmarks/benign_evaluation.py \
    --agent_model "$MODEL" \
    --method LCO \
    --num_topics $NUM_TOPICS

# 对比结果
echo ""
echo "=================================="
echo "对比结果"
echo "=================================="
python evaluation/compare_benign_results.py

echo ""
echo "评测完成！"
