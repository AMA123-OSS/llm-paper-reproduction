import json
import matplotlib.pyplot as plt
import os
import argparse
import re
import numpy as np
from scipy.stats import kendalltau, ttest_1samp, wilcoxon

def safe_kendall_tau(x, y):
    # 转成 np.array 便于处理
    x = np.array(x)
    y = np.array(y)

    # 去除 nan
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # 长度不足或无变化
    if len(y) < 2 or len(set(y)) == 1:
        return 0  # 或返回 0，根据你的定义需求
    return kendalltau(x, y)[0]  # 只返回 tau 值

def compute_kendall_tau(score_sequence):
    """
    计算一条轨迹的 Kendall's tau，表示其安全评分是否随错误轮次下降
    输入：一个长度为 n 的安全评分序列，对应错误轮次为 [0, 1, ..., n-1]
    输出：tau 值（越小表示下降趋势越强）
    """
    n = len(score_sequence)
    error_rounds = list(range(n))
    return safe_kendall_tau(error_rounds, score_sequence)


def analyze_all_trajectories(all_scores):
    """
    输入：所有轨迹的安全评分列表，每条为一个 list，例如：
        all_scores = [
            [3, 4, 1],     # 一条轨迹
            [5, 4, 2, 1],  # 第二条轨迹
            ...
        ]
    输出：
        - 每条轨迹的 tau 值
        - 显著性检验结果
    """
    taus = [compute_kendall_tau(scores) for scores in all_scores]
    taus = np.array(taus)
    # 过滤大于0的值（评估不准确的样本）
    taus = taus[taus <= 0]
    print(taus)
    # 计算趋势为负（tau < 0）的比例
    negative_trend_ratio = np.mean(taus < 0)

    # 显著性检验（判断 tau 是否整体小于 0）
    t_stat, t_pval = ttest_1samp(taus, popmean=0, alternative='less')
    try:
        wilcoxon_stat, wilcoxon_pval = wilcoxon(taus - 0, alternative='less')
    except:
        wilcoxon_stat, wilcoxon_pval = None, None  # 样本不足等

    return {
        "taus": taus,
        "negative_trend_ratio": negative_trend_ratio,
        "t_test_pval": t_pval,
        "wilcoxon_pval": wilcoxon_pval
    }


# 读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def extract_scores_from_jsonl(file_path,cnt):
    all_scores = []
    count=0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            count+=1
            task_scores = []
            try:
                segments = json.loads(line.strip())
                for segment in segments:
                    raw_score = segment.get("Overall Quantitative Score", "")
                    # 使用正则表达式提取第一个出现的数字（整数或浮点数）
                    match = re.search(r'[-+]?\d*\.\d+|\d+', str(raw_score))
                    if match:
                        score = float(match.group())
                    else:
                        score = 1.5  # 默认值
                    task_scores.append(score)
                if task_scores[0]!=0:
                    all_scores.append(task_scores)
            except Exception as e:
                print(f"解析失败：{e}")
                continue
            if count>=cnt:
                break

    return all_scores

# 提取 “Overall Quantitative Score”
def extract_scores(data):
    return [item.get("Overall Quantitative Score", None) for item in data]

# 计算平均值
def calculate_average(scores):
    valid_scores = []
    for score in scores:
        if score is not None:
            match = re.search(r'\d+', str(score))  # 提取字符串中的第一个整数
            if match:
                valid_scores.append(int(match.group()))
    return sum(valid_scores) / len(valid_scores) if valid_scores else None

# 绘制多条折线图
def plot_multiple_scores(scores_list, file_names):
    plt.figure(figsize=(10, 6))
    for scores, file_name in zip(scores_list, file_names):
        plt.plot(scores, marker="o", label=file_name)
    plt.title("Overall Quantitative Score Over Time")
    plt.xlabel("Index")
    plt.ylabel("Overall Quantitative Score")
    plt.legend()
    plt.grid(True)
    plt.show()

# 主函数
def main():
    parser = argparse.ArgumentParser(description="计算 JSON 文件中的 Overall Quantitative Score 均值")
    parser.add_argument("files", nargs="+", help="JSON 文件路径列表")
    args = parser.parse_args()

    for file_path in args.files:
        if os.path.exists(file_path):  # 检查文件是否存在
            scores = extract_scores_from_jsonl(file_path,cnt=30)
            print(scores)
        else:
            print(f"文件 {file_path} 不存在，跳过。")

    result = analyze_all_trajectories(scores)
    print("平均 Kendall Tau 值：", np.mean(result["taus"]))
    print("下降趋势比例：", result["negative_trend_ratio"])
    print("轨迹的 tau 平均值是否显著小于 0?")
    print("t检验 p 值：", result["t_test_pval"])
    print("Wilcoxon 检验 p 值：", result["wilcoxon_pval"])


if __name__ == "__main__":
    main()
