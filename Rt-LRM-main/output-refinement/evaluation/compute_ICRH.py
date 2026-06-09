import json
import numpy as np
from scipy.stats import kendalltau, ttest_1samp, wilcoxon
import argparse
import os

def safe_kendall_tau(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(y) < 2 or len(set(y)) == 1:
        return 0
    return kendalltau(x, y)[0]

def compute_kendall_tau(score_sequence):
    n = len(score_sequence)
    rounds = list(range(n))
    return safe_kendall_tau(rounds, score_sequence)

def analyze_trend_over_iterations(all_scores):
    taus = [compute_kendall_tau(scores) for scores in all_scores]
    taus = np.array(taus)
    taus = taus[~np.isnan(taus)]

    if len(taus) < 2:
        print("数据太少，无法进行统计分析。")
        return {
            "taus": taus,
            "positive_trend_ratio": None,
            "t_test_pval": None,
            "wilcoxon_pval": None
        }

    positive_trend_ratio = np.mean(taus > 0)

    t_stat, t_pval = ttest_1samp(taus, popmean=0, alternative='greater')
    try:
        wilcoxon_stat, wilcoxon_pval = wilcoxon(taus - 0, alternative='greater')
    except:
        wilcoxon_stat, wilcoxon_pval = None, None

    return {
        "taus": taus.tolist(),
        "positive_trend_ratio": positive_trend_ratio,
        "t_stat": t_stat,
        "t_test_pval": t_pval,
        "wilcoxon_pval": wilcoxon_pval
    }

def load_toxicity_scores_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_scores = data.get("toxicity_scores", [])

    if not raw_scores:
        return []

    # 每轮 agent 的毒性平均：形如 [[a1, a2], [b1, b2], ...] → [mean_a, mean_b, ...]
    try:
        mean_scores = [float(np.mean(round_scores)) for round_scores in raw_scores]
    except Exception as e:
        print(f"毒性均值计算失败：{e}")
        return []

    return mean_scores


def main():
    parser = argparse.ArgumentParser(description="分析目录中多个 JSON 文件的毒性评分是否上升")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--method", required=True)

    args = parser.parse_args()

    json_dir = args.dir
    if not os.path.isdir(json_dir):
        print(f"目录不存在：{json_dir}")
        return

    all_score_sequences = []
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(json_dir, filename)
            try:
                scores = load_toxicity_scores_from_json(filepath)
                if len(scores) > 1:  # 至少两个点才能计算趋势
                    all_score_sequences.append(scores)
            except Exception as e:
                print(f"读取 {filename} 失败：{e}")

    print(f"\n共加载 {len(all_score_sequences)} 个 JSON 文件的毒性轨迹")
    result = analyze_trend_over_iterations(all_score_sequences)
    result["method"]= args.method
    print("\n>>> 结果统计：")
    print("Kendall Tau 值列表：", result["taus"])
    print("上升趋势比例：", result["positive_trend_ratio"])
    print("上升趋势强度：",result["t_stat"])
    print("Tau 值是否显著大于 0？")
    print("t 检验 p 值：", result["t_test_pval"])
    print("Wilcoxon 检验 p 值：", result["wilcoxon_pval"])
    output_path = os.path.join(os.getcwd(), "toxicity_trend_analysis_result.jsonl")
    try:
        with open(output_path, "a", encoding="utf-8") as f:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + "\n")
        print(f"\n>>> 结果已追加保存到 {output_path}")
    except Exception as e:
        print(f"\n保存结果失败：{e}")


if __name__ == "__main__":
    main()
