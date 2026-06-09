# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# def plot_tau_distributions(jsonl_path):
#     methods = []
#     taus = []
#
#     # 读取 jsonl 文件
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 entry = json.loads(line.strip())
#                 method_name = entry.get("method", "unknown")
#                 method_taus = entry.get("taus", [])
#                 methods.extend([method_name] * len(method_taus))
#                 taus.extend(method_taus)
#             except json.JSONDecodeError:
#                 print("跳过一行格式错误的数据")
#
#     # 转为 DataFrame
#     df = pd.DataFrame({"method": methods, "tau": taus})
#
#     # KDE 曲线图
#     plt.figure(figsize=(8, 5))
#     sns.kdeplot(data=df, x="tau", hue="method", fill=True, common_norm=False, alpha=0.5)
#     plt.axvline(0, color='black', linestyle='--')
#     plt.title("Distribution of Kendall Tau Values")
#     plt.xlabel("Kendall Tau")
#     plt.ylabel("Density")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#     # 箱线图（可选）
#     plt.figure(figsize=(6, 5))
#     sns.boxplot(data=df, x="method", y="tau")
#     plt.axhline(0, color='black', linestyle='--')
#     plt.title("Boxplot of Kendall Tau Values by Method")
#     plt.ylabel("Kendall Tau")
#     plt.grid(True, axis='y')
#     plt.tight_layout()
#     plt.show()
#
# # 示例调用
# plot_tau_distributions("toxicity_trend_analysis_result.jsonl")
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

# 读取 JSONL 文件
def load_results_from_jsonl(path):
    methods = []
    taus = []
    for line in open(path, "r", encoding="utf-8"):
        data = json.loads(line)
        method = data.get("method", "Unknown")
        method_taus = data.get("taus", [])
        methods.extend([method] * len(method_taus))
        taus.extend(method_taus)
    return pd.DataFrame({"method": methods, "tau": taus})

# 显著性检验 + 标注星号
def annotate_significance(ax, df):
    grouped = df.groupby("method")
    for i, (name, group) in enumerate(grouped):
        taus = group["tau"].dropna()
        if len(taus) < 2:
            continue
        t_stat, p_val = ttest_1samp(taus, popmean=0, alternative="greater")
        print(p_val)
        # 显著性星号规则
        if p_val < 0.001:
            star = "***"
        elif p_val < 0.01:
            star = "**"
        elif p_val < 0.05:
            star = "*"
        else:
            star = "ns"  # not significant

        # 获取箱体上边界的位置，用于放置星号
        y = taus.max() + 0.02
        ax.text(i, y, star, ha="center", va="bottom", fontsize=14, color="red")

# 主绘图函数
def plot_box_with_significance(jsonl_path):
    df = load_results_from_jsonl(jsonl_path)

    plt.figure(figsize=(6, 5))
    ax = sns.boxplot(data=df, x="method", y="tau", showfliers=False)

    # 画平均值点
    sns.pointplot(data=df, x="method", y="tau", estimator="mean", color="black", markers="D", linestyles="")

    # 零线
    plt.axhline(0, color='black', linestyle='--')

    # 标注显著性
    annotate_significance(ax, df)

    # 美化图像
    plt.title("Boxplot of Kendall Tau Values by Method")
    plt.ylabel("Kendall Tau")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
plot_box_with_significance("toxicity_trend_analysis_result.jsonl")
