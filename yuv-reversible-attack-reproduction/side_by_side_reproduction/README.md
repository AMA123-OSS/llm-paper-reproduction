# 原始复现与更新优化并列展示

本目录用于在 GitHub 中把“原本复现代码/结果”和“后续更新优化代码/结果”并列放置，便于老师或评审直接对照查看。

## 目录结构

| 并列区域 | 内容 | 说明 |
|---|---|---|
| `01_original_code/` | 原始复现代码 | 包含论文复现核心代码和阶段实验脚本 |
| `02_updated_code/` | 更新优化代码 | 包含可逆恢复闭环、UV失真分析、迁移性扩展脚本 |
| `03_original_results/` | 原始复现结果 | 包含阶段5到阶段10、最终复盘、Y/UV消融、迁移性等结果CSV与记录MD |
| `04_updated_results/` | 更新优化结果 | 包含路线A/B/C更新实验CSV、sidecar恢复JSON和更新实验总结 |

## 原始复现代码

位置：

- `01_original_code/core/`
- `01_original_code/experiment_scripts/`

核心文件：

- `reversible_attack_OURS.py`：论文主流程复现代码。
- `atk.py`：YUV空间FGSM、BIM、PGD、MI-FGSM、NI-FGSM攻击实现。
- `embed_utils.py`：算术编码、PEE嵌入和原始`embed_main`。
- `EnModel.py`：InceptionV3、DenseNet161、GoogLeNet集成模型。
- `stage8_ablation_experiment.py`：Y/UV消融实验脚本。
- `stage9_transferability_experiment.py`：迁移性实验脚本。

## 更新优化代码

位置：

- `02_updated_code/`

核心文件：

- `recovery_embed.py`：补齐`code`、`dic`、`compressed_bit_length`、`T`、每轮payload长度等恢复元数据。
- `recover_from_rae.py`：根据RAE和sidecar恢复原始YUV的CLI工具。
- `route_a_recovery_selfcheck.py`：10个合成YUV样本可逆恢复闭环自检。
- `route_b_uv_ablation_selfcheck.py`：U-only、V-only、U/V分配比例、阈值策略和sidecar外置负载消融。
- `route_c_transferability_experiment.py`：10图/100图迁移性扩展实验脚本。
- `route_c_transferability_preflight.py`：环境预检和实验矩阵生成。

## 原始复现结果

位置：

- `03_original_results/`

代表性结论：

- 小规模10图复现：在低成本设置下攻击成功率达到`10/10`。
- Y/UV消融：`Y-only`在10图上保持`10/10`成功率，并取得更高图像质量；`Y+UV embed`保持攻击成功，但UV嵌入带来额外失真。
- 迁移性实验：single Inception在white-box上成功，但迁移到DenseNet161、GoogLeNet较弱；EnModel能明显提升集成成员模型上的迁移成功率。
- 可逆恢复阶段：原始流程暴露出恢复元数据未完整持久化的问题，为后续路线A改进提供依据。

## 更新优化结果

位置：

- `04_updated_results/`

代表性结论：

- 路线A：10个合成YUV样本全部满足`recovery_max_abs_error=0`、`recovery_error_pixels=0`，证明sidecar补齐元数据后可逆恢复闭环成立。
- 路线B：默认`U+V stacked`和`adaptive_T_cost`在合成样本上平均`UV PSNR=33.6774`、`UV SSIM=0.920479`；U-only/V-only因容量压力需要更高阈值，反而失真更大。
- 路线C：生成112行扩展迁移性实验矩阵，覆盖single、full EnModel、leave-one-out生成方式，以及VGG16、MobileNetV3、EfficientNet、ConvNeXt、ViT等held-out模型。

## 对照阅读建议

1. 先看`01_original_code/core/embed_utils.py`，再看`02_updated_code/recovery_embed.py`，对照理解为什么原始流程缺少恢复闭环元数据。
2. 先看`03_original_results/stage8_ablation_summary.csv`，再看`04_updated_results/route_b_uv_ablation_selfcheck.csv`，对照理解UV嵌入失真的来源。
3. 先看`03_original_results/stage9_transferability_summary.csv`，再看`04_updated_results/route_c_experiment_matrix.csv`，对照理解后续迁移性实验如何扩展到更多模型。
