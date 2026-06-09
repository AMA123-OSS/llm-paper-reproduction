# runs 目录说明

本目录保存 YUV 可逆对抗攻击复现实验的运行副本、输出、脚本和结果表。

## 记录规则

每次实验至少记录：

- 输入图片列表
- 生成参数：`steps`、`max_iteration`、`eps`、`alpha`、生成模型、验证模型
- 输出目录
- 攻击成功率
- 图像质量：SSIM、PSNR、L2、Linf
- 运行耗时
- 恢复验证字段：`recovery_exact`、`recovery_max_abs_error`、`recovery_error_pixels`

## 关键结果表

- `stage6_metrics_summary.csv`：阶段 4/5 单图指标统一汇总。
- `stage7_10img_per_image_metrics.csv`：阶段 7 逐图结果。
- `stage7_10img_summary.csv`：阶段 7 10 图汇总。
- `stage8_ablation_per_image_metrics.csv`：阶段 8 RGB/Y-only/Y+UV 消融逐图结果。
- `stage8_ablation_summary.csv`：阶段 8 消融汇总。
- `stage9_transferability_per_image.csv`：阶段 9 迁移性逐图结果。
- `stage9_transferability_summary.csv`：阶段 9 迁移性汇总。
- `stage10_recovery_preconditions.csv`：阶段 10 恢复前置条件检查。
- `stage10_recovery_observed_diffs.csv`：阶段 10 已保存 RAE 与原图差异。

## 关键脚本

- `evaluate_outputs.py`：阶段 6 通用指标脚本。
- `stage7_evaluate_runs.py`：阶段 7 10 图汇总脚本。
- `stage8_ablation_experiment.py`：阶段 8 消融实验脚本。
- `stage9_transferability_experiment.py`：阶段 9 迁移性实验脚本。
- `stage10_recovery_feasibility_check.py`：阶段 10 恢复可行性检查脚本。

## 注意事项

- 根目录主算法尽量保持原样；阶段实验优先在 `runs` 下新增脚本或副本。
- 当前公开代码尚不能完整验证 error-free recovery，原因是缺少恢复元数据保存和独立恢复入口。
- 当前 10 图结果不能写成论文全量实验结论。
