# YUV 可逆对抗攻击三路线更新实验总结

更新时间：2026-06-10

## 当前环境边界

本次更新在 `D:\大模型论文复现\yuv-reversible-attack-reproduction\update_yuv_recovery_optimization` 内完成，未修改原始论文复现主流程。当前仓库缺少 `ORI_IMG`，且本次使用的 bundled Python 环境未安装 `torch/torchvision`，因此真实 10 图/100 图 ImageNet 迁移性重跑暂时无法完成。本次已完成的是：

- 路线 A 的可逆恢复闭环代码和 10 个合成 YUV 样本无损恢复验证。
- 路线 B 的 UV 嵌入失真合成消融测试。
- 路线 C 的真实实验矩阵、环境预检和可执行脚本。

## 路线 A：补齐可逆恢复闭环

### 实现内容

- 新增 `compressfun_with_meta(signal)`，返回 `code`、`dic`、`compressed_bit_length`。
- 新增 `embed_main_with_meta(ori_yuv, advy_yuv)`，返回可逆嵌入后的 UV carrier 和恢复所需元数据。
- 每张 RAE 可保存一个 `*.recovery.json` sidecar，字段包括图像尺寸、UV 堆叠尺寸、算术编码字典、压缩比特长度、每轮阈值 `T` 和每轮 payload 长度。
- 新增 `recover_original_yuv_from_rae_yuv(rae_yuv, metadata)` 和 CLI `recover_from_rae.py`。

### 测试结果

结果文件：`results/route_a_recovery_selfcheck.csv`

| 指标 | 结果 |
|---|---:|
| 合成样本数 | 10 |
| `recovery_max_abs_error=0` | 10/10 |
| `recovery_error_pixels=0` | 10/10 |
| `recovery_exact=1` | 10/10 |

结论：原始代码中的 PEE 编码/解码原语本身具备可逆能力，真正缺口是运行时未持久化 `T_flag`、算术编码 `dic`、压缩比特长度和每轮 payload 长度。sidecar 补齐后，可逆恢复闭环成立。

## 路线 B：降低 UV 嵌入失真

### 已测策略

结果文件：`results/route_b_uv_ablation_selfcheck.csv`

| 策略 | 状态 | 平均 UV PSNR | 平均 UV SSIM | 平均 L_inf | 可逆 carrier |
|---|---|---:|---:|---:|---:|
| `uv_default` | 3/3 ok | 33.6774 | 0.920479 | 6 | 1 |
| `adaptive_T_cost` | 3/3 ok | 33.6774 | 0.920479 | 6 | 1 |
| `u_only` | 3/3 ok | 32.4706 | 0.892939 | 11 | 1 |
| `v_only` | 3/3 ok | 32.4838 | 0.893196 | 11 | 1 |
| `uv_25_75` | 3/3 ok | 31.0830 | 0.872954 | 11 | 1 |
| `uv_75_25` | 3/3 ok | 31.0764 | 0.872806 | 11 | 1 |
| `fixed_T1` | 0/3 ok | capacity failed | capacity failed | 0 | 0 |
| `sidecar_external_payload` | 3/3 ok | inf | 1.000000 | 0 | 1 |

### 分析

- U-only/V-only 不是天然更优。单通道容量更小，为装下相同压缩 payload，需要把 PEE 阈值从 6 提高到 11，导致最大扰动和整体失真上升。
- 简单 U/V 分配比例会产生两轮单通道嵌入，反而把平均 PSNR 拉低到约 31.08 dB。
- `adaptive_T_cost` 在当前合成样本上与原始 adaptive 阈值选择一致，说明原始阈值已经接近满足容量的低失真选择。
- `sidecar_external_payload` 是质量上界：UV 不承载恢复 payload 时 PSNR 为 inf、SSIM 为 1，但这会削弱论文中“RAE 自身携带恢复信息”的设定。可作为工程折中方案，而不是严格论文复现方案。

### 建议改进路线

- 优先保留 U+V 堆叠嵌入，不建议简单改成 U-only/V-only。
- 若追求质量提升，可做“混合 sidecar”：保留必要校验和少量索引在 UV，将大 payload 外置，明确区分强可逆自包含版本和工程高保真版本。
- 后续真实图像上应把 `PSNR/SSIM` 与攻击成功率一起报告，避免只优化视觉质量而破坏攻击效果。

## 路线 C：扩大迁移性实验

### 已生成内容

结果文件：

- `results/route_c_preflight.csv`
- `results/route_c_experiment_matrix.csv`

实验矩阵共 112 行，覆盖：

- 生成器：single InceptionV3、single DenseNet161、single GoogLeNet、full EnModel、去掉 InceptionV3/DenseNet161/GoogLeNet 的 leave-one-out ensemble。
- 测试模型：InceptionV3、DenseNet161、GoogLeNet、VGG16、MobileNetV3-Large、EfficientNet-B0、ConvNeXt-Tiny、ViT-B/16。
- 关系标签：`white_box`、`ensemble_member`、`held_out_black_box`。
- 图像规模：10 图和 100 图。
- 指标：攻击成功率、原始分类正确条件下成功率、PSNR、SSIM、L2、L_inf、95% CI。

### 当前预检状态

| 项目 | 状态 | 说明 |
|---|---|---|
| `ORI_IMG` | blocked | 当前 0 张 PNG，需要恢复 10/100 张真实输入 |
| Python | ok | bundled Python 3.12.13 |
| torch | blocked | 当前环境缺失 |
| torchvision | blocked | 当前环境缺失 |
| cuda | warning | 未能在缺 torch 环境下确认 |

### 后续执行命令

```powershell
python route_c_transferability_experiment.py --image-count 10
python route_c_transferability_experiment.py --image-count 100
```

该脚本会生成：

- `results/route_c_10img_generation.csv`
- `results/route_c_10img_per_image.csv`
- `results/route_c_10img_summary.csv`
- `results/route_c_100img_generation.csv`
- `results/route_c_100img_per_image.csv`
- `results/route_c_100img_summary.csv`

## 总结判断

最优先的路线 A 已经证明：通过 sidecar 保存必要恢复元数据，可以把论文最核心的“可逆恢复”闭环补齐。路线 B 的初步结果显示，降低 UV 失真的关键不在简单 U/V 分配，而在降低需要嵌入 UV 的 payload 或设计更细的 carrier 选择。路线 C 已经准备好真实迁移性实验脚本和矩阵，但需要先恢复数据集与 PyTorch 环境。
