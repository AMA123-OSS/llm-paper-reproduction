# 阶段 8：Y 通道与 UV 嵌入消融记录

日期：2026-06-09

## 阶段目标

使用同一批 10 张图片、同一组模型和同一组低成本参数，对比三种攻击/嵌入设置：

- A：普通 RGB MI-FGSM，对 RGB 三通道直接生成扰动。
- B：只使用 Y 通道扰动，不做 UV 可逆嵌入。
- C：Y 通道扰动 + UV 通道嵌入，即当前代码中的论文主方法近似复现。

本阶段目标不是证明论文全量结论，而是验证 Y 通道攻击和 UV 嵌入在本机小样本上的可观察影响。

## 固定设置

| 项 | 设置 |
| --- | --- |
| 输入图片 | 阶段 7 同一批 10 张，`0001_321.png` 到 `0010_111.png` |
| 图片格式 | RGB，299x299 |
| 生成模型 | `EnModel`，包含 DenseNet161、Inception v3、GoogLeNet |
| 验证模型 | Inception v3 |
| CAM 模型 | ResNet50 + Grad-CAM++，用于 Y-only 与 Y+UV 组 |
| seed | 2022 |
| eps | 2/255 |
| alpha | 1/255 |
| steps | 4 |
| max_iteration | 5 |

新增脚本为 `runs\stage8_ablation_experiment.py`。该脚本只放在 `runs` 下，不改动根目录主算法。

## 三组实验定义

| method | 输出目录 | 说明 |
| --- | --- | --- |
| `rgb_mi_fgsm` | `runs\stage8_ablation\rgb_mi_fgsm\output` | RGB 普通 MI-FGSM，直接扰动 RGB 图像 |
| `y_only` | `runs\stage8_ablation\y_only\output` | 生成 Y 通道扰动并替换 Y，不执行 UV 嵌入 |
| `y_uv_embed` | `runs\stage8_ablation\y_uv_embed\output` | Y 通道扰动后调用 `embed_main` 将恢复信息嵌入 U/V |

## 执行命令

```powershell
D:\大模型论文复现\yuv-reversible-attack-reproduction\.venv\Scripts\python.exe -u D:\大模型论文复现\yuv-reversible-attack-reproduction\runs\stage8_ablation_experiment.py
```

## 汇总结果

| method | 输出数 | 成功数 | 成功率 | 平均 SSIM | 平均 PSNR | 平均 L2 | 平均 Linf | 平均单图耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rgb_mi_fgsm` | 10 | 10 | 1.000000 | 0.975990 | 42.4469 | 4.0510 | 0.009412 | 0.60s |
| `y_only` | 10 | 10 | 1.000000 | 0.991483 | 46.5931 | 2.5893 | 0.016471 | 0.80s |
| `y_uv_embed` | 10 | 10 | 1.000000 | 0.977176 | 41.6994 | 4.7156 | 0.038432 | 2.07s |

完整逐图表见 `runs\stage8_ablation_per_image_metrics.csv`，包含每张图的预测类别、迭代轮次、耗时、SSIM、PSNR、L2、Linf 和输出文件大小。

## 结果分析

### 攻击成功率

三组均达到 `10/10` 成功率。因此在当前 10 张图和低成本参数下，攻击成功率不能区分三种方法的优劣，需要看质量指标和运行代价。

### Y 通道攻击效果

`y_only` 组在成功率相同的情况下取得最高平均质量：

- 平均 `SSIM=0.991483`，高于 RGB 组的 `0.975990`。
- 平均 `PSNR=46.5931`，高于 RGB 组的 `42.4469`。
- 平均 `L2=2.5893`，低于 RGB 组的 `4.0510`。

这说明在当前小样本上，把扰动集中到 Y 通道能用更小的总体像素差异达到同样误分类效果，和论文中减少 GRP（Generation-Redundant Perturbations）的动机一致。不过这只是 10 图小规模现象，还需要阶段 9 的迁移性实验进一步验证。

### UV 嵌入影响

`y_uv_embed` 与 `y_only` 的成功率相同，均为 `10/10`，说明当前样本中 UV 嵌入没有造成可观察的攻击成功率下降。也就是说，从成功率角度看，本阶段没有观察到明显 EOP（Embedding-Overwritten Perturbations）问题。

但 `y_uv_embed` 的质量指标明显低于 `y_only`：

- `SSIM` 从 `0.991483` 降到 `0.977176`。
- `PSNR` 从 `46.5931` 降到 `41.6994`。
- `L2` 从 `2.5893` 升到 `4.7156`。
- `Linf` 从 `0.016471` 升到 `0.038432`。

这说明 UV 嵌入在当前实现中带来了额外图像失真。该成本是换取后续可逆恢复能力的代价，但阶段 8 尚未验证真实恢复是否 error-free；恢复验证仍放在阶段 10。

### 运行耗时

RGB 组最快，因为不需要 CAM 和 UV 嵌入。`y_uv_embed` 最慢，主要额外成本来自 CAM、YUV 转换和 `embed_main` 可逆嵌入。耗时只统计模型加载后的逐图处理时间，不包含模型初始化。

## 输出完整性检查

- `rgb_mi_fgsm` 输出 10 张 PNG。
- `y_only` 输出 10 张 PNG。
- `y_uv_embed` 输出 10 张 PNG。
- `stage8_ablation_per_image_metrics.csv` 共 30 行，三组字段一致。
- `stage8_ablation_summary.csv` 共 3 行，参数、成功率、质量指标和耗时均可追踪。
- 标准错误日志只有 `torchvision pretrained` 弃用警告和 PyTorch hook 未来警告，未出现运行失败堆栈。

## 阶段结论

- 阶段 8 验收通过：三组实验使用同一批输入图片，输出目录分开，结果表字段一致。
- 当前小样本支持“Y 通道扰动更高效”的现象：`y_only` 在相同成功率下有最高 PSNR/SSIM 和最低 L2。
- 当前小样本也显示“UV 嵌入有质量代价”：`y_uv_embed` 保持成功率，但图像质量指标明显低于 `y_only`。
- 阶段 8 不能替代阶段 10 的可逆恢复验证；是否真正 error-free 仍需恢复脚本和逐像素比较确认。
