# Paper 信息：YUV 可逆对抗攻击最终复盘与改进建议

日期：2026-06-09  
论文：`Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`  
作者：Yucheng Fan, Zhaoxia Yin*, Jiawei Chen, Wanli Lyu  
期刊：Neurocomputing, 2025, 131088  
本地代码目录：`D:\大模型论文复现\yuv-reversible-attack-reproduction`

## 复现范围说明

本次复现基于 Windows + RTX 4060 Laptop GPU，使用项目本地 `.venv`，核心环境为：

- Python 3.11.7
- PyTorch 2.5.1+cu121
- torchvision 0.20.1+cu121
- CUDA 可用
- 输入样例：本地 `ORI_IMG` 前 10 张 ImageNet 标签命名图片

本次已完成单图冒烟、10 图小规模复现、Y/UV 消融、迁移性实验和恢复可行性检查。由于当前公开代码没有完整保存恢复元数据，本地尚不能证明 error-free recovery。

## What：论文做了什么，解决了什么问题？

论文提出一种基于 YUV 颜色空间的可逆对抗攻击方法，目标不是单纯生成对抗样本，而是同时满足两个需求：

1. 对未授权模型造成误分类。
2. 授权方可以从对抗样本恢复原图。

它重点处理两类问题：

- **GRP：Generation-Redundant Perturbations**  
  对抗扰动中存在冗余或低效部分，影响图像质量和迁移性。

- **EOP：Embedding-Overwritten Perturbations**  
  可逆嵌入阶段可能覆盖原本有效的攻击扰动，导致攻击效果下降。

论文的核心主张是：把攻击扰动主要放在 Y 亮度通道，把恢复信息嵌入 U/V 色度通道，从而让“攻击”和“恢复信息存储”尽量分工。

## How：怎么做的？

结合代码，主流程如下：

1. 从 `ORI_IMG` 读取 `序号_标签.png` 格式图片。
2. 使用 ResNet50 + Grad-CAM++ 生成注意力区域。
3. 使用集成模型 `EnModel` 生成攻击梯度，成员为 DenseNet161、Inception v3、GoogLeNet。
4. 将 RGB 转为 YUV。
5. 只在注意力区域内替换 Y 通道扰动。
6. 计算原始 Y 与扰动后 Y 的差异。
7. 将恢复所需信息通过 `embed_main` 嵌入 U/V 通道。
8. 拼回 Y/U/V，转回 RGB，保存 RAE。
9. 使用 Inception v3 验证 RAE 是否误分类。

代码中关键文件：

- `reversible_attack_OURS.py`：主入口。
- `atk.py`：YUV 空间攻击函数，阶段 5-9 主要使用 `atk_MI_YFGSM`。
- `EnModel.py`：集成生成模型。
- `embed_utils.py`：预测误差扩展嵌入与部分解码函数。
- `utils.py`：RGB/YUV 转换。
- `calMetrics.py`：SSIM、PSNR、L2、Linf 指标。

## Why：为什么能够解决？（重点）

### 1. Y 通道更接近模型依赖的结构信息

Y 通道表达亮度、边缘、纹理和形状结构。分类模型往往依赖这些结构线索，因此扰动 Y 通道能更有效影响模型判别。

阶段 8 消融结果支持这一点：在 10 张图上，Y-only 组与 RGB 组都达到 `10/10` 成功，但 Y-only 的平均质量更好：

| 方法 | 成功率 | SSIM | PSNR | L2 |
| --- | ---: | ---: | ---: | ---: |
| RGB MI-FGSM | 10/10 | 0.975990 | 42.4469 | 4.0510 |
| Y-only | 10/10 | 0.991483 | 46.5931 | 2.5893 |

这说明在当前小样本上，Y 通道扰动用更小的总体差异达到了同样攻击效果，符合减少 GRP 的动机。

### 2. U/V 通道承载恢复信息，避免直接覆盖 Y 攻击扰动

如果把恢复信息直接嵌入被攻击的 Y 通道，就可能覆盖有效扰动，产生 EOP。论文把恢复信息放入 U/V，使 Y 主要负责攻击，U/V 主要负责恢复信息承载。

阶段 8 中，Y+UV 组成功率仍为 `10/10`，说明当前样本未出现明显“嵌入导致攻击失败”的 EOP 成功率损失。但质量指标下降明显：

| 对比 | SSIM | PSNR | L2 | Linf |
| --- | ---: | ---: | ---: | ---: |
| Y-only | 0.991483 | 46.5931 | 2.5893 | 0.016471 |
| Y+UV embed | 0.977176 | 41.6994 | 4.7156 | 0.038432 |

这说明 UV 嵌入确实有额外失真成本。它可能换来可逆恢复能力，但当前代码没有完整恢复链路，阶段 10 还不能验证收益是否成立。

### 3. 集成模型降低单模型过拟合，提高部分迁移性

单模型攻击容易只对该模型有效。集成 DenseNet161、Inception v3、GoogLeNet 的梯度，可以让扰动覆盖更多模型共同使用的判别特征。

阶段 9 结果支持这一点，但需要谨慎解释：

| 生成方式 | 测试模型 | 关系 | 条件成功率 |
| --- | --- | --- | ---: |
| single Inception | Inception v3 | white-box | 10/10 |
| single Inception | DenseNet161 | transfer | 0/10 |
| single Inception | GoogLeNet | transfer | 0/10 |
| EnModel | DenseNet161 | ensemble member | 9/10 |
| EnModel | GoogLeNet | ensemble member | 5/10 |
| EnModel | ResNet50 | held-out | 2/8 |

集成生成显著提升了 DenseNet161/GoogLeNet 上的效果，但它们本身是集成成员；对真正未见的 ResNet50，本次 10 图结果没有看到提升。

## Pros：优点

- 思路清晰：Y 负责攻击，U/V 负责嵌入恢复信息，机制容易解释。
- 对 4060 本机友好：低成本参数 `steps=4, max_iteration=5` 在 10 图上已达到 `10/10`。
- Y 通道消融结果好：Y-only 在当前样本中质量最好，说明通道选择确实有效。
- 集成攻击有价值：相较单 Inception，EnModel 对 DenseNet161/GoogLeNet 的覆盖明显更好。
- 指标体系完整：可同时记录成功率、SSIM、PSNR、L2、Linf、耗时和恢复误差字段。

## Cons：不足与改进方案（重点）

### 1. 当前代码无法完整验证 error-free recovery

这是最大问题。`embed_utils.py` 中虽然有 `PE_decode`，但主脚本没有保存恢复必需元数据：

- `T_flag` 没有返回或保存。
- 算术编码字典 `dic` 被丢弃。
- 没有 sidecar 元数据文件。
- 没有 `recover_from_rae.py` 入口。

阶段 10 结论：当前本地代码/保存方式无法证明 error-free recovery。这个问题直接影响论文核心卖点。

改进方案：

- 修改 `compressfun`，返回 `code` 和 `dic`。
- 修改 `embed_main`，返回 `img_result`、`T_flag`、`dic`、压缩长度等元数据。
- 为每张 RAE 保存 `0001_321.recovery.json`。
- 新增 `recover_from_rae.py`，输入 RAE + sidecar，输出恢复图。
- 验证 `recovery_max_abs_error=0`、`recovery_error_pixels=0`。

### 2. UV 嵌入带来明显质量损失

阶段 8 显示，Y+UV 相比 Y-only：

- SSIM 从 `0.991483` 降到 `0.977176`。
- PSNR 从 `46.5931` 降到 `41.6994`。
- L2 从 `2.5893` 升到 `4.7156`。

这说明当前嵌入实现的视觉代价不小。

改进实验：

- 对比 U-only、V-only、U/V 分配比例。
- 测试不同嵌入阈值策略对 PSNR/SSIM 的影响。
- 将恢复元数据外置 sidecar，减少必须嵌入 U/V 的信息量。
- 对比 PNG 保存前后是否存在量化损失。

### 3. 迁移性结论需要更严格黑盒验证

阶段 9 中，EnModel 对 DenseNet161/GoogLeNet 好于单 Inception，但这两个模型是集成成员，不是严格黑盒。真正 held-out ResNet50 上，EnModel 与 single Inception 条件成功率都是 `2/8`。

改进实验：

- 增加未参与生成的模型：VGG16、MobileNetV3、EfficientNet、ConvNeXt、ViT。
- 区分 ensemble member、white-box、held-out black-box。
- 在 100 张或 1000 张图上统计置信区间，而不是只用 10 图。

### 4. 工程复现性不足

当前代码存在这些问题：

- 路径硬编码：`./ORI_IMG/`、`./output/rae/`。
- 尺寸硬编码：`299x299`。
- 文件名解析脆弱：必须是 `序号_标签.png`。
- `calMetrics.py` 使用旧版 `multichannel=True`，新版 `scikit-image` 报错。
- `torchvision.models(pretrained=True)` 已弃用。
- 没有 CLI 参数、requirements 文件、恢复脚本和统一实验入口。

改进方案：

- 增加 `--input-dir`、`--output-dir`、`--steps`、`--max-iteration`、`--device`。
- 增加 `requirements.txt` 或 `environment.yml`。
- 将 `calMetrics.py` 改为 `channel_axis=-1, data_range=1.0`。
- 增加统一 `scripts/run_experiment.py` 和 `scripts/evaluate_outputs.py`。

## 本地已复现的关键结果

### 阶段 7：10 图小规模

| 参数 | 成功率 | SSIM | PSNR | L2 | Linf | 平均单图耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| steps=4, iter=5 | 10/10 | 0.977174 | 41.6998 | 4.7156 | 0.038432 | 2.35s |
| steps=20, iter=50 | 10/10 | 0.976744 | 41.5805 | 4.7450 | 0.037647 | 4.59s |

### 阶段 8：Y/UV 消融

| 方法 | 成功率 | SSIM | PSNR | L2 | Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB MI-FGSM | 10/10 | 0.975990 | 42.4469 | 4.0510 | 0.009412 |
| Y-only | 10/10 | 0.991483 | 46.5931 | 2.5893 | 0.016471 |
| Y+UV embed | 10/10 | 0.977176 | 41.6994 | 4.7156 | 0.038432 |

### 阶段 9：迁移性

| 生成方式 | 测试模型 | 条件成功率 |
| --- | --- | ---: |
| single Inception | Inception v3 | 10/10 |
| single Inception | DenseNet161 | 0/10 |
| single Inception | GoogLeNet | 0/10 |
| single Inception | ResNet50 | 2/8 |
| EnModel | Inception v3 | 10/10 |
| EnModel | DenseNet161 | 9/10 |
| EnModel | GoogLeNet | 5/10 |
| EnModel | ResNet50 | 2/8 |

### 阶段 10：恢复验证

当前无法生成恢复图。已反证当前公开代码/本地保存流程缺少恢复元数据，不能证明 error-free recovery。

## 最终判断

这篇论文的核心想法是有价值的：Y 通道攻击在本地消融中确实表现出更高图像质量和更低扰动，集成生成也能提升部分模型覆盖。但当前公开代码离完整论文复现还有明显距离，尤其是 error-free recovery 缺少可运行闭环。下一步最值得投入的是补齐恢复元数据和恢复脚本，然后再扩大样本规模做严格黑盒迁移性测试。
