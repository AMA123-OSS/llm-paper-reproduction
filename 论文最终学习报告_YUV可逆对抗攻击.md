# 论文最终学习报告：YUV 可逆对抗攻击

论文：`Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`  
作者：Yucheng Fan, Zhaoxia Yin*, Jiawei Chen, Wanli Lyu  
期刊：Neurocomputing, 2025, 131088

## 1. 论文问题意识

这篇论文关注的不是普通对抗攻击，而是“可逆对抗攻击”：未授权模型看到的是对抗样本，会产生误分类；授权方应能从对抗样本恢复原图。

论文试图同时解决三件事：

- 攻击有效：RAE 能让分类模型误分类。
- 图像质量好：扰动不能过于明显。
- 可逆恢复：授权方应能无损恢复原图。

论文中特别强调两个机制性问题：

- **GRP**：生成阶段存在冗余扰动，扰动没有高效作用在关键判别信息上。
- **EOP**：嵌入恢复信息时可能覆盖有效攻击扰动，导致攻击效果下降。

## 2. 方法理解

论文方法的核心是 YUV 通道分工：

- **Y 通道**：亮度与结构信息，主要用于生成攻击扰动。
- **U/V 通道**：色度信息，主要用于嵌入恢复信息。

结合代码，主流程是：

1. 读取 ImageNet 标签命名图片。
2. 用 ResNet50 + Grad-CAM++ 生成注意力掩码。
3. 用 DenseNet161、Inception v3、GoogLeNet 组成的 `EnModel` 生成攻击扰动。
4. RGB 转 YUV。
5. 在注意力区域替换 Y 通道。
6. 将 Y 通道差异通过预测误差扩展嵌入 U/V。
7. 转回 RGB 保存 RAE。
8. 用 Inception v3 检查是否误分类。

## 3. 为什么这个方法可能有效？

第一，Y 通道更接近分类模型依赖的结构线索。边缘、纹理、形状、亮度变化通常比色度更影响分类结果，所以攻击 Y 通道更高效。

第二，U/V 通道承载恢复信息，可以降低恢复信息直接覆盖 Y 通道扰动的概率。这是论文缓解 EOP 的关键设计。

第三，集成模型生成扰动能减少单模型过拟合。扰动如果同时影响多个模型的判别边界，更可能具有迁移性。

本地阶段 8 消融支持第一点：Y-only 在 10 图上达到 `10/10` 成功，同时平均 `SSIM=0.991483`、`PSNR=46.5931`，优于 RGB baseline 和 Y+UV 组。

本地阶段 9 支持第三点的一部分：EnModel 生成对 DenseNet161 的条件成功率为 `9/10`，对 GoogLeNet 为 `5/10`；单 Inception 生成在这两个模型上均为 `0/10`。

## 4. 本地核心复现结果

### 4.1 10 图小规模复现

| 参数 | 成功率 | SSIM | PSNR | 平均单图耗时 |
| --- | ---: | ---: | ---: | ---: |
| `steps=4, max_iteration=5` | 10/10 | 0.977174 | 41.6998 | 2.35s |
| `steps=20, max_iteration=50` | 10/10 | 0.976744 | 41.5805 | 4.59s |

低成本参数在本地 10 图上已经足够有效。

### 4.2 Y/UV 消融

| 方法 | 成功率 | SSIM | PSNR | L2 | Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB MI-FGSM | 10/10 | 0.975990 | 42.4469 | 4.0510 | 0.009412 |
| Y-only | 10/10 | 0.991483 | 46.5931 | 2.5893 | 0.016471 |
| Y+UV embed | 10/10 | 0.977176 | 41.6994 | 4.7156 | 0.038432 |

结论：Y-only 质量最好；UV 嵌入没有降低成功率，但带来明显质量代价。

### 4.3 迁移性

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

结论：集成生成提升了对成员模型的覆盖，但在 held-out ResNet50 上，本地 10 图没有看到优于单 Inception 的迁移性。

## 5. 本复现环境局限性

### 5.1 样本规模有限

本次主要使用 10 张图片做小规模复现，不能代表论文全量实验。成功率、PSNR、SSIM 和迁移性都可能受样本选择影响。

### 5.2 设备与时间限制

本机为 RTX 4060 Laptop GPU，显存约 8GB。代码同时加载 Inception v3、DenseNet161、GoogLeNet、ResNet50，显存压力较高，因此优先使用低成本参数 `steps=4, max_iteration=5`。

### 5.3 代码预处理口径与标准 ImageNet 不完全一致

为了贴合原代码，本次评估使用 `Resize((299,299)) + ToTensor()`，没有额外使用 ImageNet mean/std normalization。这保证了和仓库行为一致，但不等同于 torchvision 官方评估口径。

### 5.4 当前公开代码缺少完整恢复闭环

这是最大局限。代码有 `PE_decode` 和 `Arithmetic_decode`，但主流程没有保存恢复必需的 `T_flag`、算术编码字典和 sidecar 元数据，也没有独立 `recover_from_rae.py`。

因此本地无法证明论文的 error-free recovery，只能说明当前公开代码/保存流程不完整。

### 5.5 黑盒迁移性验证还不够严格

阶段 9 中 DenseNet161、GoogLeNet 是 EnModel 成员，对集成生成来说不是严格 held-out 黑盒。真正未参与生成的 ResNet50 上，集成生成没有比单 Inception 更好。

## 6. 优点总结

- YUV 通道分工清晰，机制可解释。
- Y-only 消融结果支持“Y 通道扰动更高效”。
- 低成本参数在 4060 上可快速跑通 10 图实验。
- 集成生成对成员模型有明显覆盖优势。
- 论文问题意识明确：同时考虑攻击、质量、可逆恢复。

## 7. 不足与改进建议

最重要的改进是补齐恢复闭环：

1. 修改 `compressfun` 返回 `code` 和 `dic`。
2. 修改 `embed_main` 返回 `T_flag`、字典、压缩长度等元数据。
3. 为每张 RAE 保存 sidecar JSON。
4. 新增 `recover_from_rae.py`。
5. 验证 `recovery_max_abs_error=0`、`recovery_error_pixels=0`。

后续实验建议：

- 扩展到 100 或 1000 张图。
- 增加 held-out 模型，如 VGG、MobileNet、EfficientNet、ConvNeXt、ViT。
- 比较 U-only、V-only、U/V 分配比例。
- 对比有无 ImageNet normalization 的结果差异。
- 统计模型初始化时间和完整 wall-clock 时间。

## 8. 最终学习结论

这篇论文最值得学习的是“攻击通道”和“恢复通道”解耦的设计思路。Y 通道攻击在本地消融中确实显示出更好的扰动效率；集成生成也能提升部分模型覆盖。但当前公开代码缺少完整恢复元数据保存与恢复入口，导致本地无法验证最关键的 error-free recovery。后续若要形成更完整复现，第一优先级应是补齐恢复闭环，而不是继续堆更多攻击参数。
