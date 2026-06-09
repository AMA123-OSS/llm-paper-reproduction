# 给老师的论文学习报告：YUV 可逆对抗攻击

汇报人：本地复现学习记录  
论文：`Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`  
来源：尹朝霞老师课题组主页近期英文论文  
代码：`edu-yinzhaoxia/Efficient-and-Transferable-Reversible-Adversarial-Attacks-Utilizing-YUV-Color-Space`

## 1. 选题与阅读方法

我从课题组主页近期论文中选择了 2025 年 Neurocomputing 论文 `Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`。选择原因是该论文有公开代码，任务是图像对抗攻击与可逆嵌入，不需要训练大模型，适合在 Windows + RTX 4060 本机做小规模完整复现。

阅读方法参考了“读论文三个层次”和“至少看三遍并画进度表”的思路：

- 第一遍：快速读题目、摘要、方法图和结论，确认论文解决的问题。
- 第二遍：对照代码读方法，理解每个模块怎么落地。
- 第三遍：用实验结果反问论文机制是否成立，重点看消融、迁移性和恢复验证。

本次复现不是只跑主脚本，而是按阶段推进：环境搭建、单图冒烟、低成本参数复测、10 图小规模、Y/UV 消融、迁移性实验、恢复可行性检查。

## 2. Paper 信息

### What：论文做了什么，解决了什么问题？

论文提出一种基于 YUV 颜色空间的可逆对抗攻击方法。它要同时解决两个目标：

- 对未授权分类模型产生误导，让模型误分类。
- 授权方能够从对抗样本恢复原始图像。

论文重点处理两个问题：

- **GRP**：生成阶段存在冗余扰动，扰动不够高效。
- **EOP**：可逆嵌入阶段可能覆盖有效攻击扰动，导致攻击失效。

### How：怎么做的？

代码中的核心流程是：

1. 读取 `ORI_IMG` 中 `序号_标签.png` 图片。
2. 用 ResNet50 + Grad-CAM++ 得到注意力区域。
3. 用 DenseNet161、Inception v3、GoogLeNet 组成的 `EnModel` 生成攻击扰动。
4. 将 RGB 转成 YUV。
5. 只在注意力区域替换 Y 通道扰动。
6. 计算原图 Y 通道和扰动 Y 通道差异。
7. 用 `embed_main` 将恢复信息嵌入 U/V 通道。
8. 转回 RGB，保存 RAE。
9. 用 Inception v3 验证是否误分类。

### Why：为什么能够解决？

我理解论文方法成立的关键是“通道分工”：

- Y 通道包含亮度、边缘、结构和纹理信息，分类模型对这些信息更敏感，所以攻击 Y 通道更有效。
- U/V 通道主要表达色度信息，用来嵌入恢复信息，可以减少对 Y 通道攻击扰动的覆盖。
- 集成模型生成扰动可以减少单模型过拟合，提升部分跨模型迁移性。

本地消融结果支持这个解释：Y-only 组在 10 图上攻击成功率为 `10/10`，同时平均 `SSIM=0.991483`、`PSNR=46.5931`，质量明显优于 RGB baseline 和 Y+UV 嵌入组。这说明 Y 通道扰动在当前小样本上更高效。

## 3. 复现实验结果

### 3.1 10 图小规模复现

| 参数 | 成功率 | 平均 SSIM | 平均 PSNR | 平均单图耗时 |
| --- | ---: | ---: | ---: | ---: |
| `steps=4, max_iteration=5` | 10/10 | 0.977174 | 41.6998 | 2.35s |
| `steps=20, max_iteration=50` | 10/10 | 0.976744 | 41.5805 | 4.59s |

结论：低成本参数已能在本地 10 图上跑通，且耗时约减少一半。

### 3.2 Y/UV 消融实验

| 方法 | 成功率 | 平均 SSIM | 平均 PSNR | 平均 L2 | 平均 Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB MI-FGSM | 10/10 | 0.975990 | 42.4469 | 4.0510 | 0.009412 |
| Y-only | 10/10 | 0.991483 | 46.5931 | 2.5893 | 0.016471 |
| Y+UV embed | 10/10 | 0.977176 | 41.6994 | 4.7156 | 0.038432 |

分析：

- Y-only 以最高质量达到同样成功率，支持论文关于减少 GRP 的动机。
- Y+UV 没有降低成功率，但质量明显低于 Y-only，说明 UV 嵌入有额外失真成本。

### 3.3 迁移性实验

| 生成方式 | 测试模型 | 关系 | 条件成功率 |
| --- | --- | --- | ---: |
| single Inception | Inception v3 | white-box | 10/10 |
| single Inception | DenseNet161 | transfer | 0/10 |
| single Inception | GoogLeNet | transfer | 0/10 |
| single Inception | ResNet50 | transfer | 2/8 |
| EnModel | Inception v3 | ensemble member | 10/10 |
| EnModel | DenseNet161 | ensemble member | 9/10 |
| EnModel | GoogLeNet | ensemble member | 5/10 |
| EnModel | ResNet50 | held-out | 2/8 |

分析：

- 集成生成明显提升了 DenseNet161 和 GoogLeNet 上的攻击效果。
- 但这两个模型是集成成员，不是严格未见黑盒。
- 对 held-out ResNet50，本地 10 图没有看到集成生成优于单模型生成。

### 3.4 可逆恢复检查

当前公开代码中存在 `PE_decode` 和 `Arithmetic_decode`，但主脚本没有保存完整恢复元数据：

- 缺少 `T_flag`。
- 缺少算术编码字典。
- 缺少 sidecar 元数据。
- 缺少独立 `recover_from_rae.py`。

因此本地无法证明 error-free recovery。这个结论不是否定论文理论，而是说明当前公开代码复现链路不完整。

## 4. Pros：优点

- 通道分工思路清楚：Y 负责攻击，U/V 负责恢复信息嵌入。
- 本地消融支持 Y 通道攻击更高效。
- 在 RTX 4060 上可用低成本参数完成小规模实验。
- 集成生成能提升部分模型覆盖。
- 论文同时考虑攻击成功、质量和可逆恢复，问题设定有价值。

## 5. Cons：不足与改进空间

### 5.1 最大不足：恢复闭环缺失

当前代码不能直接验证 error-free recovery，这是复现中最重要的问题。缺少恢复元数据导致无法从 RAE 还原原图。

### 5.2 UV 嵌入质量代价明显

Y+UV 相比 Y-only：

- SSIM 从 `0.991483` 降到 `0.977176`。
- PSNR 从 `46.5931` 降到 `41.6994`。
- L2 从 `2.5893` 升到 `4.7156`。

说明可逆嵌入虽然保留攻击成功率，但引入了明显图像质量损失。

### 5.3 迁移性结论还需要更严格黑盒

EnModel 对成员模型有效，但 held-out ResNet50 上没有提升。后续需要增加 VGG、MobileNet、EfficientNet、ConvNeXt、ViT 等未参与生成模型。

### 5.4 工程代码复现性不足

代码存在路径硬编码、尺寸硬编码、旧版 `scikit-image` 参数、缺少 CLI、缺少 requirements 等问题。

## 6. 本复现环境局限性

- 本机为 RTX 4060 Laptop GPU，显存约 8GB，无法轻松扩大到全量高参数实验。
- 样本规模主要是 10 张图，小规模结果不能代表论文全量结果。
- 为贴合原代码，预处理采用 `Resize((299,299)) + ToTensor()`，未使用标准 ImageNet normalization。
- 阶段 9 中部分测试模型是集成成员，严格黑盒结论有限。
- 当前公开代码缺少恢复元数据保存，无法验证 error-free recovery。

## 7. 改进方案与具体路线

### 路线 A：优先补齐可逆恢复闭环

1. 修改 `compressfun`，返回 `code` 和 `dic`。
2. 修改 `embed_main`，返回 `img_result`、`T_flag`、`dic`、压缩长度。
3. 每张 RAE 保存一个 `recovery.json` sidecar。
4. 新增 `recover_from_rae.py`。
5. 对 10 张图验证 `recovery_max_abs_error=0`、`recovery_error_pixels=0`。

这是最优先路线，因为它直接对应论文核心卖点。

### 路线 B：降低 UV 嵌入失真

1. 测试 U-only、V-only、U/V 分配比例。
2. 测试不同阈值 `T` 选择策略。
3. 比较 sidecar 外置部分元数据后，UV 嵌入负载下降是否提升 PSNR/SSIM。
4. 做 10 图和 100 图对比。

### 路线 C：扩大迁移性实验

1. 增加 held-out 模型：VGG16、MobileNetV3、EfficientNet、ConvNeXt、ViT。
2. 明确区分 white-box、ensemble member、held-out black-box。
3. 扩展到 100 张图，报告置信区间。
4. 对比单模型生成、EnModel 生成、去掉某个成员模型生成。

### 路线 D：工程化复现

1. 增加 CLI 参数：`--input-dir`、`--output-dir`、`--steps`、`--max-iteration`、`--device`。
2. 增加 `requirements.txt`。
3. 增加 `scripts/smoke_test.ps1`。
4. 把阶段 6-10 脚本整理成正式 `scripts`。
5. 固化 `.gitignore`，不提交虚拟环境、模型权重、输出图片。

## 8. 最终结论

这篇论文最值得学习的是把对抗攻击和可逆嵌入拆到不同颜色通道中的设计。我的本地复现实验支持 Y 通道攻击更高效，也看到集成生成能提升部分模型覆盖。但当前公开代码未提供完整恢复闭环，导致 error-free recovery 不能被直接验证。下一步最有价值的工作不是继续调攻击参数，而是补齐恢复元数据和恢复脚本，再扩大样本和严格黑盒模型验证。
