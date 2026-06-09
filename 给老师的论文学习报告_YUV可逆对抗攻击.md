# 给老师的论文学习报告：YUV 可逆对抗攻击

汇报人：本地复现学习记录  
论文：`Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`  
来源：尹朝霞老师课题组主页近期英文论文  
代码：`edu-yinzhaoxia/Efficient-and-Transferable-Reversible-Adversarial-Attacks-Utilizing-YUV-Color-Space`

## 0. 给老师的简要回复

老师您好，我选择并复现学习的是 2025 年 Neurocomputing 论文 `Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`。这篇文章的核心问题是：在保证授权方可无损恢复原图的前提下，如何让 reversible adversarial examples 同时具备更好的攻击有效性、视觉质量和跨模型迁移性。

我对照论文 PDF 和公开代码完成了分阶段复现。当前在 Windows + RTX 4060 环境下已跑通 10 张 ImageNet 样例的小规模实验、Y/UV 消融、单模型与集成生成迁移性对比，并检查了恢复链路。我的主要结论是：本地结果支持“Y 通道扰动更高效”和“集成生成提升部分模型覆盖”这两个机制；但公开代码没有保存完整恢复元数据，因此我无法直接证明论文报告的 error-free recovery，这一点我在报告中单独列为最重要的复现限制和后续改进方向。

## 1. 选题与阅读方法

我从课题组主页近期论文中选择了 2025 年 Neurocomputing 论文 `Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`。选择原因是该论文有公开代码，任务是图像对抗攻击与可逆嵌入，不需要训练大模型，适合在 Windows + RTX 4060 本机做小规模完整复现。

PDF 信息核对如下：

- 作者：Yucheng Fan, Zhaoxia Yin*, Jiawei Chen, Wanli Lyu。
- 期刊：`Neurocomputing 652 (2025) 131088`。
- DOI：`10.1016/j.neucom.2025.131088`。
- 在线发表时间：2025-07-24。
- 论文公开代码地址：`https://github.com/Definitely-Maybe/Efficient-and-Transferable-Reversible-Adversarial-Attacks`。

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

根据 PDF，论文把这两个问题定义得更具体：

- **EOP** 是指嵌入恢复信息时覆盖了已生成的攻击扰动，导致 RAE 中对应像素/通道不再保持 AE 的有效扰动。
- **GRP** 是指在最终不打算用于攻击的通道或区域上也生成扰动，例如只使用 Y 通道攻击时仍在 RGB 三通道生成扰动。

论文的 Table 1 将方法放在 error-free recovery、transferability、imperceptibility、W/O EOP、W/O GRP 五个维度比较，作者声称 OURS 是唯一同时满足这五项的方法。

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

对应论文 Algorithm 1 和 Algorithm 2，可以概括为：

- Algorithm 1：先把 RGB 转为 YUV，固定 U/V，仅对 Y 通道求梯度并更新；每轮把当前 YUV 转回 RGB 后送入多个模型，聚合 logits 后计算 loss，从而生成 Y-channel ensemble adversarial example。
- Algorithm 2：从原图与 AE 的 Y 通道差异中提取 `delta_Y`，用 CAM 二值化前景 mask 约束扰动区域，压缩扰动信息，再用 PEE 将压缩 bitstream 嵌入 U/V 通道，最终得到 RAE。

论文正式实验设置也比较明确：

- 数据：1000 张 ImageNet 图片，要求同时被 InceptionV3、GoogleNet、DenseNet 正确分类，中心裁剪到 `299 x 299`。
- 白盒模型：InceptionV3、GoogleNet、DenseNet。
- 黑盒模型：ResNet、VGG。
- 攻击方法：FGSM、I-FGSM、PGD、MI-FGSM、NI-FGSM。
- 参数：`epsilon=2/255` 或 `4/255`，`alpha=1/255`；I-FGSM/MI-FGSM/NI-FGSM 迭代 20 次，PGD 迭代 10 次。
- CAM：使用 ResNet50 生成注意力图，二值化阈值 `phi=0.5`。

### Why：为什么能够解决？

我理解论文方法成立的关键是“通道分工”：

- Y 通道包含亮度、边缘、结构和纹理信息，分类模型对这些信息更敏感，所以攻击 Y 通道更有效。
- U/V 通道主要表达色度信息，用来嵌入恢复信息，可以减少对 Y 通道攻击扰动的覆盖。
- 集成模型生成扰动可以减少单模型过拟合，提升部分跨模型迁移性。

本地消融结果支持这个解释：Y-only 组在 10 图上攻击成功率为 `10/10`，同时平均 `SSIM=0.991483`、`PSNR=46.5931`，质量明显优于 RGB baseline 和 Y+UV 嵌入组。这说明 Y 通道扰动在当前小样本上更高效。

更细地说，论文的 Why 可以拆成三层：

1. **减少 GRP**：只在 Y 通道生成扰动，避免在最终不使用的色度通道上浪费扰动预算。
2. **减少 EOP**：攻击扰动作用于 Y，而恢复信息嵌入 U/V，通道分离降低嵌入过程覆盖攻击扰动的风险。
3. **提升迁移性**：集成 logits 生成扰动，使扰动不只贴合单个白盒模型，而更可能命中多个模型共享的判别特征。

论文 Table 4 对这个解释提供了强证据。例如 FGSM `epsilon=4/255`、以 InceptionV3 为白盒生成时，YUV 方法在 ResNet/VGG 上的黑盒 ASR 为 `16.79%/15.99%`，OURS 提升到 `40.48%/35.71%`；NI-FGSM `epsilon=4/255` 下，InceptionV3 生成时 ResNet/VGG 从 `16.29%/16.61%` 提升到 `46.99%/39.75%`。

论文 Table 5 的消融也说明了两个组件各自的作用：YACK 只加入双策略设计，FGSM `4/255` 黑盒 ASR 从 YUV 的 `16.79%` 到 `20.55%`；ENS 只加入集成生成，迁移性提升更明显；OURS 将双策略和 ensemble 结合，在视觉质量、迁移性和耗时之间取得折中。

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

论文恢复过程是：从 RAE 中提取 U/V 里嵌入的 bitstream，经 PEE 反向恢复 U/V，再通过算术解码恢复 `delta_Y*`，最后用 `Y' - delta_Y*` 恢复原始 Y，并转回 RGB。论文报告恢复图 SSIM 为 `1.000`。

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
- PDF 中实验设置完整，覆盖 1000 张 ImageNet、三白盒两黑盒和五种攻击，指标设计比较全面。

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

### 5.5 论文公式与代码有一个需要说明的核对点

PDF 第 5 页 Eq.13 中 V 通道公式显示为 `V = -0.500R - 0.4187G - 0.0813B + 128`，但公开代码 `utils.py` 采用的是 `V = 0.500R - 0.4187G - 0.0813B + 128`，这也更符合与 Eq.14 中 `R = Y + 1.402(V - 128)` 配套的常见 YUV 变换。本地复现以公开代码为准，报告中保留这个核对点。

## 6. 本复现环境局限性

- 本机为 RTX 4060 Laptop GPU，显存约 8GB，无法轻松扩大到全量高参数实验。
- 样本规模主要是 10 张图，小规模结果不能代表论文全量结果。
- 为贴合原代码，预处理采用 `Resize((299,299)) + ToTensor()`，未使用标准 ImageNet normalization。
- 阶段 9 中部分测试模型是集成成员，严格黑盒结论有限。
- 当前公开代码缺少恢复元数据保存，无法验证 error-free recovery。
- 论文完整实验使用 1000 张 ImageNet 图片，本地只完成 10 图小规模复现，因此只能说明机制被局部验证，不能替代论文全量统计结论。

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

这篇论文最值得学习的是把对抗攻击和可逆嵌入拆到不同颜色通道中的设计。我的本地复现实验支持 Y 通道攻击更高效，也看到集成生成能提升部分模型覆盖；论文 PDF 的 Table 4 和 Table 5 进一步说明，完整实验中迁移性提升主要来自 ensemble，效率提升主要来自减少 GRP/EOP 的双策略设计。

当前最关键的不足是公开代码未提供完整恢复闭环，导致 error-free recovery 不能被直接验证。下一步最有价值的工作不是继续调攻击参数，而是先补齐恢复元数据和恢复脚本，在 10 图上完成逐像素无损恢复验证；随后扩展到 100 图和更多 held-out 黑盒模型，重新检验迁移性与图像质量之间的折中。

## 9. 可直接继续推进的实验计划

- [ ] 补齐 `embed_main` 返回值：`T_flag`、算术编码字典、压缩 bitstream 长度。
- [ ] 为每张 RAE 保存 `*.recovery.json` sidecar。
- [ ] 新增 `recover_from_rae.py`，验证 `recovery_max_abs_error=0`。
- [ ] 增加 VGG16/VGG19、MobileNetV3、EfficientNet、ConvNeXt、ViT 作为严格 held-out 黑盒模型。
- [ ] 将样本量从 10 张扩展到 100 张，报告条件 ASR、PSNR、SSIM 和耗时均值。
