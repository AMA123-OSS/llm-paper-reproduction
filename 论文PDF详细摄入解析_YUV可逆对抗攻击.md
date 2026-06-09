# 论文 PDF 详细摄入解析：YUV 可逆对抗攻击

日期：2026-06-09  
论文：`Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`  
作者：Yucheng Fan, Zhaoxia Yin*, Jiawei Chen, Wanli Lyu  
期刊：`Neurocomputing 652 (2025) 131088`  
DOI：`10.1016/j.neucom.2025.131088`  
接收与上线：2025-07-23 接收，2025-07-24 在线发表  
本地复现目录：`D:\大模型论文复现\yuv-reversible-attack-reproduction`

## 1. 摄入结论总览

这篇论文的核心不是简单把 RGB 改成 YUV，而是利用 YUV 的通道语义完成一次明确分工：

- `Y` 通道承载攻击扰动，因为 Y 通道更接近亮度、边缘、结构和纹理信息，分类模型更敏感。
- `U/V` 通道承载可逆恢复信息，因为 U/V 主要表达色度信息，把恢复信息放在这里可以减少对 Y 通道攻击扰动的覆盖。
- 集成模型生成扰动，以降低扰动对单一白盒模型的过拟合，提升跨模型迁移性。

论文要解决的关键问题是：已有 error-free reversible adversarial attack 方法多数关注可行性、白盒攻击成功率和图像质量，但对跨模型迁移性不足；同时存在两类低效扰动，导致有限扰动预算没有被充分用于攻击。

## 2. 论文要解决的问题

### 2.1 可逆对抗攻击的任务设定

普通对抗攻击只要求让模型误分类，而可逆对抗攻击还要求授权方能从对抗样本恢复原始图像。论文将这个过程概括为：

1. 对原图 `x` 生成扰动 `delta`，得到 adversarial example。
2. 使用可逆数据隐藏方法把恢复所需信息嵌入图像，得到 reversible adversarial example，即 RAE。
3. 授权方通过反向提取嵌入信息，恢复扰动并从 RAE 中恢复原图。

论文强调 error-free recovery，即恢复图像应与原图无损一致。这一点是该类方法区别于近似恢复方法的核心价值，尤其适合医疗、军事等不接受恢复误差的场景。

### 2.2 两类 ineffective perturbations

论文给 ineffective perturbations 做了明确分类：

- `EOP`：Embedding-Overwritten Perturbations。嵌入恢复信息时覆盖了已经生成的攻击扰动，导致原本有效的攻击扰动被破坏。
- `GRP`：Generation-Redundant Perturbations。在不打算真正用于攻击的通道或区域上生成扰动。例如最终只打算修改 Y 通道，却在 RGB 三通道都生成扰动，U/V 或未使用区域的梯度就成为冗余。

这两个问题共同导致扰动预算浪费。由于可逆嵌入容量有限，扰动越不能被高效利用，越难同时保持攻击性、图像质量和可恢复性。

## 3. 方法结构

论文框架由三个模块构成：

1. `Adversarial perturbation generation`：生成对抗扰动。
2. `Reversible adversarial example generation`：将扰动作用到图像，并把恢复信息嵌入 U/V。
3. `Original image restoration`：从 RAE 中提取嵌入信息并恢复原图。

### 3.1 Y 通道集成攻击

论文将传统 FGSM、I-FGSM、PGD、MI-FGSM、NI-FGSM 扩展为 Y 通道版本，例如 YFGSM、YI-FGSM、YPGD、YMI-FGSM。

核心流程：

1. 将 RGB 图像转换到 YUV。
2. 固定原始 `U/V` 通道，只让 `Y` 参与梯度更新。
3. 每一轮把当前 `(Y, U, V)` 转回 RGB，送入多个模型。
4. 对多个模型的 logits 做加权求和，得到 ensemble logits。
5. 用交叉熵对 `Y` 通道求梯度，并按攻击方法更新 `Y`。
6. 将结果裁剪到原图的 `epsilon` 邻域内。
7. 迭代完成后，把最终 Y 与未改动的 U/V 合成 AE。

论文的集成 logits 形式可概括为：多个模型输出 logits 后按权重加权相加，再统一计算 loss。这样生成的扰动不只贴合单个模型，而是更接近多个模型共享的判别线索。

### 3.2 YUV 转换公式与代码核对点

PDF 第 5 页 Eq.13/Eq.14 给出 YUV 与 RGB 的转换公式。本地代码实现位于 `yuv-reversible-attack-reproduction\utils.py`：

```python
yuv[:, :, 0] = np.round(0.299 * R + 0.587 * G + 0.114 * B)
yuv[:, :, 1] = np.round(- 0.1687 * R - 0.3313 * G + 0.5 * B + 128)
yuv[:, :, 2] = np.round(0.5 * R - 0.4187 * G - 0.0813 * B + 128)
```

需要注意：PDF 抽取文本和渲染页中，`V` 通道公式显示为 `-0.500 * R - 0.4187 * G - 0.0813 * B + 128`；但代码和常见 YUV 反变换配套关系使用的是 `+0.500 * R - 0.4187 * G - 0.0813 * B + 128`。本地复现以公开代码实现为准，并在报告中把这一点作为论文公式与代码口径核对点记录。

### 3.3 RAE 生成

论文 Algorithm 2 的逻辑可以分成六步：

1. 用 Y-channel ensemble attack 生成 `x_adv`。
2. 将 `x_adv` 和原图 `x` 转到 YUV。
3. 提取 Y 通道扰动：`delta_Y = Y_adv - Y`。
4. 用 CAM 生成注意力图，并用阈值 `phi` 二值化为前景 mask。
5. 只保留显著区域扰动，得到 `delta_Y*`，并加到原始 Y 通道，得到 `Y'`。
6. 对 `delta_Y*` 做算术编码压缩，再用 PEE 将压缩比特流嵌入 U/V，最后转回 RGB 得到 RAE。

这里 CAM 的作用是减少需要嵌入和实际修改的扰动范围。论文的解释是：模型更关注前景或语义显著区域，把扰动集中在这些区域能提升扰动效率，同时降低需要嵌入的扰动信息量。

### 3.4 原图恢复

论文的恢复路径是：

1. 将 RAE 从 RGB 转回 YUV。
2. 对 `U'/V'` 执行 PEE 反向解码，恢复干净的 `U/V` 并提取压缩 bitstream。
3. 对 bitstream 做算术解码，恢复 `delta_Y*`。
4. 用 `Y' - delta_Y*` 恢复原始 Y。
5. 将恢复后的 `(Y, U, V)` 转回 RGB。

论文报告恢复图与原图的 SSIM 为 `1.000`，表示无结构失真。但本地公开代码中，主流程只保存 RAE PNG，没有保存恢复必需的 `T_flag`、算术编码字典、压缩流长度等 sidecar 元数据，因此本地阶段 10 无法直接验证 error-free recovery。

## 4. 实验设置摄入

### 4.1 数据集与模型

论文正式实验使用 ImageNet：

- 从 ImageNet 随机选择 1000 张图片。
- 这些图片需被 InceptionV3、GoogleNet、DenseNet 同时正确分类。
- 每张图片中心裁剪到 `299 x 299`。

模型划分：

- 白盒模型：InceptionV3、GoogleNet、DenseNet。
- 黑盒模型：ResNet、VGG。
- 集成生成模型：InceptionV3、GoogleNet、DenseNet 的 logits 集成。

### 4.2 攻击与参数

论文测试五种攻击：

- FGSM
- I-FGSM
- PGD
- MI-FGSM
- NI-FGSM

参数设置：

- FGSM：`epsilon = 2/255` 或 `4/255`，`alpha = 1/255`。
- I-FGSM、MI-FGSM、NI-FGSM：`epsilon = 2/255` 或 `4/255`，`alpha = 1/255`，迭代 20 次。
- PGD：`epsilon = 2/255` 或 `4/255`，`alpha = 1/255`，迭代 10 次。
- CAM：使用 ResNet50 生成注意力图，二值化阈值 `phi = 0.5`。

### 4.3 指标

视觉质量：

- `PSNR`：越高越好。
- `SSIM`：越高越好。
- `CIEDE2000`：越低越好。

攻击能力：

- `White-box ASR`
- `Black-box ASR`

本地复现额外记录：

- `L2`
- `Linf`
- 单图耗时
- 恢复前置条件与恢复误差字段

## 5. 论文结果摄入

### 5.1 视觉质量

论文 Table 2 显示：

- OURS 在所有设置下明显优于早期 RDH 方法。
- 与上一代 YUV 方法相比，OURS 的 PSNR/SSIM 略低，但多数设置下降小于 3%。
- CIEDE2000 上，OURS 在大多数设置中达到最好或接近最好。

代表性数值：

| 攻击设置 | YUV PSNR | OURS PSNR | YUV SSIM | OURS SSIM | YUV CIEDE2000 | OURS CIEDE2000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FGSM 2/255 | 40.754 | 40.606 | 0.976 | 0.975 | 1.586 | 1.401 |
| FGSM 4/255 | 38.136 | 38.096 | 0.961 | 0.960 | 1.886 | 1.481 |
| MI-FGSM 4/255 | 38.933 | 38.961 | 0.967 | 0.967 | 1.780 | 1.563 |
| NI-FGSM 4/255 | 38.153 | 37.761 | 0.962 | 0.959 | 1.889 | 1.709 |

解释：OURS 加入集成攻击后，为了追求跨模型迁移性，可能需要更强或更多扰动，因此单模型视觉质量略低于 YUV；但 CIEDE2000 经常更好，说明色彩感知差异并没有明显变坏。

### 5.2 白盒攻击能力

论文 Table 3 显示：

- 在 InceptionV3、GoogleNet、DenseNet 三个白盒模型上，YUV 和 OURS 的 ASR 基本接近 100%。
- RDH 方法在部分设置下明显低于 100%，原因是可逆嵌入可能覆盖已添加扰动。

这说明在白盒攻击上，OURS 没有因为通道分工和嵌入设计丢失攻击能力。

### 5.3 黑盒攻击能力

论文 Table 4 是本文最重要的证据之一。OURS 在 ResNet 和 VGG 黑盒模型上显著高于 YUV，很多设置提升超过 100%。

代表性数值：

| 攻击设置 | 生成白盒 | YUV -> ResNet | OURS -> ResNet | YUV -> VGG | OURS -> VGG |
| --- | --- | ---: | ---: | ---: | ---: |
| FGSM 2/255 | InceptionV3 | 10.40 | 28.07 | 9.63 | 24.53 |
| FGSM 4/255 | InceptionV3 | 16.79 | 40.48 | 15.99 | 35.71 |
| PGD 4/255 | InceptionV3 | 10.03 | 37.09 | 10.25 | 30.59 |
| NI-FGSM 4/255 | InceptionV3 | 16.29 | 46.99 | 16.61 | 39.75 |

解释：YUV 旧方法主要改善通道分离和可逆性，但扰动仍容易贴合单一模型；OURS 通过 logits 集成生成扰动，削弱单模型过拟合，所以黑盒 ASR 明显提高。

### 5.4 消融实验

论文 Table 5 对比了四种方法：

- `YUV`：上一代 YUV 方法。
- `YACK`：只使用双策略设计，不使用 ensemble。
- `ENS`：只用 ensemble 生成扰动，不使用完整双策略。
- `OURS`：双策略加 ensemble。

关键结论：

- `YACK` 相对 `YUV` 提升黑盒 ASR，并在多数设置下降低运行时间。论文举例：FGSM `epsilon=4/255` 时黑盒 ASR 从 `16.79%` 提升到 `20.55%`。
- `ENS` 相对 `YUV` 大幅提升黑盒 ASR，说明 ensemble 是迁移性提升主因。
- `OURS` 相对 `ENS` 运行时间更低，论文称多数设置降低 15% 以上，说明 Y 通道和显著区域策略减少了冗余计算。
- `OURS` 在视觉质量、黑盒 ASR、耗时之间取得折中。

代表性 Table 5 数值：

| 指标 | YUV | YACK | ENS | OURS |
| --- | ---: | ---: | ---: | ---: |
| FGSM 4/255 Black-box ASR | 16.79 | 20.55 | 38.10 | 40.48 |
| NI-FGSM 4/255 Black-box ASR | 16.29 | 18.80 | 45.61 | 46.99 |
| FGSM 4/255 Running Time | 2:46:33 | 1:29:57 | 2:53:43 | 1:47:06 |
| MI-FGSM 4/255 Running Time | 1:56:47 | 1:21:02 | 4:04:33 | 3:01:54 |

## 6. 与本地代码的对应关系

| 论文模块 | 本地文件 | 本地实现状态 |
| --- | --- | --- |
| YUV 转换 | `utils.py` | 已实现，V 通道符号以代码为准 |
| Y 通道攻击 | `atk.py` | 已实现 `atk_MI_YFGSM` 等核心攻击 |
| 集成模型 | `EnModel.py` | 使用 DenseNet161、Inception v3、GoogLeNet |
| CAM mask | `reversible_attack_OURS.py`，`pytorch_grad_cam` | 使用 ResNet50 layer4 与 Grad-CAM++ |
| UV 嵌入 | `embed_utils.py` | 已实现 PEE 编码，返回值不含完整恢复元数据 |
| 恢复 | `embed_utils.py` | 有 `PE_decode` 和 `Arithmetic_decode` 积木，但缺少完整恢复入口和 sidecar |
| 指标 | `calMetrics.py`，`runs/evaluate_outputs.py` | 本地已补充可复查 CSV |

## 7. 本地复现结果对照

### 7.1 10 图小规模复现

| 参数 | 成功率 | 平均 SSIM | 平均 PSNR | 平均单图耗时 |
| --- | ---: | ---: | ---: | ---: |
| `steps=4, max_iteration=5` | 10/10 | 0.977174 | 41.6998 | 2.35s |
| `steps=20, max_iteration=50` | 10/10 | 0.976744 | 41.5805 | 4.59s |

低成本参数在本地 10 张图上已经能跑通，说明 RTX 4060 可以支持小规模复现与消融。

### 7.2 Y/UV 消融

| 方法 | 成功率 | SSIM | PSNR | L2 | Linf |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGB MI-FGSM | 10/10 | 0.975990 | 42.4469 | 4.0510 | 0.009412 |
| Y-only | 10/10 | 0.991483 | 46.5931 | 2.5893 | 0.016471 |
| Y+UV embed | 10/10 | 0.977176 | 41.6994 | 4.7156 | 0.038432 |

本地结果支持论文的 GRP 观点：Y-only 在相同成功率下图像质量更好，说明扰动更集中有效。与此同时，Y+UV 嵌入带来明显质量代价，这也是后续改进重点。

### 7.3 迁移性

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

本地结果部分支持 ensemble 的价值：对集成成员模型覆盖明显提升。但对真正 held-out ResNet50，10 图实验没有看到提升。因此本地报告必须谨慎表述：已复现“成员模型覆盖提升”，还未充分复现“严格黑盒迁移性提升”。

### 7.4 恢复验证

本地阶段 10 的结论是：公开代码中存在恢复积木，但无法从保存的 RAE PNG 独立恢复原图。缺失前置条件包括：

- `T_flag_persisted`
- `arithmetic_dictionary_persisted`
- `compressed_bit_length_persisted`
- `standalone_recovery_entry`

这不是对论文理论的否定，而是代码复现链路的限制。若要完整验证论文核心卖点，必须补齐恢复 sidecar 和 `recover_from_rae.py`。

## 8. What / How / Why / Pros / Cons

### What

论文提出一种高效且具迁移性的 error-free reversible adversarial attack。它同时面向攻击、图像质量、可逆恢复和跨模型迁移性，重点解决 EOP、GRP 和单模型过拟合。

### How

方法用 YUV 通道分工实现：

- Y 通道生成并承载攻击扰动。
- CAM 限定扰动区域，降低冗余扰动和嵌入负载。
- 算术编码压缩 Y 通道扰动。
- PEE 将压缩信息嵌入 U/V。
- 集成多个白盒模型 logits 生成更具迁移性的扰动。

### Why

这个方法能够解决问题的原因有三层：

1. `Y` 通道包含亮度和结构，扰动更容易影响分类模型的判别特征，因此减少 GRP。
2. `U/V` 承载恢复信息，与 Y 攻击通道分离，减少嵌入过程覆盖攻击扰动的 EOP。
3. Ensemble logits 让扰动同时适配多个模型的决策边界，降低单模型过拟合，提高迁移性。

### Pros

- 机制清楚，可解释性强。
- 同时考虑 error-free recovery、视觉质量、攻击成功率、黑盒迁移性和运行效率。
- 论文实验口径完整，覆盖 1000 张 ImageNet 图片、三白盒两黑盒、五种攻击。
- 本地 4060 上可完成小规模复现与核心消融。
- Y-only 消融在本地结果中表现最好，支撑通道选择动机。

### Cons

- 公开代码没有完整恢复闭环，难以直接复现 error-free recovery。
- UV 嵌入在本地带来明显质量损失，需要优化嵌入容量、通道分配和 sidecar 策略。
- 本地 10 图实验无法替代论文 1000 图完整统计。
- 本地严格 held-out 黑盒迁移性尚未充分验证。
- PDF 公式与代码的 V 通道符号存在核对点，应在复现报告中说明以代码为准。

## 9. 建议的后续改进路线

### 路线 A：补齐恢复闭环

1. 修改 `compressfun`，返回 `code` 和 `dic`。
2. 修改 `embed_main`，返回 `img_result`、`T_flag`、压缩长度、字典或字典索引。
3. 每张 RAE 保存一个 `*.recovery.json`。
4. 新增 `recover_from_rae.py`。
5. 用 10 图验证 `recovery_max_abs_error=0` 和 `recovery_error_pixels=0`。

### 路线 B：降低 UV 嵌入失真

1. 对比 U-only、V-only、U/V 按容量分配。
2. 测试不同 PEE 阈值选择策略。
3. 将部分恢复元数据放入 sidecar，减少 U/V 必须承载的信息量。
4. 对比 PNG 保存前后量化误差。

### 路线 C：扩大迁移性验证

1. 增加 VGG16/19、MobileNetV3、EfficientNet、ConvNeXt、ViT。
2. 明确区分 white-box、ensemble member、held-out black-box。
3. 扩展到 100 张再到 1000 张。
4. 对每个模型报告原图正确数、条件 ASR、无条件 ASR 和置信区间。

## 10. 可写入给老师回复的核心细节

建议在回复老师时补充这些“体现认真读过 PDF 和代码”的细节：

- 正式论文使用 1000 张 ImageNet 图片，且要求同时被 InceptionV3、GoogleNet、DenseNet 正确分类，统一裁剪到 `299 x 299`。
- 白盒模型是 InceptionV3、GoogleNet、DenseNet，黑盒模型是 ResNet 和 VGG。
- CAM 使用 ResNet50，阈值 `phi=0.5`。
- 论文参数是 `epsilon=2/255` 或 `4/255`，`alpha=1/255`，I-FGSM/MI-FGSM/NI-FGSM 迭代 20 次，PGD 迭代 10 次。
- Table 4 的黑盒迁移性是最关键证据，例如 FGSM `4/255` 下，InceptionV3 生成时 ResNet ASR 从 YUV 的 `16.79%` 提升到 OURS 的 `40.48%`。
- Table 5 的消融显示，YACK 提升扰动效率，ENS 提升迁移性，OURS 是两者结合。
- 本地复现已经支持 Y 通道扰动更高效和 ensemble 对成员模型覆盖更强，但还未能验证公开代码中的 error-free recovery。
