# Paper 信息与阶段 0 结论：YUV 可逆对抗攻击

阶段：阶段 0，材料归档与主线确认  
生成日期：2026-06-09  
当前项目主线：`Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`  
目标代码目录名：`YUV_Reversible_Attack_2025`

> 阶段 0 只完成材料核验、项目主线确认、代码目录结构核验和论文方法理解；不搭建环境、不运行实验。当前本地已存在 `YUV_Reversible_Attack_2025`，并已对照顶层代码文件做静态理解，但不能记录本地复现实验结果。

## 1. 阶段 0 核验结论

- [x] 当前复现主线已确认为 2025 年 Neurocomputing 论文 `Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`。
- [x] 最终选择依据来自 `D:\大模型论文复现\最后确认文件.md`。
- [x] 旧 Rt-LRM/LCO 文件、目录和文档仅作为历史候选材料保留，不作为当前复现主线。
- [x] 目标代码目录名确认为 `D:\大模型论文复现\YUV_Reversible_Attack_2025`。
- [x] 代码仓库来源确认为 `https://github.com/edu-yinzhaoxia/Efficient-and-Transferable-Reversible-Adversarial-Attacks-Utilizing-YUV-Color-Space`。
- [x] 本地代码目录 `D:\大模型论文复现\YUV_Reversible_Attack_2025` 已存在。
- [x] 本地代码结构确认包含 `ORI_IMG`、`pytorch_grad_cam`、`EnModel.py`、`atk.py`、`calMetrics.py`、`embed_utils.py`、`helpers.py`、`reversible_attack_OURS.py`、`utils.py`。
- [x] 本地 `ORI_IMG` 中已有 1000 张按 `序号_标签.png` 命名的样例图。
- [x] 本地已有 `output`、`runs`、`.venv` 目录；环境可用性尚未测试，留到后续阶段。
- [x] 当前阶段未运行本地实验，实验结果留到阶段 4 以后记录。

## 2. 材料来源

| 类型 | 当前状态 | 说明 |
|---|---|---|
| 论文列表 | 已核验 | 殷赵霞老师主页 2025 年条目列出该论文、DOI 和代码链接 |
| DOI/论文页面 | 已核验 | ScienceDirect 条目显示期刊、卷号、文章号、摘要和 Highlights |
| 代码仓库 | 已核验远程结构 | GitHub 仓库公开，Python 100%，README 较简略 |
| 本地代码 | 已存在并核验顶层结构 | 目录为 `YUV_Reversible_Attack_2025`，无 `.git`，推测为 ZIP/复制方式获取 |
| 旧 Rt-LRM/LCO 材料 | 保留但排除 | 不参与当前 YUV 项目命令、指标和结论 |

## 3. Paper 信息

### What：论文做了什么，解决了什么问题？

论文提出一种基于 YUV 颜色空间的 error-free reversible adversarial attack 方法。它试图同时解决两个看似冲突的目标：

1. 生成能误导图像分类模型的对抗样本，使未授权模型难以正确识别图像。
2. 让授权方可以从对抗样本中无损恢复原始图像。

已有可逆对抗攻击方法通常关注能否攻击成功、视觉质量是否足够好、是否能恢复原图，但对两个问题关注不足：

- **跨模型迁移性不足**：在一个白盒模型上成功生成的对抗样本，换到未知模型上可能失效。
- **无效扰动较多**：生成的扰动可能被后续可逆嵌入覆盖，或被浪费在对攻击贡献较低的通道中。

论文把无效扰动主要拆成两类：

- **EOP：Embedding-Overwritten Perturbations**，可逆嵌入阶段覆盖了原本有用的攻击扰动。
- **GRP：Generation-Redundant Perturbations**，攻击生成阶段产生了不必要或低效的冗余扰动。

### How：怎么做的？

论文采用双策略设计：

1. **Y 通道攻击**  
   将图像从 RGB 转换到 YUV 颜色空间，只在亮度通道 Y 上生成主要对抗扰动。论文和代码中涉及的攻击思路包括 YFGSM、YI-FGSM、YPGD、YMI-FGSM 等。

2. **U/V 通道嵌入恢复信息**  
   将恢复原图所需的信息嵌入 U/V 色度通道，而不是覆盖 Y 通道。这样能尽量保留已经生成的有效 Y 通道扰动。

3. **集成模型生成扰动**  
   使用多个预训练分类模型共同生成扰动，减少扰动对单个模型结构的过拟合。本地 `EnModel.py` 显示集成了 DenseNet161、Inception v3 和 GoogLeNet，并对三个模型输出做平均。

4. **CAM 注意力区域约束**  
   本地 `reversible_attack_OURS.py` 显示主脚本使用 ResNet50 和 Grad-CAM++ 生成注意力掩码，只在 CAM 值高于阈值的区域替换 Y 通道扰动，以降低可见失真并提高扰动利用率。

代码层面的主流程为：

1. 设置随机种子。
2. 自动选择 `cuda` 或 `cpu`。
3. 加载 Inception v3 作为攻击后分类验证模型。
4. 加载 `EnModel` 作为集成攻击模型。
5. 读取 `./ORI_IMG/` 中的图像。
6. 从文件名解析 ImageNet 标签，例如 `0001_281.png`。
7. 生成 CAM 注意力掩码。
8. 调用 `atk_MI_YFGSM` 生成 Y 通道扰动。
9. 调用 `embed_main` 将 Y 通道差异信息嵌入 U/V 通道。
10. 保存 RAE 到 `./output/rae/`，并用分类模型判断攻击是否成功。

### Why：为什么能够解决？（重要）

这篇论文的关键逻辑是“通道分工”与“扰动利用率提升”。

1. **Y 通道承载亮度结构，对分类模型更关键**  
   图像分类模型往往依赖边缘、纹理、形状和亮度结构等信息。YUV 中的 Y 通道更集中地表达亮度和结构信息，因此把扰动集中在 Y 通道，理论上比在 RGB 或所有 YUV 通道平均生成扰动更高效。

2. **U/V 通道承载色度信息，适合放恢复信息**  
   U/V 通道对人眼和分类判别的影响相对不同。把恢复所需信息放到 U/V 中，可以避免直接覆盖 Y 通道的有效攻击扰动，从机制上减少 EOP。

3. **攻击扰动和恢复信息被解耦**  
   传统可逆对抗攻击容易发生“先生成扰动，再嵌入恢复信息时把扰动破坏掉”的问题。该方法让 Y 通道主要负责攻击，U/V 通道主要负责恢复信息，从流程设计上把两个目标拆开处理。

4. **减少 GRP，提高每单位扰动的有效性**  
   只在 Y 通道生成扰动，减少了在色度通道中产生低效扰动的机会。这能减少无效计算和无效修改，提高攻击速度与视觉质量。

5. **集成模型降低白盒过拟合**  
   如果扰动只针对一个模型，可能只是抓住了该模型的局部脆弱性。多个模型共同参与梯度或输出融合，能逼迫扰动学习更通用的错误诱导方向，因此更可能迁移到未知模型。

6. **CAM 区域约束让扰动集中在关键区域**  
   CAM 高响应区域通常与模型判别相关。优先在这些区域引入 Y 通道扰动，有助于在较少修改下产生更强攻击效果。

### Pros：优点

- **目标清晰**：同时追求攻击成功和原图无损恢复。
- **机制可解释**：Y 负责攻击、U/V 负责嵌入，通道职责明确。
- **更重视迁移性**：引入集成攻击策略，而不是只看白盒成功率。
- **工程复现成本适中**：基于 PyTorch、torchvision、少量图片即可先跑通。
- **适合 RTX 4060**：不需要大模型训练，不依赖 LLM API，不需要多卡。
- **指标体系完整**：可围绕攻击成功率、PSNR、SSIM、L2、Linf、恢复误差和运行时间做实验。

### Cons：不足之处以及可能改进方案（重要）

#### 不足 1：代码 README 过简，复现门槛被转移到读源码

当前仓库 README 只有项目标题，缺少依赖、运行命令、输入样例说明、预期输出和恢复验证说明。

可能改进：

- 补充 `requirements.txt`。
- 增加 `README_reproduce.md`。
- 增加 `scripts/smoke_test.ps1`。
- 增加命令行参数，减少手改脚本。

测试方案：

- 阶段 1-4 完成后，记录从空环境到单图输出所需全部命令。
- 用一台干净 Windows 环境复跑，检查文档是否足够。

#### 不足 2：硬编码较多，输入格式脆弱

远程主脚本中输入目录为 `./ORI_IMG/`，输出目录为 `./output/rae/`，并从文件名按下划线解析标签。`embed_main` 中也存在 `299*299` 和双层循环等尺寸假设。

可能改进：

- 增加 `--input-dir`、`--output-dir`、`--image-size`、`--steps`、`--max-iteration`。
- 增加文件名合法性检查。
- 增加图片尺寸自动 resize 或显式报错。

测试方案：

- 使用正确命名 `0001_281.png` 和错误命名 `cat.png` 分别测试。
- 使用 `299x299` 和非 `299x299` 图片分别测试。
- 验证脚本能给出清晰错误提示，而不是中途崩溃。

#### 不足 3：集成模型显存占用较高

`EnModel.py` 集成 DenseNet161、Inception v3、GoogLeNet。对 RTX 4060 来说可跑，但显存和首次下载成本都更高。

可能改进：

- 增加单模型模式。
- 增加轻量集成模式，例如只用 Inception v3 + GoogLeNet。
- 增加 `torch.no_grad()` 用于非梯度评估环节。
- 记录显存峰值。

测试方案：

| 设置 | 图片数 | steps | 预期记录 |
|---|---:|---:|---|
| 单模型 Inception v3 | 1 | 5 | 冒烟测试速度和成功率 |
| 轻量双模型 | 1 | 5 | 显存与迁移性折中 |
| 原始三模型集成 | 1 | 20 | 接近论文方法 |

#### 不足 4：恢复验证入口不够显式

论文强调 error-free recovery，但主脚本更偏生成 RAE 和攻击成功判断。后续应明确恢复函数入口，并做像素级恢复测试。

可能改进：

- 新增 `restore_image.py`。
- 新增 `verify_recovery.py`。
- 输出恢复图，并比较原图与恢复图最大绝对误差。

测试方案：

- 对 1 张图做恢复，记录最大像素差。
- 对 10 张图做批量恢复，记录恢复失败样本。
- 如果误差不为 0，排查 PNG/JPEG 保存、RGB/YUV 转换、浮点量化和嵌入容量。

#### 不足 5：论文级结论需要更多数据支撑

阶段 0 还没有本地代码和实验环境，因此不能给出本机攻击成功率、PSNR、SSIM 或恢复误差。即便阶段 4 单图跑通，也只能说明流程可运行，不能代表完整论文复现。

可能改进：

- 阶段 7 使用 10 张图做小规模可复查结果。
- 阶段 8 做 Y 通道与 UV 嵌入消融。
- 阶段 9 做迁移性评估。
- 阶段 10 做恢复验证。

讨论用实验表模板：

| 实验 | 图片数 | 模型 | steps | 成功率 | PSNR | SSIM | 恢复最大误差 | 结论 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 阶段 4 单图冒烟 | 1 | EnModel 生成 / Inception v3 验证 | 20 | 1/1 | 35.7848 | 0.961871 | NA | 流程跑通 |
| 阶段 7 小规模 | 10 | EnModel 生成 / Inception v3 验证 | 4 / 20 | 10/10；10/10 | 41.6998；41.5805 | 0.977174；0.976744 | NA | 低成本与近默认参数均跑通 |
| 阶段 8 消融 | 10 | RGB / Y-only / Y+UV | 4 | 10/10；10/10；10/10 | 42.4469；46.5931；41.6994 | 0.975990；0.991483；0.977176 | NA | Y-only 质量最好，UV 嵌入带来额外失真 |
| 阶段 9 迁移性 | 10 | 单 Inception / EnModel 生成，4 模型测试 | 4 | 见迁移表 | NA | NA | NA | 集成提升成员模型迁移，ResNet50 未见提升 |

阶段 8 补充结论：在当前 10 图小样本上，Y-only 组以最高 PSNR/SSIM 和最低 L2 达到同样成功率，支持“减少 GRP”的方向性现象；Y+UV 组没有出现成功率下降，但相对 Y-only 有明显质量损失，说明可逆嵌入存在额外失真成本。恢复最大误差仍为 `NA`，需阶段 10 验证。

阶段 9 补充结论：单 Inception 生成在 Inception v3 白盒上为 `10/10`，但迁移到 DenseNet161/GoogLeNet 为 `0/10`；EnModel 集成生成在 DenseNet161 为 `9/10`、GoogLeNet 为 `5/10`，显示集成生成能提升成员模型覆盖。held-out ResNet50 上两者条件成功率均为 `2/8`，当前小样本不能证明集成生成提升未见模型迁移性。

## 4. 阶段 0 是否复现实验？

没有。

原因：

1. 阶段 0 的计划目标是材料归档与主线确认，不是代码获取和实验运行。
2. 阶段 0 只做静态核验，虽然本地代码已存在，但尚未按阶段 2 验证 Python/CUDA 依赖。
3. 模型权重尚未下载或验证。
4. 主脚本默认会遍历 `ORI_IMG` 中 1000 张图片，未降载前不适合作为阶段 0 操作。

阶段 0 可给出的“实验相关结论”是：

- 本地仓库具备可复现实验入口 `reversible_attack_OURS.py`。
- 本地源码显示主流程会读取 `./ORI_IMG/`，生成 RAE 并保存到 `./output/rae/`。
- 本地源码显示当前默认攻击函数为 `atk_MI_YFGSM`，默认集成模型为 DenseNet161、Inception v3、GoogLeNet。
- 本地数值结果必须等阶段 4 以后产生。

## 5. 下一阶段入口

阶段 1 应完成：

- 记录本地代码来源：当前目录无 `.git`，应补充 ZIP/复制来源说明。
- 确认仓库顶层文件存在，当前已静态核验通过。
- 确认 `runs` 和 `output\rae` 已存在。
- 记录下载方式、下载时间、commit 或 ZIP 文件信息；若无法获得 commit，则记录文件时间戳和获取方式。

阶段 1 完成后，才能进入阶段 2 环境搭建。

## 6. 参考来源

- 殷赵霞老师论文列表：`https://edu-yinzhaoxia.github.io/publications/`
- ScienceDirect 论文页面：`https://www.sciencedirect.com/science/article/abs/pii/S0925231225017606`
- DOI：`https://doi.org/10.1016/j.neucom.2025.131088`
- GitHub 代码仓库：`https://github.com/edu-yinzhaoxia/Efficient-and-Transferable-Reversible-Adversarial-Attacks-Utilizing-YUV-Color-Space`
- 本地主入口源码：`D:\大模型论文复现\YUV_Reversible_Attack_2025\reversible_attack_OURS.py`
- 本地集成模型源码：`D:\大模型论文复现\YUV_Reversible_Attack_2025\EnModel.py`
