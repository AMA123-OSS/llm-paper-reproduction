# update_yuv_recovery_optimization

本文件夹用于承接三条后续优化路线，保持和原始复现代码隔离，避免破坏论文原始流程。

## 目标

- 路线 A：补齐可逆恢复闭环，保存每张 RAE 对应的 `recovery.json` sidecar，并验证原图可无损恢复。
- 路线 B：分析 UV 嵌入失真，比较 U-only、V-only、U/V 分配比例、阈值策略和 sidecar 外置负载方案。
- 路线 C：扩展迁移性实验矩阵，覆盖 white-box、ensemble member、held-out black-box，并支持 10 图和 100 图实验。

## 文件说明

- `recovery_embed.py`：新增带元数据的压缩、嵌入和恢复函数。
- `recover_from_rae.py`：根据 `rae_yuv.npy` 和 `recovery.json` 恢复原始 YUV 的命令行工具。
- `route_a_recovery_selfcheck.py`：10 个合成 YUV 样本的可逆恢复自检。
- `route_b_uv_payload_analysis.py`：默认 UV 嵌入的负载与失真诊断。
- `route_b_ablation_plan.py`：路线 B 的完整消融计划 CSV。
- `route_b_uv_ablation_selfcheck.py`：U-only、V-only、U/V split、阈值策略和 sidecar 外置负载的合成消融测试。
- `route_c_transferability_preflight.py`：路线 C 的环境预检和 10/100 图实验矩阵生成。
- `route_c_transferability_experiment.py`：恢复 `ORI_IMG` 和深度学习环境后可执行的真实迁移性实验脚本。
- `results/`：轻量 CSV/JSON 结果。`.npy` 和输出图片被本目录 `.gitignore` 忽略，避免上传大文件。

## 当前可复现结果

已在当前 Windows 本机环境运行：

```powershell
& 'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' route_a_recovery_selfcheck.py
& 'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' route_b_uv_payload_analysis.py
& 'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' route_b_ablation_plan.py
& 'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' route_b_uv_ablation_selfcheck.py
& 'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' route_c_transferability_preflight.py
```

核心结论：

- 路线 A：10/10 个合成样本均满足 `recovery_max_abs_error=0`、`recovery_error_pixels=0`。
- 路线 B：默认 U+V 堆叠和 `adaptive_T_cost` 在当前合成样本上表现最好，平均 `uv_psnr=33.6774 dB`、`uv_ssim=0.920479`；U-only/V-only 因容量压力需要更大阈值，PSNR 下降到约 32.47/32.48 dB。
- 路线 C：已生成 112 行实验矩阵；当前真实迁移性跑分被 `ORI_IMG` 缺失和当前 Python 环境缺少 `torch/torchvision` 阻塞。

## 恢复真实 10/100 图实验

准备条件：

- 将 ImageNet 风格输入图恢复到 `D:\大模型论文复现\yuv-reversible-attack-reproduction\ORI_IMG`。
- 文件名保持 `<编号>_<真实label>.png`，例如 `0001_123.png`。
- 恢复可用的 PyTorch、torchvision、CUDA 环境和模型权重下载能力。

运行路线 C：

```powershell
python route_c_transferability_experiment.py --image-count 10
python route_c_transferability_experiment.py --image-count 100
```

如需只跑某个生成器：

```powershell
python route_c_transferability_experiment.py --image-count 10 --variants full_EnModel
python route_c_transferability_experiment.py --image-count 10 --variants single_inception_v3,full_EnModel
```

输出会写入 `results/route_c_*`，其中包括生成记录、逐图评估和带 95% 置信区间的汇总表。
