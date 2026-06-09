# 阶段 2 环境搭建与测试记录：YUV 可逆对抗攻击

日期：2026-06-09  
项目目录：`D:\大模型论文复现\YUV_Reversible_Attack_2025`  
阶段目标：完成 Python、CUDA、PyTorch 和基础图像依赖的本地环境验收。

## 1. 阶段 2 结论

- [x] 项目本地虚拟环境 `.venv` 可用。
- [x] `.venv` Python 版本为 `Python 3.11.7`，符合阶段 2 要求。
- [x] pip 来自项目 `.venv`。
- [x] NVIDIA 驱动可用，`nvidia-smi` 能看到 RTX 4060。
- [x] PyTorch CUDA 版安装成功。
- [x] `torch.cuda.is_available()` 返回 `True`。
- [x] 基础图像依赖导入成功。

## 2. 环境信息

| 项目 | 检查结果 |
|---|---|
| 项目 Python | `D:\大模型论文复现\YUV_Reversible_Attack_2025\.venv\Scripts\python.exe` |
| Python 版本 | `3.11.7` |
| pip | `pip 26.1.2` |
| GPU | `NVIDIA GeForce RTX 4060 Laptop GPU` |
| 显存 | `8188 MiB` |
| 驱动版本 | `592.27` |
| nvidia-smi CUDA | `13.1` |
| PyTorch | `2.5.1+cu121` |
| PyTorch CUDA | `12.1` |
| torchvision | `0.20.1+cu121` |
| torchaudio | `2.5.1+cu121` |

## 3. 已执行安装

创建虚拟环境并升级 pip：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

第一次尝试安装 PyTorch CUDA 12.8 版：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

问题：

```text
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败。Error loading ... torch\lib\c10.dll
```

处理：卸载 `torch 2.11.0+cu128`、`torchvision 0.26.0+cu128`、`torchaudio 2.11.0+cu128`，切换到更保守的 CUDA 12.1 稳定组合。

最终安装 PyTorch CUDA 12.1 版：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

下载过程说明：`torch` 大包在线下载曾中断，后续用 `curl -C -` 续传到 `wheelhouse`，再从本地 wheel 安装。

最终安装结果：

```text
torch-2.5.1+cu121
torchvision-0.20.1+cu121
torchaudio-2.5.1+cu121
```

基础图像依赖在安装前已存在：

```text
numpy
pillow
opencv-python
scikit-image
matplotlib
tqdm
```

## 4. 验收命令与结果

Python 版本：

```powershell
.\.venv\Scripts\python.exe --version
```

结果：

```text
Python 3.11.7
```

pip 位置：

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

结果：

```text
pip 26.1.2 from D:\大模型论文复现\YUV_Reversible_Attack_2025\.venv\Lib\site-packages\pip (python 3.11)
```

GPU 检查：

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

结果：

```text
name, driver_version, memory.total [MiB]
NVIDIA GeForce RTX 4060 Laptop GPU, 592.27, 8188 MiB
```

PyTorch CUDA 检查：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

结果：

```text
2.5.1+cu121
True
12.1
NVIDIA GeForce RTX 4060 Laptop GPU
```

基础依赖导入检查：

```powershell
.\.venv\Scripts\python.exe -c "import torchvision, PIL, cv2, skimage, numpy, matplotlib, tqdm; print('deps ok'); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__)"
```

结果：

```text
deps ok
```

CUDA 张量计算检查：

```powershell
.\.venv\Scripts\python.exe -c "import torch; x=torch.randn(1024,1024,device='cuda'); y=x@x.T; torch.cuda.synchronize(); print(y.shape); print(torch.cuda.memory_allocated() > 0)"
```

结果：

```text
torch.Size([1024, 1024])
True
```

依赖一致性检查：

```powershell
.\.venv\Scripts\python.exe -m pip check
```

结果：

```text
No broken requirements found.
```

## 5. Git 忽略与已知注意事项

- 后续优先使用：

```powershell
.\.venv\Scripts\python.exe <script.py>
```

或先激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

- `.venv` 已被 Git 忽略。
- `wheelhouse` 已被 Git 忽略，保留为本机 PyTorch wheel 缓存，避免网络中断后反复重下。
- `output\rae` 和 `runs` 已按递归规则忽略，避免实验输出误提交。
- 阶段 4 单图冒烟测试前建议关闭不必要的 GPU 占用程序，给集成模型留出显存。

## 6. 阶段 2 验收结论

阶段 2 通过。当前环境可以进入阶段 3：代码静态梳理与运行前检查。
