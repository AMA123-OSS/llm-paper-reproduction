# Agent Rules

This file contains stable project guidance for AI coding agents working in `D:\大模型论文复现`. Do not put one-off user requests, temporary status, private credentials, rapidly changing experiment results, or copied third-party documentation here.

## Project Authority

- The current main project is the YUV reversible adversarial attack reproduction selected in `最后确认文件.md`.
- Target paper: `Efficient and Transferable Reversible Adversarial Attacks Utilizing YUV Color Space`.
- Target repository: `edu-yinzhaoxia/Efficient-and-Transferable-Reversible-Adversarial-Attacks-Utilizing-YUV-Color-Space`.
- Target local code directory: `YUV_Reversible_Attack_2025/`.
- Older Rt-LRM/LCO files and folders are historical candidate materials. Do not use their commands, metrics, or assumptions unless the user explicitly switches back to that project.

## Tech Stack

- OS: Windows.
- GPU target: NVIDIA RTX 4060.
- Language: Python.
- Core ML framework: PyTorch and torchvision.
- Image libraries: Pillow, OpenCV, scikit-image, NumPy, matplotlib.
- Model source: torchvision pretrained image classification models.
- Default input folder: `ORI_IMG/`.
- Default output folder: `output/rae/`.
- Suggested experiment log folder: `runs/`.

## Environment Rules

- Prefer a project-local virtual environment under `YUV_Reversible_Attack_2025/.venv`.
- Confirm `python` points to a real Python installation, not only the Microsoft Store launcher.
- Confirm CUDA with `torch.cuda.is_available()` before GPU experiments.
- Use the official PyTorch install selector when installing or changing CUDA wheels.
- Do not commit virtual environments, downloaded model weights, generated RAE images, cache folders, or large datasets.
- Keep model weights in the standard PyTorch cache unless the user asks for a custom cache location.

## Key Commands

From the project code directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python reversible_attack_OURS.py
```

Fallback activation-free style:

```powershell
.\.venv\Scripts\python.exe reversible_attack_OURS.py
```

Dependency smoke test:

```powershell
python -c "import torch, torchvision, PIL, cv2, skimage, numpy; print('deps ok')"
```

## Code Style

- Keep Python names in English.
- Use `snake_case` for functions and variables.
- Use `CamelCase` for classes.
- Keep imports grouped as standard library, third-party packages, local modules.
- Prefer small, focused scripts for evaluation and logging instead of heavily rewriting the original algorithm.
- Avoid broad refactors of `reversible_attack_OURS.py`, `atk.py`, `embed_utils.py`, or `EnModel.py` before the baseline run is reproduced.
- Add comments only where they clarify research logic, metric assumptions, or non-obvious image/channel transformations.

## Data And Output Conventions

- Input images should be RGB.
- First-run images should be `299x299` unless the code is updated and tested for other sizes.
- Image filenames should contain an ImageNet label after an underscore, for example `0001_281.png`.
- Keep experiment outputs separated by run directory when comparing parameters.
- Store run commands, environment versions, input image lists, metrics, and notes under `runs/`.
- Do not mix outputs from different attack settings in the same metrics file.

## Known Traps

- The repository README is minimal, so missing setup detail is expected.
- Windows PowerShell may block `.venv\Scripts\Activate.ps1`; use the activation-free Python path if needed.
- `git` may be unavailable on a fresh Windows machine; ZIP download is acceptable.
- GitHub downloads can be unstable; record whether the source came from Git clone or ZIP.
- First execution may download several torchvision model weights and appear slow.
- CPU-only PyTorch will run very slowly and is not the intended setup.
- Integrated models can exceed comfortable 4060 memory; reduce input count, attack steps, or iteration count for smoke tests.
- RGB/BGR confusion can corrupt metrics when mixing Pillow and OpenCV.
- File names without the expected underscore label can break label parsing.
- Saving intermediate images in lossy formats can break exact recovery checks.

## Testing Protocol

- Always run dependency and CUDA checks before the first attack run.
- Start with one image, then scale to 10 images, then larger batches.
- Record every parameter change before comparing results.
- Treat single-image success as a smoke test, not a paper-level reproduction.
- For every reported result, keep the command, input list, output path, model settings, and metrics.
- Verify generated images can be opened before computing metrics.
- Verify restoration with pixel-level comparison when testing error-free recovery.

## Security Boundary

- This project is for local academic reproduction only.
- Do not use generated adversarial examples against third-party, online, commercial, or unauthorized recognition systems.
- Do not add code that automates attacks against external services.
- Do not store credentials or private data in the repository.

