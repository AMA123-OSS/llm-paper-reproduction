from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import models
from torchvision import transforms
from torchvision import utils as vutils


RUNS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atk import atk_MI_YFGSM  # noqa: E402
from embed_utils import embed_main  # noqa: E402
from EnModel import EnModel  # noqa: E402
from pytorch_grad_cam import CAM  # noqa: E402
from utils import rgb2yuv, yuv2rgb  # noqa: E402


SEED = 2022
STEPS = 4
MAX_ITERATION = 5
EPS = 2 / 255
ALPHA = 1 / 255
IMAGE_LIMIT = 10

OUTPUT_ROOT = RUNS_DIR / "stage8_ablation"
PER_IMAGE_CSV = RUNS_DIR / "stage8_ablation_per_image_metrics.csv"
SUMMARY_CSV = RUNS_DIR / "stage8_ablation_summary.csv"
INPUT_LIST_CSV = RUNS_DIR / "stage8_ablation_input_list.csv"

METHODS = [
    {
        "method": "rgb_mi_fgsm",
        "display_name": "RGB MI-FGSM baseline",
        "output_dir": OUTPUT_ROOT / "rgb_mi_fgsm" / "output",
    },
    {
        "method": "y_only",
        "display_name": "Y-channel only",
        "output_dir": OUTPUT_ROOT / "y_only" / "output",
    },
    {
        "method": "y_uv_embed",
        "display_name": "Y-channel + UV embedding",
        "output_dir": OUTPUT_ROOT / "y_uv_embed" / "output",
    },
]

PER_IMAGE_FIELDS = [
    "method",
    "display_name",
    "input_filename",
    "label",
    "output_image",
    "output_exists",
    "success",
    "final_pred",
    "max_iteration_used",
    "elapsed_s",
    "ssim",
    "psnr",
    "l2",
    "linf",
    "output_size_bytes",
]

SUMMARY_FIELDS = [
    "method",
    "display_name",
    "total_images",
    "outputs",
    "success_count",
    "success_rate",
    "avg_ssim",
    "avg_psnr",
    "avg_l2",
    "avg_linf",
    "avg_elapsed_s",
    "total_image_elapsed_s",
    "steps",
    "max_iteration",
    "eps",
    "alpha",
    "seed",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def compute_metrics(original_path: Path, output_path: Path) -> dict[str, str]:
    original = read_rgb_float(original_path)
    output = read_rgb_float(output_path)
    diff = original - output
    return {
        "ssim": format_float(
            structural_similarity(original, output, channel_axis=-1, data_range=1.0)
        ),
        "psnr": format_float(peak_signal_noise_ratio(original, output, data_range=1.0), 4),
        "l2": format_float(float(np.linalg.norm(diff.ravel())), 4),
        "linf": format_float(float(np.max(np.abs(diff))), 6),
    }


def tensor_to_image_tensor(transform: transforms.Compose, image_array: np.ndarray, device: torch.device) -> torch.Tensor:
    image = Image.fromarray(np.uint8(image_array))
    return transform(image).unsqueeze(0).to(device)


def atk_mi_fgsm_rgb(
    model: nn.Module,
    device: torch.device,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    mu: float = 1.0,
) -> torch.Tensor:
    loss = nn.CrossEntropyLoss()
    original = images.detach().to(device)
    adv = original.clone().detach()
    momentum = torch.zeros_like(adv).detach().to(device)

    for _ in range(steps):
        adv.requires_grad = True
        output = model(adv)
        cost = loss(output, labels)
        model.zero_grad()
        cost.backward()
        grad = adv.grad
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True) + 1e-8
        momentum = mu * momentum + grad / grad_norm
        perturbed = adv.detach() + alpha * torch.sign(momentum)
        delta = torch.clamp(perturbed - original, min=-eps, max=eps)
        adv = torch.clamp(original + delta, min=0, max=1).detach()
    return adv


def classify(model: nn.Module, image_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs.data, 1)
    return int(pred.item())


def cam_map(cam: CAM, image: torch.Tensor) -> np.ndarray:
    return cam(input_tensor=image, method="gradcam++", target_category=None)


def binarize_cam(cam_array: np.ndarray) -> np.ndarray:
    return (cam_array > 0.5).astype(np.uint8)


def build_candidate(
    method: str,
    images: torch.Tensor,
    original_yuv: np.ndarray,
    attention_mask: np.ndarray | None,
    ens_model: nn.Module,
    transform: transforms.Compose,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if method == "rgb_mi_fgsm":
        candidate = atk_mi_fgsm_rgb(
            ens_model, device=device, images=images, labels=labels, eps=EPS, alpha=ALPHA, steps=STEPS
        )
        return candidate, candidate

    adv_images = atk_MI_YFGSM(
        ens_model, eps=EPS, alpha=ALPHA, steps=STEPS, device=device, images=images, labels=labels
    )
    adv_yuv = rgb2yuv(adv_images)
    advy_yuv = original_yuv.copy()
    for i in range(299):
        for j in range(299):
            if attention_mask[i][j] == 1:
                advy_yuv[:, :, 0][i][j] = adv_yuv[:, :, 0][i][j]

    advy_tensor = tensor_to_image_tensor(transform, yuv2rgb(advy_yuv), device)
    if method == "y_only":
        return advy_tensor, advy_tensor

    reversible_yuv = embed_main(original_yuv, advy_yuv)
    reversible_u = reversible_yuv[:299, :]
    reversible_v = reversible_yuv[299:, :]
    rae_yuv = advy_yuv.copy()
    rae_yuv[:, :, 1] = reversible_u
    rae_yuv[:, :, 2] = reversible_v
    reversible_tensor = tensor_to_image_tensor(transform, yuv2rgb(rae_yuv), device)
    return reversible_tensor, advy_tensor


def run_method(
    method_config: dict[str, object],
    image_paths: list[Path],
    model: nn.Module,
    ens_model: nn.Module,
    cam: CAM,
    transform: transforms.Compose,
    transform_to_tensor: transforms.Compose,
    device: torch.device,
) -> list[dict[str, str]]:
    method = str(method_config["method"])
    output_dir = Path(method_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for image_path in image_paths:
        start = time.perf_counter()
        images = Image.open(image_path).convert("RGB")
        images = transform_to_tensor(images).unsqueeze(0).to(device)
        label = int(image_path.stem.split("_")[1])
        labels = torch.tensor([label]).to(device)
        original_yuv = rgb2yuv(images)
        attention_mask = None
        if method in {"y_only", "y_uv_embed"}:
            attention_mask = binarize_cam(cam_map(cam, images))

        success = False
        final_pred = None
        max_iteration_used = 0
        candidate_tensor = images
        update_tensor = images
        for max_iteration in range(MAX_ITERATION):
            candidate_tensor, update_tensor = build_candidate(
                method=method,
                images=images,
                original_yuv=original_yuv,
                attention_mask=attention_mask,
                ens_model=ens_model,
                transform=transform,
                labels=labels,
                device=device,
            )
            final_pred = classify(model, candidate_tensor)
            max_iteration_used = max_iteration
            if final_pred != label:
                success = True
                break
            images = update_tensor

        output_path = output_dir / image_path.name
        vutils.save_image(candidate_tensor, output_path)
        metrics = compute_metrics(image_path, output_path)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "method": method,
                "display_name": str(method_config["display_name"]),
                "input_filename": image_path.name,
                "label": str(label),
                "output_image": str(output_path.relative_to(RUNS_DIR)),
                "output_exists": "True" if output_path.exists() else "False",
                "success": "1" if success else "0",
                "final_pred": str(final_pred),
                "max_iteration_used": str(max_iteration_used),
                "elapsed_s": format_float(elapsed, 2),
                "output_size_bytes": str(output_path.stat().st_size) if output_path.exists() else "NA",
                **metrics,
            }
        )
        print(
            f"{method}: {image_path.name}, success={int(success)}, pred={final_pred}, "
            f"iter={max_iteration_used}, elapsed={elapsed:.2f}s",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def mean_float(rows: list[dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row[key] != "NA"]
    if not values:
        return "NA"
    digits = 4 if key in {"psnr", "l2"} else 6
    return format_float(sum(values) / len(values), digits)


def summarize(method_config: dict[str, object], rows: list[dict[str, str]]) -> dict[str, str]:
    total = len(rows)
    outputs = sum(1 for row in rows if row["output_exists"] == "True")
    successes = sum(1 for row in rows if row["success"] == "1")
    elapsed_values = [float(row["elapsed_s"]) for row in rows if row["elapsed_s"]]
    total_elapsed = sum(elapsed_values)
    avg_elapsed = total_elapsed / len(elapsed_values) if elapsed_values else 0.0
    return {
        "method": str(method_config["method"]),
        "display_name": str(method_config["display_name"]),
        "total_images": str(total),
        "outputs": str(outputs),
        "success_count": str(successes),
        "success_rate": format_float(successes / total if total else 0.0),
        "avg_ssim": mean_float(rows, "ssim"),
        "avg_psnr": mean_float(rows, "psnr"),
        "avg_l2": mean_float(rows, "l2"),
        "avg_linf": mean_float(rows, "linf"),
        "avg_elapsed_s": format_float(avg_elapsed, 2),
        "total_image_elapsed_s": format_float(total_elapsed, 2),
        "steps": str(STEPS),
        "max_iteration": str(MAX_ITERATION),
        "eps": format_float(EPS),
        "alpha": format_float(ALPHA),
        "seed": str(SEED),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted((PROJECT_ROOT / "ORI_IMG").glob("*.png"))[:IMAGE_LIMIT]
    write_csv(
        INPUT_LIST_CSV,
        ["input_filename", "label", "size", "mode"],
        [
            {
                "input_filename": path.name,
                "label": path.stem.split("_")[1],
                "size": "299x299",
                "mode": "RGB",
            }
            for path in image_paths
        ],
    )

    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])

    print(f"device={device}", flush=True)
    print("loading models...", flush=True)
    model = models.inception_v3(pretrained=True).to(device)
    model.eval()
    ens_model = EnModel().to(device)
    ens_model.eval()
    cam_model = models.resnet50(pretrained=True).to(device)
    cam_model.eval()
    target_layer = cam_model.layer4[-1]
    cam = CAM(model=cam_model, target_layer=target_layer, use_cuda=(device.type == "cuda"))

    all_rows: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    for method_config in METHODS:
        print(f"running {method_config['method']}...", flush=True)
        rows = run_method(
            method_config=method_config,
            image_paths=image_paths,
            model=model,
            ens_model=ens_model,
            cam=cam,
            transform=transform,
            transform_to_tensor=transform_to_tensor,
            device=device,
        )
        all_rows.extend(rows)
        summaries.append(summarize(method_config, rows))

    write_csv(PER_IMAGE_CSV, PER_IMAGE_FIELDS, all_rows)
    write_csv(SUMMARY_CSV, SUMMARY_FIELDS, summaries)
    print(f"wrote {len(all_rows)} per-image rows to {PER_IMAGE_CSV}", flush=True)
    print(f"wrote {len(summaries)} summary rows to {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
