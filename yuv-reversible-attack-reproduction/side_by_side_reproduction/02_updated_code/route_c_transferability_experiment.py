from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atk import atk_MI_YFGSM  # noqa: E402
from recovery_embed import embed_main_with_meta, save_recovery_sidecar  # noqa: E402
from utils import rgb2yuv, yuv2rgb  # noqa: E402


SEED = 2022
ORIGINAL_ENSEMBLE = ["inception_v3", "densenet161", "googlenet"]
HELD_OUT_MODELS = ["vgg16", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny", "vit_b_16"]
TEST_MODELS = ORIGINAL_ENSEMBLE + HELD_OUT_MODELS
GENERATION_VARIANTS = [
    ("single_inception_v3", "single_model", ["inception_v3"]),
    ("single_densenet161", "single_model", ["densenet161"]),
    ("single_googlenet", "single_model", ["googlenet"]),
    ("full_EnModel", "ensemble", ORIGINAL_ENSEMBLE),
    ("leave_one_out_without_inception_v3", "leave_one_out", ["densenet161", "googlenet"]),
    ("leave_one_out_without_densenet161", "leave_one_out", ["inception_v3", "googlenet"]),
    ("leave_one_out_without_googlenet", "leave_one_out", ["inception_v3", "densenet161"]),
]


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_name: str, device: Any) -> Any:
    import torch.nn as nn
    from torchvision import models

    constructors = {
        "inception_v3": lambda: models.inception_v3(pretrained=True),
        "densenet161": lambda: models.densenet161(pretrained=True),
        "googlenet": lambda: models.googlenet(pretrained=True),
        "vgg16": lambda: models.vgg16(pretrained=True),
        "mobilenet_v3_large": lambda: models.mobilenet_v3_large(pretrained=True),
        "efficientnet_b0": lambda: models.efficientnet_b0(pretrained=True),
        "convnext_tiny": lambda: models.convnext_tiny(pretrained=True),
        "vit_b_16": lambda: models.vit_b_16(pretrained=True),
    }
    if model_name not in constructors:
        raise ValueError(f"unknown model: {model_name}")
    model = constructors[model_name]()
    if isinstance(model, nn.Module):
        model = model.to(device)
        model.eval()
    return model


class AveragedEnsemble:
    def __init__(self, models: list[Any]):
        self.models = models

    def zero_grad(self) -> None:
        for model in self.models:
            model.zero_grad()

    def eval(self) -> "AveragedEnsemble":
        for model in self.models:
            model.eval()
        return self

    def __call__(self, inputs: Any) -> Any:
        outputs = [model(inputs) for model in self.models]
        total = outputs[0]
        for output in outputs[1:]:
            total = total + output
        return total / len(outputs)


def relation_for(members: list[str], generation_type: str, test_model: str) -> str:
    if generation_type == "single_model" and test_model in members:
        return "white_box"
    if test_model in members:
        return "ensemble_member"
    return "held_out_black_box"


def read_label(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"filename must end with _<imagenet_label>.png: {path.name}") from exc


def classify(model: Any, image_tensor: Any) -> int:
    import torch

    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs.data, 1)
    return int(pred.item())


def simple_y_attention_mask(original_yuv: np.ndarray, ratio: float = 0.35) -> np.ndarray:
    y_channel = original_yuv[:, :, 0].astype(float)
    gy, gx = np.gradient(y_channel)
    saliency = np.abs(gx) + np.abs(gy)
    threshold = float(np.quantile(saliency, 1.0 - ratio))
    return (saliency >= threshold).astype(np.uint8)


def tensor_from_rgb_array(transform: Any, image_array: np.ndarray, device: Any) -> Any:
    image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255))).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def psnr_from_arrays(original: np.ndarray, candidate: np.ndarray) -> float:
    original_f = original.astype(float) / 255.0
    candidate_f = candidate.astype(float) / 255.0
    mse = float(np.mean(np.square(original_f - candidate_f)))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def global_ssim_fallback(original: np.ndarray, candidate: np.ndarray) -> float:
    original_f = original.astype(float)
    candidate_f = candidate.astype(float)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    values: list[float] = []
    for channel in range(3):
        x = original_f[:, :, channel]
        y = candidate_f[:, :, channel]
        mu_x = float(np.mean(x))
        mu_y = float(np.mean(y))
        var_x = float(np.var(x))
        var_y = float(np.var(y))
        cov_xy = float(np.mean((x - mu_x) * (y - mu_y)))
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        if denominator:
            values.append(((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / denominator)
    return float(sum(values) / len(values)) if values else float("nan")


def image_metrics(original_rgb: np.ndarray, candidate_rgb: np.ndarray) -> dict[str, str]:
    original_f = original_rgb.astype(float) / 255.0
    candidate_f = candidate_rgb.astype(float) / 255.0
    diff = original_f - candidate_f
    try:
        from skimage.metrics import structural_similarity

        ssim = structural_similarity(original_f, candidate_f, channel_axis=-1, data_range=1.0)
    except Exception:
        ssim = global_ssim_fallback(original_rgb, candidate_rgb)
    return {
        "psnr": f"{psnr_from_arrays(original_rgb, candidate_rgb):.4f}",
        "ssim": f"{float(ssim):.6f}",
        "l2": f"{float(np.linalg.norm(diff.ravel())):.4f}",
        "linf": f"{float(np.max(np.abs(diff))):.6f}",
    }


def generate_rae(
    image_path: Path,
    label: int,
    generator_model: Any,
    transform: Any,
    transform_to_tensor: Any,
    device: Any,
    output_dir: Path,
    steps: int,
    eps: float,
    alpha: float,
) -> dict[str, str]:
    import torch
    from torchvision import utils as vutils

    start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir = output_dir / "sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    pil_image = Image.open(image_path).convert("RGB")
    original_rgb = np.asarray(pil_image.resize((299, 299)), dtype=np.uint8)
    image_tensor = transform_to_tensor(pil_image).unsqueeze(0).to(device)
    labels = torch.tensor([label]).to(device)
    original_yuv = rgb2yuv(image_tensor)
    attention_mask = simple_y_attention_mask(original_yuv)

    adv_tensor = atk_MI_YFGSM(
        generator_model,
        eps=eps,
        alpha=alpha,
        steps=steps,
        device=device,
        images=image_tensor,
        labels=labels,
        channel=0,
    )
    adv_yuv = rgb2yuv(adv_tensor)
    advy_yuv = original_yuv.copy()
    for i in range(299):
        for j in range(299):
            if attention_mask[i][j] == 1:
                advy_yuv[:, :, 0][i][j] = adv_yuv[:, :, 0][i][j]

    reversible_uv, metadata = embed_main_with_meta(original_yuv, advy_yuv)
    rae_yuv = advy_yuv.copy()
    rae_yuv[:, :, 1] = reversible_uv[:299, :]
    rae_yuv[:, :, 2] = reversible_uv[299:, :]
    rae_rgb = np.uint8(np.clip(yuv2rgb(rae_yuv), 0, 255))
    rae_tensor = tensor_from_rgb_array(transform, rae_rgb, device)

    output_path = output_dir / image_path.name
    sidecar_path = sidecar_dir / f"{image_path.stem}.recovery.json"
    vutils.save_image(rae_tensor, output_path)
    save_recovery_sidecar(sidecar_path, metadata)
    metrics = image_metrics(original_rgb, rae_rgb)
    return {
        "input_filename": image_path.name,
        "label": str(label),
        "output_image": str(output_path.relative_to(BASE_DIR)),
        "sidecar": str(sidecar_path.relative_to(BASE_DIR)),
        "compressed_bit_length": str(metadata["compressed_bit_length"]),
        "round_count": str(len(metadata["rounds"])),
        "elapsed_s": f"{time.perf_counter() - start:.2f}",
        **metrics,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def confidence_interval_95(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    margin = 1.96 * math.sqrt(p * (1.0 - p) / total)
    return max(0.0, p - margin), min(1.0, p + margin)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    keys = sorted({(row["generation_variant"], row["test_model"], row["relation"]) for row in rows})
    for generation_variant, test_model, relation in keys:
        group = [
            row
            for row in rows
            if row["generation_variant"] == generation_variant
            and row["test_model"] == test_model
            and row["relation"] == relation
        ]
        total = len(group)
        original_correct = sum(1 for row in group if row["original_correct"] == "1")
        successes = sum(1 for row in group if row["attack_success_vs_label"] == "1")
        conditional_successes = sum(
            1 for row in group if row["original_correct"] == "1" and row["attack_success_vs_label"] == "1"
        )
        ci_low, ci_high = confidence_interval_95(successes, total)
        summary_rows.append(
            {
                "generation_variant": generation_variant,
                "test_model": test_model,
                "relation": relation,
                "total_images": str(total),
                "original_correct_count": str(original_correct),
                "attack_success_count_vs_label": str(successes),
                "attack_success_rate_vs_label": f"{successes / total if total else 0.0:.6f}",
                "attack_success_rate_95ci_low": f"{ci_low:.6f}",
                "attack_success_rate_95ci_high": f"{ci_high:.6f}",
                "conditional_success_count": str(conditional_successes),
                "conditional_success_rate_on_original_correct": (
                    f"{conditional_successes / original_correct:.6f}" if original_correct else "0.000000"
                ),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Route C transferability experiment for updated YUV RAE pipeline.")
    parser.add_argument("--image-dir", default=str(PROJECT_ROOT / "ORI_IMG"))
    parser.add_argument("--image-count", type=int, default=10, choices=[10, 100])
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--eps", type=float, default=2 / 255)
    parser.add_argument("--alpha", type=float, default=1 / 255)
    parser.add_argument("--variants", default="all", help="comma-separated generation variants, or all")
    args = parser.parse_args()

    import torch
    from torchvision import transforms

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted(Path(args.image_dir).glob("*.png"))[: args.image_count]
    if len(image_paths) < args.image_count:
        raise FileNotFoundError(
            f"Need {args.image_count} PNG images in {args.image_dir}, found {len(image_paths)}"
        )

    selected = {value.strip() for value in args.variants.split(",")} if args.variants != "all" else None
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    transform_to_tensor = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])

    all_eval_rows: list[dict[str, str]] = []
    generation_rows: list[dict[str, str]] = []
    test_model_cache: dict[str, Any] = {}

    for generation_variant, generation_type, members in GENERATION_VARIANTS:
        if selected is not None and generation_variant not in selected:
            continue
        print(f"loading generator {generation_variant}: {members}", flush=True)
        member_models = [load_model(model_name, device) for model_name in members]
        generator_model = member_models[0] if len(member_models) == 1 else AveragedEnsemble(member_models).eval()
        output_dir = RESULTS_DIR / "route_c_outputs" / f"{args.image_count}img" / generation_variant

        for image_path in image_paths:
            label = read_label(image_path)
            row = generate_rae(
                image_path=image_path,
                label=label,
                generator_model=generator_model,
                transform=transform,
                transform_to_tensor=transform_to_tensor,
                device=device,
                output_dir=output_dir,
                steps=args.steps,
                eps=args.eps,
                alpha=args.alpha,
            )
            row.update(
                {
                    "generation_variant": generation_variant,
                    "generation_type": generation_type,
                    "generator_members": "+".join(members),
                }
            )
            generation_rows.append(row)

        for test_model_name in TEST_MODELS:
            if test_model_name not in test_model_cache:
                print(f"loading evaluator {test_model_name}", flush=True)
                test_model_cache[test_model_name] = load_model(test_model_name, device)
            evaluator = test_model_cache[test_model_name]
            for gen_row in [row for row in generation_rows if row["generation_variant"] == generation_variant]:
                image_path = Path(args.image_dir) / gen_row["input_filename"]
                output_path = BASE_DIR / gen_row["output_image"]
                label = int(gen_row["label"])
                original_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                adv_tensor = transform(Image.open(output_path).convert("RGB")).unsqueeze(0).to(device)
                original_pred = classify(evaluator, original_tensor)
                adv_pred = classify(evaluator, adv_tensor)
                all_eval_rows.append(
                    {
                        "generation_variant": generation_variant,
                        "generation_type": generation_type,
                        "generator_members": "+".join(members),
                        "test_model": test_model_name,
                        "relation": relation_for(members, generation_type, test_model_name),
                        "input_filename": gen_row["input_filename"],
                        "label": str(label),
                        "original_pred": str(original_pred),
                        "original_correct": "1" if original_pred == label else "0",
                        "adv_pred": str(adv_pred),
                        "attack_success_vs_label": "1" if adv_pred != label else "0",
                        "changed_from_original": "1" if adv_pred != original_pred else "0",
                        "output_image": gen_row["output_image"],
                        "sidecar": gen_row["sidecar"],
                        "psnr": gen_row["psnr"],
                        "ssim": gen_row["ssim"],
                        "l2": gen_row["l2"],
                        "linf": gen_row["linf"],
                    }
                )

        for model in member_models:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(RESULTS_DIR / f"route_c_{args.image_count}img_generation.csv", generation_rows)
    write_csv(RESULTS_DIR / f"route_c_{args.image_count}img_per_image.csv", all_eval_rows)
    write_csv(RESULTS_DIR / f"route_c_{args.image_count}img_summary.csv", summarize(all_eval_rows))
    print(f"wrote Route C results for {len(image_paths)} images to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
