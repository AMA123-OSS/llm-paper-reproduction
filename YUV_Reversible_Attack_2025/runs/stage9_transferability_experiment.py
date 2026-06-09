from __future__ import annotations

import csv
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import models
from torchvision import transforms
from torchvision import utils as vutils


RUNS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atk import atk_MI_YFGSM  # noqa: E402
from embed_utils import embed_main  # noqa: E402
from pytorch_grad_cam import CAM  # noqa: E402
from utils import rgb2yuv, yuv2rgb  # noqa: E402


SEED = 2022
STEPS = 4
MAX_ITERATION = 5
EPS = 2 / 255
ALPHA = 1 / 255
IMAGE_LIMIT = 10

OUTPUT_ROOT = RUNS_DIR / "stage9_transferability"
SINGLE_OUTPUT_DIR = OUTPUT_ROOT / "single_inception_y_uv" / "output"
ENSEMBLE_OUTPUT_DIR = OUTPUT_ROOT / "ensemble_y_uv_from_stage8" / "output"
ENSEMBLE_SOURCE_DIR = RUNS_DIR / "stage8_ablation" / "y_uv_embed" / "output"

INPUT_LIST_CSV = RUNS_DIR / "stage9_transferability_input_list.csv"
GENERATION_CSV = RUNS_DIR / "stage9_single_inception_generation.csv"
PER_IMAGE_CSV = RUNS_DIR / "stage9_transferability_per_image.csv"
SUMMARY_CSV = RUNS_DIR / "stage9_transferability_summary.csv"
MODEL_VERSION_CSV = RUNS_DIR / "stage9_model_versions.csv"

TEST_MODELS = ["inception_v3", "resnet50", "densenet161", "googlenet"]
GENERATOR_OUTPUTS = [
    {
        "generator": "single_inception_y_uv",
        "output_dir": SINGLE_OUTPUT_DIR,
        "generator_desc": "Y+UV generated with single Inception v3",
    },
    {
        "generator": "ensemble_y_uv",
        "output_dir": ENSEMBLE_OUTPUT_DIR,
        "generator_desc": "Y+UV generated with EnModel, copied from stage8",
    },
]

GEN_FIELDS = [
    "generator",
    "input_filename",
    "label",
    "output_image",
    "success_on_inception",
    "final_pred",
    "max_iteration_used",
    "elapsed_s",
    "output_size_bytes",
]

PER_IMAGE_FIELDS = [
    "generator",
    "generator_desc",
    "test_model",
    "relation",
    "input_filename",
    "label",
    "original_pred",
    "original_correct",
    "adv_pred",
    "attack_success_vs_label",
    "changed_from_original",
    "output_image",
]

SUMMARY_FIELDS = [
    "generator",
    "test_model",
    "relation",
    "total_images",
    "outputs",
    "original_correct_count",
    "attack_success_count_vs_label",
    "attack_success_rate_vs_label",
    "conditional_success_count",
    "conditional_success_rate_on_original_correct",
    "changed_from_original_count",
    "changed_from_original_rate",
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


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def tensor_to_image_tensor(transform: transforms.Compose, image_array: np.ndarray, device: torch.device) -> torch.Tensor:
    image = Image.fromarray(np.uint8(image_array))
    return transform(image).unsqueeze(0).to(device)


def classify(model: torch.nn.Module, image_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs.data, 1)
    return int(pred.item())


def cam_map(cam: CAM, image: torch.Tensor) -> np.ndarray:
    return cam(input_tensor=image, method="gradcam++", target_category=None)


def binarize_cam(cam_array: np.ndarray) -> np.ndarray:
    return (cam_array > 0.5).astype(np.uint8)


def build_y_uv_candidate(
    generator_model: torch.nn.Module,
    images: torch.Tensor,
    original_yuv: np.ndarray,
    attention_mask: np.ndarray,
    transform: transforms.Compose,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    adv_images = atk_MI_YFGSM(
        generator_model, eps=EPS, alpha=ALPHA, steps=STEPS, device=device, images=images, labels=labels
    )
    adv_yuv = rgb2yuv(adv_images)
    advy_yuv = original_yuv.copy()
    for i in range(299):
        for j in range(299):
            if attention_mask[i][j] == 1:
                advy_yuv[:, :, 0][i][j] = adv_yuv[:, :, 0][i][j]

    advy_tensor = tensor_to_image_tensor(transform, yuv2rgb(advy_yuv), device)
    reversible_yuv = embed_main(original_yuv, advy_yuv)
    reversible_u = reversible_yuv[:299, :]
    reversible_v = reversible_yuv[299:, :]
    rae_yuv = advy_yuv.copy()
    rae_yuv[:, :, 1] = reversible_u
    rae_yuv[:, :, 2] = reversible_v
    reversible_tensor = tensor_to_image_tensor(transform, yuv2rgb(rae_yuv), device)
    return reversible_tensor, advy_tensor


def load_test_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if model_name == "inception_v3":
        model = models.inception_v3(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "densenet161":
        model = models.densenet161(pretrained=True)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)
    model.eval()
    return model


def relation_for(generator: str, test_model: str) -> str:
    if generator == "single_inception_y_uv":
        return "white_box" if test_model == "inception_v3" else "transfer_black_box"
    if test_model == "resnet50":
        return "held_out_black_box"
    if test_model in {"inception_v3", "densenet161", "googlenet"}:
        return "ensemble_member"
    return "unknown"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_ensemble_outputs(image_paths: list[Path]) -> None:
    ENSEMBLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        source = ENSEMBLE_SOURCE_DIR / image_path.name
        target = ENSEMBLE_OUTPUT_DIR / image_path.name
        if not source.exists():
            raise FileNotFoundError(f"Missing stage8 ensemble output: {source}")
        shutil.copy2(source, target)


def generate_single_inception_outputs(
    image_paths: list[Path],
    transform: transforms.Compose,
    transform_to_tensor: transforms.Compose,
    device: torch.device,
) -> list[dict[str, str]]:
    SINGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading single Inception generator and CAM model...", flush=True)
    generator_model = models.inception_v3(pretrained=True).to(device)
    generator_model.eval()
    cam_model = models.resnet50(pretrained=True).to(device)
    cam_model.eval()
    cam = CAM(model=cam_model, target_layer=cam_model.layer4[-1], use_cuda=(device.type == "cuda"))

    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        start = time.perf_counter()
        images = Image.open(image_path).convert("RGB")
        images = transform_to_tensor(images).unsqueeze(0).to(device)
        label = int(image_path.stem.split("_")[1])
        labels = torch.tensor([label]).to(device)
        original_yuv = rgb2yuv(images)
        attention_mask = binarize_cam(cam_map(cam, images))

        success = False
        final_pred = None
        max_iteration_used = 0
        candidate_tensor = images
        update_tensor = images
        for max_iteration in range(MAX_ITERATION):
            candidate_tensor, update_tensor = build_y_uv_candidate(
                generator_model=generator_model,
                images=images,
                original_yuv=original_yuv,
                attention_mask=attention_mask,
                transform=transform,
                labels=labels,
                device=device,
            )
            final_pred = classify(generator_model, candidate_tensor)
            max_iteration_used = max_iteration
            if final_pred != label:
                success = True
                break
            images = update_tensor

        output_path = SINGLE_OUTPUT_DIR / image_path.name
        vutils.save_image(candidate_tensor, output_path)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "generator": "single_inception_y_uv",
                "input_filename": image_path.name,
                "label": str(label),
                "output_image": str(output_path.relative_to(RUNS_DIR)),
                "success_on_inception": "1" if success else "0",
                "final_pred": str(final_pred),
                "max_iteration_used": str(max_iteration_used),
                "elapsed_s": format_float(elapsed, 2),
                "output_size_bytes": str(output_path.stat().st_size),
            }
        )
        print(
            f"single_inception_y_uv: {image_path.name}, success={int(success)}, "
            f"pred={final_pred}, iter={max_iteration_used}, elapsed={elapsed:.2f}s",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del generator_model
    del cam_model
    del cam
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def evaluate_transfer(
    image_paths: list[Path],
    transform: transforms.Compose,
    device: torch.device,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model_name in TEST_MODELS:
        print(f"evaluating {model_name}...", flush=True)
        model = load_test_model(model_name, device)
        original_preds: dict[str, int] = {}
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            original_preds[image_path.name] = classify(model, image_tensor)

        for generator_config in GENERATOR_OUTPUTS:
            generator = str(generator_config["generator"])
            output_dir = Path(generator_config["output_dir"])
            for image_path in image_paths:
                label = int(image_path.stem.split("_")[1])
                output_path = output_dir / image_path.name
                image = Image.open(output_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)
                adv_pred = classify(model, image_tensor)
                original_pred = original_preds[image_path.name]
                rows.append(
                    {
                        "generator": generator,
                        "generator_desc": str(generator_config["generator_desc"]),
                        "test_model": model_name,
                        "relation": relation_for(generator, model_name),
                        "input_filename": image_path.name,
                        "label": str(label),
                        "original_pred": str(original_pred),
                        "original_correct": "1" if original_pred == label else "0",
                        "adv_pred": str(adv_pred),
                        "attack_success_vs_label": "1" if adv_pred != label else "0",
                        "changed_from_original": "1" if adv_pred != original_pred else "0",
                        "output_image": str(output_path.relative_to(RUNS_DIR)),
                    }
                )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    keys = sorted({(row["generator"], row["test_model"], row["relation"]) for row in rows})
    for generator, test_model, relation in keys:
        group = [
            row for row in rows
            if row["generator"] == generator and row["test_model"] == test_model and row["relation"] == relation
        ]
        total = len(group)
        outputs = total
        original_correct = sum(1 for row in group if row["original_correct"] == "1")
        attack_success = sum(1 for row in group if row["attack_success_vs_label"] == "1")
        conditional_success = sum(
            1 for row in group
            if row["original_correct"] == "1" and row["attack_success_vs_label"] == "1"
        )
        changed = sum(1 for row in group if row["changed_from_original"] == "1")
        summaries.append(
            {
                "generator": generator,
                "test_model": test_model,
                "relation": relation,
                "total_images": str(total),
                "outputs": str(outputs),
                "original_correct_count": str(original_correct),
                "attack_success_count_vs_label": str(attack_success),
                "attack_success_rate_vs_label": format_float(attack_success / total if total else 0.0),
                "conditional_success_count": str(conditional_success),
                "conditional_success_rate_on_original_correct": format_float(
                    conditional_success / original_correct if original_correct else 0.0
                ),
                "changed_from_original_count": str(changed),
                "changed_from_original_rate": format_float(changed / total if total else 0.0),
            }
        )
    return summaries


def write_metadata() -> None:
    rows = [
        {
            "item": "torch",
            "version": torch.__version__,
            "weights": "N/A",
            "preprocess": "Resize((299,299)) + ToTensor(), no normalization, code-consistent",
        },
        {
            "item": "torchvision",
            "version": torchvision.__version__,
            "weights": "pretrained=True, torchvision legacy IMAGENET1K equivalent",
            "preprocess": "Resize((299,299)) + ToTensor(), no normalization, code-consistent",
        },
    ]
    write_csv(MODEL_VERSION_CSV, ["item", "version", "weights", "preprocess"], rows)


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = sorted((PROJECT_ROOT / "ORI_IMG").glob("*.png"))[:IMAGE_LIMIT]
    transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])

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
    write_metadata()

    print(f"device={device}", flush=True)
    copy_ensemble_outputs(image_paths)
    generation_rows = generate_single_inception_outputs(image_paths, transform, transform_to_tensor, device)
    write_csv(GENERATION_CSV, GEN_FIELDS, generation_rows)

    transfer_rows = evaluate_transfer(image_paths, transform, device)
    write_csv(PER_IMAGE_CSV, PER_IMAGE_FIELDS, transfer_rows)
    summary_rows = summarize(transfer_rows)
    write_csv(SUMMARY_CSV, SUMMARY_FIELDS, summary_rows)
    print(f"wrote {len(transfer_rows)} transfer rows to {PER_IMAGE_CSV}", flush=True)
    print(f"wrote {len(summary_rows)} summary rows to {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
