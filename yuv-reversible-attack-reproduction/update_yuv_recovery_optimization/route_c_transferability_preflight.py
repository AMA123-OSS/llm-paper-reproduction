from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
PREFLIGHT_CSV = RESULTS_DIR / "route_c_preflight.csv"
MATRIX_CSV = RESULTS_DIR / "route_c_experiment_matrix.csv"

ORIGINAL_ENSEMBLE = ["inception_v3", "densenet161", "googlenet"]
HELD_OUT_MODELS = ["vgg16", "mobilenet_v3_large", "efficientnet_b0", "convnext_tiny", "vit_b_16"]
TEST_MODELS = ORIGINAL_ENSEMBLE + HELD_OUT_MODELS

GENERATION_VARIANTS = [
    {
        "generation_variant": "single_inception_v3",
        "generation_type": "single_model",
        "members": ["inception_v3"],
    },
    {
        "generation_variant": "single_densenet161",
        "generation_type": "single_model",
        "members": ["densenet161"],
    },
    {
        "generation_variant": "single_googlenet",
        "generation_type": "single_model",
        "members": ["googlenet"],
    },
    {
        "generation_variant": "full_EnModel",
        "generation_type": "ensemble",
        "members": ORIGINAL_ENSEMBLE,
    },
    {
        "generation_variant": "leave_one_out_without_inception_v3",
        "generation_type": "leave_one_out",
        "members": ["densenet161", "googlenet"],
    },
    {
        "generation_variant": "leave_one_out_without_densenet161",
        "generation_type": "leave_one_out",
        "members": ["inception_v3", "googlenet"],
    },
    {
        "generation_variant": "leave_one_out_without_googlenet",
        "generation_type": "leave_one_out",
        "members": ["inception_v3", "densenet161"],
    },
]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def relation_for(members: list[str], generation_type: str, test_model: str) -> str:
    if generation_type == "single_model" and test_model in members:
        return "white_box"
    if test_model in members:
        return "ensemble_member"
    return "held_out_black_box"


def preflight_rows() -> list[dict[str, str]]:
    ori_img_dir = PROJECT_ROOT / "ORI_IMG"
    ori_images = sorted(ori_img_dir.glob("*.png")) if ori_img_dir.exists() else []
    torch_available = module_available("torch")
    torchvision_available = module_available("torchvision")
    cuda_value = "unknown"
    if torch_available:
        try:
            import torch

            cuda_value = str(torch.cuda.is_available())
        except Exception as exc:
            cuda_value = f"import_failed: {exc}"

    rows = [
        {
            "item": "ORI_IMG",
            "status": "ok" if len(ori_images) >= 10 else "blocked",
            "value": str(ori_img_dir),
            "observed": f"{len(ori_images)} png images",
            "required_action": "restore at least 10 labeled ImageNet-style PNG files, 100 for full confidence interval run",
        },
        {
            "item": "python",
            "status": "ok",
            "value": sys.executable,
            "observed": sys.version.split()[0],
            "required_action": "none",
        },
        {
            "item": "torch",
            "status": "ok" if torch_available else "blocked",
            "value": str(torch_available),
            "observed": "available" if torch_available else "missing",
            "required_action": "install torch matching CUDA 4060 environment" if not torch_available else "none",
        },
        {
            "item": "torchvision",
            "status": "ok" if torchvision_available else "blocked",
            "value": str(torchvision_available),
            "observed": "available" if torchvision_available else "missing",
            "required_action": "install torchvision model zoo support" if not torchvision_available else "none",
        },
        {
            "item": "cuda",
            "status": "ok" if cuda_value == "True" else "warning",
            "value": cuda_value,
            "observed": "cuda availability checked without loading model weights",
            "required_action": "GPU is recommended for 10/100 image transferability runs",
        },
    ]
    return rows


def matrix_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for image_count in [10, 100]:
        for generator in GENERATION_VARIANTS:
            members = list(generator["members"])
            generation_type = str(generator["generation_type"])
            for test_model in TEST_MODELS:
                relation = relation_for(members, generation_type, test_model)
                rows.append(
                    {
                        "image_count": str(image_count),
                        "generation_variant": str(generator["generation_variant"]),
                        "generation_type": generation_type,
                        "generator_members": "+".join(members),
                        "test_model": test_model,
                        "relation": relation,
                        "attack": "MI-YFGSM in Y channel + reversible UV embedding",
                        "metrics": "success_rate, conditional_success_rate, PSNR, SSIM, L2, L_inf, 95% CI",
                        "planned_output": (
                            f"route_c_{image_count}img_"
                            f"{generator['generation_variant']}_{test_model}.csv"
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    preflight = preflight_rows()
    matrix = matrix_rows()
    write_csv(PREFLIGHT_CSV, preflight)
    write_csv(MATRIX_CSV, matrix)
    for row in preflight:
        print(row)
    print(f"wrote {len(matrix)} route C matrix rows to {MATRIX_CSV}")


if __name__ == "__main__":
    main()
