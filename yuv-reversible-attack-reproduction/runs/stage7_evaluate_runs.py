from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


RUNS_DIR = Path(__file__).resolve().parent

RUN_CONFIGS = [
    {
        "run_name": "stage7_steps4_iter5",
        "run_dir": "stage7_10img_steps4_iter5",
        "attack_steps": "4",
        "max_iteration_limit": "5",
    },
    {
        "run_name": "stage7_steps20_iter50",
        "run_dir": "stage7_10img_steps20_iter50",
        "attack_steps": "20",
        "max_iteration_limit": "50",
    },
]

PER_IMAGE_FIELDS = [
    "run_name",
    "run_dir",
    "attack_steps",
    "max_iteration_limit",
    "input_filename",
    "label",
    "output_filename",
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
    "run_name",
    "run_dir",
    "attack_steps",
    "max_iteration_limit",
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
]


def read_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def image_metrics(original_path: Path, output_path: Path) -> dict[str, str]:
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


def resolve_output(run_path: Path, output_text: str) -> Path:
    output_path = Path(output_text)
    if output_path.is_absolute():
        return output_path
    return run_path / output_path


def evaluate_run(config: dict[str, str]) -> list[dict[str, str]]:
    run_path = RUNS_DIR / config["run_dir"]
    runtime_path = run_path / "stage7_runtime.csv"
    rows: list[dict[str, str]] = []

    with runtime_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for runtime_row in csv.DictReader(handle):
            input_path = run_path / "ORI_IMG" / runtime_row["input_filename"]
            output_path = resolve_output(run_path, runtime_row["output_image"])
            output_exists = output_path.exists()
            metric_values = {"ssim": "NA", "psnr": "NA", "l2": "NA", "linf": "NA"}
            if input_path.exists() and output_exists:
                metric_values = image_metrics(input_path, output_path)

            rows.append(
                {
                    "run_name": config["run_name"],
                    "run_dir": config["run_dir"],
                    "attack_steps": config["attack_steps"],
                    "max_iteration_limit": config["max_iteration_limit"],
                    "input_filename": runtime_row["input_filename"],
                    "label": runtime_row["label"],
                    "output_filename": output_path.name,
                    "output_exists": "True" if output_exists else "False",
                    "success": runtime_row["success"],
                    "final_pred": runtime_row["final_pred"],
                    "max_iteration_used": runtime_row["max_iteration"],
                    "elapsed_s": runtime_row["elapsed_s"],
                    "output_size_bytes": str(output_path.stat().st_size) if output_exists else "NA",
                    **metric_values,
                }
            )
    return rows


def mean_float(rows: list[dict[str, str]], key: str) -> str:
    values = [float(row[key]) for row in rows if row[key] != "NA"]
    if not values:
        return "NA"
    digits = 4 if key in {"psnr", "l2"} else 6
    return format_float(sum(values) / len(values), digits)


def summarize_run(config: dict[str, str], rows: list[dict[str, str]]) -> dict[str, str]:
    total_images = len(rows)
    outputs = sum(1 for row in rows if row["output_exists"] == "True")
    success_count = sum(1 for row in rows if row["success"] == "1")
    elapsed_values = [float(row["elapsed_s"]) for row in rows if row["elapsed_s"]]
    success_rate = success_count / total_images if total_images else 0.0
    total_elapsed = sum(elapsed_values)
    avg_elapsed = total_elapsed / len(elapsed_values) if elapsed_values else 0.0
    return {
        "run_name": config["run_name"],
        "run_dir": config["run_dir"],
        "attack_steps": config["attack_steps"],
        "max_iteration_limit": config["max_iteration_limit"],
        "total_images": str(total_images),
        "outputs": str(outputs),
        "success_count": str(success_count),
        "success_rate": format_float(success_rate),
        "avg_ssim": mean_float(rows, "ssim"),
        "avg_psnr": mean_float(rows, "psnr"),
        "avg_l2": mean_float(rows, "l2"),
        "avg_linf": mean_float(rows, "linf"),
        "avg_elapsed_s": format_float(avg_elapsed, 2),
        "total_image_elapsed_s": format_float(total_elapsed, 2),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    all_rows: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    for config in RUN_CONFIGS:
        rows = evaluate_run(config)
        all_rows.extend(rows)
        summaries.append(summarize_run(config, rows))

    write_csv(RUNS_DIR / "stage7_10img_per_image_metrics.csv", PER_IMAGE_FIELDS, all_rows)
    write_csv(RUNS_DIR / "stage7_10img_summary.csv", SUMMARY_FIELDS, summaries)
    print(f"Wrote {len(all_rows)} per-image rows and {len(summaries)} summary rows.")


if __name__ == "__main__":
    main()
