from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


RUNS_DIR = Path(__file__).resolve().parent

FIELDNAMES = [
    "run_name",
    "phase",
    "attack_steps",
    "max_iteration",
    "elapsed_s",
    "success_count",
    "total_images",
    "success_rate",
    "attack_success",
    "input_image",
    "output_image",
    "restored_image",
    "output_exists",
    "ssim",
    "psnr",
    "l2",
    "linf",
    "recovery_checked",
    "recovery_image_exists",
    "recovery_exact",
    "recovery_max_abs_error",
    "recovery_error_pixels",
    "output_size_bytes",
    "notes",
]


@dataclass(frozen=True)
class RunRecord:
    run_name: str
    phase: str
    attack_steps: str
    max_iteration: str
    elapsed_s: str
    success_count: str
    total_images: str
    input_image: str
    output_image: str
    restored_image: str = ""
    notes: str = ""


DEFAULT_RECORDS = [
    RunRecord(
        run_name="baseline_steps20_iter50",
        phase="stage4",
        attack_steps="20",
        max_iteration="50",
        elapsed_s="23.19",
        success_count="1",
        total_images="1",
        input_image="stage4_smoke_single_0001/ORI_IMG/0001_321.png",
        output_image="stage4_smoke_single_0001/output/rae/0001_321.png",
        notes="stage4 smoke baseline; root algorithm unchanged",
    ),
    RunRecord(
        run_name="steps1_iter5",
        phase="stage5",
        attack_steps="1",
        max_iteration="5",
        elapsed_s="10.10",
        success_count="0",
        total_images="1",
        input_image="stage5_lowcost_single_0001_steps1_iter5/ORI_IMG/0001_321.png",
        output_image="stage5_lowcost_single_0001_steps1_iter5/output/rae/0001_321.png",
        notes="low-cost probe; generated image but attack failed",
    ),
    RunRecord(
        run_name="steps2_iter5",
        phase="stage5",
        attack_steps="2",
        max_iteration="5",
        elapsed_s="10.11",
        success_count="0",
        total_images="1",
        input_image="stage5_lowcost_single_0001_steps2_iter5/ORI_IMG/0001_321.png",
        output_image="stage5_lowcost_single_0001_steps2_iter5/output/rae/0001_321.png",
        notes="low-cost probe; generated image but attack failed",
    ),
    RunRecord(
        run_name="steps3_iter5",
        phase="stage5",
        attack_steps="3",
        max_iteration="5",
        elapsed_s="11.11",
        success_count="0",
        total_images="1",
        input_image="stage5_lowcost_single_0001_steps3_iter5/ORI_IMG/0001_321.png",
        output_image="stage5_lowcost_single_0001_steps3_iter5/output/rae/0001_321.png",
        notes="low-cost probe; generated image but attack failed",
    ),
    RunRecord(
        run_name="steps4_iter5",
        phase="stage5",
        attack_steps="4",
        max_iteration="5",
        elapsed_s="12.12",
        success_count="1",
        total_images="1",
        input_image="stage5_lowcost_single_0001_steps4_iter5/ORI_IMG/0001_321.png",
        output_image="stage5_lowcost_single_0001_steps4_iter5/output/rae/0001_321.png",
        notes="current lowest effective single-image candidate",
    ),
    RunRecord(
        run_name="steps5_iter5",
        phase="stage5",
        attack_steps="5",
        max_iteration="5",
        elapsed_s="12.12",
        success_count="1",
        total_images="1",
        input_image="stage5_lowcost_single_0001_steps5_iter5/ORI_IMG/0001_321.png",
        output_image="stage5_lowcost_single_0001_steps5_iter5/output/rae/0001_321.png",
        notes="low-cost successful comparison",
    ),
]


def resolve_run_path(path_text: str) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = RUNS_DIR / path
    return path


def read_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def read_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def success_rate(success_count: str, total_images: str) -> str:
    try:
        success = int(success_count)
        total = int(total_images)
    except ValueError:
        return "NA"
    if total == 0:
        return "NA"
    return format_float(success / total)


def evaluate_record(record: RunRecord) -> dict[str, str]:
    input_path = resolve_run_path(record.input_image)
    output_path = resolve_run_path(record.output_image)
    restored_path = resolve_run_path(record.restored_image)
    row = {
        "run_name": record.run_name,
        "phase": record.phase,
        "attack_steps": record.attack_steps,
        "max_iteration": record.max_iteration,
        "elapsed_s": record.elapsed_s,
        "success_count": record.success_count,
        "total_images": record.total_images,
        "success_rate": success_rate(record.success_count, record.total_images),
        "attack_success": "True" if success_rate(record.success_count, record.total_images) == "1.000000" else "False",
        "input_image": record.input_image,
        "output_image": record.output_image,
        "restored_image": record.restored_image or "NA",
        "output_exists": "False",
        "ssim": "NA",
        "psnr": "NA",
        "l2": "NA",
        "linf": "NA",
        "recovery_checked": "False",
        "recovery_image_exists": "False",
        "recovery_exact": "NA",
        "recovery_max_abs_error": "NA",
        "recovery_error_pixels": "NA",
        "output_size_bytes": "NA",
        "notes": record.notes,
    }

    if input_path is None or output_path is None or not input_path.exists() or not output_path.exists():
        return row

    original = read_rgb_float(input_path)
    adversarial = read_rgb_float(output_path)
    if original.shape != adversarial.shape:
        row["notes"] = f"{row['notes']}; shape mismatch {original.shape} vs {adversarial.shape}"
        return row

    diff = original - adversarial
    row.update(
        {
            "output_exists": "True",
            "ssim": format_float(
                structural_similarity(original, adversarial, channel_axis=-1, data_range=1.0)
            ),
            "psnr": format_float(peak_signal_noise_ratio(original, adversarial, data_range=1.0), 4),
            "l2": format_float(float(np.linalg.norm(diff.ravel())), 4),
            "linf": format_float(float(np.max(np.abs(diff))), 6),
            "output_size_bytes": str(output_path.stat().st_size),
        }
    )

    if restored_path is not None:
        row["recovery_checked"] = "True"
        row["recovery_image_exists"] = "True" if restored_path.exists() else "False"
        if restored_path.exists():
            original_u8 = read_rgb_uint8(input_path)
            restored_u8 = read_rgb_uint8(restored_path)
            if original_u8.shape == restored_u8.shape:
                recovery_abs = np.abs(original_u8.astype(np.int16) - restored_u8.astype(np.int16))
                max_error = int(recovery_abs.max())
                error_pixels = int(np.count_nonzero(np.any(recovery_abs != 0, axis=-1)))
                row["recovery_exact"] = "True" if max_error == 0 and error_pixels == 0 else "False"
                row["recovery_max_abs_error"] = str(max_error)
                row["recovery_error_pixels"] = str(error_pixels)
            else:
                row["recovery_exact"] = "False"
                row["notes"] = f"{row['notes']}; recovery shape mismatch"

    return row


def load_records(path: Path | None) -> list[RunRecord]:
    if path is None:
        return DEFAULT_RECORDS

    records: list[RunRecord] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            records.append(
                RunRecord(
                    run_name=row.get("run_name", ""),
                    phase=row.get("phase", ""),
                    attack_steps=row.get("attack_steps", ""),
                    max_iteration=row.get("max_iteration", ""),
                    elapsed_s=row.get("elapsed_s", ""),
                    success_count=row.get("success_count", ""),
                    total_images=row.get("total_images", ""),
                    input_image=row.get("input_image", ""),
                    output_image=row.get("output_image", ""),
                    restored_image=row.get("restored_image", ""),
                    notes=row.get("notes", ""),
                )
            )
    return records


def write_csv(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YUV reversible attack outputs.")
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="Optional CSV with run records. Relative image paths are resolved from the runs directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "stage6_metrics_summary.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.records)
    rows = [evaluate_record(record) for record in records]
    output_path = args.output if args.output.is_absolute() else RUNS_DIR / args.output
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
