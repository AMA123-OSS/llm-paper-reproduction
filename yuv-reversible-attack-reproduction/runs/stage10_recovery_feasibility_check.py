from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image


RUNS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNS_DIR.parent

RECOVERY_PRECONDITION_CSV = RUNS_DIR / "stage10_recovery_preconditions.csv"
RECOVERY_OBSERVED_DIFF_CSV = RUNS_DIR / "stage10_recovery_observed_diffs.csv"

OUTPUT_GROUPS = [
    {
        "group": "stage7_steps4_iter5_y_uv",
        "output_dir": RUNS_DIR / "stage7_10img_steps4_iter5" / "output" / "rae",
        "input_dir": RUNS_DIR / "stage7_10img_steps4_iter5" / "ORI_IMG",
        "note": "Y+UV output generated in stage7 with steps=4,max_iteration=5",
    },
    {
        "group": "stage8_y_uv_embed",
        "output_dir": RUNS_DIR / "stage8_ablation" / "y_uv_embed" / "output",
        "input_dir": PROJECT_ROOT / "ORI_IMG",
        "note": "Y+UV output generated in stage8 ablation",
    },
    {
        "group": "stage9_single_inception_y_uv",
        "output_dir": RUNS_DIR / "stage9_transferability" / "single_inception_y_uv" / "output",
        "input_dir": PROJECT_ROOT / "ORI_IMG",
        "note": "Y+UV output generated in stage9 with single Inception",
    },
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def has_pattern(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE) is not None


def inspect_recovery_preconditions() -> list[dict[str, str]]:
    embed_text = read_text(PROJECT_ROOT / "embed_utils.py")
    main_text = read_text(PROJECT_ROOT / "reversible_attack_OURS.py")
    return [
        {
            "item": "PE_decode_function",
            "status": "present" if "def PE_decode" in embed_text else "missing",
            "evidence": "embed_utils.py contains PE_decode, which can extract embedded bits if T is known.",
        },
        {
            "item": "Arithmetic_decode_function",
            "status": "present" if "def Arithmetic_decode" in read_text(PROJECT_ROOT / "helpers.py") else "missing",
            "evidence": "helpers.py contains Arithmetic_decode, but decoding needs the original frequency dictionary.",
        },
        {
            "item": "T_flag_persisted",
            "status": "missing" if not has_pattern(main_text, r"T_flag|threshold") else "present_or_partial",
            "evidence": "reversible_attack_OURS.py saves only PNG RAE; embed_main keeps T_flag local and does not return it.",
        },
        {
            "item": "arithmetic_dictionary_persisted",
            "status": "missing",
            "evidence": "compressfun returns only code from embed_main path; dic is discarded and not saved with RAE.",
        },
        {
            "item": "compressed_bit_length_persisted",
            "status": "missing",
            "evidence": "embed_main prepends per-round length bits inside UV payload, but the arithmetic dictionary and T sequence are still unavailable.",
        },
        {
            "item": "standalone_recovery_entry",
            "status": "missing",
            "evidence": "No script or function is called by reversible_attack_OURS.py to reconstruct RGB original from saved PNG RAE.",
        },
    ]


def read_rgb_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def diff_image_pair(input_path: Path, output_path: Path, group: str, note: str) -> dict[str, str]:
    original = read_rgb_uint8(input_path)
    rae = read_rgb_uint8(output_path)
    abs_diff = np.abs(original.astype(np.int16) - rae.astype(np.int16))
    return {
        "group": group,
        "input_filename": input_path.name,
        "output_filename": output_path.name,
        "output_exists": "True",
        "max_abs_diff_original_vs_rae": str(int(abs_diff.max())),
        "different_pixels_original_vs_rae": str(int(np.count_nonzero(np.any(abs_diff != 0, axis=-1)))),
        "different_values_original_vs_rae": str(int(np.count_nonzero(abs_diff))),
        "recovered_image_exists": "False",
        "recovery_exact": "NA",
        "recovery_max_abs_error": "NA",
        "recovery_error_pixels": "NA",
        "note": note,
    }


def inspect_observed_diffs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group_config in OUTPUT_GROUPS:
        output_dir = Path(group_config["output_dir"])
        input_dir = Path(group_config["input_dir"])
        for output_path in sorted(output_dir.glob("*.png")):
            input_path = input_dir / output_path.name
            if not input_path.exists():
                rows.append(
                    {
                        "group": str(group_config["group"]),
                        "input_filename": output_path.name,
                        "output_filename": output_path.name,
                        "output_exists": "True",
                        "max_abs_diff_original_vs_rae": "NA",
                        "different_pixels_original_vs_rae": "NA",
                        "different_values_original_vs_rae": "NA",
                        "recovered_image_exists": "False",
                        "recovery_exact": "NA",
                        "recovery_max_abs_error": "NA",
                        "recovery_error_pixels": "NA",
                        "note": "input image not found for this output",
                    }
                )
                continue
            rows.append(
                diff_image_pair(
                    input_path=input_path,
                    output_path=output_path,
                    group=str(group_config["group"]),
                    note=str(group_config["note"]),
                )
            )
    return rows


def main() -> None:
    precondition_rows = inspect_recovery_preconditions()
    write_csv(
        RECOVERY_PRECONDITION_CSV,
        ["item", "status", "evidence"],
        precondition_rows,
    )
    diff_rows = inspect_observed_diffs()
    write_csv(
        RECOVERY_OBSERVED_DIFF_CSV,
        [
            "group",
            "input_filename",
            "output_filename",
            "output_exists",
            "max_abs_diff_original_vs_rae",
            "different_pixels_original_vs_rae",
            "different_values_original_vs_rae",
            "recovered_image_exists",
            "recovery_exact",
            "recovery_max_abs_error",
            "recovery_error_pixels",
            "note",
        ],
        diff_rows,
    )
    missing = [row["item"] for row in precondition_rows if row["status"] == "missing"]
    print(f"wrote {len(precondition_rows)} precondition rows to {RECOVERY_PRECONDITION_CSV}")
    print(f"wrote {len(diff_rows)} observed diff rows to {RECOVERY_OBSERVED_DIFF_CSV}")
    print("missing recovery prerequisites:", ", ".join(missing))


if __name__ == "__main__":
    main()
