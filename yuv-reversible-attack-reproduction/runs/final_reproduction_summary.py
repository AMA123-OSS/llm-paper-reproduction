from __future__ import annotations

import csv
from pathlib import Path


RUNS_DIR = Path(__file__).resolve().parent
OUTPUT_MD = RUNS_DIR / "final_reproduction_summary.md"
OUTPUT_CSV = RUNS_DIR / "final_reproduction_summary.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["section", "item", "value", "note"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    stage7 = read_csv(RUNS_DIR / "stage7_10img_summary.csv")
    stage8 = read_csv(RUNS_DIR / "stage8_ablation_summary.csv")
    stage9 = read_csv(RUNS_DIR / "stage9_transferability_summary.csv")
    stage10_pre = read_csv(RUNS_DIR / "stage10_recovery_preconditions.csv")

    rows: list[dict[str, str]] = []
    for row in stage7:
        rows.append(
            {
                "section": "stage7",
                "item": row["run_name"],
                "value": f"success={row['success_count']}/{row['total_images']}, "
                f"ssim={row['avg_ssim']}, psnr={row['avg_psnr']}, "
                f"avg_elapsed_s={row['avg_elapsed_s']}",
                "note": "10-image reproduction",
            }
        )
    for row in stage8:
        rows.append(
            {
                "section": "stage8",
                "item": row["method"],
                "value": f"success={row['success_count']}/{row['total_images']}, "
                f"ssim={row['avg_ssim']}, psnr={row['avg_psnr']}, l2={row['avg_l2']}",
                "note": "Y/UV ablation",
            }
        )
    for row in stage9:
        rows.append(
            {
                "section": "stage9",
                "item": f"{row['generator']} -> {row['test_model']}",
                "value": f"success_vs_label={row['attack_success_count_vs_label']}/{row['total_images']}, "
                f"conditional={row['conditional_success_count']}/{row['original_correct_count']}",
                "note": row["relation"],
            }
        )
    missing = [row["item"] for row in stage10_pre if row["status"] == "missing"]
    rows.append(
        {
            "section": "stage10",
            "item": "error-free recovery",
            "value": "not verified",
            "note": "missing prerequisites: " + ", ".join(missing),
        }
    )
    write_csv(OUTPUT_CSV, rows)

    stage7_rows = [
        [row["run_name"], row["success_rate"], row["avg_ssim"], row["avg_psnr"], row["avg_elapsed_s"]]
        for row in stage7
    ]
    stage8_rows = [
        [row["method"], row["success_rate"], row["avg_ssim"], row["avg_psnr"], row["avg_l2"]]
        for row in stage8
    ]
    stage9_rows = [
        [
            row["generator"],
            row["test_model"],
            row["relation"],
            row["attack_success_rate_vs_label"],
            row["conditional_success_rate_on_original_correct"],
        ]
        for row in stage9
    ]

    md = f"""# 最终复现实验自动摘要

本文件由 `runs/final_reproduction_summary.py` 从阶段 7-10 的 CSV 自动生成。

## 阶段 7：10 图小规模复现

{md_table(["run", "success_rate", "avg_ssim", "avg_psnr", "avg_elapsed_s"], stage7_rows)}

## 阶段 8：Y/UV 消融

{md_table(["method", "success_rate", "avg_ssim", "avg_psnr", "avg_l2"], stage8_rows)}

## 阶段 9：迁移性

{md_table(["generator", "test_model", "relation", "success_vs_label", "conditional_success"], stage9_rows)}

## 阶段 10：恢复验证

当前未能验证 error-free recovery。缺失前置条件：

{", ".join(missing)}
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUTPUT_MD}")
    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
