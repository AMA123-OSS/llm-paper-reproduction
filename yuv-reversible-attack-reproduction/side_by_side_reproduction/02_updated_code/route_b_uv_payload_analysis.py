from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from recovery_embed import embed_main_with_meta


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_CSV = RESULTS_DIR / "route_b_uv_payload_analysis.csv"


def synthetic_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ori_yuv = np.zeros((299, 299, 3), dtype=float)
    ori_yuv[:, :, 0] = rng.integers(32, 224, size=(299, 299))
    ori_yuv[:, :, 1] = rng.integers(112, 145, size=(299, 299))
    ori_yuv[:, :, 2] = rng.integers(112, 145, size=(299, 299))
    advy_yuv = ori_yuv.copy()
    mask = rng.random((299, 299)) < 0.08
    advy_yuv[:, :, 0] = np.clip(advy_yuv[:, :, 0] + mask * rng.integers(-2, 3, size=(299, 299)), 0, 255)
    return ori_yuv, advy_yuv


def psnr_from_diff(diff: np.ndarray) -> float:
    mse = float(np.mean(np.square(diff)))
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def analyze_case(seed: int) -> dict[str, str]:
    ori_yuv, advy_yuv = synthetic_pair(seed)
    reversible_uv, metadata = embed_main_with_meta(ori_yuv, advy_yuv)
    uv_before = np.vstack((advy_yuv[:, :, 1], advy_yuv[:, :, 2]))
    uv_diff = reversible_uv - uv_before
    u_diff = uv_diff[:299, :]
    v_diff = uv_diff[299:, :]
    changed_u = int(np.count_nonzero(np.round(u_diff)))
    changed_v = int(np.count_nonzero(np.round(v_diff)))
    total_changed = changed_u + changed_v
    return {
        "case": f"synthetic_{seed:02d}",
        "compressed_bit_length": str(metadata["compressed_bit_length"]),
        "round_count": str(len(metadata["rounds"])),
        "changed_u_pixels": str(changed_u),
        "changed_v_pixels": str(changed_v),
        "changed_uv_pixels": str(total_changed),
        "u_changed_ratio": f"{changed_u / (299 * 299):.6f}",
        "v_changed_ratio": f"{changed_v / (299 * 299):.6f}",
        "uv_linf": f"{float(np.max(np.abs(uv_diff))):.6f}",
        "uv_l2": f"{float(np.linalg.norm(uv_diff.ravel())):.4f}",
        "uv_psnr": f"{psnr_from_diff(uv_diff):.4f}",
        "sidecar_externalized_fields": "T_flag,dic,compressed_bit_length,round_payload_len",
        "interpretation": "sidecar closes recovery metadata gap; payload bits are still embedded in UV in this prototype",
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [analyze_case(seed) for seed in range(3)]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
