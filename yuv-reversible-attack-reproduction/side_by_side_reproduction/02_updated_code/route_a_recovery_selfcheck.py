from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from recovery_embed import (
    embed_main_with_meta,
    recover_original_yuv_from_rae_yuv,
    save_recovery_sidecar,
)


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_CSV = RESULTS_DIR / "route_a_recovery_selfcheck.csv"
CASE_COUNT = 10


def synthetic_yuv(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ori_yuv = np.zeros((299, 299, 3), dtype=float)
    ori_yuv[:, :, 0] = rng.integers(32, 224, size=(299, 299))
    ori_yuv[:, :, 1] = rng.integers(96, 160, size=(299, 299))
    ori_yuv[:, :, 2] = rng.integers(96, 160, size=(299, 299))

    advy_yuv = ori_yuv.copy()
    mask = rng.random((299, 299)) < 0.08
    delta = rng.integers(-2, 3, size=(299, 299))
    advy_yuv[:, :, 0] = np.clip(advy_yuv[:, :, 0] + mask * delta, 0, 255)
    return ori_yuv, advy_yuv


def run_case(seed: int) -> dict[str, str]:
    ori_yuv, advy_yuv = synthetic_yuv(seed)
    reversible_uv, metadata = embed_main_with_meta(ori_yuv, advy_yuv)
    rae_yuv = advy_yuv.copy()
    rae_yuv[:, :, 1] = reversible_uv[:299, :]
    rae_yuv[:, :, 2] = reversible_uv[299:, :]

    sidecar = RESULTS_DIR / f"synthetic_{seed:02d}.recovery.json"
    save_recovery_sidecar(sidecar, metadata)
    np.save(RESULTS_DIR / f"synthetic_{seed:02d}.rae_yuv.npy", rae_yuv)

    recovered_yuv = recover_original_yuv_from_rae_yuv(rae_yuv, metadata)
    diff = np.abs(np.round(recovered_yuv) - np.round(ori_yuv))
    return {
        "case": f"synthetic_{seed:02d}",
        "sidecar": str(sidecar.relative_to(BASE_DIR)),
        "round_count": str(len(metadata["rounds"])),
        "compressed_bit_length": str(metadata["compressed_bit_length"]),
        "recovery_max_abs_error": str(int(diff.max())),
        "recovery_error_pixels": str(int(np.count_nonzero(diff))),
        "recovery_exact": "1" if int(diff.max()) == 0 and int(np.count_nonzero(diff)) == 0 else "0",
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_case(seed) for seed in range(CASE_COUNT)]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
