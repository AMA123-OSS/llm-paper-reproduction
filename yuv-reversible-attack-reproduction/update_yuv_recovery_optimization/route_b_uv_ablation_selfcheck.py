from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed_utils import (
    EmbeddingHistogramShifting,
    PE_decode,
    PE_encode,
    calculate_threshold,
    crossPrediction,
    dotPrediction,
)
from recovery_embed import compressfun_with_meta


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_CSV = RESULTS_DIR / "route_b_uv_ablation_selfcheck.csv"

IMAGE_SIZE = 299
LEN_BITS = 18


@dataclass
class EmbedResult:
    carrier: np.ndarray
    status: str
    round_count: int
    thresholds: list[int]
    embedded_payload_bits: int
    recovery_exact: bool


def synthetic_pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ori_yuv = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=float)
    ori_yuv[:, :, 0] = rng.integers(32, 224, size=(IMAGE_SIZE, IMAGE_SIZE))
    ori_yuv[:, :, 1] = rng.integers(112, 145, size=(IMAGE_SIZE, IMAGE_SIZE))
    ori_yuv[:, :, 2] = rng.integers(112, 145, size=(IMAGE_SIZE, IMAGE_SIZE))
    advy_yuv = ori_yuv.copy()
    mask = rng.random((IMAGE_SIZE, IMAGE_SIZE)) < 0.08
    advy_yuv[:, :, 0] = np.clip(
        advy_yuv[:, :, 0] + mask * rng.integers(-2, 3, size=(IMAGE_SIZE, IMAGE_SIZE)),
        0,
        255,
    )
    return ori_yuv, advy_yuv


def int_to_bits(value: int, width: int = LEN_BITS) -> list[int]:
    if value < 0 or value >= 2**width:
        raise ValueError(f"value {value} cannot be represented by {width} bits")
    bits = bin(value)[2:]
    return [0] * (width - len(bits)) + [int(bit) for bit in bits]


def bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = value * 2 + int(bit)
    return value


def capacity_for_threshold(carrier: np.ndarray, threshold: int) -> int:
    zeros = [0] * carrier.size
    cross_pred, cross_error, cross_mask = crossPrediction(carrier)
    cross_stego, cross_capacity = EmbeddingHistogramShifting(
        cross_pred, zeros, threshold, cross_error, cross_mask
    )
    dot_pred, dot_error, dot_mask = dotPrediction(cross_stego)
    _, dot_capacity = EmbeddingHistogramShifting(
        dot_pred, zeros[cross_capacity:], threshold, dot_error, dot_mask
    )
    return int(cross_capacity + dot_capacity)


def psnr_from_diff(diff: np.ndarray) -> float:
    mse = float(np.mean(np.square(diff)))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim_or_na(before: np.ndarray, after: np.ndarray) -> str:
    try:
        from skimage.metrics import structural_similarity

        value = structural_similarity(before, after, data_range=255.0)
        return f"{float(value):.6f}"
    except Exception:
        before_f = before.astype(float)
        after_f = after.astype(float)
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        mu_x = float(np.mean(before_f))
        mu_y = float(np.mean(after_f))
        var_x = float(np.var(before_f))
        var_y = float(np.var(after_f))
        cov_xy = float(np.mean((before_f - mu_x) * (after_f - mu_y)))
        denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        if denominator == 0:
            return "NA"
        value = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / denominator
        return f"{value:.6f}"


def choose_threshold(carrier: np.ndarray, required_len: int, mode: str, data: list[int]) -> tuple[int, int, str]:
    if mode == "adaptive_original":
        threshold = int(calculate_threshold(carrier, data, required_len))
        return threshold, capacity_for_threshold(carrier, threshold), "ok"

    if mode == "fixed_T1":
        threshold = 1
        capacity = capacity_for_threshold(carrier, threshold)
        if capacity < required_len:
            return threshold, capacity, "capacity_failed"
        return threshold, capacity, "ok"

    if mode == "adaptive_T_cost":
        best: tuple[float, int, int] | None = None
        for threshold in range(1, 82, 5):
            capacity = capacity_for_threshold(carrier, threshold)
            if capacity < required_len:
                continue
            stego = PE_encode(carrier, threshold, data)
            mse = float(np.mean(np.square(stego - carrier)))
            if best is None or mse < best[0]:
                best = (mse, threshold, capacity)
        if best is None:
            threshold = int(calculate_threshold(carrier, data, required_len))
            return threshold, capacity_for_threshold(carrier, threshold), "fallback_original_threshold"
        return best[1], best[2], "ok"

    raise ValueError(f"unknown threshold mode: {mode}")


def embed_payload(carrier: np.ndarray, payload_bits: list[int], threshold_mode: str) -> EmbedResult:
    current = carrier.astype(float).copy()
    remaining = [int(bit) for bit in payload_bits]
    thresholds: list[int] = []
    embedded_payload_bits = 0
    recovery_exact = True

    while remaining:
        payload_len_guess = min(len(remaining), max(1, current.size - LEN_BITS))
        provisional_data = int_to_bits(payload_len_guess) + remaining[:payload_len_guess]
        provisional_data += [0] * max(0, current.size - len(provisional_data))
        threshold, capacity, status = choose_threshold(
            current, LEN_BITS + payload_len_guess, threshold_mode, provisional_data
        )
        if capacity <= LEN_BITS:
            return EmbedResult(current, status, len(thresholds), thresholds, embedded_payload_bits, False)

        payload_len = min(len(remaining), capacity - LEN_BITS)
        required_len = LEN_BITS + payload_len
        data = int_to_bits(payload_len) + remaining[:payload_len]
        data += [0] * max(0, current.size - len(data))

        if status != "ok" and status != "fallback_original_threshold":
            return EmbedResult(current, status, len(thresholds), thresholds, embedded_payload_bits, False)

        stego = PE_encode(current, threshold, data)
        recovered, extracted = PE_decode(threshold, stego)
        extracted = [int(bit) for bit in extracted]
        round_ok = (
            len(extracted) >= required_len
            and bits_to_int(extracted[:LEN_BITS]) == payload_len
            and extracted[LEN_BITS:required_len] == remaining[:payload_len]
            and np.array_equal(np.round(recovered), np.round(current))
        )
        recovery_exact = recovery_exact and round_ok
        thresholds.append(int(threshold))
        embedded_payload_bits += int(payload_len)
        current = stego
        remaining = remaining[payload_len:]

        if not round_ok:
            return EmbedResult(current, "decode_check_failed", len(thresholds), thresholds, embedded_payload_bits, False)

    return EmbedResult(current, "ok", len(thresholds), thresholds, embedded_payload_bits, recovery_exact)


def build_uv_by_experiment(
    experiment: str,
    uv_before: np.ndarray,
    payload_bits: list[int],
) -> tuple[np.ndarray, EmbedResult, str]:
    u_before = uv_before[:IMAGE_SIZE, :]
    v_before = uv_before[IMAGE_SIZE:, :]

    if experiment == "uv_default":
        result = embed_payload(uv_before, payload_bits, "adaptive_original")
        return result.carrier, result, "U+V stacked"

    if experiment == "u_only":
        result = embed_payload(u_before, payload_bits, "adaptive_original")
        uv_after = np.vstack((result.carrier, v_before))
        return uv_after, result, "U only"

    if experiment == "v_only":
        result = embed_payload(v_before, payload_bits, "adaptive_original")
        uv_after = np.vstack((u_before, result.carrier))
        return uv_after, result, "V only"

    if experiment in {"uv_25_75", "uv_75_25"}:
        u_ratio = 0.25 if experiment == "uv_25_75" else 0.75
        split = int(round(len(payload_bits) * u_ratio))
        u_result = embed_payload(u_before, payload_bits[:split], "adaptive_original")
        v_result = embed_payload(v_before, payload_bits[split:], "adaptive_original")
        uv_after = np.vstack((u_result.carrier, v_result.carrier))
        result = EmbedResult(
            carrier=uv_after,
            status="ok" if u_result.status == "ok" and v_result.status == "ok" else "partial_failed",
            round_count=u_result.round_count + v_result.round_count,
            thresholds=u_result.thresholds + v_result.thresholds,
            embedded_payload_bits=u_result.embedded_payload_bits + v_result.embedded_payload_bits,
            recovery_exact=u_result.recovery_exact and v_result.recovery_exact,
        )
        return uv_after, result, f"U/V split {int(u_ratio * 100)}/{int((1 - u_ratio) * 100)}"

    if experiment == "fixed_T1":
        result = embed_payload(uv_before, payload_bits, "fixed_T1")
        return result.carrier, result, "U+V stacked, fixed T=1"

    if experiment == "adaptive_T_cost":
        result = embed_payload(uv_before, payload_bits, "adaptive_T_cost")
        return result.carrier, result, "U+V stacked, cost search"

    if experiment == "sidecar_external_payload":
        result = EmbedResult(
            carrier=uv_before.copy(),
            status="ok_external_payload",
            round_count=0,
            thresholds=[],
            embedded_payload_bits=0,
            recovery_exact=True,
        )
        return uv_before.copy(), result, "UV unchanged; payload stored in sidecar"

    raise ValueError(f"unknown experiment: {experiment}")


def analyze_case(seed: int, experiment: str) -> dict[str, str]:
    ori_yuv, advy_yuv = synthetic_pair(seed)
    ori_y = ori_yuv[:, :, 0]
    advy_y = advy_yuv[:, :, 0]
    err = (ori_y - advy_y).reshape(-1, 1)
    payload_bits, _, compressed_bit_length = compressfun_with_meta(err)
    uv_before = np.vstack((advy_yuv[:, :, 1], advy_yuv[:, :, 2]))
    uv_after, result, carrier_desc = build_uv_by_experiment(experiment, uv_before, payload_bits)
    uv_diff = uv_after - uv_before
    changed_u = int(np.count_nonzero(np.round(uv_diff[:IMAGE_SIZE, :])))
    changed_v = int(np.count_nonzero(np.round(uv_diff[IMAGE_SIZE:, :])))
    return {
        "case": f"synthetic_{seed:02d}",
        "experiment": experiment,
        "carrier": carrier_desc,
        "status": result.status,
        "compressed_bit_length": str(compressed_bit_length),
        "embedded_payload_bits": str(result.embedded_payload_bits),
        "round_count": str(result.round_count),
        "thresholds": "|".join(str(value) for value in result.thresholds),
        "carrier_recovery_exact": "1" if result.recovery_exact else "0",
        "changed_u_pixels": str(changed_u),
        "changed_v_pixels": str(changed_v),
        "changed_uv_pixels": str(changed_u + changed_v),
        "uv_linf": f"{float(np.max(np.abs(uv_diff))):.6f}",
        "uv_l2": f"{float(np.linalg.norm(uv_diff.ravel())):.4f}",
        "uv_psnr": f"{psnr_from_diff(uv_diff):.4f}",
        "uv_ssim": ssim_or_na(uv_before, uv_after),
        "note": "synthetic carrier ablation; real-image confirmation requires ORI_IMG",
    }


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    experiments = sorted({row["experiment"] for row in rows})
    for experiment in experiments:
        group = [row for row in rows if row["experiment"] == experiment]
        ok_rows = [row for row in group if row["status"].startswith("ok")]
        psnr_values = [float(row["uv_psnr"]) for row in ok_rows if row["uv_psnr"] != "inf"]
        ssim_values = [float(row["uv_ssim"]) for row in ok_rows if row["uv_ssim"] != "NA"]
        summary_rows.append(
            {
                "case": "SUMMARY",
                "experiment": experiment,
                "carrier": group[0]["carrier"],
                "status": f"{len(ok_rows)}/{len(group)} ok",
                "compressed_bit_length": "avg="
                + f"{sum(float(row['compressed_bit_length']) for row in group) / len(group):.1f}",
                "embedded_payload_bits": "avg="
                + f"{sum(float(row['embedded_payload_bits']) for row in group) / len(group):.1f}",
                "round_count": "avg="
                + f"{sum(float(row['round_count']) for row in group) / len(group):.2f}",
                "thresholds": "varies",
                "carrier_recovery_exact": "1"
                if all(row["carrier_recovery_exact"] == "1" for row in group)
                else "0",
                "changed_u_pixels": "avg="
                + f"{sum(float(row['changed_u_pixels']) for row in group) / len(group):.1f}",
                "changed_v_pixels": "avg="
                + f"{sum(float(row['changed_v_pixels']) for row in group) / len(group):.1f}",
                "changed_uv_pixels": "avg="
                + f"{sum(float(row['changed_uv_pixels']) for row in group) / len(group):.1f}",
                "uv_linf": "avg=" + f"{sum(float(row['uv_linf']) for row in group) / len(group):.4f}",
                "uv_l2": "avg=" + f"{sum(float(row['uv_l2']) for row in group) / len(group):.2f}",
                "uv_psnr": "avg=" + (f"{sum(psnr_values) / len(psnr_values):.4f}" if psnr_values else "inf"),
                "uv_ssim": "avg=" + (f"{sum(ssim_values) / len(ssim_values):.6f}" if ssim_values else "NA"),
                "note": "summary across synthetic seeds",
            }
        )
    return summary_rows


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    experiments = [
        "uv_default",
        "u_only",
        "v_only",
        "uv_25_75",
        "uv_75_25",
        "fixed_T1",
        "adaptive_T_cost",
        "sidecar_external_payload",
    ]
    rows = [analyze_case(seed, experiment) for seed in range(3) for experiment in experiments]
    rows_with_summary = rows + summarize(rows)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_with_summary[0].keys()))
        writer.writeheader()
        writer.writerows(rows_with_summary)
    for row in rows_with_summary:
        print(row)


if __name__ == "__main__":
    main()
