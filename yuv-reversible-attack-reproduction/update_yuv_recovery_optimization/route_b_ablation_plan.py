from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
PLAN_CSV = RESULTS_DIR / "route_b_ablation_plan.csv"


ROWS = [
    {
        "experiment": "uv_default",
        "carrier": "U+V stacked",
        "payload_strategy": "all compressed Y-diff bits embedded in UV, metadata saved as sidecar",
        "expected_effect": "baseline for updated recovery-capable method",
    },
    {
        "experiment": "u_only",
        "carrier": "U only",
        "payload_strategy": "embed into U channel only; V unchanged",
        "expected_effect": "tests whether concentrating chroma changes is less visible but may exceed capacity",
    },
    {
        "experiment": "v_only",
        "carrier": "V only",
        "payload_strategy": "embed into V channel only; U unchanged",
        "expected_effect": "checks asymmetric sensitivity between U and V",
    },
    {
        "experiment": "uv_25_75",
        "carrier": "U/V split",
        "payload_strategy": "25% payload in U, 75% payload in V",
        "expected_effect": "searches lower-distortion chroma allocation",
    },
    {
        "experiment": "uv_75_25",
        "carrier": "U/V split",
        "payload_strategy": "75% payload in U, 25% payload in V",
        "expected_effect": "paired allocation comparison",
    },
    {
        "experiment": "fixed_T_small",
        "carrier": "U+V stacked",
        "payload_strategy": "fixed low T if capacity allows",
        "expected_effect": "lower local distortion but possible capacity failure",
    },
    {
        "experiment": "adaptive_T_cost",
        "carrier": "U+V stacked",
        "payload_strategy": "choose lowest T that satisfies capacity and minimizes estimated distortion",
        "expected_effect": "main candidate for PSNR/SSIM improvement",
    },
    {
        "experiment": "sidecar_external_payload",
        "carrier": "sidecar + optional UV checksum",
        "payload_strategy": "move more recovery data outside UV into sidecar",
        "expected_effect": "upper-bound test for quality; weakens pure self-contained RAE assumption",
    },
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PLAN_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(ROWS)
    for row in ROWS:
        print(row)


if __name__ == "__main__":
    main()
