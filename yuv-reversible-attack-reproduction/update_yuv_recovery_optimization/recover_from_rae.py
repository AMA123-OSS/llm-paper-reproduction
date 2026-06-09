from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from recovery_embed import load_recovery_sidecar, recover_original_yuv_from_rae_yuv


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover original YUV data from an RAE YUV sidecar pair.")
    parser.add_argument("--rae-yuv-npy", required=True, help="Path to a saved RAE YUV .npy array.")
    parser.add_argument("--sidecar", required=True, help="Path to recovery.json sidecar.")
    parser.add_argument("--output-yuv-npy", required=True, help="Path for recovered original YUV .npy array.")
    parser.add_argument("--output-preview", default="", help="Optional uint8 grayscale Y-channel preview image.")
    args = parser.parse_args()

    rae_yuv = np.load(args.rae_yuv_npy)
    metadata = load_recovery_sidecar(Path(args.sidecar))
    recovered_yuv = recover_original_yuv_from_rae_yuv(rae_yuv, metadata)

    output_path = Path(args.output_yuv_npy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, recovered_yuv)

    if args.output_preview:
        y_channel = np.clip(np.round(recovered_yuv[:, :, 0]), 0, 255).astype(np.uint8)
        Image.fromarray(y_channel, mode="L").save(args.output_preview)


if __name__ == "__main__":
    main()
