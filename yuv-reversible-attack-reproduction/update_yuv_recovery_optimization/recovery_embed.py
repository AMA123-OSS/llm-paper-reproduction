from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embed_utils import PE_decode, PE_encode, calculate_threshold  # noqa: E402
from helpers import Arithmetic_decode, Arithmetic_encode  # noqa: E402


IMAGE_SIZE = 299
LEN_BITS = 18


def _to_int_list(values: list[Any]) -> list[int]:
    result: list[int] = []
    for value in values:
        if isinstance(value, str) and value == "!":
            continue
        result.append(int(round(float(value))))
    return result


def _int_to_bits(value: int, width: int = LEN_BITS) -> list[int]:
    if value < 0 or value >= 2 ** width:
        raise ValueError(f"value {value} cannot be represented by {width} bits")
    bits = bin(value)[2:]
    return [0] * (width - len(bits)) + [int(bit) for bit in bits]


def _bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = value * 2 + int(bit)
    return value


def compressfun_with_meta(signal: np.ndarray) -> tuple[list[int], dict[str, int], int]:
    flat_signal = signal[:, 0].astype(int).tolist()
    code, dic = Arithmetic_encode(flat_signal, precision=32)
    serializable_dic = {str(key): int(value) for key, value in dic.items()}
    return [int(bit) for bit in code], serializable_dic, len(code)


def _decode_signal(code: list[int], dic: dict[str, int]) -> list[int]:
    decode_dic: dict[Any, int] = {}
    for key, value in dic.items():
        if key == "!":
            decode_dic[key] = int(value)
        else:
            decode_dic[int(key)] = int(value)
    decoded = Arithmetic_decode([int(bit) for bit in code], decode_dic, precision=32)
    return _to_int_list(decoded)


def embed_main_with_meta(ori_yuv: np.ndarray, advy_yuv: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if ori_yuv.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE) or advy_yuv.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError("current implementation follows the original fixed 299x299 pipeline")

    ori_y = ori_yuv[:, :, 0]
    advy_y = advy_yuv[:, :, 0]
    err = np.zeros((IMAGE_SIZE * IMAGE_SIZE, 1))
    index = 0
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            err[index] = int(round(float(ori_y[i, j] - advy_y[i, j])))
            index += 1

    code, dic, compressed_bit_length = compressfun_with_meta(err)

    img_cb = advy_yuv[:, :, 1]
    img_cr = advy_yuv[:, :, 2]
    img = np.vstack((img_cb, img_cr)).astype(float)

    m, n = img.shape
    remaining = code.copy()
    rounds: list[dict[str, int]] = []

    while remaining:
        cover = img.copy()
        len_data = len(remaining)

        if len_data >= (m * n - LEN_BITS):
            payload_len = m * n - LEN_BITS
            embed_data = _int_to_bits(payload_len) + remaining[:payload_len]
            remaining = remaining[payload_len:]
        else:
            payload_len = len_data
            embed_data = _int_to_bits(payload_len) + remaining
            embed_data += [0] * (m * n - len(embed_data))
            remaining = []

        t_value = int(calculate_threshold(cover, embed_data, LEN_BITS + payload_len))
        rounds.append({"T": t_value, "payload_len": int(payload_len)})
        img = PE_encode(cover, t_value, embed_data)

    metadata = {
        "image_size": IMAGE_SIZE,
        "uv_stacked_shape": [int(m), int(n)],
        "compressed_bit_length": int(compressed_bit_length),
        "dic": dic,
        "rounds": rounds,
        "err_shape": [int(err.shape[0]), int(err.shape[1])],
        "recovery_order": "reverse rounds; PE_decode recovers previous UV carrier and extracts length-prefixed bits",
    }
    return img.copy(), metadata


def recover_original_yuv_from_rae_yuv(rae_yuv: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    image_size = int(metadata["image_size"])
    stacked_shape = tuple(int(value) for value in metadata["uv_stacked_shape"])
    if image_size != IMAGE_SIZE or stacked_shape != (IMAGE_SIZE * 2, IMAGE_SIZE):
        raise ValueError("metadata shape does not match the fixed 299x299 implementation")

    current_uv = np.vstack((rae_yuv[:, :, 1], rae_yuv[:, :, 2])).astype(float)
    extracted_code_reversed: list[int] = []

    for round_meta in reversed(metadata["rounds"]):
        recovered_uv, recovered_data = PE_decode(int(round_meta["T"]), current_uv)
        payload_len_from_bits = _bits_to_int([int(bit) for bit in recovered_data[:LEN_BITS]])
        payload_len = int(round_meta["payload_len"])
        if payload_len_from_bits != payload_len:
            raise ValueError(
                f"payload length mismatch: sidecar={payload_len}, extracted={payload_len_from_bits}"
            )
        payload = [int(bit) for bit in recovered_data[LEN_BITS:LEN_BITS + payload_len]]
        extracted_code_reversed = payload + extracted_code_reversed
        current_uv = recovered_uv

    compressed_bit_length = int(metadata["compressed_bit_length"])
    code = extracted_code_reversed[:compressed_bit_length]
    err_flat = _decode_signal(code, metadata["dic"])
    expected_len = IMAGE_SIZE * IMAGE_SIZE
    if len(err_flat) != expected_len:
        raise ValueError(f"decoded Y error length {len(err_flat)} != {expected_len}")

    recovered_y = rae_yuv[:, :, 0].astype(float).copy()
    index = 0
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            recovered_y[i, j] = recovered_y[i, j] + err_flat[index]
            index += 1

    recovered_yuv = rae_yuv.copy().astype(float)
    recovered_yuv[:, :, 0] = recovered_y
    recovered_yuv[:, :, 1] = current_uv[:IMAGE_SIZE, :]
    recovered_yuv[:, :, 2] = current_uv[IMAGE_SIZE:, :]
    return recovered_yuv


def save_recovery_sidecar(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_recovery_sidecar(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
