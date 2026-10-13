# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Row-wise quantization of embedding tables into runtime sidecars.

NumPy only, so the ONNX exporter and the direct builder share one encoder.
Tables are processed in row chunks and streamed to disk, so a multi-GB
per-layer embedding table never has to fit in host memory.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_FORMATS = ("fp16", "fp8", "int8", "int4", "nvfp4")
EMBEDDING_DESCRIPTOR_METADATA_KEY = "tensorrt_edge_llm.embedding_descriptor"

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
INT8_MAX = 127
INT4_MAX = 7
INT4_ZERO_POINT = 8
FP8_GROUP_SIZE = 128
INT4_GROUP_SIZE = 128
NVFP4_GROUP_SIZE = 16
# Groups whose absolute maximum is below this floor share one fixed scale.
AMAX_FLOOR = 1e-4
_CHUNK_BYTES = 64 << 20

# Non-negative E4M3 magnitudes for codes 0x00..0x7E (0x7F is NaN).
_E4M3_MAGNITUDES = np.array([m * 2.0**-9 for m in range(8)] +
                            [(1.0 + m / 8.0) * 2.0**(e - 7)
                             for e in range(1, 16) for m in range(8)][:119],
                            dtype=np.float32)
_E4M3_MIDPOINTS = (_E4M3_MAGNITUDES[:-1] + _E4M3_MAGNITUDES[1:]) / 2
_E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                            dtype=np.float32)
_E2M1_MIDPOINTS = (_E2M1_MAGNITUDES[:-1] + _E2M1_MAGNITUDES[1:]) / 2


def _round_to_codes(magnitude: np.ndarray,
                    midpoints: np.ndarray) -> np.ndarray:
    """Nearest code of a sorted magnitude table, ties to the even code."""
    codes = np.searchsorted(midpoints, magnitude, side="left")
    tie = midpoints[np.minimum(codes, midpoints.size - 1)] == magnitude
    return (codes + (tie & (codes % 2 == 1))).astype(np.uint8)


def encode_fp8_e4m3(values: np.ndarray) -> np.ndarray:
    """Encode float32 values as saturating round-to-nearest-even E4M3 bytes."""
    magnitude = np.minimum(np.abs(values), FP8_E4M3_MAX)
    codes = _round_to_codes(magnitude, _E4M3_MIDPOINTS)
    return codes | (np.signbit(values).astype(np.uint8) << 7)


def decode_fp8_e4m3(codes: np.ndarray) -> np.ndarray:
    """Decode finite E4M3 bytes to float32."""
    magnitude = _E4M3_MAGNITUDES[np.minimum(codes & 0x7F, 0x7E)]
    return np.where(codes & 0x80, -magnitude, magnitude)


def _pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    """Pack ``[rows, cols]`` nibbles two per byte, even column in the low nibble."""
    return nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)


def table_group_size(fmt: str, cols: int, layer_width: int = 0) -> int:
    """Scale group of one table; per-layer tables keep groups inside a layer.

    FP16 tables have no scales and return 0.
    """
    if fmt == "fp16":
        return 0
    span = layer_width or cols
    group = {
        "fp8": FP8_GROUP_SIZE,
        "int8": span,
        "int4": INT4_GROUP_SIZE,
        "nvfp4": NVFP4_GROUP_SIZE,
    }[fmt]
    if span % group or cols % span:
        raise ValueError(
            f"{fmt} embedding group {group} does not tile a table of width "
            f"{cols} with layer width {span}")
    return group


def quantize_symmetric(
        rows: np.ndarray,
        group: int,
        qmax: int,
        scale_dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric round-half-even integer quantization with one scale per group.

    Returns signed values in ``[-qmax, qmax]`` and ``[rows, cols / group]``
    scales such that ``value ~= q * scale``.
    """
    num_rows, cols = rows.shape
    grouped = rows.reshape(num_rows, cols // group, group)
    amax = np.maximum(np.abs(grouped).max(axis=2), AMAX_FLOOR)
    scale = (amax / qmax).astype(scale_dtype)
    quantized = np.rint(grouped / scale.astype(np.float32)[..., None])
    quantized = np.clip(quantized, -qmax, qmax).astype(np.int8)
    return quantized.reshape(num_rows, cols), scale


def nvfp4_global_scale(amax: float) -> float:
    """Per-table NVFP4 multiplier that maps ``amax`` to the largest block code."""
    return max(float(amax), AMAX_FLOOR) / (FP4_E2M1_MAX * FP8_E4M3_MAX)


def quantize_rows(
        rows: np.ndarray,
        fmt: str,
        group: int,
        global_scale: float = 1.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Quantize ``[rows, cols]`` float32 values to one sidecar format.

    Returns ``(data, scale)``. ``scale`` is FP32 ``[rows, cols / group]`` for
    fp8/int8/int4, E4M3 bytes ``[rows, cols / 16]`` for nvfp4, and None for
    fp16. int4 and nvfp4 pack two values per byte, even column in the low
    nibble; int4 stores ``q + 8``. nvfp4 decodes as
    ``e2m1 * e4m3(scale) * global_scale``.
    """
    rows = np.asarray(rows, dtype=np.float32)
    if fmt == "fp16":
        return rows.astype(np.float16), None
    if rows.ndim != 2 or group <= 0 or rows.shape[1] % group:
        raise ValueError(
            f"{fmt} quantization needs a 2D table whose width is a multiple "
            f"of the group {group}, got shape {rows.shape}")
    if fmt == "int8":
        return quantize_symmetric(rows, group, INT8_MAX)
    if fmt == "int4":
        quantized, scale = quantize_symmetric(rows, group, INT4_MAX)
        nibbles = (quantized + INT4_ZERO_POINT).astype(np.uint8)
        return _pack_nibbles(nibbles), scale
    num_rows, cols = rows.shape
    grouped = rows.reshape(num_rows, cols // group, group)
    amax = np.abs(grouped).max(axis=2)
    if fmt == "fp8":
        scale = np.maximum(amax, AMAX_FLOOR) / FP8_E4M3_MAX
        data = encode_fp8_e4m3(grouped / scale[..., None])
        return data.reshape(num_rows, cols), scale
    if fmt == "nvfp4":
        if group != NVFP4_GROUP_SIZE:
            raise ValueError(
                f"nvfp4 uses {NVFP4_GROUP_SIZE}-element groups, got {group}")
        block_scale = encode_fp8_e4m3(
            amax / (FP4_E2M1_MAX * np.float32(global_scale)))
        divisor = decode_fp8_e4m3(block_scale) * np.float32(global_scale)
        safe = np.where(divisor > 0, divisor, 1.0)[..., None]
        normalized = np.where(divisor[..., None] > 0, grouped / safe, 0.0)
        magnitude = np.minimum(np.abs(normalized), FP4_E2M1_MAX)
        nibbles = (_round_to_codes(magnitude, _E2M1_MIDPOINTS)
                   | (np.signbit(normalized).astype(np.uint8) << 3))
        return _pack_nibbles(nibbles.reshape(num_rows, cols)), block_scale
    raise ValueError(f"unsupported embedding format {fmt!r}; supported: "
                     f"{', '.join(EMBEDDING_FORMATS)}")


def _sidecar_layout(name: str, rows: int, cols: int, fmt: str, group: int):
    """Return ``(tensor name, safetensors dtype, shape, item bytes)`` tuples."""
    if fmt == "fp16":
        return [(name, "F16", (rows, cols), 2)]
    if fmt in ("int4", "nvfp4"):
        data = (name, "U8", (rows, cols // 2), 1)
    else:
        data = (name, "F8_E4M3" if fmt == "fp8" else "I8", (rows, cols), 1)
    if fmt == "nvfp4":
        return [
            data,
            (f"{name}_block_scale", "F8_E4M3", (rows, cols // group), 1),
            (f"{name}_multiplier_scale", "F32", (1, ), 4),
        ]
    return [data, (f"{name}_scale", "F32", (rows, cols // group), 4)]


def write_quantized_table(path: str, name: str, shape: Tuple[int, int],
                          read_rows: Callable[[int, int], np.ndarray],
                          fmt: str, group: int) -> None:
    """Stream a ``[rows, cols]`` table into a quantized safetensors sidecar.

    ``read_rows(start, stop)`` returns the already model-scaled rows as
    float32. The header is written first and each chunk is written at its
    final offset, so memory stays bounded by one chunk.
    """
    rows, cols = (int(dim) for dim in shape)
    chunk_rows = max(1, _CHUNK_BYTES // (cols * 4 * 4))
    global_scale = 1.0
    if fmt == "nvfp4":
        amax = 0.0
        for start in range(0, rows, chunk_rows):
            chunk = read_rows(start, min(rows, start + chunk_rows))
            amax = max(amax, float(np.abs(chunk).max()))
        global_scale = nvfp4_global_scale(amax)

    layout = _sidecar_layout(name, rows, cols, fmt, group)
    header = {}
    offsets = []
    offset = 0
    for tensor_name, dtype, tensor_shape, item_bytes in layout:
        nbytes = int(np.prod(tensor_shape)) * item_bytes
        header[tensor_name] = {
            "dtype": dtype,
            "shape": list(tensor_shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offsets.append(offset)
        offset += nbytes
    if fmt == "nvfp4":
        header["__metadata__"] = {
            EMBEDDING_DESCRIPTOR_METADATA_KEY:
            json.dumps({
                "logical_shape": [rows, cols],
                "group_size": group
            })
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    payload_start = 8 + len(header_bytes)

    with open(path, "wb") as sidecar:
        sidecar.write(len(header_bytes).to_bytes(8, "little"))
        sidecar.write(header_bytes)
        sidecar.truncate(payload_start + offset)
        descriptor = sidecar.fileno()
        if fmt == "nvfp4":
            os.pwrite(descriptor,
                      np.float32(global_scale).tobytes(),
                      payload_start + offsets[2])
        for start in range(0, rows, chunk_rows):
            stop = min(rows, start + chunk_rows)
            outputs = quantize_rows(read_rows(start, stop), fmt, group,
                                    global_scale)
            for array, tensor_offset in zip(outputs, offsets):
                if array is None:
                    continue
                row_bytes = array.nbytes // (stop - start)
                os.pwrite(descriptor,
                          np.ascontiguousarray(array).tobytes(),
                          payload_start + tensor_offset + start * row_bytes)
    logger.info("Wrote %s %s [%d, %d] to %s", fmt, name, rows, cols, path)
