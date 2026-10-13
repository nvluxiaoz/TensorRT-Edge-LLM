# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Embedding artifact writers."""

import logging
import os
import shutil

import numpy as np

from tensorrt_edgellm.checkpoint.embedding_quantization import (
    table_group_size, write_quantized_table)

from .. import contracts
from ..config import DeviceConfig
from ..safetensors_np import SafetensorsStore, load_safetensors_tensor
from ..weight_policy import (CHECKPOINT_BINDING_ENGINE_EMBEDDING,
                             CHECKPOINT_BINDING_ENGINE_PLE,
                             CHECKPOINT_BINDING_ROLE_EMBEDDING,
                             CHECKPOINT_BINDING_ROLE_PLE, EXTERNAL_WEIGHT_FP16)
from ..weights import Weights
from .tensors import save_safetensors

EMBEDDING_KEY = "model.embed_tokens.weight"

logger = logging.getLogger(__name__)


def externalizes_embedding(args, weight_conversion) -> bool:
    """Whether the runtime owns this component's embedding artifact.

    Components that pull embeddings from a second checkpoint keep their
    sidecar: the runtime only opens one ``--checkpointDir``.
    """
    if not args.weight_policy.externalizes_embedding:
        return False
    if getattr(weight_conversion, "runtime_embedding_model_dir",
               None) is not None:
        return False
    externalizes = getattr(weight_conversion, "externalizes_runtime_embedding",
                           None)
    if externalizes is not None and not externalizes(args):
        return False
    writes_embedding = getattr(weight_conversion, "writes_runtime_embedding",
                               None)
    return writes_embedding is None or writes_embedding(args)


def write_cached_draft_embedding(output_dir: str, draft_model_dir: str,
                                 mask_token_id: int,
                                 embedding_scale: float) -> None:
    """Fold a cached-draft trained mask-token row into the base embedding."""
    with SafetensorsStore(draft_model_dir) as draft:
        draft_key = next(
            (key
             for key in ("embed_tokens.weight", "model.embed_tokens.weight")
             if draft.has(key)), None)
        if draft_key is None:
            logger.info("Cached draft shares the base embedding table")
            return
        draft_shape = draft.shape(draft_key)
        draft_row = draft.get_f16_row(draft_key, mask_token_id)

    path = os.path.join(output_dir, "embedding.safetensors")
    weight = load_safetensors_tensor(path, "embedding")
    if weight.dtype != np.float16:
        raise TypeError("cached-draft embedding fold requires FP16 embedding")
    if weight.ndim != 2 or not 0 <= mask_token_id < weight.shape[0]:
        raise ValueError(
            f"mask_token_id {mask_token_id} is outside embedding shape "
            f"{weight.shape}")
    if draft_shape != weight.shape:
        raise ValueError(
            f"cached draft embedding shape {draft_shape} does not match "
            f"base embedding shape {weight.shape}")

    weight[mask_token_id] = np.asarray(draft_row * np.float16(embedding_scale),
                                       dtype=np.float16)
    temporary_path = path + ".tmp"
    save_safetensors(temporary_path, {"embedding": weight})
    os.replace(temporary_path, path)


def embedding_binding(weights: Weights, cfg: DeviceConfig) -> dict:
    """Checkpoint binding standing in for ``embedding.safetensors``."""
    key = weights.find(EMBEDDING_KEY)
    binding = {
        "engine_name": CHECKPOINT_BINDING_ENGINE_EMBEDDING,
        "role": CHECKPOINT_BINDING_ROLE_EMBEDDING,
        "checkpoint_keys": [key],
        "source_layout": "fp16",
        # This describes the runtime tensor, not the provider payload. The
        # checkpoint reader converts BF16/F32 embeddings to FP16 while loading.
        "dtype": "F16",
        "shape": [int(dim) for dim in weights.store.shape(key)],
        "embedding_scale": float(np.float16(cfg.embedding_scale)),
    }
    locations = weights.checkpoint_locations([key])
    if locations:
        binding["checkpoint_locations"] = locations
    return binding


def externalizes_ple(args, cfg: DeviceConfig) -> bool:
    """Whether Gemma4 PLE stays in the provider checkpoint."""
    return (cfg.hidden_size_per_layer_input > 0
            and args.ple_quantization == "fp16"
            and args.weight_policy.wants(EXTERNAL_WEIGHT_FP16))


def ple_embedding_binding(weights: Weights, cfg: DeviceConfig) -> dict:
    """Checkpoint binding standing in for ``ple_embedding.safetensors``."""
    key = weights.find("model.embed_tokens_per_layer.weight")
    binding = {
        "engine_name":
        CHECKPOINT_BINDING_ENGINE_PLE,
        "role":
        CHECKPOINT_BINDING_ROLE_PLE,
        "checkpoint_keys": [key],
        "source_layout":
        "fp16",
        "dtype":
        "F16",
        "shape": [int(dim) for dim in weights.store.shape(key)],
        "embedding_scale":
        float(np.float16(np.sqrt(cfg.hidden_size_per_layer_input))),
    }
    locations = weights.checkpoint_locations([key])
    if locations:
        binding["checkpoint_locations"] = locations
    return binding


def _scaled_rows(store, key: str, scale: float):
    """Row reader over one dense checkpoint table, scaled in FP32."""
    multiplier = np.float32(scale)

    def read_rows(start: int, stop: int) -> np.ndarray:
        return store.get_f32_rows(key, start, stop) * multiplier

    return read_rows


def write_embedding(weights: Weights, cfg: DeviceConfig, args,
                    engine_dir: str) -> None:
    fmt = args.embedding_quantization
    scale = np.float16(cfg.embedding_scale)
    path = os.path.join(engine_dir, "embedding.safetensors")
    # Runtime embeddings are FP16, including dequantized NVFP4 checkpoints.
    if weights.is_nvfp4("model.embed_tokens"):
        weight = weights.linear_fp16("model.embed_tokens")[0]
    elif fmt == "fp16":
        weight = weights.f16(EMBEDDING_KEY)
    else:
        key = weights.find(EMBEDDING_KEY)
        shape = weights.store.shape(key)
        write_quantized_table(path, "embedding", shape,
                              _scaled_rows(weights.store, key, scale), fmt,
                              table_group_size(fmt, shape[1]))
        return
    if fmt == "fp16":
        save_safetensors(path,
                         {"embedding": np.ascontiguousarray(weight * scale)})
        return
    write_quantized_table(
        path, "embedding", weight.shape, lambda start, stop:
        (weight[start:stop] * scale).astype(np.float32), fmt,
        table_group_size(fmt, weight.shape[1]))


def write_ple_embedding(weights: Weights, cfg: DeviceConfig, args,
                        engine_dir: str) -> None:
    fmt = args.ple_quantization
    key = weights.find("model.embed_tokens_per_layer.weight")
    shape = weights.store.shape(key)
    layer_width = cfg.hidden_size_per_layer_input
    write_quantized_table(
        os.path.join(engine_dir, "ple_embedding.safetensors"), "weight", shape,
        _scaled_rows(weights.store, key, np.float16(np.sqrt(layer_width))),
        fmt, table_group_size(fmt, shape[1], layer_width))


def copy_vocab_artifacts(args, engine_dir: str) -> None:
    source_dir = (args.draft_reduced_vocab_dir if args.resolved_spec_role
                  == contracts.SpecRole.DRAFT else args.reduced_vocab_dir)
    if not source_dir:
        return
    draft = args.resolved_spec_role == contracts.SpecRole.DRAFT
    destination_map = ("draft_vocab_map.safetensors"
                       if draft else "vocab_map.safetensors")
    source_map = os.path.join(source_dir, "vocab_map.safetensors")
    vocab_map = load_safetensors_tensor(source_map,
                                        "vocab_map").astype(np.int32)
    save_safetensors(os.path.join(engine_dir, destination_map),
                     {"vocab_map": vocab_map})
    source_info = os.path.join(source_dir, "reduced_vocab.json")
    if os.path.isfile(source_info):
        shutil.copy2(
            source_info,
            os.path.join(
                engine_dir,
                "draft_reduced_vocab.json" if draft else "reduced_vocab.json"))
