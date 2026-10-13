# Embedding Quantization

## Overview

The token embedding table (`embedding.safetensors`) and the Gemma 4 per-layer embedding table (`ple_embedding.safetensors`) can be stored quantized. The runtime dequantizes only the looked-up rows, so the output is FP16 and the tables stay compact in memory. This matters most for large vocabularies: Gemma 4 E2B has a 768 MiB FP16 token table and a 4480 MiB FP16 per-layer table.

| Format | Data | Scales | Group | Size vs FP16 |
|---|---|---|---|---|
| `fp16` (default) | FP16 | – | – | 100% |
| `fp8` | FP8 E4M3 | FP32 | 128 | ~52% |
| `int8` | INT8, symmetric | FP32 | one row (token); 256, one per layer (per-layer table) | ~50% |
| `int4` | INT4, symmetric, two per byte | FP32 | 128 | ~27% |
| `nvfp4` | FP4 E2M1, two per byte | FP8 E4M3 + one FP32 global scale | 16 | ~28% |

Quantization is direct round-to-nearest on the absolute maximum of each group; no calibration data is needed. The gather is plain CUDA and runs on every supported GPU, including Jetson Orin.

---

## Workflow

### ONNX Export

```bash
tensorrt-edgellm-export \
  /path/to/gemma-4-E2B-it \
  /tmp/gemma4_e2b_onnx \
  --embedding-quantization int8 \
  --ple-quantization int4
```

`--fp8-embedding` is an alias for `--embedding-quantization fp8`. The flags only change the sidecar files; the ONNX graph and the checkpoint weights are unchanged. Build and run the engine as usual: the format is detected from the sidecar tensors.

### Direct Engine Builder and Server

The direct builder and `tensorrt-edgellm-serve` accept the same `--embedding-quantization` and `--ple-quantization` flags. By default the runtime reads both FP16 tables from the checkpoint; a quantized choice writes the sidecar into the engine directory instead.

For models with tied word embeddings, `--lm-head-quantization int4` quantizes the LM head into the engine (symmetric, group 128, FP16 scales) for the INT4 GEMM plugin. It is opt-in: check accuracy on your task before deploying it. With an FP16 tied head, the runtime embedding shares the head's storage, so no second copy of the table is loaded.

---

## Technical Details

Both sidecars store the table already multiplied by the model's embedding scale. Tensor names, with `embedding` for the token table and `weight` for the per-layer table:

| Format | Tensors |
|---|---|
| `fp16` | `embedding` |
| `fp8`, `int8`, `int4` | `embedding`, `embedding_scale` (`[rows, cols / group]`) |
| `nvfp4` | `embedding`, `embedding_block_scale`, `embedding_multiplier_scale` |

- Packed formats store the even element in the low nibble. INT4 stores `q + 8`.
- NVFP4 sidecars carry their logical shape and group size in the safetensors metadata.
- Per-layer table groups never cross a layer boundary.

The writer streams the table in row chunks, so even the multi-GB per-layer table is quantized with bounded host memory.

---

## Limitations

- Embedding quantization applies to the LLM thinker only. For TTS, the Qwen3-Omni talker and code predictor, a warning is logged and FP16 embeddings are used.
- Speculative-decoding draft engines use the base model's embedding sidecar.
- `--lm-head-quantization int4` requires tied word embeddings and does not combine with `--reduced-vocab-dir` or tensor parallelism.
