# Gemma 4 E2B on Jetson Orin Nano 8 GB

Gemma 4 E2B INT4 fits an 8 GB Orin Nano with three build options:

- `--ple-residency cached` keeps the per-layer embedding (PLE) table in the
  checkpoint file and reads only the rows of each step through the OS page
  cache, instead of uploading the table to the GPU.
- `--lm-head-quantization int4` quantizes the tied output head to INT4. The
  token embedding stays in its own table. This is optional: the model fits
  with an FP16 head, but INT4 decodes faster and leaves more memory free, at a
  small accuracy cost.
- `--embedding-quantization int8` stores that token embedding as INT8 with one
  scale per row, saving about 0.4 GiB.

Vision and audio encoders are built on the first request that carries images
or audio, so a text-only session does not load them.

## 1. Quantize (on a GPU host)

Do not quantize on the device. Run this on a GPU host and copy
`gemma4-e2b-int4` to the device:

```bash
tensorrt-edgellm-quantize llm --model_dir google/gemma-4-E2B-it \
  --output_dir gemma4-e2b-int4 --quantization int4_awq \
  --awq_full --num_samples 512 --calib_seq_len 2048
```

The recipe quantizes all text linears to INT4 (including `down_proj`) with
`awq_full`, calibrated on 512 text blocks of 2048 tokens. The per-layer gate
and projection linears stay FP16.

The community checkpoint
`Vishva007/gemma-4-E2B-it-W4A16-AutoRound-GPTQ` is also supported. Download it
instead of quantizing. It is plain round-to-nearest quantization and less
accurate than the recipe above.

## 2. Build Edge-LLM (on the Orin Nano)

From the repository root, with the JetPack 7.2 CUDA toolkit and TensorRT
installed:

```bash
python3 -m venv --system-site-packages .venv && . .venv/bin/activate && \
python -m pip install pybind11==3.0.4 && \
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin -DCUDA_CTK_VERSION=13.2 -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON -Dpybind11_DIR=$(python -m pybind11 --cmakedir) && \
cmake --build build -j2 && python -m pip install -e ".[server]"
```

`-j2` keeps the build within 8 GB of memory.

## 3. Serve

```bash
tensorrt-edgellm-serve gemma4-e2b-int4 \
  --max-input-len 2048 --max-kv-cache-capacity 2560 \
  --lm-head-quantization int4 --ple-residency cached \
  --embedding-quantization int8
```

The server listens on 0.0.0.0:8000; add `--host 127.0.0.1` to keep it local.
Run it in the shell where `.venv` is active. The first start builds the
engines into `~/.cache/tensorrt-edgellm` (about 20 minutes on the Orin Nano);
later starts reuse them. The checkpoint must stay in place while the server
runs, because the PLE rows are read from it.

To serve the community checkpoint, pass its directory instead; the options are
the same:

```bash
tensorrt-edgellm-serve gemma-4-E2B-it-W4A16-AutoRound-GPTQ \
  --max-input-len 2048 --max-kv-cache-capacity 2560 \
  --lm-head-quantization int4 --ple-residency cached \
  --embedding-quantization int8
```
