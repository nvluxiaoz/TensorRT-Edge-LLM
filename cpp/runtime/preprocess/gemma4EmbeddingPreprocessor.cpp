/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <numeric>
#include <unistd.h>

namespace trt_edgellm
{
namespace rt
{

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir,
    LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream,
    std::optional<Tensor> checkpointTable, std::optional<PleFileSource> fileSource)
    : mConfig(config)
    , mFileSource(std::move(fileSource))
{
    ELLM_CHECK(mConfig.pleEnabled, "Gemma4EmbeddingPreprocessor constructed while PLE is disabled");
    ELLM_CHECK(maxBatchSize > 0, "Gemma4EmbeddingPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Gemma4EmbeddingPreprocessor requires positive max sequence length");

    if (mFileSource.has_value())
    {
        PleFileSource const& source = *mFileSource;
        ELLM_CHECK(source.dtype == nvinfer1::DataType::kHALF || source.dtype == nvinfer1::DataType::kBF16,
            "File-backed PLE table must be FP16 or BF16");
        ELLM_CHECK(source.cols == static_cast<int64_t>(mConfig.numPleInputs) * mConfig.pleHiddenSize,
            "PLE table second dimension must equal num_ple_inputs * ple_hidden_size");
        mFileDescriptor = ::open(source.file.c_str(), O_RDONLY | O_CLOEXEC);
        ELLM_CHECK(mFileDescriptor >= 0, "Failed to open PLE file " + source.file.string());
        // Rows are scattered across the table, so readahead would only fill the page cache with unused rows.
        ::posix_fadvise(mFileDescriptor, 0, 0, POSIX_FADV_RANDOM);

        int64_t const capacity = static_cast<int64_t>(maxBatchSize) * maxSeqLen;
        mHostIds = Tensor({capacity}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "Gemma4Ple::mHostIds");
        mHostRows
            = Tensor({capacity, source.cols}, DeviceType::kCPU, nvinfer1::DataType::kHALF, "Gemma4Ple::mHostRows");
        mDeviceRows
            = Tensor({capacity, source.cols}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "Gemma4Ple::mDeviceRows");
        mDeviceIota = Tensor({capacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4Ple::mDeviceIota");
        int32_t* hostIds = mHostIds.dataPointer<int32_t>();
        std::iota(hostIds, hostIds + capacity, 0);
        CUDA_CHECK(cudaMemcpyAsync(
            mDeviceIota.rawPointer(), hostIds, capacity * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    else if (checkpointTable.has_value())
    {
        mPleTable = std::move(*checkpointTable);
    }
    else
    {
        std::filesystem::path const plePath = engineDir / binding_names::kPleEmbeddingFileName;
        std::vector<Tensor> pleTensors;
        ELLM_CHECK(safetensors::loadSafetensors(plePath, pleTensors, stream),
            "Failed to load " + std::string(binding_names::kPleEmbeddingFileName)
                + " from model directory: " + engineDir.string());
        ELLM_CHECK(pleTensors.size() == 1, "ple_embedding.safetensors must contain exactly one tensor named weight");
        ELLM_CHECK(pleTensors[0].getName() == "weight", "ple_embedding.safetensors tensor must be named weight");
        mPleTable = std::move(pleTensors[0]);
    }

    if (!mFileSource.has_value())
    {
        auto const pleShape = mPleTable.getShape();
        ELLM_CHECK(pleShape.getNumDims() == 2, "PLE table must be 2D [vocab, num_layers * hidden]");
        ELLM_CHECK(pleShape[1] == static_cast<int64_t>(mConfig.numPleInputs) * mConfig.pleHiddenSize,
            "PLE table second dimension must equal num_ple_inputs * ple_hidden_size");
        ELLM_CHECK(mPleTable.getDataType() == nvinfer1::DataType::kHALF
                || mPleTable.getDataType() == nvinfer1::DataType::kBF16,
            "PLE table must be FP16 or BF16");
    }
    nvinfer1::DataType const outputType = mFileSource.has_value() ? nvinfer1::DataType::kHALF : mPleTable.getDataType();

    mPleOutputBuffer = Tensor({mConfig.numPleInputs, maxBatchSize, maxSeqLen, mConfig.pleHiddenSize}, DeviceType::kGPU,
        outputType, "Gemma4EmbeddingPreprocessor::mPleOutputBuffer");

    mPleOutputViews.reserve(mConfig.numPleInputs);
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews.emplace_back(makeTokenMajorOutputViewForLayer(idx, maxBatchSize * maxSeqLen));
        tensorMap.set(mPleOutputViews.back().getName(), mPleOutputViews.back());
    }

    LOG_INFO("Initialized Gemma4 PLE preprocessor: table=%s outputBuffer=%s numPleInputs=%d pleHiddenSize=%d",
        mFileSource.has_value() ? "file-backed" : mPleTable.getShape().formatString().c_str(),
        mPleOutputBuffer.getShape().formatString().c_str(), mConfig.numPleInputs, mConfig.pleHiddenSize);
}

Gemma4EmbeddingPreprocessor::~Gemma4EmbeddingPreprocessor() noexcept
{
    if (mFileDescriptor >= 0)
    {
        ::close(mFileDescriptor);
    }
}

Tensor Gemma4EmbeddingPreprocessor::makeTokenMajorOutputViewForLayer(int32_t layerIdx, int64_t physicalTokens)
{
    ELLM_CHECK(layerIdx >= 0 && layerIdx < mConfig.numPleInputs, "Gemma4 PLE layer index out of range");
    auto const outputShape = mPleOutputBuffer.getShape();
    ELLM_CHECK(physicalTokens > 0, "Gemma4 PLE physical token count must be positive");
    ELLM_CHECK(
        physicalTokens <= outputShape[1] * outputShape[2], "Gemma4 PLE physical token count exceeds buffer capacity");

    int64_t const layerOutputCapacityBytes = outputShape[1] * outputShape[2] * mConfig.pleHiddenSize
        * static_cast<int64_t>(utils::getTypeSize(mPleOutputBuffer.getDataType()));
    void* const layerOutputPtr
        = static_cast<void*>(static_cast<char*>(mPleOutputBuffer.rawPointer()) + layerIdx * layerOutputCapacityBytes);
    return Tensor(layerOutputPtr, Coords{physicalTokens, mConfig.pleHiddenSize}, DeviceType::kGPU,
        mPleOutputBuffer.getDataType(), binding_names::formatPleTokenEmbedsName(layerIdx));
}

void Gemma4EmbeddingPreprocessor::reshapeOutputsTokenMajor(int64_t physicalTokens)
{
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews[idx] = makeTokenMajorOutputViewForLayer(idx, physicalTokens);
    }
}

void Gemma4EmbeddingPreprocessor::embed(Tensor const& tokenIds, cudaStream_t stream)
{
    auto const tokenShape = tokenIds.getShape();
    ELLM_CHECK(tokenShape.getNumDims() == 2, "Gemma4 PLE token IDs must be [batch, seq_len]");
    if (mFileSource.has_value())
    {
        embedFromFile(tokenIds, stream);
    }
    else
    {
        kernel::gemma4PleGather(tokenIds, mPleTable, mPleOutputBuffer, mConfig.numPleInputs, mConfig.pleHiddenSize,
            mConfig.imageTokenId, mConfig.audioTokenId, stream);
    }
    reshapeOutputsTokenMajor(tokenShape[0] * tokenShape[1]);
}

void Gemma4EmbeddingPreprocessor::embedFromFile(Tensor const& tokenIds, cudaStream_t stream)
{
    PleFileSource const& source = *mFileSource;
    auto const tokenShape = tokenIds.getShape();
    int64_t const numTokens = tokenShape[0] * tokenShape[1];
    ELLM_CHECK(numTokens <= mHostIds.getShape()[0], "Gemma4 PLE token count exceeds the staging capacity");

    // The sync also guarantees the previous step finished reading the pinned rows.
    CUDA_CHECK(cudaMemcpyAsync(
        mHostIds.rawPointer(), tokenIds.rawPointer(), numTokens * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    bool const isBf16 = source.dtype == nvinfer1::DataType::kBF16;
    std::vector<uint16_t> raw(static_cast<size_t>(source.cols));
    auto* rows = mHostRows.dataPointer<half>();
    int32_t const* ids = mHostIds.dataPointer<int32_t>();
    for (int64_t token = 0; token < numTokens; ++token)
    {
        half* row = rows + token * source.cols;
        int32_t const id = ids[token];
        // Image and audio positions carry no per-layer embedding.
        if (id < 0 || id >= source.rows || id == mConfig.imageTokenId || id == mConfig.audioTokenId)
        {
            std::memset(row, 0, static_cast<size_t>(source.cols) * sizeof(half));
            continue;
        }
        size_t const rowBytes = raw.size() * sizeof(uint16_t);
        ssize_t const bytesRead = ::pread(mFileDescriptor, raw.data(), rowBytes, source.offset + id * rowBytes);
        ELLM_CHECK(bytesRead == static_cast<ssize_t>(rowBytes), "Failed to read PLE row from " + source.file.string());
        for (int64_t col = 0; col < source.cols; ++col)
        {
            float value{};
            if (isBf16)
            {
                __nv_bfloat16_raw const bits{raw[col]};
                value = __bfloat162float(__nv_bfloat16(bits));
            }
            else
            {
                __half_raw const bits{raw[col]};
                value = __half2float(half(bits));
            }
            row[col] = __float2half(value * source.scale);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(mDeviceRows.rawPointer(), mHostRows.rawPointer(), numTokens * source.cols * sizeof(half),
        cudaMemcpyHostToDevice, stream));
    Tensor const table(
        mDeviceRows.rawPointer(), Coords{numTokens, source.cols}, DeviceType::kGPU, nvinfer1::DataType::kHALF);
    Tensor const rowIndex(mDeviceIota.rawPointer(), tokenShape, DeviceType::kGPU, nvinfer1::DataType::kINT32);
    kernel::gemma4PleGather(
        rowIndex, table, mPleOutputBuffer, mConfig.numPleInputs, mConfig.pleHiddenSize, -1, -1, stream);
}

} // namespace rt
} // namespace trt_edgellm
