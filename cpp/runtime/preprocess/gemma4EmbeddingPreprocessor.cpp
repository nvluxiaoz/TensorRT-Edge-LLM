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
#include "common/mmapReader.h"
#include "common/safetensorsUtils.h"
#include "common/stringUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <unistd.h>

namespace trt_edgellm
{
namespace rt
{
namespace
{

//! pread exactly `bytes` bytes, retrying interrupted and short reads.
void readFileRange(int32_t fd, void* destination, size_t bytes, int64_t offset, std::filesystem::path const& file)
{
    auto* out = static_cast<uint8_t*>(destination);
    while (bytes > 0)
    {
        ssize_t const count = ::pread(fd, out, bytes, static_cast<off_t>(offset));
        if (count < 0 && errno == EINTR)
        {
            continue;
        }
        ELLM_CHECK(count > 0,
            format::fmtstr("Failed to read %zu PLE bytes at offset %ld of %s: %s", bytes, offset, file.c_str(),
                count == 0 ? "unexpected end of file" : std::strerror(errno)));
        out += count;
        bytes -= static_cast<size_t>(count);
        offset += count;
    }
}

//! Location of one tensor in a safetensors file; with onlyTensor the file must hold nothing else.
PleFileSource locateFileTensor(std::filesystem::path const& file, std::string const& tensorName, bool onlyTensor)
{
    file_io::MmapReader const reader(file);
    safetensors::FileMetadata const metadata
        = safetensors::parseMetadata(reader.getData(), reader.getSize(), file.string());
    auto const tensor = std::find_if(metadata.tensors.begin(), metadata.tensors.end(),
        [&tensorName](safetensors::TensorMetadata const& candidate) { return candidate.name == tensorName; });
    ELLM_CHECK(tensor != metadata.tensors.end(), "PLE tensor " + tensorName + " not found in " + file.string());
    ELLM_CHECK(!onlyTensor || metadata.tensors.size() == 1,
        "Cached PLE residency reads F16 or BF16 tables, but " + file.string()
            + " holds a quantized table; use resident PLE");
    ELLM_CHECK(tensor->shape.getNumDims() == 2,
        "PLE tensor " + tensorName + " must be 2D, got " + tensor->shape.formatString());
    return PleFileSource{tensorName, file, static_cast<int64_t>(metadata.dataOffset + tensor->offset), tensor->dataType,
        tensor->shape[0], tensor->shape[1], 1.0F};
}

} // namespace

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir,
    LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream,
    std::optional<Tensor> checkpointTable, PleRuntimeConfig const& pleConfig, std::optional<PleFileSource> fileSource)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.pleEnabled, "Gemma4EmbeddingPreprocessor constructed while PLE is disabled");
    ELLM_CHECK(maxBatchSize > 0, "Gemma4EmbeddingPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Gemma4EmbeddingPreprocessor requires positive max sequence length");

    int64_t pleColumns{0};
    nvinfer1::DataType outputType{nvinfer1::DataType::kHALF};
    if (pleConfig.residency == PleResidency::kCACHED)
    {
        ELLM_CHECK(!checkpointTable.has_value(), "Cached PLE residency cannot use a resident PLE table");
        // Checkpoint rows feed an FP16 engine input; sidecar rows keep their dtype as in resident residency.
        bool const fromCheckpoint = fileSource.has_value();
        mFileSource = std::move(fileSource);
        if (!fromCheckpoint)
        {
            mFileSource = locateFileTensor(engineDir / binding_names::kPleEmbeddingFileName, "weight", true);
        }
        if (!pleConfig.filePath.empty())
        {
            PleFileSource relocated = locateFileTensor(pleConfig.filePath, mFileSource->tensorName, false);
            ELLM_CHECK(relocated.dtype == mFileSource->dtype && relocated.rows == mFileSource->rows
                    && relocated.cols == mFileSource->cols,
                "PLE tensor " + relocated.tensorName + " in " + pleConfig.filePath
                    + " differs in dtype or shape from the recorded PLE source " + mFileSource->file.string());
            relocated.scale = mFileSource->scale;
            mFileSource = std::move(relocated);
        }
        PleFileSource const& source = *mFileSource;
        ELLM_CHECK(source.dtype == nvinfer1::DataType::kHALF || source.dtype == nvinfer1::DataType::kBF16,
            format::fmtstr("Cached PLE residency reads F16 or BF16 rows, got dtype %d in %s",
                static_cast<int32_t>(source.dtype), source.file.c_str()));
        ELLM_CHECK(source.rows > 0 && source.rows <= std::numeric_limits<int32_t>::max(),
            format::fmtstr("PLE vocabulary %ld is outside the INT32 token range", source.rows));
        int64_t const rowBytes = source.cols * static_cast<int64_t>(utils::getTypeSize(source.dtype));
        auto const fileBytes = static_cast<int64_t>(std::filesystem::file_size(source.file));
        ELLM_CHECK(
            source.offset >= 0 && source.offset <= fileBytes && source.rows <= (fileBytes - source.offset) / rowBytes,
            format::fmtstr("PLE tensor [%ld, %ld] at offset %ld exceeds the %ld bytes of %s", source.rows, source.cols,
                source.offset, fileBytes, source.file.c_str()));
        pleColumns = source.cols;
        outputType = fromCheckpoint ? nvinfer1::DataType::kHALF : source.dtype;

        mFileDescriptor = ::open(source.file.c_str(), O_RDONLY | O_CLOEXEC);
        ELLM_CHECK(
            mFileDescriptor >= 0, "Failed to open PLE file " + source.file.string() + ": " + std::strerror(errno));
        // Rows are scattered across the table; readahead would only fill the page cache with unused rows.
        int32_t const adviceStatus = ::posix_fadvise(mFileDescriptor, 0, 0, POSIX_FADV_RANDOM);
        if (adviceStatus != 0)
        {
            LOG_WARNING("posix_fadvise(RANDOM) failed for %s: %s", source.file.c_str(), std::strerror(adviceStatus));
        }

        int64_t const capacity = static_cast<int64_t>(maxBatchSize) * maxSeqLen;
        mHostRows
            = Tensor({capacity, source.cols}, DeviceType::kCPU, source.dtype, "Gemma4EmbeddingPreprocessor::mHostRows");
        mDeviceRows = Tensor(
            {capacity, source.cols}, DeviceType::kGPU, source.dtype, "Gemma4EmbeddingPreprocessor::mDeviceRows");
        mHostRowIndex = Tensor(
            {capacity}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "Gemma4EmbeddingPreprocessor::mHostRowIndex");
        mDeviceRowIndex = Tensor(
            {capacity}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "Gemma4EmbeddingPreprocessor::mDeviceRowIndex");
        mStagedSlot.assign(static_cast<size_t>(source.rows), -1);
        mStagedTokens.resize(static_cast<size_t>(capacity));
        LOG_INFO("Gemma4 PLE rows are read from %s (tensor %s, offset %ld, scale %g)", source.file.c_str(),
            source.tensorName.c_str(), source.offset, source.scale);
    }
    else
    {
        ELLM_CHECK(!fileSource.has_value() && pleConfig.filePath.empty(), "A PLE file requires cached PLE residency");
        if (checkpointTable.has_value())
        {
            mPleTable.table = std::move(*checkpointTable);
        }
        else
        {
            mPleTable = loadEmbeddingTable(engineDir / binding_names::kPleEmbeddingFileName, "weight", stream);
        }
        pleColumns = kernel::validateEmbeddingTable(mPleTable.descriptor());
        outputType = mPleTable.table.getDataType() == nvinfer1::DataType::kBF16 ? nvinfer1::DataType::kBF16
                                                                                : nvinfer1::DataType::kHALF;
    }
    ELLM_CHECK(pleColumns == static_cast<int64_t>(mConfig.numPleInputs) * mConfig.pleHiddenSize,
        format::fmtstr("PLE table width %ld must equal num_ple_inputs * ple_hidden_size = %d * %d", pleColumns,
            mConfig.numPleInputs, mConfig.pleHiddenSize));

    mPleOutputBuffer = Tensor({mConfig.numPleInputs, maxBatchSize, maxSeqLen, mConfig.pleHiddenSize}, DeviceType::kGPU,
        outputType, "Gemma4EmbeddingPreprocessor::mPleOutputBuffer");

    mPleOutputViews.reserve(mConfig.numPleInputs);
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews.emplace_back(makeTokenMajorOutputViewForLayer(idx, maxBatchSize * maxSeqLen));
        tensorMap.set(mPleOutputViews.back().getName(), mPleOutputViews.back());
    }

    LOG_INFO("Initialized Gemma4 PLE preprocessor: table=%s outputBuffer=%s numPleInputs=%d pleHiddenSize=%d",
        mPleTable.table.getShape().formatString().c_str(), mPleOutputBuffer.getShape().formatString().c_str(),
        mConfig.numPleInputs, mConfig.pleHiddenSize);
}

Gemma4EmbeddingPreprocessor::~Gemma4EmbeddingPreprocessor() noexcept
{
    if (mFileDescriptor >= 0)
    {
        static_cast<void>(::close(mFileDescriptor));
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
    embed(tokenIds, nullptr, stream);
}

void Gemma4EmbeddingPreprocessor::embed(Tensor const& tokenIds, int32_t const* hostTokenIds, cudaStream_t stream)
{
    auto const tokenShape = tokenIds.getShape();
    ELLM_CHECK(tokenShape.getNumDims() == 2, "Gemma4 PLE token IDs must be [batch, seq_len]");
    int64_t const numTokens = tokenShape[0] * tokenShape[1];
    if (!mFileSource.has_value())
    {
        kernel::gemma4PleGather(tokenIds, mPleTable.descriptor(), mPleOutputBuffer, mConfig.numPleInputs,
            mConfig.pleHiddenSize, mConfig.imageTokenId, mConfig.audioTokenId, stream);
    }
    else
    {
        ELLM_CHECK(hostTokenIds != nullptr, "Cached PLE residency requires the host token IDs");
        ELLM_CHECK(numTokens <= mHostRowIndex.getShape()[0],
            format::fmtstr(
                "Gemma4 PLE token count %ld exceeds the staging capacity %ld", numTokens, mHostRowIndex.getShape()[0]));
        int32_t const stagedRows = stageRows(hostTokenIds, numTokens, stream);
        Tensor const rows(mDeviceRows.rawPointer(), Coords{std::max(stagedRows, 1), mFileSource->cols},
            DeviceType::kGPU, mFileSource->dtype);
        Tensor const rowIndex(mDeviceRowIndex.rawPointer(), tokenShape, DeviceType::kGPU, nvinfer1::DataType::kINT32);
        kernel::gemma4PleGather(tokenIds, kernel::EmbeddingTableDescriptor{rows, std::nullopt, mFileSource->scale},
            mPleOutputBuffer, mConfig.numPleInputs, mConfig.pleHiddenSize, mConfig.imageTokenId, mConfig.audioTokenId,
            stream, rowIndex);
    }
    reshapeOutputsTokenMajor(numTokens);
}

int32_t Gemma4EmbeddingPreprocessor::stageRows(int32_t const* hostTokenIds, int64_t numTokens, cudaStream_t stream)
{
    PleFileSource const& source = *mFileSource;
    // The previous embed() may still be uploading from the pinned buffers.
    mStagingFence.wait();
    int32_t* rowIndex = mHostRowIndex.dataPointer<int32_t>();
    int32_t staged{0};
    for (int64_t token = 0; token < numTokens; ++token)
    {
        int32_t const tokenId = hostTokenIds[token];
        bool const zeroFill = tokenId < 0 || tokenId >= source.rows
            || (mConfig.imageTokenId >= 0 && tokenId == mConfig.imageTokenId)
            || (mConfig.audioTokenId >= 0 && tokenId == mConfig.audioTokenId);
        if (zeroFill)
        {
            rowIndex[token] = -1;
            continue;
        }
        int32_t& slot = mStagedSlot[static_cast<size_t>(tokenId)];
        if (slot < 0)
        {
            slot = staged;
            mStagedTokens[static_cast<size_t>(staged++)] = tokenId;
        }
        rowIndex[token] = slot;
    }

    int64_t const rowBytes = source.cols * static_cast<int64_t>(utils::getTypeSize(source.dtype));
    auto* hostRows = static_cast<uint8_t*>(mHostRows.rawPointer());
    for (int32_t slot = 0; slot < staged; ++slot)
    {
        mStagedSlot[static_cast<size_t>(mStagedTokens[static_cast<size_t>(slot)])] = -1;
    }
    for (int32_t slot = 0; slot < staged; ++slot)
    {
        int64_t const tokenId = mStagedTokens[static_cast<size_t>(slot)];
        readFileRange(mFileDescriptor, hostRows + slot * rowBytes, static_cast<size_t>(rowBytes),
            source.offset + tokenId * rowBytes, source.file);
    }

    if (staged > 0)
    {
        CUDA_CHECK(
            cudaMemcpyAsync(mDeviceRows.rawPointer(), hostRows, staged * rowBytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        mDeviceRowIndex.rawPointer(), rowIndex, numTokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    mStagingFence.record(stream);
    return staged;
}

} // namespace rt
} // namespace trt_edgellm
