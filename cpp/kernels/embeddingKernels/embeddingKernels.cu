/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "common/checkMacros.h"
#include "common/stringUtils.h"
#include "embeddingKernels.h"
#include "kernels/common/vectorizedTypes.cuh"
#include <cmath>
#include <cub/block/block_scan.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>
#include <type_traits>
#if SUPPORTS_FP8
#include <cuda_fp8.h>
#endif

namespace trt_edgellm
{
namespace kernel
{

namespace
{

enum class TableFormat
{
    kFP16,
    kBF16,
    kFP8,
    kINT8,
    kINT4,
    kNVFP4,
};

//! Each thread decodes one 16-byte load of packed elements per step.
constexpr int64_t kLoadBytes = 16;
//! Elements per output vector store; scale groups are a multiple of this.
constexpr int64_t kStoreElems = DVec<half>::vec_size;
constexpr int32_t kInt4ZeroPoint = 8;
constexpr uint32_t kNibbleBits = 4;
constexpr uint32_t kNibbleMask = 0xFU;
constexpr int64_t kGatherWarpSize = 32;
constexpr int64_t kLookupWarpsPerBlock = 4;
constexpr int64_t kPleGatherThreads = 256;

template <TableFormat F>
constexpr int64_t kBitsPerElement = (F == TableFormat::kFP16 || F == TableFormat::kBF16) ? 16
    : (F == TableFormat::kFP8 || F == TableFormat::kINT8)                                ? 8
                                                                                         : 4;

template <TableFormat F>
constexpr int64_t kChunkElems = kLoadBytes * 8 / kBitsPerElement<F>;

//! Validated device view of an EmbeddingTableDescriptor; cols is the logical row width.
struct TableView
{
    TableFormat format{TableFormat::kFP16};
    uint8_t const* data{nullptr};
    void const* scale{nullptr};
    int64_t rows{0};
    int64_t cols{0};
    int64_t rowBytes{0};
    int64_t group{0};
    int64_t groupsPerRow{0};
    float multiplier{1.0F};
};

template <typename Fn>
void dispatchFormat(TableFormat format, Fn&& fn)
{
    switch (format)
    {
    case TableFormat::kFP16: fn(std::integral_constant<TableFormat, TableFormat::kFP16>{}); break;
    case TableFormat::kBF16: fn(std::integral_constant<TableFormat, TableFormat::kBF16>{}); break;
    case TableFormat::kFP8: fn(std::integral_constant<TableFormat, TableFormat::kFP8>{}); break;
    case TableFormat::kINT8: fn(std::integral_constant<TableFormat, TableFormat::kINT8>{}); break;
    case TableFormat::kINT4: fn(std::integral_constant<TableFormat, TableFormat::kINT4>{}); break;
    case TableFormat::kNVFP4: fn(std::integral_constant<TableFormat, TableFormat::kNVFP4>{}); break;
    }
}

int64_t chunkElems(TableFormat format)
{
    int64_t elems{0};
    dispatchFormat(format, [&](auto tag) { elems = kChunkElems<decltype(tag)::value>; });
    return elems;
}

TableView makeTableView(EmbeddingTableDescriptor const& table)
{
    rt::Tensor const& data = table.data;
    auto const shape = data.getShape();
    nvinfer1::DataType const dataType = data.getDataType();
    check::check(shape.getNumDims() == 2,
        format::fmtstr("embedding table must be 2D [rows, cols], got %s", shape.formatString().c_str()));
    check::check(std::isfinite(table.multiplier), "embedding table multiplier must be finite");

    TableView view;
    view.data = static_cast<uint8_t const*>(data.rawPointer());
    view.rows = shape[0];
    view.rowBytes = shape[1] * static_cast<int64_t>(rt::utils::getTypeSize(dataType));
    view.cols = shape[1];
    view.multiplier = table.multiplier;
    if (!table.scale.has_value())
    {
        check::check(dataType == nvinfer1::DataType::kHALF || dataType == nvinfer1::DataType::kBF16,
            format::fmtstr(
                "embedding table without scales must be FP16 or BF16, got dtype %d", static_cast<int32_t>(dataType)));
        view.format = dataType == nvinfer1::DataType::kHALF ? TableFormat::kFP16 : TableFormat::kBF16;
    }
    else
    {
        rt::Tensor const& scale = table.scale->get();
        nvinfer1::DataType const scaleType = scale.getDataType();
        auto const scaleShape = scale.getShape();
        if (dataType == nvinfer1::DataType::kUINT8)
        {
            view.format = scaleType == nvinfer1::DataType::kFP8 ? TableFormat::kNVFP4 : TableFormat::kINT4;
            view.cols = shape[1] * 2;
        }
        else
        {
            check::check(dataType == nvinfer1::DataType::kFP8 || dataType == nvinfer1::DataType::kINT8,
                format::fmtstr(
                    "scaled embedding table must be FP8, INT8 or UINT8, got dtype %d", static_cast<int32_t>(dataType)));
            view.format = dataType == nvinfer1::DataType::kFP8 ? TableFormat::kFP8 : TableFormat::kINT8;
        }
        check::check(
            scaleType == (view.format == TableFormat::kNVFP4 ? nvinfer1::DataType::kFP8 : nvinfer1::DataType::kFLOAT),
            format::fmtstr("embedding scale dtype %d does not match the table dtype %d",
                static_cast<int32_t>(scaleType), static_cast<int32_t>(dataType)));
        check::check(scaleShape.getNumDims() == 2 && scaleShape[0] == view.rows && scaleShape[1] > 0
                && view.cols % scaleShape[1] == 0,
            format::fmtstr("embedding scale shape %s does not divide table [%ld, %ld] into row groups",
                scaleShape.formatString().c_str(), view.rows, view.cols));
        view.scale = scale.rawPointer();
        view.groupsPerRow = scaleShape[1];
        view.group = view.cols / view.groupsPerRow;
        check::check(view.group % kStoreElems == 0,
            format::fmtstr("embedding scale group %ld must be a multiple of %ld", view.group, kStoreElems));
    }
#if !SUPPORTS_FP8
    check::check(view.format != TableFormat::kFP8 && view.format != TableFormat::kNVFP4,
        "FP8 and NVFP4 embedding tables require a build with FP8 support (SUPPORTS_FP8=0)");
#endif
    int64_t const elems = chunkElems(view.format);
    check::check(view.cols % elems == 0,
        format::fmtstr("embedding table width %ld must be a multiple of %ld for vectorized access", view.cols, elems));
    return view;
}

__device__ __forceinline__ float decodeE2m1(uint32_t nibble)
{
    // E2M1 magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6}: codes >= 2 map to FP32 by shifting exponent and mantissa.
    uint32_t const magnitude = nibble & 0x7U;
    float const value = magnitude == 0U ? 0.0F
        : magnitude == 1U               ? 0.5F
                                        : __uint_as_float(0x3F000000U + (magnitude << 22U));
    return (nibble & 0x8U) != 0U ? -value : value;
}

template <TableFormat F>
__device__ __forceinline__ float decodeElement(uint8_t const* bytes, uint32_t idx)
{
    if constexpr (F == TableFormat::kFP16)
    {
        return __half2float(reinterpret_cast<half const*>(bytes)[idx]);
    }
    else if constexpr (F == TableFormat::kBF16)
    {
        return __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(bytes)[idx]);
    }
    else if constexpr (F == TableFormat::kINT8)
    {
        return static_cast<float>(reinterpret_cast<int8_t const*>(bytes)[idx]);
    }
    else if constexpr (F == TableFormat::kFP8)
    {
#if SUPPORTS_FP8
        return static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const*>(bytes)[idx]);
#else
        return 0.0F;
#endif
    }
    else
    {
        uint32_t const nibble = (bytes[idx / 2] >> ((idx % 2) * kNibbleBits)) & kNibbleMask;
        if constexpr (F == TableFormat::kINT4)
        {
            return static_cast<float>(static_cast<int32_t>(nibble) - kInt4ZeroPoint);
        }
        else
        {
            return decodeE2m1(nibble);
        }
    }
}

//! Scale times multiplier for the group holding `col`.
template <TableFormat F>
__device__ __forceinline__ float groupScale(TableView const& table, int64_t row, int64_t col)
{
    if constexpr (F == TableFormat::kFP16 || F == TableFormat::kBF16)
    {
        return table.multiplier;
    }
    else
    {
        int64_t const idx = row * table.groupsPerRow + col / table.group;
        if constexpr (F == TableFormat::kNVFP4)
        {
#if SUPPORTS_FP8
            return static_cast<float>(static_cast<__nv_fp8_e4m3 const*>(table.scale)[idx]) * table.multiplier;
#else
            return 0.0F;
#endif
        }
        else
        {
            return static_cast<float const*>(table.scale)[idx] * table.multiplier;
        }
    }
}

//! Decode kChunkElems<F> consecutive elements of one row starting at `col` from a single 16-byte load.
template <TableFormat F>
__device__ __forceinline__ void loadChunk(
    TableView const& table, int64_t row, int64_t col, float (&values)[kChunkElems<F>])
{
    uint4 const packed
        = *reinterpret_cast<uint4 const*>(table.data + row * table.rowBytes + col * kBitsPerElement<F> / 8);
    uint8_t const* bytes = reinterpret_cast<uint8_t const*>(&packed);
#pragma unroll
    for (uint32_t base = 0; base < kChunkElems<F>; base += kStoreElems)
    {
        float const scale = groupScale<F>(table, row, col + base);
#pragma unroll
        for (uint32_t i = 0; i < kStoreElems; ++i)
        {
            values[base + i] = decodeElement<F>(bytes, base + i) * scale;
        }
    }
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ half fromFloat<half>(float value)
{
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <typename T, int64_t N>
__device__ __forceinline__ void storeChunk(float const (&values)[N], T* dst)
{
#pragma unroll
    for (uint32_t base = 0; base < N; base += kStoreElems)
    {
        DVec<T> out;
#pragma unroll
        for (uint32_t i = 0; i < kStoreElems; ++i)
        {
            out[i] = fromFloat<T>(values[base + i]);
        }
        out.store(dst + base);
    }
}

template <typename T, int64_t N>
__device__ __forceinline__ void zeroChunk(T* dst)
{
#pragma unroll
    for (uint32_t base = 0; base < N; base += kStoreElems)
    {
        DVec<T> out;
        out.data = make_uint4(0U, 0U, 0U, 0U);
        out.store(dst + base);
    }
}

// CUDA kernel for assembling deepstack embeddings (FP16 only)
// Extracts image-token embeddings from deepstack features. Positions whose token id == imageTokenId
// are mapped to a deepstack feature row via multimodalIndices; all other positions get zero embeddings.
__global__ void assembleDeepstackEmbeddingKernel(int32_t const* inputIds, half const* deepstackFeatures, half* output,
    int64_t batchSize, int64_t seqLen, int32_t imageTokenId, int64_t hiddenSize, int64_t numImageTokens,
    int32_t const* multimodalIndices)
{
    // Each warp handles one hidden state (one token's embedding)
    // Each thread processes 8 FP16 elements (128-bit granularity)
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    constexpr uint32_t warpSize = 32;

    // Use 2D CTA: (32, 4) - warp index directly from blockIdx.x * blockDim.y + threadIdx.y
    uint32_t const warpId = blockIdx.x * blockDim.y + threadIdx.y;
    uint32_t const laneId = threadIdx.x;

    if (warpId >= batchSize * seqLen)
    {
        return;
    }

    // Get token ID (warpId == batchIdx * seqLen + tokenIdx)
    int64_t const pos = static_cast<int64_t>(warpId);
    int32_t const tokenId = inputIds[pos];

    // Image tokens are identified by the explicit imageTokenId; their deepstack
    // feature row is selected via multimodalIndices.
    bool const isImageToken = (imageTokenId > 0 && tokenId == imageTokenId);

    // Calculate base indices for this warp's work
    uint32_t const baseOutputIdx = warpId * hiddenSize;

    // Each thread processes vecSize elements, loop until we cover the entire hidden state
    for (uint32_t offset = laneId * vecSize; offset < hiddenSize; offset += warpSize * vecSize)
    {
        DVec<half> embeddingVec;

        if (isImageToken)
        {
            // Select the deepstack feature row for this image position.
            int32_t const deepstackIdx = multimodalIndices[pos];

            // Validate that deepstackIdx is within bounds
            if (deepstackIdx >= 0 && deepstackIdx < numImageTokens)
            {
                // Load embedding data from deepstack features
                uint32_t const embeddingOffset = deepstackIdx * hiddenSize + offset;
                embeddingVec.load(deepstackFeatures + embeddingOffset);
            }
            else
            {
                // Out-of-bounds image token, use zero embedding
#pragma unroll
                for (uint32_t i = 0; i < vecSize; ++i)
                {
                    embeddingVec[i] = __float2half(0.0f);
                }
            }
        }
        else
        {
            // Not an image token, use zero embedding
#pragma unroll
            for (uint32_t i = 0; i < vecSize; ++i)
            {
                embeddingVec[i] = __float2half(0.0f);
            }
        }

        // Store to output
        uint32_t const outputIdx = baseOutputIdx + offset;
        embeddingVec.store(output + outputIdx);
    }
}

//! \brief Embedding lookup kernel (text + optional image/audio); one warp per token.
//! Text tokens are decoded from the table, image/audio tokens copy FP16 embeds.
template <TableFormat F>
__global__ void embeddingLookupKernelImpl(int32_t const* inputIds, TableView table, int32_t const* multimodalIndices,
    int32_t imageTokenId, half const* imageEmbeds, int64_t imageTokenLen, int32_t audioTokenId, half const* audioEmbeds,
    int64_t audioTokenLen, half* output, int64_t numTokens)
{
    constexpr int64_t kElems = kChunkElems<F>;
    int64_t const tokenIdx = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    int64_t const laneId = threadIdx.x;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    int32_t const tokenId = inputIds[tokenIdx];
    int64_t const hiddenSize = table.cols;
    half const* mediaRow = nullptr;
    bool isValidTextToken = false;
    if (imageEmbeds != nullptr && tokenId == imageTokenId)
    {
        int32_t const multimodalIdx = multimodalIndices[tokenIdx];
        mediaRow
            = multimodalIdx >= 0 && multimodalIdx < imageTokenLen ? imageEmbeds + multimodalIdx * hiddenSize : nullptr;
    }
    else if (audioEmbeds != nullptr && tokenId == audioTokenId)
    {
        int32_t const multimodalIdx = multimodalIndices[tokenIdx];
        mediaRow
            = multimodalIdx >= 0 && multimodalIdx < audioTokenLen ? audioEmbeds + multimodalIdx * hiddenSize : nullptr;
    }
    else
    {
        isValidTextToken = tokenId >= 0 && tokenId < table.rows;
    }

    for (int64_t col = laneId * kElems; col < hiddenSize; col += kGatherWarpSize * kElems)
    {
        half* dst = output + tokenIdx * hiddenSize + col;
        if (isValidTextToken)
        {
            float values[kElems];
            loadChunk<F>(table, tokenId, col, values);
            storeChunk(values, dst);
        }
        else if (mediaRow != nullptr)
        {
#pragma unroll
            for (int64_t base = 0; base < kElems; base += kStoreElems)
            {
                DVec<half> media;
                media.load(mediaRow + col + base);
                media.store(dst + base);
            }
        }
        else
        {
            zeroChunk<half, kElems>(dst);
        }
    }
}

//! \brief Gemma4 PLE gather; one block per token decodes the token's whole [numLayers * pleHiddenSize] row
//! and writes each layer slice to its layer-major output. A non-null rowIndex selects the table row per token.
template <TableFormat F, typename T>
__global__ void gemma4PleGatherKernel(int32_t const* inputIds, int32_t const* rowIndex, TableView table,
    T* outputBuffer, int64_t layerOutputCapacity, int32_t pleHiddenSize, int32_t imageTokenId, int32_t audioTokenId)
{
    constexpr int64_t kElems = kChunkElems<F>;
    int64_t const tokenOffset = blockIdx.x;
    int32_t const tokenId = inputIds[tokenOffset];
    int32_t const row = rowIndex == nullptr ? tokenId : rowIndex[tokenOffset];
    bool const zeroFill = row < 0 || row >= table.rows || (imageTokenId >= 0 && tokenId == imageTokenId)
        || (audioTokenId >= 0 && tokenId == audioTokenId);

    for (int64_t col = threadIdx.x * kElems; col < table.cols; col += blockDim.x * kElems)
    {
        int64_t const layerIdx = col / pleHiddenSize;
        T* dst = outputBuffer + layerIdx * layerOutputCapacity + tokenOffset * pleHiddenSize + col % pleHiddenSize;
        if (zeroFill)
        {
            zeroChunk<T, kElems>(dst);
        }
        else
        {
            float values[kElems];
            loadChunk<F>(table, row, col, values);
            storeChunk(values, dst);
        }
    }
}

} // namespace

constexpr int32_t kMetadataScanThreads{256};

template <bool PerBatchOffsets>
__global__ void generateMultimodalIndicesKernel(int32_t const* inputIds, int32_t* multimodalIndices, int64_t seqLen,
    int32_t batchSize, int32_t imageTokenId, int32_t audioTokenId, int32_t const* imageBaseOffsets,
    int32_t const* audioBaseOffsets)
{
    using BlockScan = cub::BlockScan<int32_t, kMetadataScanThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t imageBase;
    __shared__ int32_t audioBase;

    int32_t const batch = PerBatchOffsets ? static_cast<int32_t>(blockIdx.x) : 0;
    int64_t const begin = PerBatchOffsets ? static_cast<int64_t>(batch) * seqLen : 0;
    int64_t const end = PerBatchOffsets ? begin + seqLen : static_cast<int64_t>(batchSize) * seqLen;
    if (threadIdx.x == 0)
    {
        imageBase = PerBatchOffsets && imageBaseOffsets != nullptr ? imageBaseOffsets[batch] : 0;
        audioBase = PerBatchOffsets && audioBaseOffsets != nullptr ? audioBaseOffsets[batch] : 0;
    }
    __syncthreads();

    for (int64_t tile = begin; tile < end; tile += kMetadataScanThreads)
    {
        int64_t const position = tile + threadIdx.x;
        bool const valid = position < end;
        int32_t const tokenId = valid ? inputIds[position] : -1;
        int32_t const isAudio = valid && audioTokenId >= 0 && tokenId == audioTokenId;
        int32_t const isImage = valid && !isAudio && imageTokenId >= 0 && tokenId == imageTokenId;
        int32_t imagePrefix{};
        int32_t imageCount{};
        BlockScan(scanStorage).ExclusiveSum(isImage, imagePrefix, imageCount);
        if (valid && isImage)
        {
            multimodalIndices[position] = imageBase + imagePrefix;
        }
        __syncthreads();
        int32_t audioPrefix{};
        int32_t audioCount{};
        BlockScan(scanStorage).ExclusiveSum(isAudio, audioPrefix, audioCount);
        if (valid)
        {
            multimodalIndices[position] = isAudio ? audioBase + audioPrefix
                : isImage                         ? multimodalIndices[position]
                                                  : -1;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            imageBase += imageCount;
            audioBase += audioCount;
        }
        __syncthreads();
    }
}

template <bool PerBatchOffsets>
__global__ void generateSingleModalityIndicesKernel(int32_t const* inputIds, int32_t* multimodalIndices, int64_t seqLen,
    int32_t batchSize, int32_t tokenId, int32_t const* baseOffsets)
{
    using BlockScan = cub::BlockScan<int32_t, kMetadataScanThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t base;

    int32_t const batch = PerBatchOffsets ? static_cast<int32_t>(blockIdx.x) : 0;
    int64_t const begin = PerBatchOffsets ? static_cast<int64_t>(batch) * seqLen : 0;
    int64_t const end = PerBatchOffsets ? begin + seqLen : static_cast<int64_t>(batchSize) * seqLen;
    if (threadIdx.x == 0)
    {
        base = PerBatchOffsets ? baseOffsets[batch] : 0;
    }
    __syncthreads();

    for (int64_t tile = begin; tile < end; tile += kMetadataScanThreads)
    {
        int64_t const position = tile + threadIdx.x;
        bool const matches = position < end && inputIds[position] == tokenId;
        int32_t prefix{};
        int32_t count{};
        BlockScan(scanStorage).ExclusiveSum(static_cast<int32_t>(matches), prefix, count);
        if (matches)
        {
            multimodalIndices[position] = base + prefix;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            base += count;
        }
        __syncthreads();
    }
}

__global__ void generateVisionBlockIdsKernel(
    int32_t const* inputIds, int32_t* visionBlockIds, int64_t batchSize, int64_t seqLen, int32_t imageTokenId)
{
    using BlockScan = cub::BlockScan<int32_t, kMetadataScanThreads>;
    __shared__ typename BlockScan::TempStorage scanStorage;
    __shared__ int32_t blockBase;
    int64_t const batch = blockIdx.x;
    if (batch >= batchSize)
    {
        return;
    }

    int64_t const base = batch * seqLen;
    if (threadIdx.x == 0)
    {
        blockBase = 0;
    }
    __syncthreads();
    for (int64_t tile = 0; tile < seqLen; tile += kMetadataScanThreads)
    {
        int64_t const position = tile + threadIdx.x;
        bool const valid = position < seqLen;
        bool const isVision = valid && inputIds[base + position] == imageTokenId;
        bool const startsBlock = isVision && (position == 0 || inputIds[base + position - 1] != imageTokenId);
        int32_t precedingBlocks{};
        int32_t tileBlocks{};
        BlockScan(scanStorage).ExclusiveSum(static_cast<int32_t>(startsBlock), precedingBlocks, tileBlocks);
        if (valid)
        {
            int32_t const currentBlock = blockBase + precedingBlocks - (startsBlock ? 0 : 1);
            visionBlockIds[base + position] = isVision ? currentBlock : -1;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            blockBase += tileBlocks;
        }
        __syncthreads();
    }
}

void generateMultimodalIndices(rt::Tensor const& inputIds, rt::Tensor& multimodalIndices,
    std::optional<int32_t> imageTokenId, std::optional<int32_t> audioTokenId, cudaStream_t stream,
    int32_t const* imageBaseOffsets, int32_t const* audioBaseOffsets)
{
    auto const inputShape = inputIds.getShape();
    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(multimodalIndices.getShape().volume() == inputShape.volume(),
        "multimodalIndices must have the same element count as inputIds");
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(multimodalIndices.getDataType() == nvinfer1::DataType::kINT32, "multimodalIndices must be INT32");

    int32_t const batchSize = static_cast<int32_t>(inputShape[0]);
    int64_t const seqLen = inputShape[1];
    bool const imageHasOffsets = imageBaseOffsets != nullptr;
    bool const audioHasOffsets = audioBaseOffsets != nullptr;
    if (imageHasOffsets == audioHasOffsets)
    {
        if (imageHasOffsets)
        {
            generateMultimodalIndicesKernel<true><<<batchSize, kMetadataScanThreads, 0, stream>>>(
                inputIds.dataPointer<int32_t>(), multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize,
                imageTokenId.value_or(-1), audioTokenId.value_or(-1), imageBaseOffsets, audioBaseOffsets);
        }
        else
        {
            generateMultimodalIndicesKernel<false><<<1, kMetadataScanThreads, 0, stream>>>(
                inputIds.dataPointer<int32_t>(), multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize,
                imageTokenId.value_or(-1), audioTokenId.value_or(-1), nullptr, nullptr);
        }
    }
    else
    {
        CUDA_CHECK(cudaMemsetAsync(
            multimodalIndices.rawPointer(), 0xFF, static_cast<size_t>(inputShape.volume()) * sizeof(int32_t), stream));
        if (imageTokenId.has_value())
        {
            if (imageHasOffsets)
            {
                generateSingleModalityIndicesKernel<true>
                    <<<batchSize, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *imageTokenId, imageBaseOffsets);
            }
            else
            {
                generateSingleModalityIndicesKernel<false>
                    <<<1, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *imageTokenId, nullptr);
            }
        }
        if (audioTokenId.has_value())
        {
            if (audioHasOffsets)
            {
                generateSingleModalityIndicesKernel<true>
                    <<<batchSize, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *audioTokenId, audioBaseOffsets);
            }
            else
            {
                generateSingleModalityIndicesKernel<false>
                    <<<1, kMetadataScanThreads, 0, stream>>>(inputIds.dataPointer<int32_t>(),
                        multimodalIndices.dataPointer<int32_t>(), seqLen, batchSize, *audioTokenId, nullptr);
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void generateVisionBlockIds(
    rt::Tensor const& inputIds, rt::Tensor& visionBlockIds, int32_t imageTokenId, cudaStream_t stream)
{
    auto const shape = inputIds.getShape();
    check::check(shape.getNumDims() == 2, "inputIds must be 2D tensor");
    check::check(inputIds.getDeviceType() == rt::DeviceType::kGPU, "inputIds must be a GPU tensor");
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must have INT32 dtype");
    check::check(visionBlockIds.getDeviceType() == rt::DeviceType::kGPU
            && visionBlockIds.getDataType() == nvinfer1::DataType::kINT32
            && visionBlockIds.getShape().volume() == shape.volume(),
        "visionBlockIds must be an equally-sized GPU INT32 tensor");
    check::check(imageTokenId >= 0, "imageTokenId must be non-negative");

    int64_t const batchSize = shape[0];
    int64_t const seqLen = shape[1];
    generateVisionBlockIdsKernel<<<batchSize, kMetadataScanThreads, 0, stream>>>(
        inputIds.dataPointer<int32_t>(), visionBlockIds.dataPointer<int32_t>(), batchSize, seqLen, imageTokenId);
    CUDA_CHECK(cudaGetLastError());
}

void assembleDeepstackEmbedding(rt::Tensor const& inputIds, rt::Tensor const& deepstackFeatures,
    rt::Tensor& deepstackEmbeds, cudaStream_t stream, int32_t imageTokenId, rt::OptionalInputTensor multimodalIndices)
{
    // Validate input shapes
    auto const inputShape = inputIds.getShape();
    auto const featuresShape = deepstackFeatures.getShape();
    auto const outputShape = deepstackEmbeds.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(featuresShape.getNumDims() == 2, "deepstackFeatures must be 2D tensor [numImageTokens, hiddenSize]");
    check::check(outputShape.getNumDims() == 3, "deepstackEmbeds must be 3D tensor [batchSize, seqLen, hiddenSize]");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int64_t const numImageTokens = featuresShape[0];
    int64_t const hiddenSize = featuresShape[1];

    check::check(outputShape[0] == batchSize, "Output batch size mismatch");
    check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
    check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");

    // Validate data types
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(deepstackFeatures.getDataType() == nvinfer1::DataType::kHALF, "deepstackFeatures must be FP16");
    check::check(deepstackEmbeds.getDataType() == nvinfer1::DataType::kHALF, "deepstackEmbeds must be FP16");

    // Get device pointers
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    half const* deepstackFeaturesPtr = deepstackFeatures.dataPointer<half>();
    half* outputPtr = deepstackEmbeds.dataPointer<half>();

    // Launch kernel
    constexpr uint32_t vecSize = DVec<half>::vec_size;
    uint32_t const totalTokens = batchSize * seqLen;

    // Validate that hiddenSize is a multiple of vecSize to avoid partial loads
    check::check(hiddenSize % vecSize == 0,
        format::fmtstr("hiddenSize must be a multiple of %d for efficient vectorized access", vecSize));

    // Image positions are mapped to deepstack feature rows via multimodalIndices.
    check::check(multimodalIndices.has_value(), "assembleDeepstackEmbedding requires multimodalIndices");
    int32_t const* multimodalIndicesPtr = multimodalIndices.value().get().dataPointer<int32_t>();

    // Use 2D CTA: (32, 4) - 4 warps per block
    dim3 const threadsPerBlock(32, 4);               // (32, 4) = 128 threads total
    uint32_t const gridSize = (totalTokens + 3) / 4; // 4 warps per block

    assembleDeepstackEmbeddingKernel<<<gridSize, threadsPerBlock, 0, stream>>>(inputIdsPtr, deepstackFeaturesPtr,
        outputPtr, batchSize, seqLen, imageTokenId, hiddenSize, numImageTokens, multimodalIndicesPtr);
}

int64_t validateEmbeddingTable(EmbeddingTableDescriptor const& table)
{
    return makeTableView(table).cols;
}

void embeddingLookup(rt::Tensor const& inputIds, EmbeddingTableDescriptor const& table, rt::Tensor& output,
    cudaStream_t stream, rt::OptionalInputTensor multimodalIndices, std::optional<int32_t> imageTokenId,
    rt::OptionalInputTensor imageEmbeds, std::optional<int32_t> audioTokenId, rt::OptionalInputTensor audioEmbeds)
{
    TableView const view = makeTableView(table);
    auto const inputShape = inputIds.getShape();
    auto const outputShape = output.getShape();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(outputShape.getNumDims() == 2 || outputShape.getNumDims() == 3,
        "output must be token-major [physicalTokens, hiddenSize] or [batchSize, seqLen, hiddenSize]");
    check::check(
        view.rows <= std::numeric_limits<int32_t>::max(), "embedding table vocab dimension exceeds int32 range");

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    int64_t const hiddenSize = view.cols;

    // Validate output shape
    if (outputShape.getNumDims() == 2)
    {
        check::check(outputShape[0] == batchSize * seqLen, "Output physical token count mismatch");
        check::check(outputShape[1] == hiddenSize, "Output hidden size mismatch");
    }
    else
    {
        check::check(outputShape[0] == batchSize, "Output batch size mismatch");
        check::check(outputShape[1] == seqLen, "Output sequence length mismatch");
        check::check(outputShape[2] == hiddenSize, "Output hidden size mismatch");
    }

    // Validate common data types for required inputs
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(output.getDataType() == nvinfer1::DataType::kHALF, "output must be FP16");

    // Handle optional image parameters
    bool const hasImage = imageTokenId.has_value() && imageEmbeds.has_value();
    if (hasImage)
    {
        auto const imageShape = imageEmbeds->get().getShape();
        check::check(imageShape.getNumDims() == 2, "imageEmbeds must be 2D tensor [imageTokenLen, hiddenSize]");
        check::check(imageShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and imageEmbeds");
        check::check(imageEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "imageEmbeds must be FP16");
    }

    // Handle optional audio parameters
    bool const hasAudio = audioTokenId.has_value() && audioEmbeds.has_value();
    if (hasAudio)
    {
        auto const audioShape = audioEmbeds->get().getShape();
        check::check(audioShape.getNumDims() == 2, "audioEmbeds must be 2D tensor [audioTokenLen, hiddenSize]");
        check::check(audioShape[1] == hiddenSize, "Hidden size mismatch between embeddingTable and audioEmbeds");
        check::check(audioEmbeds->get().getDataType() == nvinfer1::DataType::kHALF, "audioEmbeds must be FP16");
    }

    // Validate that imageTokenId and audioTokenId are different when both are present
    if (hasImage && hasAudio)
    {
        check::check(*imageTokenId != *audioTokenId, "imageTokenId and audioTokenId must be different");
    }

    // Validate multimodalIndices if any multimodal input is present
    bool const hasMultimodal = hasImage || hasAudio;
    if (hasMultimodal)
    {
        check::check(
            multimodalIndices.has_value(), "multimodalIndices is required when image or audio inputs are provided");
        auto const multimodalIndicesShape = multimodalIndices->get().getShape();
        check::check(
            multimodalIndicesShape.getNumDims() == 2, "multimodalIndices must be 2D tensor [batchSize, seqLen]");
        check::check(multimodalIndicesShape[0] == batchSize, "multimodalIndices batch size mismatch");
        check::check(multimodalIndicesShape[1] == seqLen, "multimodalIndices sequence length mismatch");
        check::check(
            multimodalIndices->get().getDataType() == nvinfer1::DataType::kINT32, "multimodalIndices must be INT32");
    }

    // Extract values for kernel - use safe defaults when modalities are absent
    int32_t const imageTokenIdValue = hasImage ? *imageTokenId : -1;
    half const* imageEmbedsPtr = hasImage ? imageEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const imageTokenLen = hasImage ? imageEmbeds->get().getShape()[0] : 0;

    int32_t const audioTokenIdValue = hasAudio ? *audioTokenId : -1;
    half const* audioEmbedsPtr = hasAudio ? audioEmbeds->get().dataPointer<half>() : nullptr;
    int64_t const audioTokenLen = hasAudio ? audioEmbeds->get().getShape()[0] : 0;

    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    int32_t const* multimodalIndicesPtr = hasMultimodal ? multimodalIndices->get().dataPointer<int32_t>() : nullptr;
    half* outputPtr = output.dataPointer<half>();
    int64_t const numTokens = batchSize * seqLen;
    if (numTokens == 0)
    {
        return;
    }

    dim3 const threadsPerBlock(kGatherWarpSize, kLookupWarpsPerBlock);
    int64_t const gridSize = (numTokens + kLookupWarpsPerBlock - 1) / kLookupWarpsPerBlock;
    dispatchFormat(view.format, [&](auto format) {
        embeddingLookupKernelImpl<decltype(format)::value><<<gridSize, threadsPerBlock, 0, stream>>>(inputIdsPtr, view,
            multimodalIndicesPtr, imageTokenIdValue, imageEmbedsPtr, imageTokenLen, audioTokenIdValue, audioEmbedsPtr,
            audioTokenLen, outputPtr, numTokens);
    });
    CUDA_CHECK(cudaGetLastError());
}

void embeddingLookup(rt::Tensor const& inputIds, rt::Tensor const& embeddingTable, rt::OptionalInputTensor scales,
    rt::Tensor& output, cudaStream_t stream, rt::OptionalInputTensor multimodalIndices,
    std::optional<int32_t> imageTokenId, rt::OptionalInputTensor imageEmbeds, std::optional<int32_t> audioTokenId,
    rt::OptionalInputTensor audioEmbeds)
{
    embeddingLookup(inputIds, EmbeddingTableDescriptor{embeddingTable, scales}, output, stream, multimodalIndices,
        imageTokenId, imageEmbeds, audioTokenId, audioEmbeds);
}

void gemma4PleGather(rt::Tensor const& inputIds, EmbeddingTableDescriptor const& pleTable, rt::Tensor& outputBuffer,
    int32_t numLayers, int32_t pleHiddenSize, int32_t imageTokenId, int32_t audioTokenId, cudaStream_t stream,
    rt::OptionalInputTensor rowIndex)
{
    TableView const view = makeTableView(pleTable);
    auto const inputShape = inputIds.getShape();
    auto const outputShape = outputBuffer.getShape();
    nvinfer1::DataType const outputType = outputBuffer.getDataType();

    check::check(inputShape.getNumDims() == 2, "inputIds must be 2D tensor [batchSize, seqLen]");
    check::check(outputShape.getNumDims() == 4,
        "outputBuffer must be 4D tensor [numLayers, maxBatchSize, maxSeqLen, pleHiddenSize]");
    check::check(inputIds.getDataType() == nvinfer1::DataType::kINT32, "inputIds must be INT32");
    check::check(outputType == nvinfer1::DataType::kHALF
            || (outputType == nvinfer1::DataType::kBF16 && view.format == TableFormat::kBF16),
        "outputBuffer dtype must be FP16, or BF16 for a BF16 pleTable");
    check::check(numLayers > 0, "numLayers must be positive");
    check::check(pleHiddenSize > 0, "pleHiddenSize must be positive");
    check::check(outputShape[0] == numLayers, "outputBuffer first dimension must match numLayers");
    check::check(outputShape[3] == pleHiddenSize, "outputBuffer hidden dimension must match pleHiddenSize");
    check::check(view.cols == static_cast<int64_t>(numLayers) * pleHiddenSize,
        "pleTable second dimension must match numLayers * pleHiddenSize");
    check::check(view.rows <= std::numeric_limits<int32_t>::max(), "pleTable vocab dimension exceeds int32 range");
    int64_t const elems = chunkElems(view.format);
    check::check(pleHiddenSize % elems == 0,
        format::fmtstr("pleHiddenSize %d must be a multiple of %ld for vectorized access", pleHiddenSize, elems));
    check::check(view.group == 0 || pleHiddenSize % view.group == 0,
        format::fmtstr("pleTable scale group %ld must divide pleHiddenSize %d", view.group, pleHiddenSize));

    int64_t const batchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    check::check(batchSize <= outputShape[1], "inputIds batch size exceeds outputBuffer capacity");
    check::check(seqLen <= outputShape[2], "inputIds sequence length exceeds outputBuffer capacity");
    if (rowIndex.has_value())
    {
        check::check(
            rowIndex->get().getShape() == inputShape && rowIndex->get().getDataType() == nvinfer1::DataType::kINT32,
            "rowIndex must be INT32 with the inputIds shape");
    }
    int32_t const* rowIndexPtr = rowIndex.has_value() ? rowIndex->get().dataPointer<int32_t>() : nullptr;
    int64_t const numTokens = batchSize * seqLen;
    if (numTokens == 0)
    {
        return;
    }

    int64_t const layerOutputCapacity = outputShape[1] * outputShape[2] * pleHiddenSize;
    int32_t const* inputIdsPtr = inputIds.dataPointer<int32_t>();
    auto launch = [&](auto format, auto* outputPtr) {
        using T = std::remove_pointer_t<decltype(outputPtr)>;
        gemma4PleGatherKernel<decltype(format)::value, T><<<numTokens, kPleGatherThreads, 0, stream>>>(
            inputIdsPtr, rowIndexPtr, view, outputPtr, layerOutputCapacity, pleHiddenSize, imageTokenId, audioTokenId);
    };
    dispatchFormat(view.format, [&](auto format) {
        if constexpr (decltype(format)::value == TableFormat::kBF16)
        {
            if (outputType == nvinfer1::DataType::kBF16)
            {
                launch(format, outputBuffer.dataPointer<__nv_bfloat16>());
                return;
            }
        }
        launch(format, outputBuffer.dataPointer<half>());
    });
    CUDA_CHECK(cudaGetLastError());
}

} // namespace kernel
} // namespace trt_edgellm
