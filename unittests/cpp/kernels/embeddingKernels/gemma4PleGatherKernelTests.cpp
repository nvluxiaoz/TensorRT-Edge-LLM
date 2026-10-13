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

#include "common/cudaUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"
#include "testUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

constexpr float kRTol = 1e-3F;
constexpr float kATol = 1e-3F;
constexpr float kSentinelValue = -7.0F;

template <typename T>
struct Gemma4PleGatherTraits;

template <>
struct Gemma4PleGatherTraits<half>
{
    static constexpr DataType kDataType = DataType::kHALF;

    static half fromFloat(float value)
    {
        return __float2half(value);
    }

    static float toFloat(half value)
    {
        return __half2float(value);
    }
};

template <>
struct Gemma4PleGatherTraits<__nv_bfloat16>
{
    static constexpr DataType kDataType = DataType::kBF16;

    static __nv_bfloat16 fromFloat(float value)
    {
        return __float2bfloat16(value);
    }

    static float toFloat(__nv_bfloat16 value)
    {
        return __bfloat162float(value);
    }
};

template <typename T>
T makeScalar(float value)
{
    return Gemma4PleGatherTraits<T>::fromFloat(value);
}

template <typename T>
float scalarToFloat(T value)
{
    return Gemma4PleGatherTraits<T>::toFloat(value);
}

float tableValue(int32_t tokenId, int32_t layerIdx, int32_t hiddenIdx)
{
    return static_cast<float>(tokenId * 100 + layerIdx * 10 + hiddenIdx);
}

template <typename T>
std::vector<T> makePleTable(int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize)
{
    std::vector<T> table(static_cast<size_t>(vocabSize) * numLayers * pleHiddenSize);
    for (int32_t tokenId = 0; tokenId < vocabSize; ++tokenId)
    {
        for (int32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
        {
            for (int32_t hiddenIdx = 0; hiddenIdx < pleHiddenSize; ++hiddenIdx)
            {
                size_t const offset = (static_cast<size_t>(tokenId) * numLayers + layerIdx) * pleHiddenSize + hiddenIdx;
                table[offset] = makeScalar<T>(tableValue(tokenId, layerIdx, hiddenIdx));
            }
        }
    }
    return table;
}

template <typename T>
std::vector<T> makeFilledOutput(int32_t numLayers, int32_t maxBatchSize, int32_t maxSeqLen, int32_t pleHiddenSize)
{
    return std::vector<T>(
        static_cast<size_t>(numLayers) * maxBatchSize * maxSeqLen * pleHiddenSize, makeScalar<T>(kSentinelValue));
}

bool shouldZeroFill(int32_t tokenId, int32_t vocabSize, int32_t imageTokenId, int32_t audioTokenId)
{
    return tokenId < 0 || tokenId >= vocabSize || (imageTokenId >= 0 && tokenId == imageTokenId)
        || (audioTokenId >= 0 && tokenId == audioTokenId);
}

template <typename T>
void verifyOutput(std::vector<T> const& output, std::vector<int32_t> const& inputIds, int32_t batchSize, int32_t seqLen,
    int32_t maxBatchSize, int32_t maxSeqLen, int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize,
    int32_t imageTokenId, int32_t audioTokenId)
{
    int64_t const layerCapacity = static_cast<int64_t>(maxBatchSize) * maxSeqLen * pleHiddenSize;
    std::vector<bool> written(output.size(), false);

    for (int32_t layerIdx = 0; layerIdx < numLayers; ++layerIdx)
    {
        for (int32_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (int32_t seqIdx = 0; seqIdx < seqLen; ++seqIdx)
            {
                size_t const tokenOffset = static_cast<size_t>(batchIdx) * seqLen + seqIdx;
                int32_t const tokenId = inputIds[tokenOffset];
                for (int32_t hiddenIdx = 0; hiddenIdx < pleHiddenSize; ++hiddenIdx)
                {
                    size_t const outputOffset
                        = static_cast<size_t>(layerIdx * layerCapacity) + (tokenOffset * pleHiddenSize) + hiddenIdx;
                    written[outputOffset] = true;
                    float const expected = shouldZeroFill(tokenId, vocabSize, imageTokenId, audioTokenId)
                        ? 0.0F
                        : scalarToFloat(makeScalar<T>(tableValue(tokenId, layerIdx, hiddenIdx)));
                    EXPECT_NEAR(scalarToFloat(output[outputOffset]), expected, kATol + kRTol * std::abs(expected))
                        << "layer=" << layerIdx << " batch=" << batchIdx << " seq=" << seqIdx
                        << " hidden=" << hiddenIdx;
                }
            }
        }
    }

    for (size_t idx = 0; idx < output.size(); ++idx)
    {
        if (!written[idx])
        {
            EXPECT_NEAR(scalarToFloat(output[idx]), kSentinelValue, kATol) << "unwritten index=" << idx;
        }
    }
}

template <typename T>
void runGatherTest(std::vector<int32_t> const& inputIds, int32_t batchSize, int32_t seqLen, int32_t maxBatchSize,
    int32_t maxSeqLen, int32_t vocabSize, int32_t numLayers, int32_t pleHiddenSize, int32_t imageTokenId,
    int32_t audioTokenId)
{
    rt::Tensor inputIdsDevice({batchSize, seqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor pleTableDevice(
        {vocabSize, numLayers * pleHiddenSize}, rt::DeviceType::kGPU, Gemma4PleGatherTraits<T>::kDataType);
    rt::Tensor outputDevice(
        {numLayers, maxBatchSize, maxSeqLen, pleHiddenSize}, rt::DeviceType::kGPU, Gemma4PleGatherTraits<T>::kDataType);

    std::vector<T> const table = makePleTable<T>(vocabSize, numLayers, pleHiddenSize);
    std::vector<T> const initialOutput = makeFilledOutput<T>(numLayers, maxBatchSize, maxSeqLen, pleHiddenSize);

    copyHostToDevice(inputIdsDevice, inputIds);
    copyHostToDevice(pleTableDevice, table);
    copyHostToDevice(outputDevice, initialOutput);

    kernel::gemma4PleGather(inputIdsDevice, kernel::EmbeddingTableDescriptor{pleTableDevice}, outputDevice, numLayers,
        pleHiddenSize, imageTokenId, audioTokenId, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<T> const output = copyDeviceToHost<T>(outputDevice);
    verifyOutput(output, inputIds, batchSize, seqLen, maxBatchSize, maxSeqLen, vocabSize, numLayers, pleHiddenSize,
        imageTokenId, audioTokenId);
}

//! Gathers an INT8 or INT4 PLE table with FP32 scales per (row, group) and checks every output element bit-exactly.
void runQuantizedGatherTest(DataType dataType, int32_t group)
{
    constexpr int32_t kBatchSize = 2;
    constexpr int32_t kSeqLen = 2;
    constexpr int32_t kMaxBatchSize = 2;
    constexpr int32_t kMaxSeqLen = 3;
    constexpr int32_t kVocabSize = 5;
    constexpr int32_t kNumLayers = 3;
    constexpr int32_t kPleHiddenSize = 256;
    constexpr int32_t kImageTokenId = 3;
    constexpr int32_t kRowWidth = kNumLayers * kPleHiddenSize;
    std::vector<int32_t> const inputIds{0, 4, kImageTokenId, 1};
    bool const isInt4 = dataType == DataType::kUINT8;
    int32_t const groupsPerRow = kRowWidth / group;

    std::vector<int8_t> quantized(static_cast<size_t>(kVocabSize) * kRowWidth);
    std::vector<uint8_t> packed;
    std::vector<float> scales(static_cast<size_t>(kVocabSize) * groupsPerRow);
    for (size_t idx = 0; idx < quantized.size(); ++idx)
    {
        quantized[idx] = static_cast<int8_t>(
            isInt4 ? static_cast<int32_t>(idx * 7 % 16) - 8 : static_cast<int32_t>(idx * 37 % 255) - 127);
        if (idx % 2 == 1)
        {
            packed.push_back(static_cast<uint8_t>((quantized[idx - 1] + 8) | ((quantized[idx] + 8) << 4)));
        }
    }
    for (size_t idx = 0; idx < scales.size(); ++idx)
    {
        scales[idx] = 0.01F * static_cast<float>(idx + 1);
    }

    rt::Tensor inputIdsDevice({kBatchSize, kSeqLen}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor tableDevice({kVocabSize, isInt4 ? kRowWidth / 2 : kRowWidth}, rt::DeviceType::kGPU, dataType);
    rt::Tensor scaleDevice({kVocabSize, groupsPerRow}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor outputDevice(
        {kNumLayers, kMaxBatchSize, kMaxSeqLen, kPleHiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(inputIdsDevice, inputIds);
    if (isInt4)
    {
        copyHostToDevice(tableDevice, packed);
    }
    else
    {
        copyHostToDevice(tableDevice, quantized);
    }
    copyHostToDevice(scaleDevice, scales);
    copyHostToDevice(outputDevice, makeFilledOutput<half>(kNumLayers, kMaxBatchSize, kMaxSeqLen, kPleHiddenSize));

    kernel::gemma4PleGather(inputIdsDevice, kernel::EmbeddingTableDescriptor{tableDevice, scaleDevice}, outputDevice,
        kNumLayers, kPleHiddenSize, kImageTokenId, /* audioTokenId = */ -1, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> const output = copyDeviceToHost<half>(outputDevice);
    int64_t const layerCapacity = static_cast<int64_t>(kMaxBatchSize) * kMaxSeqLen * kPleHiddenSize;
    for (int32_t layerIdx = 0; layerIdx < kNumLayers; ++layerIdx)
    {
        for (size_t token = 0; token < inputIds.size(); ++token)
        {
            int32_t const tokenId = inputIds[token];
            for (int32_t hiddenIdx = 0; hiddenIdx < kPleHiddenSize; ++hiddenIdx)
            {
                int64_t const col = static_cast<int64_t>(layerIdx) * kPleHiddenSize + hiddenIdx;
                int64_t const element = tokenId * kRowWidth + col;
                float const expected = tokenId == kImageTokenId
                    ? 0.0F
                    : __half2float(__float2half(
                          static_cast<float>(quantized[element]) * scales[tokenId * groupsPerRow + col / group]));
                EXPECT_EQ(__half2float(output[layerIdx * layerCapacity + token * kPleHiddenSize + hiddenIdx]), expected)
                    << "layer=" << layerIdx << " token=" << token << " hidden=" << hiddenIdx;
            }
        }
        for (int64_t idx = static_cast<int64_t>(inputIds.size()) * kPleHiddenSize; idx < layerCapacity; ++idx)
        {
            EXPECT_EQ(__half2float(output[layerIdx * layerCapacity + idx]), kSentinelValue)
                << "unwritten index=" << idx;
        }
    }
}

} // namespace

TEST(Gemma4PleGatherKernelTest, GathersInt8TableWithGroup256)
{
    runQuantizedGatherTest(DataType::kINT8, /* group = */ 256);
}

TEST(Gemma4PleGatherKernelTest, GathersInt4TableWithGroup128)
{
    runQuantizedGatherTest(DataType::kUINT8, /* group = */ 128);
}

TEST(Gemma4PleGatherKernelTest, RejectsScaleGroupsSpanningLayers)
{
    rt::Tensor inputIds({1, 1}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor table({4, 512}, rt::DeviceType::kGPU, DataType::kINT8);
    rt::Tensor perRowScale({4, 1}, rt::DeviceType::kGPU, DataType::kFLOAT);
    rt::Tensor output({2, 1, 1, 256}, rt::DeviceType::kGPU, DataType::kHALF);
    EXPECT_THROW(kernel::gemma4PleGather(inputIds, kernel::EmbeddingTableDescriptor{table, perRowScale}, output,
                     /* numLayers = */ 2, /* pleHiddenSize = */ 256, -1, -1, nullptr),
        std::runtime_error);
}

TEST(Gemma4PleGatherKernelTest, GathersFp16LayerOutputsWithCompactRuntimeShape)
{
    constexpr int32_t kBatchSize = 2;
    constexpr int32_t kSeqLen = 3;
    constexpr int32_t kMaxBatchSize = 4;
    constexpr int32_t kMaxSeqLen = 5;
    constexpr int32_t kVocabSize = 7;
    constexpr int32_t kNumLayers = 3;
    constexpr int32_t kPleHiddenSize = 16;
    std::vector<int32_t> const inputIds{0, 1, 2, 3, 4, 5};

    runGatherTest<half>(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers,
        kPleHiddenSize, /* imageTokenId = */ -1, /* audioTokenId = */ -1);
}

TEST(Gemma4PleGatherKernelTest, ZeroFillsInvalidAndMultimodalTokensFp16)
{
    constexpr int32_t kBatchSize = 1;
    constexpr int32_t kSeqLen = 5;
    constexpr int32_t kMaxBatchSize = 2;
    constexpr int32_t kMaxSeqLen = 6;
    constexpr int32_t kVocabSize = 5;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 8;
    constexpr int32_t kImageTokenId = 3;
    constexpr int32_t kAudioTokenId = 4;
    std::vector<int32_t> const inputIds{0, -1, kVocabSize, kImageTokenId, kAudioTokenId};

    runGatherTest<half>(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers,
        kPleHiddenSize, kImageTokenId, kAudioTokenId);
}

TEST(Gemma4PleGatherKernelTest, GathersBfloat16LayerOutputs)
{
    constexpr int32_t kBatchSize = 2;
    constexpr int32_t kSeqLen = 2;
    constexpr int32_t kMaxBatchSize = 2;
    constexpr int32_t kMaxSeqLen = 3;
    constexpr int32_t kVocabSize = 6;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 8;
    std::vector<int32_t> const inputIds{1, 2, 3, 4};

    runGatherTest<__nv_bfloat16>(inputIds, kBatchSize, kSeqLen, kMaxBatchSize, kMaxSeqLen, kVocabSize, kNumLayers,
        kPleHiddenSize, /* imageTokenId = */ -1, /* audioTokenId = */ -1);
}

// Given a BF16 PLE table and a staging table holding only the distinct valid rows of a token batch,
// when both are gathered to FP16 with a multiplier, the staged one through a per-token row index,
// then the outputs are identical, including duplicates and zero-filled invalid and media tokens.
TEST(Gemma4PleGatherKernelTest, StagedRowIndexMatchesResidentGather)
{
    constexpr int32_t kVocabSize = 6;
    constexpr int32_t kNumLayers = 2;
    constexpr int32_t kPleHiddenSize = 16;
    constexpr int32_t kRowWidth = kNumLayers * kPleHiddenSize;
    constexpr int32_t kImageTokenId = 5;
    constexpr float kMultiplier = 16.0F;
    std::vector<int32_t> const inputIds{3, 1, 3, -1, kVocabSize, kImageTokenId, 1, 0};
    std::vector<int32_t> const stagedTokens{3, 1, 0};
    // The image token points at a valid staged row; the kernel must still zero-fill it.
    std::vector<int32_t> const rowIndex{0, 1, 0, -1, -1, 0, 1, 2};
    int32_t const numTokens = static_cast<int32_t>(inputIds.size());

    std::vector<__nv_bfloat16> table(static_cast<size_t>(kVocabSize) * kRowWidth);
    for (size_t idx = 0; idx < table.size(); ++idx)
    {
        table[idx] = __float2bfloat16(0.37F * static_cast<float>(idx % 29) - 4.1F);
    }
    std::vector<__nv_bfloat16> staged;
    for (int32_t tokenId : stagedTokens)
    {
        staged.insert(staged.end(), table.begin() + tokenId * kRowWidth, table.begin() + (tokenId + 1) * kRowWidth);
    }

    rt::Tensor inputIdsDevice({1, numTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor rowIndexDevice({1, numTokens}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor tableDevice({kVocabSize, kRowWidth}, rt::DeviceType::kGPU, DataType::kBF16);
    rt::Tensor stagedDevice(
        {static_cast<int64_t>(stagedTokens.size()), kRowWidth}, rt::DeviceType::kGPU, DataType::kBF16);
    rt::Tensor residentOutput({kNumLayers, 1, numTokens + 1, kPleHiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor stagedOutput({kNumLayers, 1, numTokens + 1, kPleHiddenSize}, rt::DeviceType::kGPU, DataType::kHALF);
    copyHostToDevice(inputIdsDevice, inputIds);
    copyHostToDevice(rowIndexDevice, rowIndex);
    copyHostToDevice(tableDevice, table);
    copyHostToDevice(stagedDevice, staged);
    std::vector<half> const sentinel = makeFilledOutput<half>(kNumLayers, 1, numTokens + 1, kPleHiddenSize);
    copyHostToDevice(residentOutput, sentinel);
    copyHostToDevice(stagedOutput, sentinel);

    kernel::gemma4PleGather(inputIdsDevice, kernel::EmbeddingTableDescriptor{tableDevice, std::nullopt, kMultiplier},
        residentOutput, kNumLayers, kPleHiddenSize, kImageTokenId, /* audioTokenId = */ -1, nullptr);
    kernel::gemma4PleGather(inputIdsDevice, kernel::EmbeddingTableDescriptor{stagedDevice, std::nullopt, kMultiplier},
        stagedOutput, kNumLayers, kPleHiddenSize, kImageTokenId, /* audioTokenId = */ -1, nullptr, rowIndexDevice);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> const resident = copyDeviceToHost<half>(residentOutput);
    std::vector<half> const stagedResult = copyDeviceToHost<half>(stagedOutput);
    int64_t const layerCapacity = static_cast<int64_t>(numTokens + 1) * kPleHiddenSize;
    for (size_t idx = 0; idx < resident.size(); ++idx)
    {
        int64_t const token = static_cast<int64_t>(idx) % layerCapacity / kPleHiddenSize;
        int64_t const col
            = static_cast<int64_t>(idx) / layerCapacity * kPleHiddenSize + static_cast<int64_t>(idx) % kPleHiddenSize;
        float expected = kSentinelValue;
        if (token < numTokens)
        {
            int32_t const tokenId = inputIds[token];
            bool const valid = tokenId >= 0 && tokenId < kVocabSize && tokenId != kImageTokenId;
            expected = valid
                ? __half2float(__float2half(__bfloat162float(table[tokenId * kRowWidth + col]) * kMultiplier))
                : 0.0F;
        }
        EXPECT_EQ(__half2float(stagedResult[idx]), __half2float(resident[idx])) << "index=" << idx;
        EXPECT_EQ(__half2float(stagedResult[idx]), expected) << "index=" << idx;
    }
}
