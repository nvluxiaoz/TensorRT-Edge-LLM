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

#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <filesystem>
#include <string>

namespace trt_edgellm
{
namespace rt
{

//! Storage of the Gemma4 per-layer embedding (PLE) table.
enum class PleResidency
{
    kRESIDENT, //!< Whole table in GPU memory.
    kCACHED,   //!< Rows read from the source file per step through the OS page cache.
};

struct PleRuntimeConfig
{
    PleResidency residency{PleResidency::kRESIDENT};
    //! Safetensors file holding the recorded PLE tensor; empty selects the checkpoint or engine-directory source.
    std::string filePath{};
};

//! One F16 or BF16 safetensors tensor of shape [rows, cols] read row by row in cached residency.
struct PleFileSource
{
    std::string tensorName;
    std::filesystem::path file;
    int64_t offset{0}; //!< Absolute byte offset of row 0 in `file`.
    nvinfer1::DataType dtype{nvinfer1::DataType::kHALF};
    int64_t rows{0};
    int64_t cols{0};
    float scale{1.0F}; //!< Multiplier applied to every element in FP32.
};

} // namespace rt
} // namespace trt_edgellm
