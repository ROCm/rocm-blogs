// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <torch/torch.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/fill.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough    = ck::tensor_operation::element_wise::PassThrough;
using RowMajor = ck::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck::tensor_layout::gemm::ColumnMajor;


using F16 = ck::half_t;
using F32 = float;
using I8 = int8_t;
using I32 = int32_t;

struct ProblemSize final
{
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA  = 4096;
    ck::index_t StrideB  = 4096;
    ck::index_t StrideD0 = 4096;
    ck::index_t StrideE  = 4096;
};

struct ExecutionConfig final
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;
};

#define checkHipErrors( a ) do { \
    if (hipSuccess != (a)) { \
    fprintf(stderr, "Hip runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, hipGetErrorString(hipGetLastError()) ); \
    /*exit(EXIT_FAILURE);*/ \
    } \
    } while(0);

