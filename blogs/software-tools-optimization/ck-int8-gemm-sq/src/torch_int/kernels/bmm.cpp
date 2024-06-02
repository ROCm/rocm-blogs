#include "include/common.hpp"
#include "include/bmm.h"

struct Scale {
  template <typename Y, typename X>
  __host__ __device__ constexpr void operator()(Y&, const X&) const;


  template <>
  __host__ __device__ constexpr void
  operator()<I8, I32>(I8& e, const I32& c) const
    {
      F32 temp = ck::type_convert<F32>(c) * alpha;
      temp = temp > 127 ? 127 : temp;
      temp = temp < -128? -128 : temp;
      e = ck::type_convert<I8>(temp);
    }

  template <>
  __host__ __device__ constexpr void
  operator()<F32, I32>(F32& e, const I32& c) const
    {
      const F32 c_scale = ck::type_convert<F32>(c) * alpha;
      e = c_scale;
    }

  F32 alpha;
};


torch::Tensor bmm_abe_i8(torch::Tensor A,
			 torch::Tensor B,
			 float alpha)
{
  int batch_count = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);
  
  int stride_A = K;
  int stride_B = K;
  int stride_E = N;
  long long int batch_stride_A = M * K;
  long long int batch_stride_B = K * N;
  long long int batch_stride_E = M * N;

  
  auto E = torch::empty({batch_count, M, N}, torch::dtype(torch::kInt8).device(A.device()));
  
  using ADataType        = I8;
  using BDataType        = I8;
  using AccDataType      = I32;
  using CShuffleDataType = I32;
  using DsDataType       = ck::Tuple<>;
  using EDataType        = I8;

  using ALayout  = RowMajor;
  using BLayout  = ColumnMajor;
  using DsLayout = ck::Tuple<>;
  using ELayout  = RowMajor;
  
  using AElementOp   = PassThrough;
  using BElementOp   = PassThrough;
  using CDEElementOp = Scale;

  static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

  using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl
        < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   128,   64,    64,  16,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,              16>;

  auto A_ref = A.data_ptr<ADataType>();
  auto B_ref = B.data_ptr<BDataType>();
  auto E_ref = E.data_ptr<EDataType>();

  auto device_op    = DeviceOpInstance{};
  auto invoker = device_op.MakeInvoker();
  auto argument = device_op.MakeArgument(A_ref,
					 B_ref,
					 {},
					 E_ref,
					 M,
					 N,
					 K,
					 batch_count,
					 stride_A,
					 stride_B,
					 {},
					 stride_E,
					 batch_stride_A,
					 batch_stride_B,
					 {},
					 batch_stride_E,
					 AElementOp{},
					 BElementOp{},
					 CDEElementOp{alpha});
  
  if(!device_op.IsSupportedArgument(argument))
    {
      throw std::runtime_error(
			       "wrong! device_gemm with the specified compilation parameters does "
			       "not support this GEMM problem");
    }
  
  invoker.Run(argument, StreamConfig{nullptr, 0});

  return E;
}


torch::Tensor bmm_ab_i8_e_f32(torch::Tensor A,
			      torch::Tensor B,
			      float alpha)
{
  int batch_count = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  int stride_A = K;
  int stride_B = K;
  int stride_E = N;
  long long int batch_stride_A = M * K;
  long long int batch_stride_B = K * N;
  long long int batch_stride_E = M * N;

  auto E = torch::empty({batch_count, M, N}, torch::dtype(torch::kFloat32).device(A.device()));
  
  using ADataType        = I8;
  using BDataType        = I8;
  using AccDataType      = I32;
  using CShuffleDataType = I32;
  using DsDataType       = ck::Tuple<>;
  using EDataType        = F32;

  using ALayout  = RowMajor;
  using BLayout  = ColumnMajor;
  using DsLayout = ck::Tuple<>;
  using ELayout  = RowMajor;
  
  using AElementOp   = PassThrough;
  using BElementOp   = PassThrough;
  using CDEElementOp = Scale;

  static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

    using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl
    < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   128,   64,    64,  16,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,       
    2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,              16>;

  
  auto A_ref = A.data_ptr<ADataType>();
  auto B_ref = B.data_ptr<BDataType>();
  auto E_ref = E.data_ptr<EDataType>();
  
  auto device_op    = DeviceOpInstance{};
  auto invoker = device_op.MakeInvoker();
  auto argument = device_op.MakeArgument(A_ref,
					 B_ref,
					 {},
					 E_ref,
					 M,
					 N,
					 K,
					 batch_count,
					 stride_A,
					 stride_B,
					 {},
					 stride_E,
					 batch_stride_A,
					 batch_stride_B,
					 {},
					 batch_stride_E,
					 AElementOp{},
					 BElementOp{},
					 CDEElementOp{alpha});
  
  if(!device_op.IsSupportedArgument(argument))
    {
      throw std::runtime_error(
			       "wrong! device_gemm with the specified compilation parameters does "
			       "not support this GEMM problem");
    }
  
  invoker.Run(argument, StreamConfig{nullptr, 0});

  return E;
}

