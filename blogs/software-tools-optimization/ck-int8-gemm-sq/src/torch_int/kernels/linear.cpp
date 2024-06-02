#include "include/common.hpp"
#include "include/linear.h"

struct ScaleScaleAddRelu {
  template <typename Y, typename X0, typename X1>
  __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&) const;

  template <>
  __host__ __device__ constexpr void
  operator()<I8, I32, I8>(I8& e, const I32& c, const I8& d) const
    {
      const F32 c_scale = ck::type_convert<F32>(c) * alpha;
      const F32 d_scale = ck::type_convert<F32>(d) * beta;
      F32 temp = c_scale + d_scale;
      
      // RELU
      temp = temp > 0 ? temp : 0;

      // INT8 range
      temp = temp > 127 ? 127 : temp;
      
      e = ck::type_convert<I8>(temp);
    }
    
  F32 alpha;
  F32 beta;
};

struct ScaleScaleAdd {
  template <typename Y, typename X0, typename X1>
  __host__ __device__ constexpr void operator()(Y&, const X0&, const X1&) const;
  
  template <>
  __host__ __device__ constexpr void operator()<I8, I32, I8>(I8& e, const I32& c, const I8& d) const
    {
      const F32 c_scale = ck::type_convert<F32>(c) * alpha;
      const F32 d_scale = ck::type_convert<F32>(d) * beta;
      F32 temp = c_scale + d_scale;

      temp = temp > 127 ? 127 : temp;
      temp = temp < -128? -128 : temp;

      e = ck::type_convert<I8>(temp);	
    }

  template <>
  __host__ __device__ constexpr void operator()<F32, I32, F32>(F32& e, const I32& c, const F32& d) const
    {
      const F32 c_scale = ck::type_convert<F32>(c) * alpha;
      const F32 d_scale = d * beta;
      e = c_scale + d_scale;
    }
    
  F32 alpha;
  F32 beta;
};


torch::Tensor linear_ab_i8_de_f32(torch::Tensor A_,
				  torch::Tensor B_,
				  torch::Tensor D_,
				  float alpha,
				  float beta)
{
  auto A = A_.unsqueeze(0);
  auto B = B_.unsqueeze(0);
    
  int batch_count = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  int stride_A = K;
  int stride_B = K;
  int stride_D0 = N;
  int stride_E = N;
  long long int batch_stride_A = M * K;
  long long int batch_stride_B = K * N;
  int batch_stride_D0 = M * N;
  long long int batch_stride_E = M * N;
  auto D = D_.view({1,-1}).repeat({M, 1});
  auto E = torch::empty({batch_count, M, N}, torch::dtype(torch::kFloat32).device(A.device()));

  using ADataType        = I8;
  using BDataType        = I8;
  using AccDataType      = I32;
  using CShuffleDataType = I32;
  using D0DataType = F32;
  using DsDataType       = ck::Tuple<D0DataType>;
  using EDataType        = F32;

  using ALayout  = RowMajor;
  using BLayout  = ColumnMajor;
  using D0Layout = RowMajor;
  using DsLayout = ck::Tuple<D0Layout>;
  using ELayout  = RowMajor;
  
  using AElementOp   = PassThrough;
  using BElementOp   = PassThrough;
  using CDEElementOp = ScaleScaleAdd;

  static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

  using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl
    < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   64,   64,    64,  16,  16,   32,   32,    1,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,       
    2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,              16>;
  
  auto A_ref = A.data_ptr<ADataType>();
  auto B_ref = B.data_ptr<BDataType>();
  auto D0_ref = D.data_ptr<D0DataType>();
  auto E_ref = E.data_ptr<EDataType>();
  
  auto device_op    = DeviceOpInstance{};
  auto invoker = device_op.MakeInvoker();
  auto argument = device_op.MakeArgument(A_ref,
					 B_ref,
					 {D0_ref},
					 E_ref,
					 M,
					 N,
					 K,
					 batch_count,
					 stride_A,
					 stride_B,
					 {stride_D0},
					 stride_E,
					 batch_stride_A,
					 batch_stride_B,
					 {batch_stride_D0},
					 batch_stride_E,
					 AElementOp{},
					 BElementOp{},
					 CDEElementOp{alpha, beta});
  
  if(!device_op.IsSupportedArgument(argument))
    {
      throw std::runtime_error(
			       "wrong! device_gemm with the specified compilation parameters does "
			       "not support this GEMM problem");
    }
  
  invoker.Run(argument, StreamConfig{nullptr, 0});

  return E.squeeze(0);

}

torch::Tensor linear_abde_i8(torch::Tensor A_,
				  torch::Tensor B_,
				  torch::Tensor D_,
				  float alpha,
				  float beta)
{
  auto A = A_.unsqueeze(0);
  auto B = B_.unsqueeze(0);
    
  int batch_count = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  int stride_A = K;
  int stride_B = K;
  int stride_D0 = N;
  int stride_E = N;
  long long int batch_stride_A = M * K;
  long long int batch_stride_B = K * N;
  int batch_stride_D0 = M * N;
  long long int batch_stride_E = M * N;
  auto D = D_.view({1,-1}).repeat({M, 1});
  auto E = torch::empty({batch_count, M, N}, torch::dtype(torch::kInt8).device(A.device()));

  using ADataType        = I8;
  using BDataType        = I8;
  using AccDataType      = I32;
  using CShuffleDataType = I32;
  using D0DataType = I8;
  using DsDataType       = ck::Tuple<D0DataType>;
  using EDataType        = I8;

  using ALayout  = RowMajor;
  using BLayout  = ColumnMajor;
  using D0Layout = RowMajor;
  using DsLayout = ck::Tuple<D0Layout>;
  using ELayout  = RowMajor;
  
  using AElementOp   = PassThrough;
  using BElementOp   = PassThrough;
  using CDEElementOp = ScaleScaleAdd;

  static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

  using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl
    < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   64,   64,    64,  16,  16,   32,   32,    1,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,       
    2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,              16>;
  
  auto A_ref = A.data_ptr<ADataType>();
  auto B_ref = B.data_ptr<BDataType>();
  auto D0_ref = D.data_ptr<D0DataType>();
  auto E_ref = E.data_ptr<EDataType>();
  
  auto device_op    = DeviceOpInstance{};
  auto invoker = device_op.MakeInvoker();
  auto argument = device_op.MakeArgument(A_ref,
					 B_ref,
					 {D0_ref},
					 E_ref,
					 M,
					 N,
					 K,
					 batch_count,
					 stride_A,
					 stride_B,
					 {stride_D0},
					 stride_E,
					 batch_stride_A,
					 batch_stride_B,
					 {batch_stride_D0},
					 batch_stride_E,
					 AElementOp{},
					 BElementOp{},
					 CDEElementOp{alpha, beta});
  
  if(!device_op.IsSupportedArgument(argument))
    {
      throw std::runtime_error(
			       "wrong! device_gemm with the specified compilation parameters does "
			       "not support this GEMM problem");
    }
  
  invoker.Run(argument, StreamConfig{nullptr, 0});

  return E.squeeze(0);

}

torch::Tensor linear_relu_abde_i8(torch::Tensor A_,
				  torch::Tensor B_,
				  torch::Tensor D_,
				  float alpha,
				  float beta)
{
  auto A = A_.unsqueeze(0);
  auto B = B_.unsqueeze(0);
    
  int batch_count = A.size(0);
  int M = A.size(1);
  int N = B.size(1);
  int K = A.size(2);

  int stride_A = K;
  int stride_B = K;
  int stride_D0 = N;
  int stride_E = N;
  long long int batch_stride_A = M * K;
  long long int batch_stride_B = K * N;
  int batch_stride_D0 = M * N;
  long long int batch_stride_E = M * N;
  auto D = D_.view({1,-1}).repeat({M, 1});
  auto E = torch::empty({batch_count, M, N}, torch::dtype(torch::kInt8).device(A.device()));

  using ADataType        = I8;
  using BDataType        = I8;
  using AccDataType      = I32;
  using CShuffleDataType = I32;
  using D0DataType = I8;
  using DsDataType       = ck::Tuple<D0DataType>;
  using EDataType        = I8;

  using ALayout  = RowMajor;
  using BLayout  = ColumnMajor;
  using D0Layout = RowMajor;
  using DsLayout = ck::Tuple<D0Layout>;
  using ELayout  = RowMajor;
  
  using AElementOp   = PassThrough;
  using BElementOp   = PassThrough;
  using CDEElementOp = ScaleScaleAddRelu;

  static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

  using DeviceOpInstance = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl
    < ALayout, BLayout, DsLayout, ELayout, ADataType, BDataType, AccDataType, CShuffleDataType, DsDataType, EDataType,  AElementOp,  BElementOp, CDEElementOp,    GemmDefault,        1,   256,   64,   64,    64,  16,  16,   32,   32,    1,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,       
    2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,              16>;

  auto A_ref = A.data_ptr<ADataType>();
  auto B_ref = B.data_ptr<BDataType>();
  auto D0_ref = D.data_ptr<D0DataType>();
  auto E_ref = E.data_ptr<EDataType>();
  
  auto device_op    = DeviceOpInstance{};
  auto invoker = device_op.MakeInvoker();
  auto argument = device_op.MakeArgument(A_ref,
					 B_ref,
					 {D0_ref},
					 E_ref,
					 M,
					 N,
					 K,
					 batch_count,
					 stride_A,
					 stride_B,
					 {stride_D0},
					 stride_E,
					 batch_stride_A,
					 batch_stride_B,
					 {batch_stride_D0},
					 batch_stride_E,
					 AElementOp{},
					 BElementOp{},
					 CDEElementOp{alpha, beta});
  
  if(!device_op.IsSupportedArgument(argument))
    {
      throw std::runtime_error(
			       "wrong! device_gemm with the specified compilation parameters does "
			       "not support this GEMM problem");
    }
  
  invoker.Run(argument, StreamConfig{nullptr, 0});

  return E.squeeze(0);

}



