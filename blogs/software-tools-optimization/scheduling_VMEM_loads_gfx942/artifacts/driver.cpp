#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"

#include <iostream>
#include <vector>

using std::vector;

#define HIP_CHECK(func)                                                        \
  do {                                                                         \
    hipError_t status = func;                                                  \
    if (status != hipSuccess) {                                                \
      std::cerr << "HIP error: " << hipGetErrorString(status)                  \
                << " calling " #func "at Line: " << __LINE__ << "\n";          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static constexpr uint32_t numWorkgroups = 608;
static constexpr uint32_t lanesPerWorkgroup = 256;
static constexpr uint32_t shared_memory = 65536;

template <typename T>
void fillArray(vector<T> &buffer, const vector<T> pattern) {
  for (size_t i = 0, e = buffer.size(); i < e; ++i) {
    buffer[i] = pattern[i % pattern.size()];
  }
}

void timeKernel(void *xDevice, void *yDevice, void *zDevice, void *biasDevice,
                int32_t M, int32_t N, int32_t K, int32_t strideA,
                int32_t strideB, int32_t strideC, int32_t stride_bias,
                const char *filename, void *global_scratch) {

  hipModule_t Module;
  hipFunction_t Function;

  HIP_CHECK(hipModuleLoad(&Module, filename));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, "matmul_kernel"));

  // Kernel expects 12 args (72 bytes):
  // offset 0:  ptr xDevice
  // offset 8:  ptr yDevice
  // offset 16: ptr zDevice
  // offset 24: ptr biasDevice
  // offset 32: i32 M
  // offset 36: i32 N
  // offset 40: i32 K
  // offset 44: i32 strideA
  // offset 48: i32 strideB
  // offset 52: i32 strideC
  // offset 56: i32 stride_bias
  // offset 64: ptr global_scratch
  void *params[] = {&xDevice,    &yDevice,  &zDevice,     &biasDevice,
                    &M,          &N,        &K,           &strideA,
                    &strideB,    &strideC,  &stride_bias, &global_scratch};

  float eventMs = 0;
  auto minMs = std::numeric_limits<float>::max();
  static constexpr auto run_count = 20u;

  // warm up call
  HIP_CHECK(hipModuleLaunchKernel(Function, numWorkgroups, 1, 1,
                                  lanesPerWorkgroup, 1, 1, shared_memory,
                                  nullptr, params, 0));
  for (auto i = 0u; i < run_count; i++) {
    float curMs = 0.0f;
    hipEvent_t start, stop;
    (void)hipEventCreate(&start);
    (void)hipEventCreate(&stop);
    (void)hipEventRecord(start, NULL);
    HIP_CHECK(hipModuleLaunchKernel(Function, numWorkgroups, 1, 1,
                                    lanesPerWorkgroup, 1, 1, shared_memory,
                                    nullptr, params, 0));
    (void)hipEventRecord(stop, NULL);
    (void)hipEventSynchronize(stop);
    (void)hipEventElapsedTime(&curMs, start, stop);
    eventMs += curMs;
    minMs = std::min(minMs, curMs);
  }
  auto avgMs = eventMs / run_count;
  printf(
      "kernel_time avg over %d runs (hipEventElapsedTime) =%6.3fms (%6fns)\n",
      run_count, avgMs, avgMs * 1e6);
  printf(
      "kernel_time min over %d runs (hipEventElapsedTime) =%6.3fms (%6fns)\n",
      run_count, minMs, minMs * 1e6);
  std::cout << "avg kernel_flops ="
            << (2 * double(M) * double(K) * double(N) / 1e9) / avgMs
            << "TFlops\n";
}

template <typename T> void *pushToDevice(const vector<T> &hostVal) {
  void *ret = nullptr;
  size_t nBytes = hostVal.size() * sizeof(T);
  HIP_CHECK(hipMalloc((void **)&ret, nBytes));
  HIP_CHECK(hipMemcpyHtoD(ret, (void *)hostVal.data(), nBytes));
  return ret;
}

template <typename T>
vector<T> slurpFromDevice(void *deviceBuf, size_t length) {
  vector<T> ret(length, 0);
  HIP_CHECK(hipMemcpyDtoH(ret.data(), deviceBuf, length * sizeof(T)));
  return ret;
}

bool test(const char *filename) {
  constexpr uint32_t M = 4864;
  constexpr uint32_t N = 8192;
  constexpr uint32_t K = 4160;
  constexpr uint32_t xSize = M * K;
  constexpr uint32_t ySize = K * N;
  constexpr uint32_t zSize = M * N;

  vector<_Float16> x(xSize, 0.0);
  vector<_Float16> y(ySize, 0.0);
  vector<_Float16> z(zSize, 0.0);
  vector<_Float16> bias(zSize, 0.0);
  vector<_Float16> scratch(zSize, 0.0);

  const vector<_Float16> patternA = {1.0, 1.0, 1.0};
  const vector<_Float16> patternB = {1.0, 1.0, 1.0};
  const vector<_Float16> patternC = {0.0};

  fillArray(x, patternA);
  fillArray(y, patternB);
  fillArray(z, patternC);
  fillArray(bias, patternC);
  fillArray(scratch, patternC);

  void *xDevice = pushToDevice(x);
  void *yDevice = pushToDevice(y);
  void *zDevice = pushToDevice(z);
  void *biasDevice = pushToDevice(bias);
  void *scratchDevice = pushToDevice(scratch);

  bool success = true;

  // Transpose = T N
  timeKernel(xDevice, yDevice, zDevice, biasDevice, M, N, K, K, K, N, 0,
             filename, scratchDevice);

  HIP_CHECK(hipFree(zDevice));
  HIP_CHECK(hipFree(xDevice));
  HIP_CHECK(hipFree(yDevice));
  HIP_CHECK(hipFree(biasDevice));
  HIP_CHECK(hipFree(scratchDevice));

  return success;
}

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " path_to_assembled_module.hsaco\n";
    return 1;
  }
  const char *filename = argv[1];
  std::cout << filename << " ...\n";
  bool ret = test(filename);
  std::cout << (ret ? "pass\n" : "FAIL!\n");
  return !ret;
}
