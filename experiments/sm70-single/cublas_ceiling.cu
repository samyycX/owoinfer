// FP16-input / FP32-accumulator cuBLASLt timing reference. Conversion and quantized epilogues are NOT timed.
// Outputs are zero in this ceiling probe; it is not a replacement for exact INT8 inference.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "core/cuda_check.h"

namespace {
void lt_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasLtGetStatusString(status));
}
struct Buffer {
  void* p = nullptr;
  explicit Buffer(std::size_t bytes) { MI_CUDA_CHECK(cudaMalloc(&p, bytes)); MI_CUDA_CHECK(cudaMemset(p, 0, bytes)); }
  ~Buffer() { cudaFree(p); }
};
template <class F>
float timing(F&& f, int count) {
  for (int i = 0; i < 3; ++i) f();
  cudaEvent_t start, end;
  MI_CUDA_CHECK(cudaEventCreate(&start));
  MI_CUDA_CHECK(cudaEventCreate(&end));
  MI_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < count; ++i) f();
  MI_CUDA_CHECK(cudaEventRecord(end));
  MI_CUDA_CHECK(cudaEventSynchronize(end));
  float ms;
  MI_CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / count;
}
void measure(cublasLtHandle_t handle, int m, int n, int k, cudaDataType_t output_type) {
  constexpr std::size_t workspace_bytes = 32u << 20;
  Buffer a(std::size_t(n) * k * 2), b(std::size_t(m) * k * 2), c(std::size_t(m) * n * 4), scratch(workspace_bytes);
  cublasLtMatmulDesc_t desc;
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatmulPreference_t pref;
  lt_check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t transpose = CUBLAS_OP_T;
  lt_check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
  lt_check(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, k, n, k));
  lt_check(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, k, m, k));
  lt_check(cublasLtMatrixLayoutCreate(&lc, output_type, n, m, n));
  lt_check(cublasLtMatmulPreferenceCreate(&pref));
  lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                              &workspace_bytes, sizeof(workspace_bytes)));
  std::array<cublasLtMatmulHeuristicResult_t, 32> algorithms{};
  int found = 0;
  lt_check(cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, pref, algorithms.size(), algorithms.data(), &found));
  if (found == 0) throw std::runtime_error("no FP16 cuBLASLt algorithm");
  const float alpha = 1.f, beta = 0.f;
  const auto run = [&](int i) {
    lt_check(cublasLtMatmul(handle, desc, &alpha, a.p, la, b.p, lb, &beta, c.p, lc, c.p, lc,
                           &algorithms[i].algo, scratch.p, workspace_bytes, nullptr));
  };
  int best = 0;
  float best_ms = std::numeric_limits<float>::infinity();
  for (int i = 0; i < found; ++i) {
    if (algorithms[i].state != CUBLAS_STATUS_SUCCESS) continue;
    const float ms = timing([&] { run(i); }, 5);
    if (ms < best_ms) { best = i; best_ms = ms; }
  }
  const float ms = timing([&] { run(best); }, 20);
  std::printf("cublasLt M=%d N=%d K=%d output=%s: %.3f ms %.2f TFLOP/s (choice %d/%d, workspace %zu)\n",
              m, n, k, output_type == CUDA_R_32F ? "fp32" : "fp16", ms, 2.0 * m * n * k / (ms * 1e9),
              best, found, algorithms[best].workspaceSize);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(lc);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatmulDescDestroy(desc);
}
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  cublasLtHandle_t handle;
  lt_check(cublasLtCreate(&handle));
  for (auto [n, k] : {std::pair{9216, 3072}, {3072, 3072}, {16384, 3072}, {3072, 8192}}) {
    measure(handle, 8222, n, k, CUDA_R_16F);
    measure(handle, 8222, n, k, CUDA_R_32F);
  }
  cublasLtDestroy(handle);
}
