// Instrumentation only: synchronizes the caller stream and copies exact integer row norms to the host.
// Include in a separately linked GEMM probe, never in the installed inference kernel.
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "core/cuda_check.h"
#include "kernels/quant/quant.h"

namespace mi::kernels::sm70::probe {
__global__ void row_norms(const std::int8_t* x, std::int64_t stride, int k, std::uint32_t* norms) {
  __shared__ int partial[8];
  const auto* row = reinterpret_cast<const int*>(x + std::int64_t(blockIdx.x) * stride);
  int sum = 0;
  for (int i = threadIdx.x; i < k / 4; i += blockDim.x) {
    const int v = row[i];
    sum = __dp4a(v, v, sum);
  }
  for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    sum = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
    for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
    if (threadIdx.x == 0) norms[blockIdx.x] = sum;
  }
}

inline std::vector<std::uint32_t> norms(const std::int8_t* x, int rows, std::int64_t stride, int k,
                                        cudaStream_t stream) {
  std::uint32_t* device = nullptr;
  MI_CUDA_CHECK(cudaMalloc(&device, std::size_t(rows) * sizeof(*device)));
  row_norms<<<rows, 256, 0, stream>>>(x, stride, k, device);
  MI_CUDA_CHECK_LAUNCH();
  std::vector<std::uint32_t> host(rows);
  MI_CUDA_CHECK(cudaMemcpyAsync(host.data(), device, host.size() * sizeof(*device), cudaMemcpyDeviceToHost, stream));
  MI_CUDA_CHECK(cudaStreamSynchronize(stream));
  MI_CUDA_CHECK(cudaFree(device));
  return host;
}

inline void sample(const quant::Int8Operands& p, int epilogue, cudaStream_t stream) {
  const char* path = std::getenv("SM70_NORM_LOG");
  if (!path || p.m < 4096 || (p.k != 3072 && p.k != 8192)) return;
  // This probe is for a single-threaded resident-weight CLI run: pointers remain stable for that run.
  using Key = std::tuple<const std::int8_t*, int, int>;
  static std::map<Key, std::vector<std::uint32_t>> weights;
  static int call = 0;
  static FILE* file = std::fopen(path, "w");
  if (!file) throw std::runtime_error("cannot open SM70_NORM_LOG");
  if (call == 0) std::fprintf(file, "call,epilogue,m,n,k,bm,bn,safe_tiles,total_tiles,x_norm_max,w_norm_max,bound_max\n");
  const Key key{p.w, int(p.n), int(p.k)};
  auto it = weights.find(key);
  if (it == weights.end()) it = weights.emplace(key, norms(p.w, p.n, p.k, p.k, stream)).first;
  const auto& wn = it->second;
  const auto xn = norms(p.x_q, p.m, p.ldx, p.k, stream);
  const auto xmax = *std::max_element(xn.begin(), xn.end()), wmax = *std::max_element(wn.begin(), wn.end());
  for (auto [bm, bn] : {std::pair{128, 128}, {128, 256}, {256, 128}}) {
    int safe = 0, total = 0;
    std::uint64_t maximum = 0;
    for (int r0 = 0; r0 < p.m; r0 += bm) {
      const auto x = *std::max_element(xn.begin() + r0, xn.begin() + std::min<std::int64_t>(r0 + bm, p.m));
      for (int c0 = 0; c0 < p.n; c0 += bn) {
        std::uint32_t w = 0;
        for (int c = c0; c < std::min<std::int64_t>(c0 + bn, p.n); ++c) {
          const int col = epilogue == 2 ? (c / 8) * 4 + c % 4 + ((c / 4) % 2) * (p.n / 2) : c;
          w = std::max(w, wn[col]);
        }
        const std::uint64_t bound = std::uint64_t(x) * w;
        safe += bound <= (std::uint64_t{1} << 48);
        maximum = std::max(maximum, bound);
        ++total;
      }
    }
    std::fprintf(file, "%d,%d,%lld,%lld,%lld,%d,%d,%d,%d,%u,%u,%llu\n", call, epilogue,
                 (long long)p.m, (long long)p.n, (long long)p.k, bm, bn, safe, total, xmax, wmax,
                 (unsigned long long)maximum);
  }
  std::fflush(file);
  ++call;
}
}  // namespace mi::kernels::sm70::probe
