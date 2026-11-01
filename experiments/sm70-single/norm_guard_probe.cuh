// Experimental scratch ownership: single-device CLI only. A production implementation must use caller scratch.
#pragma once
#include "core/cuda_check.h"
#include "kernels/quant/quant.h"

namespace mi::kernels::sm70::probe {
__global__ void norm_bounds(quant::Int8Operands p, unsigned* bounds) {
  __shared__ unsigned xs[8], ws[8];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int row = blockIdx.x * 8 + warp;
  int sum = 0;
  if (row < p.m + p.n) {
    const auto* source = row < p.m ? p.x_q + std::int64_t(row) * p.ldx : p.w + std::int64_t(row - p.m) * p.k;
    const auto* packed = reinterpret_cast<const int*>(source);
    for (int k = lane; k < p.k / 4; k += 32) {
      const int v = packed[k];
      sum = __dp4a(v, v, sum);
    }
  }
  for (int d = 16; d > 0; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
  if (lane == 0) {
    xs[warp] = row < p.m ? sum : 0;
    ws[warp] = row >= p.m ? sum : 0;
  }
  __syncthreads();
  if (warp == 0) {
    unsigned x = lane < 8 ? xs[lane] : 0, w = lane < 8 ? ws[lane] : 0;
    for (int d = 16; d > 0; d /= 2) {
      x = max(x, __shfl_down_sync(0xffffffffu, x, d));
      w = max(w, __shfl_down_sync(0xffffffffu, w, d));
    }
    if (lane == 0) { atomicMax(bounds, x); atomicMax(bounds + 1, w); }
  }
}
inline unsigned* prepare(const quant::Int8Operands& p, cudaStream_t stream) {
  static unsigned* bounds = [] {
    unsigned* ptr = nullptr;
    MI_CUDA_CHECK(cudaMalloc(&ptr, 2 * sizeof(unsigned)));
    return ptr;
  }();
  MI_CUDA_CHECK(cudaMemsetAsync(bounds, 0, 2 * sizeof(unsigned), stream));
  norm_bounds<<<(p.m + p.n + 7) / 8, 256, 0, stream>>>(p, bounds);
  MI_CUDA_CHECK_LAUNCH();
  return bounds;
}
}  // namespace mi::kernels::sm70::probe
