/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// Volta INT8 ConvRot GEMMs: fp16 tensor-core partials, exact int32 sums and the existing quant epilogues.
#include "kernels/sm70/sm70.h"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

#include "kernels/quant/int8_gemm_impl.cuh"

namespace mi::kernels::sm70 {
namespace qd = quant::detail;

namespace {

// The SiTU factor tables of the quantized epilogue: the sm_80 tables (int8_gemm_situ.cu) are filled by a kernel built for
// sm_80+ only, so the sm_70 path keeps its own 256 KiB copy, filled once per device by a kernel built for 70. The
// factors are the same functions, so a lookup is bit-identical to the sm_80 table's value.
constexpr int kBf16Values = 1 << 16;
__device__ std::uint16_t g_situ_gate[kBf16Values];
__device__ std::uint16_t g_situ_hidden[kBf16Values];

__global__ void situ_tables_sm70_kernel() {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const float x = __uint_as_float(static_cast<std::uint32_t>(i) << 16);
  g_situ_gate[i] = __bfloat16_as_ushort(__float2bfloat16_rn(qd::situ_gate_factor(x)));
  g_situ_hidden[i] = __bfloat16_as_ushort(__float2bfloat16_rn(qd::situ_hidden_factor(x)));
}

// Fill once per device (see quant::detail::situ_factor_tables; the same lock and capture handling).
qd::SituFactorTables situ_tables(cudaStream_t stream) {
  constexpr int kMaxDevices = 64;
  struct Device {
    std::atomic<bool> filled{false};
    std::atomic<const std::uint16_t*> gate{nullptr}, hidden{nullptr};
  };
  static Device devices[kMaxDevices];
  static std::mutex fill_mutex;
  int dev = 0;
  MI_CUDA_CHECK(cudaGetDevice(&dev));
  if (dev < 0 || dev >= kMaxDevices) throw std::runtime_error("mi::kernels::sm70: device ordinal out of range");
  Device& d = devices[dev];
  const auto fill = [&] {
    constexpr int kThreads = 256;
    situ_tables_sm70_kernel<<<kBf16Values / kThreads, kThreads, 0, stream>>>();
    MI_CUDA_CHECK_LAUNCH();
  };
  if (!d.filled.load(std::memory_order_acquire)) {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    MI_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
      fill();
    } else {
      const std::lock_guard<std::mutex> lock(fill_mutex);
      if (!d.filled.load(std::memory_order_relaxed)) {
        fill();
        MI_CUDA_CHECK(cudaStreamSynchronize(stream));
        d.filled.store(true, std::memory_order_release);
      }
    }
  }
  if (d.gate.load(std::memory_order_acquire) == nullptr) {
    void *gate = nullptr, *hidden = nullptr;
    MI_CUDA_CHECK(cudaGetSymbolAddress(&gate, g_situ_gate));
    MI_CUDA_CHECK(cudaGetSymbolAddress(&hidden, g_situ_hidden));
    d.hidden.store(static_cast<const std::uint16_t*>(hidden), std::memory_order_release);
    d.gate.store(static_cast<const std::uint16_t*>(gate), std::memory_order_release);
  }
  return {d.gate.load(std::memory_order_acquire), d.hidden.load(std::memory_order_acquire)};
}

// The main loop uses CUTLASS's Volta shared-memory swizzle and pipelined register loads. Global iterators still
// read INT8; MmaPipelined's NumericArrayConverter stages exact fp16 integers. Fold every 1024 products into int32:
// each fp32 partial is bounded by 2^24, preserving integer sums even with cancellation at the maximum supported K.
using Sm70SmallTile = qd::TileCfg<64, 128, 32, 32, 64, 2>;
using Sm70LargeTile = qd::TileCfg<128, 128, 32, 32, 64, 2>;
using Sm70Instr = cutlass::gemm::GemmShape<8, 8, 4>;

template <class Base>
struct ExactMma : Base {
  using typename Base::FragmentC;
  using typename Base::IteratorA;
  using typename Base::IteratorB;
  using typename Base::SharedStorage;
  CUTLASS_DEVICE ExactMma(SharedStorage& s, int t, int w, int l) : Base(s, t, w, l) {}

  CUTLASS_DEVICE void operator()(int iterations, FragmentC& accum, IteratorA a, IteratorB b, const FragmentC& src) {
    this->prologue(a, b, iterations);
    this->gmem_wait();
    typename Base::Operator::FragmentA wa[2];
    typename Base::Operator::FragmentB wb[2];
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(wa[0]);
    this->warp_tile_iterator_B_.load(wb[0]);
    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;
    typename Base::FragmentA ta;
    typename Base::FragmentB tb;
    a.clear_mask(iterations <= 1);
    b.clear_mask(iterations <= 1);
    FragmentC partial;
    partial.clear();
    cutlass::Array<int, FragmentC::kElements> sums;
    sums.clear();
    static_assert(1024 % Base::Shape::kK == 0 && Base::Policy::kPartitionsK == 1);
    constexpr int kFold = 1024 / Base::Shape::kK;
    constexpr int kWarpIterations = Base::kWarpGemmIterations;
    // Fold only at the chunk boundary; both shared-memory stages and the prefetched warp fragments stay live.
    while (iterations > 0) {
      const int chunk = min(kFold, iterations);
      CUTLASS_GEMM_LOOP
      for (int inner = 0; inner < chunk; ++inner, --iterations) {
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < kWarpIterations; ++k) {
          if (k == kWarpIterations - 1) {
            this->smem_iterator_A_.store(this->transform_A_(ta));
            this->smem_iterator_B_.store(this->transform_B_(tb));
            this->gmem_wait();
            this->advance_smem_stages();
          }
          this->warp_tile_iterator_A_.set_kgroup_index((k + 1) % kWarpIterations);
          this->warp_tile_iterator_B_.set_kgroup_index((k + 1) % kWarpIterations);
          this->warp_tile_iterator_A_.load(wa[(k + 1) % 2]);
          this->warp_tile_iterator_B_.load(wb[(k + 1) % 2]);
          ++this->warp_tile_iterator_A_;
          ++this->warp_tile_iterator_B_;
          if (k == 0) {
            ta.clear();
            tb.clear();
            a.load(ta);
            b.load(tb);
            ++a;
            ++b;
            a.clear_mask(iterations <= 2);
            b.clear_mask(iterations <= 2);
          }
          this->warp_mma(partial, wa[k % 2], wb[k % 2], partial);
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentC::kElements; ++i) sums[i] += __float2int_rn(partial[i]);
      partial.clear();
    }
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentC::kElements; ++i) accum[i] = __int2float_rn(sums[i]) + src[i];
  }
};

// Each int8 row norm fits uint32 under kMaxInt8GemmK. Their product fits uint64. Cauchy-Schwarz bounds
// every partial dot product, in any reduction order: max(sum(x*x)) * max(sum(w*w)) <= 2^48 guarantees
// that all integer fp32 accumulators stay within [-2^24, 2^24]. Unsafe matrices retain the exact integer folds.
__global__ void norm_bounds_kernel(quant::Int8Operands p, unsigned* bounds) {
  __shared__ unsigned xs[8], ws[8];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const std::int64_t row = std::int64_t(blockIdx.x) * 8 + warp;
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

enum class Epilogue { kPlain, kResidual, kSitu };
using qd::add;
using qd::mul;
using qd::rbf;
using qd::dequant;

template <Epilogue E, typename Tile, bool Fast = false>
struct Kernel {
  using Core = cutlass::gemm::threadblock::DefaultMmaCore<
      typename Tile::Threadblock, typename Tile::Warp, Sm70Instr, cutlass::half_t, qd::RowMajor,
      cutlass::half_t, qd::ColumnMajor, float, qd::RowMajor, cutlass::arch::OpClassTensorOp, 2>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<Core::Shape::kM, Core::Shape::kK>, std::int8_t, qd::RowMajor, 1,
      typename Core::IteratorThreadMapA, 8>;
  using PermuteB = std::conditional_t<E == Epilogue::kSitu, qd::UpGatePermuteB, cutlass::layout::NoPermute>;
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<Core::Shape::kK, Core::Shape::kN>, std::int8_t, qd::ColumnMajor, 0,
      typename Core::IteratorThreadMapB, 8, false, PermuteB>;
  using BaseMma = cutlass::gemm::threadblock::MmaPipelined<
      typename Core::Shape, IteratorA, typename Core::SmemIteratorA, IteratorB, typename Core::SmemIteratorB,
      float, qd::RowMajor, typename Core::MmaPolicy>;
  using Mma = std::conditional_t<Fast, BaseMma, ExactMma<BaseMma>>;
  using SharedStorage = typename Mma::SharedStorage;
};

template <Epilogue E, typename Tile, bool Fast = false>
__global__ void int8_kernel(quant::Int8Operands p, typename Kernel<E, Tile, Fast>::IteratorA::Params pa,
                             typename Kernel<E, Tile, Fast>::IteratorB::Params pb, const __nv_bfloat16* bias,
                             const __nv_bfloat16* gate, __nv_bfloat16* out, std::int64_t ldo,
                             qd::SituFactorTables tables, int log_tile, const unsigned* bounds) {
  using K = Kernel<E, Tile, Fast>;
  constexpr int BM = Tile::Threadblock::kM, BN = Tile::Threadblock::kN;
  extern __shared__ __align__(16) char buffer[];
  auto& storage = *reinterpret_cast<typename K::SharedStorage*>(buffer);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const auto tile = qd::Swizzle::get_tile_offset(log_tile);
  const int m0 = tile.m() * BM, n0 = tile.n() * BN;
  if (m0 >= p.m || n0 >= p.n) return;
  if (p.x_norm && p.w_norm) {
    __shared__ bool safe_tile;
    if (warp == 0) {
      // Fast and folded SiTU tiles have different widths. Both use the same canonical 128x256 decision tile,
      // so a rejected fast tile is covered by exactly two folded 128x128 tiles, without double writes or holes.
      constexpr int GN = E == Epilogue::kSitu ? 256 : 128;
      const int nbase = (n0 / GN) * GN;
      unsigned xn = 0, wn = 0;
      for (int r = lane; r < 128; r += 32) if (m0 + r < p.m) xn = max(xn, p.x_norm[m0 + r]);
      for (int c = lane; c < GN; c += 32) {
        const int col = nbase + c;
        if (col < p.n) wn = max(wn, p.w_norm[E == Epilogue::kSitu ? qd::up_gate_row(col, p.n / 2) : col]);
      }
      for (int d = 16; d > 0; d /= 2) {
        xn = max(xn, __shfl_down_sync(0xffffffffu, xn, d));
        wn = max(wn, __shfl_down_sync(0xffffffffu, wn, d));
      }
      if (lane == 0) safe_tile = std::uint64_t(xn) * wn <= (std::uint64_t{1} << 48);
    }
    __syncthreads();
    if (safe_tile != Fast) return;
  } else if (bounds) {
    const bool safe = std::uint64_t(bounds[0]) * bounds[1] <= (std::uint64_t{1} << 48);
    if (safe != Fast) return;
  }
  typename K::IteratorA a(pa, const_cast<std::int8_t*>(p.x_q), {int(p.m), int(p.k)}, tid, {m0, 0});
  typename K::IteratorB b(pb, const_cast<std::int8_t*>(p.w), {int(p.k), int(p.n)}, tid, {0, n0});
  typename K::Mma mma(storage, tid, warp, lane);
  typename K::Mma::FragmentC accum, zero;
  zero.clear();
  mma((p.k + Tile::Threadblock::kK - 1) / Tile::Threadblock::kK, accum, a, b, zero);

  // CUTLASS's Volta FP32 accumulator is interleaved in 32x32 tiles. Each lane owns two rows and adjacent
  // column pairs per MMA. Decode that layout directly, avoiding a full float tile and two CTA barriers.
  constexpr int WM = Tile::Warp::kM, WN = Tile::Warp::kN;
  static_assert(WM % 32 == 0 && WN % 32 == 0);
  static_assert(fused::detail::kUpGateGroup == 4);
  const int quad = lane >> 2;
  const int r0 = m0 + (warp % (BM / WM)) * WM + (((quad & 4) >> 1) + (quad & 1)) * 8 + (lane & 1);
  const int c0 = (warp / (BM / WM)) * WN + ((quad >> 1) & 1) * 8 + (lane & 2);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < K::Mma::FragmentC::kElements; ++i) {
    // Grouped up/gate columns are four apart: the matching gate is 16 registers after each hidden value.
    if constexpr (E == Epilogue::kSitu) { if (i & 16) continue; }
    const int r = r0 + ((i / 32) % (WM / 32)) * 32 + ((i >> 3) & 1) * 4 + ((i >> 1) & 1) * 2;
    const int c = c0 + (i / 32 / (WM / 32)) * 32 + ((i >> 4) & 1) * 4 + ((i >> 2) & 1) * 16 + (i & 1);
    const int col = E == Epilogue::kSitu ? n0 / 2 + (c / 8) * 4 + c % 8 : n0 + c;
    if (r >= p.m || col >= (E == Epilogue::kSitu ? p.n / 2 : p.n)) continue;
    const float xs = p.x_scale[r];
    float value;
    if constexpr (E == Epilogue::kSitu) {
      const float h = rbf(dequant(accum[i], xs, p.w_scale[col], 0.f));
      const float gt = rbf(dequant(accum[i + 16], xs, p.w_scale[col + p.n / 2], 0.f));
      value = mul(qd::bf16_entry(tables.hidden, qd::bf16_index(h)),
                  qd::bf16_entry(tables.gate, qd::bf16_index(gt)));
    } else {
      value = dequant(accum[i], xs, p.w_scale[col], bias ? __bfloat162float(bias[col]) : 0.f);
      if constexpr (E == Epilogue::kResidual) {
        value = add(__bfloat162float(out[r * ldo + col]), rbf(mul(rbf(value), __bfloat162float(gate[col]))));
      }
    }
    out[r * ldo + col] = __float2bfloat16_rn(value);
  }
}

template <Epilogue E, typename Tile, bool Fast = false>
void launch_tile(const quant::Int8Operands& p, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
             __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream, const unsigned* bounds = nullptr) {
  constexpr int kBM = Tile::Threadblock::kM, kBN = Tile::Threadblock::kN;
  constexpr int kThreads = (kBM / Tile::Warp::kM) * (kBN / Tile::Warp::kN) * 32;
  // Reuse the quantized kernels' eight-column CTA swizzle to improve L2 reuse of activation tiles.
  const cutlass::gemm::GemmCoord tiled((p.m + kBM - 1) / kBM, (p.n + kBN - 1) / kBN, 1);
  const dim3 grid = qd::Swizzle::get_grid_shape(tiled);
  const int log_tile = qd::Swizzle::get_log_tile(tiled);
  const typename Kernel<E, Tile, Fast>::IteratorA::Params pa{qd::RowMajor(p.ldx)};
  const typename Kernel<E, Tile, Fast>::IteratorB::Params pb{qd::ColumnMajor(p.k)};
  qd::SituFactorTables tables{};
  if constexpr (E == Epilogue::kSitu) tables = situ_tables(stream);
  constexpr int bytes = sizeof(typename Kernel<E, Tile, Fast>::SharedStorage);
  auto kernel = &int8_kernel<E, Tile, Fast>;
  // The per-tile guard also uses static shared memory, so a 48 KiB dynamic tile needs opt-in.
  if constexpr (bytes >= 48 * 1024) MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  kernel<<<grid, kThreads, bytes, stream>>>(p, pa, pb, bias, gate, out, ldo, tables, log_tile, bounds);
  MI_CUDA_CHECK_LAUNCH();
}

template <Epilogue E>
void launch(const quant::Int8Operands& p, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
             __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  if (p.m >= 1024 && (p.norm_bounds || (p.x_norm && p.w_norm))) {
    using FastTile = std::conditional_t<E == Epilogue::kSitu, qd::TileCfg<128, 256, 32, 64, 64, 2>,
                                                               qd::TileCfg<128, 128, 32, 64, 64, 2>>;
    const unsigned* bounds = nullptr;
    if (!p.x_norm) {
      MI_CUDA_CHECK(cudaMemsetAsync(p.norm_bounds, 0, 2 * sizeof(std::uint32_t), stream));
      norm_bounds_kernel<<<(p.m + p.n + 7) / 8, 256, 0, stream>>>(p, p.norm_bounds);
      MI_CUDA_CHECK_LAUNCH();
      bounds = p.norm_bounds;
    }
    if constexpr (E == Epilogue::kResidual) {
      if (p.k <= 3072) launch_tile<E, Sm70LargeTile, true>(p, bias, gate, out, ldo, stream, bounds);
      else launch_tile<E, FastTile, true>(p, bias, gate, out, ldo, stream, bounds);
    } else {
      launch_tile<E, FastTile, true>(p, bias, gate, out, ldo, stream, bounds);
    }
    launch_tile<E, Sm70LargeTile>(p, bias, gate, out, ldo, stream, bounds);
  } else if (p.m >= 1024) {
    launch_tile<E, Sm70LargeTile>(p, bias, gate, out, ldo, stream);
  } else {
    // Row-norm callers can also reach short shapes. These retain the folded kernel without conditional dispatch.
    auto small = p;
    small.x_norm = small.w_norm = nullptr;
    launch_tile<E, Sm70SmallTile>(small, bias, gate, out, ldo, stream);
  }
}

}  // namespace

void int8_gemm(const quant::Int8Operands& op, const __nv_bfloat16* bias, __nv_bfloat16* out, std::int64_t ldo,
               cudaStream_t stream) {
  constexpr const char* name = "sm70::int8_gemm";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "out", out, ldo, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  if (p.m == 0) return;
  launch<Epilogue::kPlain>(op, bias, nullptr, out, ldo, stream);
}

void int8_gemm_gated_residual(const quant::Int8Operands& op, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
                              __nv_bfloat16* residual, std::int64_t ldr, cudaStream_t stream) {
  constexpr const char* name = "sm70::int8_gemm_gated_residual";
  const qd::Problem p = qd::check_operands(name, op);
  qd::check_output(name, "residual", residual, ldr, p.n);
  qd::check_vector(name, "bias", bias, /*optional=*/true);
  qd::check_vector(name, "gate", gate, /*optional=*/false);
  if (p.m == 0) return;
  launch<Epilogue::kResidual>(op, bias, gate, residual, ldr, stream);
}

void int8_gemm_situ_glu(const quant::Int8Operands& op, __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  constexpr const char* name = "sm70::int8_gemm_situ_glu";
  if (op.n % (2 * qd::kUpGateGroup) != 0) {
    kernels::detail::fail(name, "n = 2F needs F % 4 == 0, got n " + std::to_string(op.n));
  }
  const qd::Problem p = qd::check_operands(name, op);
  if (out == nullptr || reinterpret_cast<std::uintptr_t>(out) % 8 != 0 || ldo < p.n / 2 || ldo % 4 != 0) {
    kernels::detail::fail(name, "out must be 8-byte aligned with a row stride >= F and a multiple of 4, got " +
                                    std::to_string(ldo));
  }
  if (p.m == 0) return;
  launch<Epilogue::kSitu>(op, nullptr, nullptr, out, ldo, stream);
}

}  // namespace mi::kernels::sm70
