// Exact varlen/GQA attention on Volta: fp16 WMMA with fp32 scores, online softmax and accumulation. Q/K/V stay
// bf16 in global memory; only shared-memory tiles are converted. No quadratic score buffer or hidden allocation.
// A warp owns 16 query rows. The sm_70 accumulator row groups use lane XOR 2/8 (unlike Turing's XOR 1/2);
// coordinates are loaded through WMMA itself, so output stores follow its fragment layout. D=128 keeps the
// softmax tile in registers and shuffles it into the next MMA input; D=256 retains its shared-memory path.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cmath>
#include <stdexcept>

#include "core/cuda_check.h"
#include "kernels/sm70/sm70.h"

namespace mi::kernels::sm70 {
namespace {

namespace wmma = nvcuda::wmma;
constexpr float kLog2e = 1.4426950408889634f;

struct Params {
  const __nv_bfloat16* q = nullptr;
  const __nv_bfloat16* k = nullptr;
  const __nv_bfloat16* v = nullptr;
  __nv_bfloat16* o = nullptr;
  float* lse = nullptr;
  std::int64_t q_row = 0, q_head = 0;
  std::int64_t k_row = 0, k_head = 0;
  std::int64_t v_row = 0, v_head = 0;
  std::int64_t o_row = 0, o_head = 0;
  const int* cu_q = nullptr;
  const int* cu_k = nullptr;
  int heads = 0, kv_heads = 0, total_q = 0;
  float scale = 1.f;
  bool causal = false;
};

// SparkAttention's warp layout transform, adapted to Volta's 16x16 WMMA fragments (FP32 accumulators).
// C gives a lane two rows and four columns; A needs one complete row, duplicated across lane bit 3.
// Pack adjacent columns, exchange row pieces with XOR 2, then collect the two column halves with XOR 8.
// The fp32 softmax and fp16 input rounding are unchanged. See arxiv.org/html/2502.12784v1#S3.SS2.SSS2.
template <class Acc, class A>
__device__ __forceinline__ A probability_fragment(const Acc& s, int lane) {
  static_assert(Acc::num_elements == 8 && A::num_elements == 16);
  union Packed { __half2 h; unsigned u; };
  Packed p0, p2, p4, p6;
  p0.h = __floats2half2_rn(s.x[0], s.x[1]);
  p2.h = __floats2half2_rn(s.x[2], s.x[3]);
  p4.h = __floats2half2_rn(s.x[4], s.x[5]);
  p6.h = __floats2half2_rn(s.x[6], s.x[7]);
  const bool second = (lane & 2) != 0;
  const unsigned own0 = second ? p2.u : p0.u, own4 = second ? p6.u : p4.u;
  const unsigned peer0 = __shfl_xor_sync(0xffffffffu, second ? p0.u : p2.u, 2);
  const unsigned peer4 = __shfl_xor_sync(0xffffffffu, second ? p4.u : p6.u, 2);
  const unsigned pairs[4] = {second ? peer0 : own0, second ? own0 : peer0,
                              second ? peer4 : own4, second ? own4 : peer4};
  A a;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned other = __shfl_xor_sync(0xffffffffu, pairs[i], 8);
    Packed lo, hi;
    lo.u = (lane & 8) ? other : pairs[i];
    hi.u = (lane & 8) ? pairs[i] : other;
    a.x[2 * i] = __low2half(lo.h);
    a.x[2 * i + 1] = __high2half(lo.h);
    a.x[2 * i + 8] = __low2half(hi.h);
    a.x[2 * i + 9] = __high2half(hi.h);
  }
  return a;
}

template <int D, int BN, int kBM>
__global__ __launch_bounds__(kBM * 2) void attention_kernel(const Params p) {
  constexpr int kThreads = kBM * 2;
  constexpr int kQStride = D + 8, kVStride = BN + 8;
  using Acc = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
  using A = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
  using B = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
  extern __shared__ __align__(32) char storage[];
  __half* qs = reinterpret_cast<__half*>(storage);
  __half* ks = qs + kBM * kQStride;
  __half* vt = ks + BN * kQStride;
  __half* ps = vt + D * kVStride;
  const int tid = threadIdx.x, warp = tid / 32;
  const int batch = blockIdx.y, head = blockIdx.z;
  const int q_start = p.cu_q[batch], k_start = p.cu_k[batch];
  const int len_q = p.cu_q[batch + 1] - q_start, len_k = p.cu_k[batch + 1] - k_start;
  const int row_base = blockIdx.x * kBM;
  if (row_base >= len_q) return;
  const int rows = min(kBM, len_q - row_base);
  const int kv_head = head / (p.heads / p.kv_heads), shift = len_k - len_q;

  // K/V have not been loaded yet. Encode coordinates into an accumulator fragment once.
  float* coords = reinterpret_cast<float*>(D == 128 ? vt : ps);
  for (int i = tid; i < 256; i += kThreads) coords[i] = static_cast<float>(i);
  __syncthreads();
  Acc positions;
  wmma::load_matrix_sync(positions, coords, 16, wmma::mem_row_major);
  int rr[Acc::num_elements], cc[Acc::num_elements];
#pragma unroll
  for (int e = 0; e < Acc::num_elements; ++e) {
    rr[e] = static_cast<int>(positions.x[e]) / 16;
    cc[e] = static_cast<int>(positions.x[e]) % 16;
  }
  for (int i = tid; i < kBM * D; i += kThreads) {
    const int r = i / D, d = i % D;
    qs[r * kQStride + d] = __float2half_rn(r < rows
        ? __bfloat162float(p.q[(q_start + row_base + r) * p.q_row + head * p.q_head + d]) : 0.f);
  }
  __syncthreads();
  Acc o[D / 16];
#pragma unroll
  for (int d = 0; d < D / 16; ++d) wmma::fill_fragment(o[d], 0.f);
  float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f};
  const int end = p.causal ? min(len_k, max(0, row_base + rows + shift)) : len_k;
  for (int j0 = 0; j0 < end; j0 += BN) {
    for (int i = tid; i < BN * D; i += kThreads) {
      const int j = i / D, d = i % D, token = k_start + j0 + j;
      const bool valid = j0 + j < len_k;
      ks[j * kQStride + d] = __float2half_rn(valid ? __bfloat162float(p.k[token * p.k_row + kv_head * p.k_head + d]) : 0.f);
      vt[d * kVStride + j] = __float2half_rn(valid ? __bfloat162float(p.v[token * p.v_row + kv_head * p.v_head + d]) : 0.f);
    }
    __syncthreads();
    Acc s[BN / 16];
#pragma unroll
    for (int j = 0; j < BN / 16; ++j) wmma::fill_fragment(s[j], 0.f);
#pragma unroll
    for (int d = 0; d < D; d += 16) {
      A a;
      wmma::load_matrix_sync(a, qs + warp * 16 * kQStride + d, kQStride);
#pragma unroll
      for (int j = 0; j < BN / 16; ++j) {
        B b;
        wmma::load_matrix_sync(b, ks + j * 16 * kQStride + d, kQStride);
        wmma::mma_sync(s[j], a, b, s[j]);
      }
    }
    // Keep 32-key online softmax updates even when staging 64 keys: changing this grouping changes rounding.
#pragma unroll
    for (int part = 0; part < BN / 32; ++part) {
      float mx[2] = {-1e30f, -1e30f};
#pragma unroll
      for (int j = part * 2; j < part * 2 + 2; ++j) {
#pragma unroll
        for (int e = 0; e < Acc::num_elements; ++e) {
          const int key = j0 + j * 16 + cc[e], row = row_base + warp * 16 + rr[e];
          float value = s[j].x[e] * p.scale;
          if (key >= len_k || (p.causal && key > row + shift)) value = -1e30f;
          s[j].x[e] = value;
          mx[(e & 2) / 2] = fmaxf(mx[(e & 2) / 2], value);
        }
      }
      float alpha[2];
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
        mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 8));
        mx[r] = fmaxf(mx[r], m[r]);
        alpha[r] = m[r] <= -1e29f ? 0.f : exp2f((m[r] - mx[r]) * kLog2e);
        m[r] = mx[r];
      }
      float sum[2] = {0.f, 0.f};
#pragma unroll
      for (int j = part * 2; j < part * 2 + 2; ++j) {
#pragma unroll
        for (int e = 0; e < Acc::num_elements; ++e) {
          const int r = (e & 2) / 2;
          const float value = s[j].x[e] <= -1e29f ? 0.f : exp2f((s[j].x[e] - m[r]) * kLog2e);
          sum[r] += value;
          if constexpr (D == 128) s[j].x[e] = value;
          else ps[(warp * 16 + rr[e]) * kVStride + j * 16 + cc[e]] = __float2half_rn(value);
        }
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 2);
        sum[r] += __shfl_xor_sync(0xffffffffu, sum[r], 8);
        l[r] = l[r] * alpha[r] + sum[r];
      }
#pragma unroll
      for (int d = 0; d < D / 16; ++d) {
#pragma unroll
        for (int e = 0; e < Acc::num_elements; ++e) o[d].x[e] *= alpha[(e & 2) / 2];
      }
      if constexpr (D != 128) __syncwarp();  // only this warp reads/writes its P rows
#pragma unroll
      for (int j = part * 32; j < part * 32 + 32; j += 16) {
        A a;
        if constexpr (D == 128) a = probability_fragment<Acc, A>(s[j / 16], tid % 32);
        else wmma::load_matrix_sync(a, ps + warp * 16 * kVStride + j, kVStride);
#pragma unroll
        for (int d = 0; d < D / 16; ++d) {
          B b;
          wmma::load_matrix_sync(b, vt + d * 16 * kVStride + j, kVStride);
          wmma::mma_sync(o[d], a, b, o[d]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int e = 0; e < Acc::num_elements; ++e) {
    const int r = (e & 2) / 2, row = warp * 16 + rr[e];
    if (row < rows) {
      const float inv = l[r] > 0.f ? 1.f / l[r] : 0.f;
#pragma unroll
      for (int d = 0; d < D / 16; ++d) {
        p.o[(q_start + row_base + row) * p.o_row + head * p.o_head + d * 16 + cc[e]] = __float2bfloat16_rn(o[d].x[e] * inv);
      }
      if (cc[e] == 0 && p.lse) p.lse[static_cast<std::int64_t>(head) * p.total_q + q_start + row_base + row] =
          l[r] > 0.f ? m[r] + logf(l[r]) : INFINITY;
    }
  }
}

template <int D, int BN, int kBM>
void launch(const Params& p, int max_seqlen, int batch, cudaStream_t stream) {
  constexpr int kThreads = kBM * 2;
  constexpr int bytes = ((kBM + BN) * (D + 8) + (D + (D == 128 ? 0 : kBM)) * (BN + 8)) * sizeof(__half);
  static_assert(bytes <= 96 * 1024);
  auto kernel = &attention_kernel<D, BN, kBM>;
  if constexpr (bytes > 48 * 1024) MI_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  const dim3 grid(static_cast<unsigned>((max_seqlen + kBM - 1) / kBM), static_cast<unsigned>(batch),
                  static_cast<unsigned>(p.heads));
  kernel<<<grid, kThreads, bytes, stream>>>(p);
  MI_CUDA_CHECK_LAUNCH();
}

bool head_dim_supported(int d) { return d == 128 || d == 256; }

}  // namespace

void flash_fwd(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out, const Tensor& cu_seqlens_q,
               const Tensor& cu_seqlens_k, int max_seqlen, bool causal, float scale, void* workspace,
               std::size_t workspace_bytes, cudaStream_t stream) {
  const std::int64_t total_q = q.dim(0);
  if (total_q == 0) return;
  const std::int64_t heads = q.dim(1);
  const std::int64_t head_dim = q.dim(2);
  const std::int64_t kv_heads = k.dim(1);
  if (!head_dim_supported(static_cast<int>(head_dim))) {
    throw std::invalid_argument("sm70::flash_fwd: head_dim must be 128 or 256");
  }
  if ((head_dim == 128 && causal) || (head_dim == 256 && !causal)) {
    throw std::invalid_argument("sm70::flash_fwd: head_dim 128 is non-causal, 256 is causal");
  }
  if (workspace_bytes < static_cast<std::size_t>(total_q) * heads * sizeof(float)) {
    throw std::invalid_argument("sm70::flash_fwd: workspace too small for the log-sum-exp");
  }

  Params p;
  p.q = static_cast<const __nv_bfloat16*>(q.data);
  p.k = static_cast<const __nv_bfloat16*>(k.data);
  p.v = static_cast<const __nv_bfloat16*>(v.data);
  p.o = static_cast<__nv_bfloat16*>(out.data);
  p.lse = static_cast<float*>(workspace);
  p.q_row = q.stride[0];
  p.q_head = q.stride[1];
  p.k_row = k.stride[0];
  p.k_head = k.stride[1];
  p.v_row = v.stride[0];
  p.v_head = v.stride[1];
  p.o_row = out.stride[0];
  p.o_head = out.stride[1];
  p.cu_q = cu_seqlens_q.ptr<int>();
  p.cu_k = cu_seqlens_k.ptr<int>();
  p.heads = static_cast<int>(heads);
  p.kv_heads = static_cast<int>(kv_heads);
  p.total_q = static_cast<int>(total_q);
  p.scale = scale;
  p.causal = causal;
  const int batch = static_cast<int>(cu_seqlens_q.dim(0) - 1);

  if (head_dim == 128) {
    if (max_seqlen >= 1024) launch<128, 64, 192>(p, max_seqlen, batch, stream);
    else launch<128, 32, 128>(p, max_seqlen, batch, stream);
  } else {
    launch<256, 32, 64>(p, max_seqlen, batch, stream);
  }
}

}  // namespace mi::kernels::sm70
