"""One-time K/V conversion diagnostic. Single-device prototype scratch; no production inclusion."""
from pathlib import Path
import subprocess
s=subprocess.check_output(['git','show','efa8971:src/kernels/sm70/attention_sm70.cu'],text=True)
s=s.replace('  float* lse = nullptr;', '  float* lse = nullptr;\n  const __half* kh = nullptr;\n  const __half* vh = nullptr;\n  int total_k = 0;')
a=s.index('// SparkAttention')
s=s[:a]+'''__global__ void preconvert(Params p, __half* kh, __half* vh) {
  const std::int64_t i = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const int d = i % 128, row = i / 128 / p.kv_heads, head = i / 128 % p.kv_heads;
  if (row >= p.total_k) return;
  const uint4 kr = *reinterpret_cast<const uint4*>(p.k + row * p.k_row + head * p.k_head + d);
  const uint4 vr = *reinterpret_cast<const uint4*>(p.v + row * p.v_row + head * p.v_head + d);
  uint4 ko, vo;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    reinterpret_cast<__half*>(&ko)[e] = __float2half_rn(__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&kr)[e]));
    reinterpret_cast<__half*>(&vo)[e] = __float2half_rn(__bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&vr)[e]));
  }
  *reinterpret_cast<uint4*>(kh + (std::int64_t(head) * p.total_k + row) * 128 + d) = ko;
  *reinterpret_cast<uint4*>(vh + (std::int64_t(head) * p.total_k + row) * 128 + d) = vo;
}

'''+s[a:]
s=s.replace('ks[j * kQStride + d] = __float2half_rn(valid ? __bfloat162float(p.k[token * p.k_row + kv_head * p.k_head + d]) : 0.f);', '''if constexpr (D == 128) ks[j * kQStride + d] = valid ? p.kh[(std::int64_t(kv_head) * p.total_k + token) * D + d] : __float2half_rn(0.f);
      else ks[j * kQStride + d] = __float2half_rn(valid ? __bfloat162float(p.k[token * p.k_row + kv_head * p.k_head + d]) : 0.f);''')
s=s.replace('vt[d * kVStride + j] = __float2half_rn(valid ? __bfloat162float(p.v[token * p.v_row + kv_head * p.v_head + d]) : 0.f);', '''if constexpr (D == 128) vt[d * kVStride + j] = valid ? p.vh[(std::int64_t(kv_head) * p.total_k + token) * D + d] : __float2half_rn(0.f);
      else vt[d * kVStride + j] = __float2half_rn(valid ? __bfloat162float(p.v[token * p.v_row + kv_head * p.v_head + d]) : 0.f);''')
s=s.replace('  if (head_dim == 128) {\n    if (max_seqlen', '''  if (head_dim == 128) {
    // Prototype only: caller ownership is required before production integration.
    static __half* scratch = nullptr;
    static std::size_t capacity = 0;
    const std::size_t elements = std::size_t(k.dim(0)) * kv_heads * 128;
    if (elements * 4 > capacity) {
      if (scratch) MI_CUDA_CHECK(cudaFree(scratch));
      MI_CUDA_CHECK(cudaMalloc(&scratch, elements * 4));
      capacity = elements * 4;
    }
    p.total_k = k.dim(0); p.kh = scratch; p.vh = scratch + elements;
    preconvert<<<(elements / 8 + 255) / 256, 256, 0, stream>>>(p, scratch, scratch + elements);
    MI_CUDA_CHECK_LAUNCH();
    if (max_seqlen''')
Path('build-tools/attention-preconvert.cu').write_text(s)
# Row-major V keeps the same 16-wide MMA reduction order; only its shared-memory representation changes.
s=s.replace('  using B = wmma::fragment<wmma::matrix_b', '  using V = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;\n  using B = wmma::fragment<wmma::matrix_b',1)
s=s.replace('vt[d * kVStride + j] =', 'vt[j * kQStride + d] =')
s=s.replace('          B b;\n          wmma::load_matrix_sync(b, vt + d * 16 * kVStride + j, kVStride);', '          V b;\n          wmma::load_matrix_sync(b, vt + j * kQStride + d * 16, kQStride);')
# Existing V region D*(BN+8) >= BN*(D+8) for these shapes.
Path('build-tools/attention-preconvert-vrow.cu').write_text(s)
