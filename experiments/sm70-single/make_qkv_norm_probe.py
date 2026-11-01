"""Fuse Q/K norm + RoPE after the shared BF16 epilogue, preserving the reference's 4-values/lane reduction."""
from pathlib import Path
s=Path('build-tools/gemm-coalesced.cu').read_text()
s=s.replace('template <Epilogue E, typename Tile, bool Fast = false>', 'template <Epilogue E, typename Tile, bool Fast = false, bool Qk = false>')
s=s.replace('Kernel<E, Tile, Fast>', 'Kernel<E, Tile, Fast, Qk>')
s=s.replace('&int8_kernel<E, Tile, Fast>', '&int8_kernel<E, Tile, Fast, Qk>')
s=s.replace('enum class Epilogue', '''struct QkParams { const __nv_bfloat16 *qn = nullptr, *kn = nullptr, *rope = nullptr; int heads = 0; };

enum class Epilogue''',1)
s=s.replace('int log_tile, const unsigned* bounds)', 'int log_tile, const unsigned* bounds, QkParams qk)')
s=s.replace('const unsigned* bounds = nullptr) {','const unsigned* bounds = nullptr, QkParams qk = {}) {')
s=s.replace('tables, log_tile, bounds);','tables, log_tile, bounds, qk);')
a=s.index('  const int columns = E == Epilogue::kSitu')
s=s[:a]+'''  if constexpr (Qk) {
    static_assert(E == Epilogue::kPlain && BN == 128);
    if (n0 < qk.heads * 256) {
      const auto* weights = n0 < qk.heads * 128 ? qk.qn : qk.kn;
      for (int r = warp; r < BM && m0 + r < p.m; r += blockDim.x / 32) {
        float x[4], y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) x[j] = __bfloat162float(staged[r * 128 + 4 * lane + j]);
        float sum = add(add(add(mul(x[0], x[0]), mul(x[1], x[1])), mul(x[2], x[2])), mul(x[3], x[3]));
#pragma unroll
        for (int d = 16; d > 0; d /= 2) sum = add(sum, __shfl_down_sync(0xffffffffu, sum, d));
        const float total = __shfl_sync(0xffffffffu, sum, 0);
        const float inv = rsqrtf(add(mul(total, 1.f / 128.f), 1e-6f));
#pragma unroll
        for (int j = 0; j < 4; ++j) y[j] = rbf(mul(rbf(mul(x[j], inv)), __bfloat162float(weights[4 * lane + j])));
        const auto* tab = qk.rope + std::int64_t(m0 + r) * 128;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float other = __shfl_xor_sync(0xffffffffu, y[j], 16);
          const float lo = lane < 16 ? y[j] : other, hi = lane < 16 ? other : y[j];
          const int t = 4 * (lane & 15) + j;
          const float si = __bfloat162float(tab[t]), co = __bfloat162float(tab[64 + t]);
          const float value = lane < 16 ? add(rbf(mul(lo, co)), rbf(mul(-hi, si)))
                                       : add(rbf(mul(hi, co)), rbf(mul(lo, si)));
          staged[r * 128 + 4 * lane + j] = __float2bfloat16_rn(value);
        }
      }
    }
    __syncthreads();
  }
'''+s[a:]
a=s.index('void int8_gemm(const')
s=s[:a]+'''void probe_qkv(const quant::Int8Operands& p, const __nv_bfloat16* qn, const __nv_bfloat16* kn,
                const __nv_bfloat16* rope, __nv_bfloat16* out, cudaStream_t stream) {
  const QkParams args{qn, kn, rope, int(p.n / 384)};
  using Tile = qd::TileCfg<128, 128, 32, 64, 64, 2>;
  launch_tile<Epilogue::kPlain, Tile, true, true>(p, nullptr, nullptr, out, p.n, stream, nullptr, args);
  launch_tile<Epilogue::kPlain, Sm70LargeTile, false, true>(p, nullptr, nullptr, out, p.n, stream, nullptr, args);
}

'''+s[a:]
s+='''
#include "kernels/fused/qk_norm_rope.h"
#include <cstdio>
#include <vector>
#include <cmath>
struct B {
  void* p{};
  explicit B(std::size_t n) { MI_CUDA_CHECK(cudaMalloc(&p,n)); MI_CUDA_CHECK(cudaMemset(p,0,n)); }
  ~B() { cudaFree(p); }
  template<class T> T* as() { return static_cast<T*>(p); }
};
template<class F> float timing(F run) {
  for(int i=0;i<3;++i)run();
  cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
  for(int i=0;i<20;++i)run();
  cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);cudaEventDestroy(a);cudaEventDestroy(b);return ms/20;
}
int main() {
  using namespace mi;
  constexpr int m=8222,n=9216,k=3072,h=24;
  B x(std::size_t(m)*k), w(std::size_t(n)*k), xs(m*4), ws(n*4), xn(m*4), wn(n*4), qn(256), kn(256), rope(m*256);
  B baseline(std::size_t(m)*n*2), result(std::size_t(m)*n*2);
  std::vector<std::int8_t> bytes(std::size_t(n)*k);
  for(std::size_t i=0;i<bytes.size();++i)bytes[i]=int((i*17+i/31)%15)-7;
  cudaMemcpy(w.p,bytes.data(),bytes.size(),cudaMemcpyHostToDevice);cudaMemcpy(x.p,bytes.data(),std::size_t(m)*k,cudaMemcpyHostToDevice);
  std::vector<float> scales(n,0.03125f);cudaMemcpy(xs.p,scales.data(),m*4,cudaMemcpyHostToDevice);cudaMemcpy(ws.p,scales.data(),n*4,cudaMemcpyHostToDevice);
  std::vector<__nv_bfloat16> values(m*128);
  for(int i=0;i<128;++i)values[i]=__float2bfloat16(0.75f+float(i%7)/16);
  cudaMemcpy(qn.p,values.data(),256,cudaMemcpyHostToDevice);cudaMemcpy(kn.p,values.data(),256,cudaMemcpyHostToDevice);
  for(int i=0;i<m*128;++i)values[i]=__float2bfloat16(std::sin(float(i%61)*0.17f));
  cudaMemcpy(rope.p,values.data(),values.size()*2,cudaMemcpyHostToDevice);
  kernels::quant::int8_row_norms(x.as<std::int8_t>(),m,k,k,xn.as<unsigned>(),nullptr);
  kernels::quant::int8_row_norms(w.as<std::int8_t>(),n,k,k,wn.as<unsigned>(),nullptr);
  kernels::quant::Int8Operands p{x.as<std::int8_t>(),k,xs.as<float>(),w.as<std::int8_t>(),ws.as<float>(),m,n,k};
  p.x_norm=xn.as<unsigned>();p.w_norm=wn.as<unsigned>();
  auto view=[](B& b,int rows,int cols){return Tensor::view(b.p,DType::kBF16,Device::kCUDA,{rows,cols});};
  auto normq=Tensor::view(qn.p,DType::kBF16,Device::kCUDA,{128}), normk=Tensor::view(kn.p,DType::kBF16,Device::kCUDA,{128});
  auto reference=[&]{kernels::sm70::int8_gemm(p,nullptr,baseline.as<__nv_bfloat16>(),n,nullptr);kernels::fused::qk_norm_rope_moss(view(baseline,m,n),h,normq,normk,view(rope,m,128),nullptr);};
  auto fused=[&]{kernels::sm70::probe_qkv(p,qn.as<__nv_bfloat16>(),kn.as<__nv_bfloat16>(),rope.as<__nv_bfloat16>(),result.as<__nv_bfloat16>(),nullptr);};
  reference();fused();MI_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<unsigned short> a(std::size_t(m)*n),b(a.size());cudaMemcpy(a.data(),baseline.p,a.size()*2,cudaMemcpyDeviceToHost);cudaMemcpy(b.data(),result.p,b.size()*2,cudaMemcpyDeviceToHost);
  std::size_t differences=0;for(std::size_t i=0;i<a.size();++i)differences+=a[i]!=b[i];
  std::printf("qkv + norm/rope: unequal BF16 elements %zu\\n",differences);
  if(differences)return 1;
  std::printf("separate %.3f ms, fused %.3f ms\\n",timing(reference),timing(fused));
}
'''
Path('build-tools/gemm-qkv-norm.cu').write_text(s)
