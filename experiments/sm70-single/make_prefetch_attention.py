"""Generate register-prefetch attention probes without changing the 32-key softmax grouping."""
from pathlib import Path
import subprocess
s = subprocess.check_output(['git','show','efa8971:src/kernels/sm70/attention_sm70.cu'],text=True)
a = s.index('    for (int i = tid; i < BN * D; i += kThreads) {')
b = s.index('    __syncthreads();',a)
s=s[:a]+'''    if constexpr (D == 128) {
#pragma unroll
      for (int t = 0; t < kPrefetch; ++t) {
        const int i = tid + t * kThreads;
        if (i < BN * D) {
          ks[(i / D) * kQStride + i % D] = rk[t];
          vt[(i % D) * kVStride + i / D] = rv[t];
        }
      }
      prefetch(j0 + BN);
    } else {
'''+s[a:b]+'''    }
'''+s[b:]
a=s.index('  for (int j0 = 0; j0 < end; j0 += BN) {')
s=s[:a]+'''  constexpr int kPrefetch = D == 128 ? (BN * D + kThreads - 1) / kThreads : 1;
  __half rk[kPrefetch], rv[kPrefetch];
  const auto prefetch = [&](int j0) {
    if constexpr (D == 128) {
#pragma unroll
      for (int t = 0; t < kPrefetch; ++t) {
        const int i = tid + t * kThreads, j = i / D, d = i % D;
        const int token = k_start + j0 + j;
        const bool valid = i < BN * D && j0 + j < end;
        rk[t] = __float2half_rn(valid ? __bfloat162float(p.k[token * p.k_row + kv_head * p.k_head + d]) : 0.f);
        rv[t] = __float2half_rn(valid ? __bfloat162float(p.v[token * p.v_row + kv_head * p.v_head + d]) : 0.f);
      }
    }
  };
  prefetch(0);
'''+s[a:]
for bm,bn in [(128,32),(128,64),(192,32),(192,64)]:
    p=s.replace('launch<128, 64, 192>',f'launch<128, {bn}, {bm}>')
    Path(f'build-tools/attention-prefetch{bm}n{bn}.cu').write_text(p)
