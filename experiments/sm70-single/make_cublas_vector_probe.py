"""Vectorize the complete cuBLAS probe's conversion and epilogue, keeping its FP32 intermediate."""
from pathlib import Path
s=Path('experiments/sm70-single/cublas_pipeline_probe.cuh').read_text()
a=s.index('__global__ void convert');b=s.index('template <Epilogue E>\nvoid run',a)
s=s[:a]+'''__global__ void convert(quant::Int8Operands p, __half* x, __half* w) {
  const std::int64_t i = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  using Bytes = cutlass::Array<std::int8_t, 8>;
  using Halves = cutlass::Array<cutlass::half_t, 8>;
  cutlass::NumericArrayConverter<cutlass::half_t, std::int8_t, 8> convert;
  if (i < p.m * p.k) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p.x_q + (i / p.k) * p.ldx + i % p.k);
    const Halves out = convert(*reinterpret_cast<const Bytes*>(&raw));
    *reinterpret_cast<uint4*>(x + i) = *reinterpret_cast<const uint4*>(&out);
  }
  if (i < p.n * p.k) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p.w + i);
    const Halves out = convert(*reinterpret_cast<const Bytes*>(&raw));
    *reinterpret_cast<uint4*>(w + i) = *reinterpret_cast<const uint4*>(&out);
  }
}
template <Epilogue E>
__global__ void epilogue(quant::Int8Operands p, const float* acc, const __nv_bfloat16* bias,
                         const __nv_bfloat16* gate, __nv_bfloat16* out, std::int64_t ldo) {
  if (std::uint64_t(p.norm_bounds[0]) * p.norm_bounds[1] > (std::uint64_t{1} << 48)) return;
  const std::int64_t i = (std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i >= p.m * p.n) return;
  const std::int64_t r = i / p.n, c = i % p.n;
  const float xs = p.x_scale[r];
  uint4 packed;
  auto* dst = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float value = dequant(acc[i+j], xs, p.w_scale[c+j], bias ? __bfloat162float(bias[c+j]) : 0.f);
    if constexpr (E == Epilogue::kResidual) value = add(__bfloat162float(out[r * ldo + c+j]), rbf(mul(rbf(value), __bfloat162float(gate[c+j]))));
    dst[j] = __float2bfloat16_rn(value);
  }
  *reinterpret_cast<uint4*>(out + r * ldo + c) = packed;
}
'''+s[b:]
s=s.replace('(std::max(p.m, p.n) * p.k + 255) / 256', '(std::max(p.m, p.n) * p.k / 8 + 255) / 256')
s=s.replace('(p.m * p.n + 255) / 256', '(p.m * p.n / 8 + 255) / 256')
Path('build-tools/cublas_vector_probe.cuh').write_text(s)
s=Path('build-tools/gemm-cublas-pipeline.cu').read_text().replace('../experiments/sm70-single/cublas_pipeline_probe.cuh','cublas_vector_probe.cuh')
Path('build-tools/gemm-cublas-vector.cu').write_text(s)
