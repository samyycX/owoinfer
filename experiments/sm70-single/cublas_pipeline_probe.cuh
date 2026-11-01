// Single-device, single-thread prototype only: owns scratch/plans to measure the complete staging + GEMM + epilogue.
// Production integration must use caller-owned scratch and the existing handle/plan lifetime.
#include <cublasLt.h>
#include <map>
#include <memory>
#include <tuple>

namespace pipeline_probe {
inline void check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(cublasLtGetStatusString(s));
}
struct Plan {
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t a{}, b{}, c{};
  cublasLtMatmulAlgo_t algo{};
  __half *x{}, *w{};
  float* acc{};
  void* scratch{};
  static constexpr std::size_t bytes = 32u << 20;
  Plan(int m, int n, int k) {
    check(cublasLtCreate(&lt));
    check(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const auto tr = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &tr, sizeof(tr)));
    check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16F, k, n, k));
    check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16F, k, m, k));
    check(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, n, m, n));
    cublasLtMatmulPreference_t pref{};
    check(cublasLtMatmulPreferenceCreate(&pref));
    check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
    cublasLtMatmulHeuristicResult_t result{};
    int count = 0;
    check(cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, &result, &count));
    if (!count) throw std::runtime_error("no fp16/fp32 algorithm");
    algo = result.algo;
    cublasLtMatmulPreferenceDestroy(pref);
    MI_CUDA_CHECK(cudaMalloc(&x, std::size_t(m) * k * 2));
    MI_CUDA_CHECK(cudaMalloc(&w, std::size_t(n) * k * 2));
    MI_CUDA_CHECK(cudaMalloc(&acc, std::size_t(m) * n * 4));
    MI_CUDA_CHECK(cudaMalloc(&scratch, bytes));
  }
  ~Plan() {
    cudaFree(x); cudaFree(w); cudaFree(acc); cudaFree(scratch);
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulDescDestroy(desc); cublasLtDestroy(lt);
  }
};
__global__ void convert(quant::Int8Operands p, __half* x, __half* w) {
  const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < p.m * p.k) x[i] = __int2half_rn(p.x_q[(i / p.k) * p.ldx + i % p.k]);
  if (i < p.n * p.k) w[i] = __int2half_rn(p.w[i]);
}
template <Epilogue E>
__global__ void epilogue(quant::Int8Operands p, const float* acc, const __nv_bfloat16* bias,
                         const __nv_bfloat16* gate, __nv_bfloat16* out, std::int64_t ldo) {
  if (std::uint64_t(p.norm_bounds[0]) * p.norm_bounds[1] > (std::uint64_t{1} << 48)) return;
  const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p.m * p.n) return;
  const std::int64_t r = i / p.n, c = i % p.n;
  float value = dequant(acc[i], p.x_scale[r], p.w_scale[c], bias ? __bfloat162float(bias[c]) : 0.f);
  if constexpr (E == Epilogue::kResidual) {
    value = add(__bfloat162float(out[r * ldo + c]), rbf(mul(rbf(value), __bfloat162float(gate[c]))));
  }
  out[r * ldo + c] = __float2bfloat16_rn(value);
}
template <Epilogue E>
void run(const quant::Int8Operands& p, const __nv_bfloat16* bias, const __nv_bfloat16* gate,
         __nv_bfloat16* out, std::int64_t ldo, cudaStream_t stream) {
  static std::map<std::tuple<int,int,int>, std::unique_ptr<Plan>> plans;
  auto& ptr = plans[{int(p.m), int(p.n), int(p.k)}];
  if (!ptr) ptr = std::make_unique<Plan>(p.m, p.n, p.k);
  auto& plan = *ptr;
  convert<<<(std::max(p.m, p.n) * p.k + 255) / 256, 256, 0, stream>>>(p, plan.x, plan.w);
  MI_CUDA_CHECK_LAUNCH();
  const float alpha = 1.f, beta = 0.f;
  check(cublasLtMatmul(plan.lt, plan.desc, &alpha, plan.w, plan.a, plan.x, plan.b, &beta,
                      plan.acc, plan.c, plan.acc, plan.c, &plan.algo, plan.scratch, Plan::bytes, stream));
  epilogue<E><<<(p.m * p.n + 255) / 256, 256, 0, stream>>>(p, plan.acc, bias, gate, out, ldo);
  MI_CUDA_CHECK_LAUNCH();
}
}  // namespace pipeline_probe
