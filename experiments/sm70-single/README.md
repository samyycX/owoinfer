# Single-V100 denoise follow-up

Baseline: `a1126dc`. GPU 0 only, Tesla V100-SXM2-32GB, CUDA 12.9.86, driver 580.196.07,
GCC 14.2, Release, default clocks. Measurements: 2026-11-01/02. No approximate step cache, CFG truncation,
lower-precision accumulation, or multi-GPU execution was enabled.

## First-stage implementation (`a8cc10c`)

- Keep the INT8 GEMM load pipeline alive across exact 1024-product FP32-to-INT32 folds. A nested chunk loop
  avoids testing the fold boundary on every K tile.
- Decode the Volta accumulator layout in the fused epilogue and write BF16 results directly. This removes the
  full FP32 shared output tile and two CTA barriers. The 128x128 GEMM tile uses 32 KiB instead of 64 KiB of shared
  memory; the register limit still prevents assuming that occupancy doubles.
- Use the quantized layer's existing eight-column CTA swizzle for activation reuse, including its narrow-grid
  fallback and padded-grid guard.
- Transfer attention probabilities from FP32 accumulators to FP16 MMA inputs with packed warp shuffles.
  Stage 64 keys for long D=128 attention, but retain consecutive 32-key softmax updates. D=256 keeps its previous
  shared-memory probability path.

All changes stay inside the SM70 kernel family. The current compatibility-layer interfaces, weight formats,
allocator behavior, and exact-integer/epilogue rounding contracts are preserved.

## Kernel timings

`sm70_bench 8222 4111`: CUDA-event means of 20 launches after five warm-ups, zero-filled operands. Each candidate
was first checked against the host INT64 GEMM references, including all three epilogues and cancellation at
K=131056. Timings are milliseconds; different experimental timings below are individual runs, not confidence
intervals.

| Implementation | QKV | O/residual | Up/gate | Down/residual |
|---|---:|---:|---:|---:|
| Baseline, repeated with final build | 7.457 | 3.050 | 12.002 | 5.788 |
| Continuous pipeline, branch every K tile | 10.620 | 3.846 | 16.764 | 8.573 |
| Continuous pipeline, nested fold loop | 7.462 | 3.055 | 11.602 | 5.770 |
| Direct scalar epilogue | 7.689 | 2.745 | 11.857 | 5.462 |
| Direct paired epilogue | 7.461 | 2.927 | 11.967 | 5.619 |
| Direct epilogue, 32x32 warp / 128x64 CTA | 11.579 | 3.993 | 19.239 | 8.451 |
| Same smaller warp, 128-register limit | 8.681 | 3.028 | 13.703 | 6.610 |
| Baseline plus eight-column CTA swizzle | 7.180 | 2.778 | 11.549 | 5.638 |
| Nested pipeline plus CTA swizzle | 7.152 | 2.773 | 11.001 | 5.600 |
| Direct epilogue plus CTA swizzle | 7.260 | 2.602 | 11.230 | 5.503 |
| Final combined implementation | **7.148** | **2.433** | **10.789** | **5.253** |

The four shapes are respectively `(M,N,K)` = `(8222,9216,3072)`, `(8222,3072,3072)`,
`(8222,16384,3072)`, `(8222,3072,8192)`. Larger K tiles and combinations with paired output stores also failed to
beat the final configuration. Forced register caps introduced local-memory spills. A resource-count reduction
alone was not a reliable predictor of speed.

DiT attention `(B=2,L=4111,H=24,D=128)` changes from **14.682 to 12.127 ms**. Causal text attention
`(B=2,L=282,H=16,KV=4,D=256)` stays at **0.259–0.260 ms**. A previous candidate using a single 64-key softmax
update changed the final image and was excluded; the retained implementation preserves 32-key updates.

## End-to-end comparisons

MOSS-Image-SFT with Comfy-Org INT8 ConvRot DiT, Qwen3.5 9B W4A8 text encoder, `high`, exact attention,
Euler, CFG 3, all requested steps executed. Final `.npy` FP32 latents compare byte-for-byte equal in every paired
run below; decoded RGB pixels also match exactly.

| Prompt / seed / mode | Size / steps | Baseline denoise | Optimized denoise | Optimized mean step |
|---|---|---:|---:|---:|
| Apple / 42 / `max_speed` | 1024² / 20 | 37.7679 s | 33.6368 s | 1.6818 s |
| Coastal city / 7 / `max_speed` | 1024² / 20 | 37.9839 s | 33.7060 s | 1.6853 s |
| Green cup / 123 / `min_memory` | 256² / 3 | 1.12913 s | 1.13228 s | 0.3774 s |

Prompts:

1. `a red apple on a wooden table, natural light, detailed photograph`
2. `a futuristic coastal city at sunset, intricate architecture, cinematic lighting`
3. `a small green ceramic cup on a stone table, soft window light`

The two 1024² pairs reduce denoise time by **10.9% and 11.3%**. The tiny streamed case has no measured gain.
An earlier combined apple run recorded 33.4613 s denoise (1.6731 s/step); its PNG also exactly matched the saved
baseline. These measurements do not establish a hardware limit or guarantee equality for every possible input.
Model-load/text-encoding cache states and intermittent telemetry overhead make total process times less useful
than denoise for comparing these kernels. Raw reports include those stages separately.

## Reproduction and local artifacts

Build and run the commands in [the SM70 guide](../../docs/sm70.md). Run `sm70_check all`, `sm70_check big`, and the
`Sm70Build.*:GpuSm70.*:GpuSm75Convert.*` tests before timing. New coverage includes large-tile fold tails,
row/column tails across a swizzle group, and strided varlen/GQA attention with Graph replay and empty keys.

The development workspace retains the ignored `build-tools/single-*.log`, `single-*.csv`,
`build-sm70/validation/single-*.json`, `.npy`, and `.png` artifacts. `build-tools/validate-single.py` reproduces
the paired generations using the local model cache and the separately saved baseline executable. No models or
executables are committed. The paused attention and multi-GPU prototypes remain on their separate branches.

## Second stage: exact norm-guarded GEMMs (`8c21627`)

The INT8-to-FP16 converter already uses CUTLASS's packed PRMT conversion. The useful follow-up is to avoid
integer folds only when an exact GPU-computed bound proves that FP32 accumulation cannot lose integer units.
For `X = max_row sum(x*x)` and `W = max_row sum(w*w)`, `X * W <= 2^48` bounds every partial dot product by
`2^24`, by Cauchy-Schwarz. Unsafe inputs keep the integer-fold kernel. No host decision or synchronization is
needed: both kernels are launched in order and exactly one writes the result.

The fast QKV tile is 128x128x32 with a 64x64 warp; up/gate uses 128x256x32 with the same warp. O/down retain
the existing folded implementation. DiT supplies eight bytes of its inactive attention workspace through the
optional `Int8Operands::norm_bounds`; callers without scratch retain the existing path. No persistent allocation
or weight-pointer cache is introduced. Boundary tests cover equality at `2^48`, overflow of that bound, odd-product
cancellation, strided tails, scratch canaries, SiTU pairing, and CUDA Graph replay.

The instrumented apple/seed-42 1024²/20-step run sampled 3360 DiT GEMMs; all passed even the whole-matrix bound.
The largest square-root bounds were 5,956,317 (QKV), 5,404,664 (O), 5,902,101 (up/gate), and 13,795,929 (down),
below 16,777,216. This is an observed workload result, not an assumption made by the production kernel.

| Probe, ms | QKV | O | Up/gate | Down |
|---|---:|---:|---:|---:|
| Unguarded 64x64 warp, 128x128 CTA | 6.294 | 2.608 | 9.685 | 4.906 |
| Unguarded 64x64 warp, 128x256 CTA | 6.355 | 2.670 | 8.951 | 5.033 |
| Guarded selected tiles, including norm pass | 6.419 | 2.439 | 9.218 | 5.248 |
| cuBLASLt FP16 input / FP32 output | 4.727 | 1.579 | 8.201 | 3.895 |

The cuBLAS figures exclude conversion, dequantization and fused epilogues and therefore are only a ceiling probe,
not a replacement with measured end-to-end benefit. Unguarded probes do not satisfy the generic numerical contract.
The guarded prototype completed 20 apple steps in **31.99 s (~1.60 s/step)**; its final FP32 latent and decoded PNG
matched the baseline exactly. The integrated caller-workspace version reproduced that result:

| Prompt / seed / mode | Size / steps | Baseline denoise | Guarded denoise | Guarded mean step |
|---|---|---:|---:|---:|
| Apple / 42 / `max_speed` | 1024² / 20 | 37.7679 s | 32.0059 s | 1.6003 s |
| Coastal city / 7 / `max_speed` | 1024² / 20 | 37.9839 s | 32.1140 s | 1.6057 s |
| Green cup / 123 / `min_memory` | 256² / 3 | 1.12913 s | 1.13068 s | 0.3769 s |

All three final FP32 latents are byte-identical to the saved baseline, and all decoded RGB pixels are identical.
The two large cases reduce denoise time by **15.3% and 15.5%** relative to `a1126dc` (4.8% and 4.7% relative to
`a8cc10c`). Their total generation times are 33.3361 s and 33.3690 s; peak device memory remains 6.75 GiB.
GPU 0 peaked at 51 C / 54 C without reported thermal throttling. The small case still has no measured speedup.
The final integrated kernel timings are 6.403 / 2.439 / 9.235 / 5.259 ms, with DiT attention 12.129 ms.
Logs and telemetry are `build-tools/guard-*`; generated reports, latents and images are
`build-sm70/validation/guard-{apple,city,cup}-opt*`.

Validation passes all 10 SM70/build/conversion tests, both host-reference checks (`all`/`big`), memcheck on
INT8/attention (zero errors), racecheck on the norm guard and fast/fallback tiles (zero errors/warnings), 517 CPU
cases with normal fixture skips, the CMP0007 wrapper, and nine default-off backend-probe tests. The complete
SM70 build also links the shared library and server.

Probe sources and generation/build helpers are retained here. With the configured `build-sm70` directory:

```sh
python3 experiments/sm70-single/make_nofold.py
python3 experiments/sm70-single/make_guarded.py
python3 experiments/sm70-single/make_norm_probe.py
python3 experiments/sm70-single/build_gemm_probe.py gemm-guarded
python3 experiments/sm70-single/build_probe.py experiments/sm70-single/cublas_ceiling.cu
```

Generators intentionally read the first-stage source from `a8cc10c`. Generated sources and executables live in
ignored `build-tools/`. `norm_probe.cuh` synchronizes and logs for measurement; `norm_guard_probe.cuh` has a
single-device scratch allocation for prototyping. Neither is included by the production implementation.

## Third stage: cached row norms and prepared K/V

Resident INT8 weights now own exact uint32 squared row norms in the model arena. Loading and both LoRA merge
paths refresh that metadata after weight changes. Streamed/recycled buffers retain per-call scanning. The
quantizers optionally emit activation norms from their packed INT8 registers, without rereading the input.
Volta GEMMs bound each 128x128 tile independently; SiTU uses a canonical 128x256 decision region shared by its
fast CTA and its two folded fallback CTAs. Exactly one path writes every result, including gated residuals.
O uses the existing 32x64 warp without folds when safe; down uses a 64x64 warp. Short shapes retain integer folds.
Other architectures ignore the optional metadata, and quantizers without a norm output retain their old variant.

Long D=128 attention optionally converts K/V once into head-major FP16 buffers. The caller reserves this space
in its activation plan; the compatibility kernel never allocates or synchronizes. Minimum-workspace callers,
unaligned layouts, short sequences and D=256 retain tile-local conversion. The conversion, FP32 accumulation,
32-key softmax updates and output rounding are unchanged. On the 8222-token DiT shape the two buffers use about
96.35 MiB, in addition to LSE; resident weight norms use about 5.1 MiB across 42 blocks.

The baseline Nsight Systems trace measured the old norm scan at **6.984 ms/step**, rather than the suggested
15–30 ms estimate. Fused norm/modulate/quantize, QK norm/RoPE, down-input quantization and O-input quantization
accounted for approximately 37.640, 25.868, 18.750 and 6.689 ms/step respectively. These are traced measurements;
use uninstrumented runs for end-to-end comparisons. Hardware counters remain unavailable under the existing
Nsight Compute permissions.

The initial integration measured 31.5596 s/20 steps with cached norms and residual fast paths (1.5780 s/step),
then **30.9699 s/20 steps with prepared K/V (1.5485 s/step)**. Both final FP32 latents matched the original baseline
byte-for-byte. This is a modest measured gain over 1.60 s/step; the suggested 1.30–1.35 s/step was not achieved.

### Final selected build

| Prompt / seed / mode | Size / steps | Denoise | Mean step | Total generation |
|---|---|---:|---:|---:|
| Apple / 42 / `max_speed` | 1024² / 20 | 31.0510 s | 1.5526 s | 32.6042 s |
| Coastal city / 7 / `max_speed` | 1024² / 20 | 31.2554 s | 1.5628 s | 33.1865 s |
| Green cup / 123 / `min_memory` | 256² / 3 | 1.1369 s | 0.3790 s | 2.4649 s |

All three final FP32 latents are byte-identical to `a1126dc`; all decoded RGB pixels match exactly. Large-case
denoise reductions are 17.8% / 17.7% versus that baseline and 3.0% / 2.7% versus `efa8971`. The small streamed
case has no measured benefit. Reported peak device memory remains 6.75 GiB for the large cases: the additional
K/V storage fits below another phase's allocation peak. GPU 0 peaked at 51 C / 55 C without thermal slowdown.

Final kernel timings are 6.506 / 2.424 / 9.063 / 4.937 ms (QKV / O / up / down), DiT attention 11.508 ms and text
attention 0.267 ms. GEMM timing includes the tile guard and inactive fallback launch, but not the preceding
quantizer's fused norm calculation or the one-time weight norm scan. Attention timing includes K/V preconversion.

Raw reports, FP32 latents and PNGs are `build-sm70/validation/phase3-selected-{apple,city,cup}-opt*`; telemetry
and validation logs are `build-tools/phase3-selected-*`. `build-tools/validate-phase3-selected.py` reproduces
these three cases against the locally saved baseline artifacts. Earlier probe artifacts use the `phase3-` prefix.

### Other candidates measured and excluded

| Probe | Result / decision |
|---|---|
| Same CUTLASS loop, FP16 inputs | 64x64 warp: 6.164 / 2.503 / 9.511 / 4.634 ms for QKV / O / up / down. Removing INT8 conversion does not close the cuBLAS gap. |
| Complete cuBLAS pipeline, vectorized conversion and epilogue | QKV/O microbench: 5.937 / 2.039 ms, including conversion and FP32 intermediate. A caller-workspace integration preserved latent/PNG bits, but did not give a stable whole-image gain. |
| Paired native / cuBLAS / cuBLAS / native | 31.3109 / 31.4121 / 31.5073 / 31.5699 s for 20 apple steps; means 1.5720 vs 1.5730 s/step. Both used the same extracted tile-guard helper for this comparison. Removed the cuBLAS production path and its extra scratch/API. |
| Shared BF16 staging plus coalesced stores | 6.207 / 2.614 / 9.132 / 5.033 ms; mixed results, no general improvement. |
| QKV + Q/K norm/RoPE fusion | Zero differing BF16 elements, but 6.512 ms versus 6.210 ms separately. |
| Prepared K/V, transposed vs row-major shared V | 11.507 vs 12.503 ms. Retain the existing transposed V layout. |
| Register prefetch, 128/192 queries and 32/64 keys | 15.466–33.625 ms versus about 12.13 ms without it; high register pressure/spills. |

The cuBLAS A/B telemetry peaked at 53–57 C and 1470–1515 MHz in the native runs, with no reported thermal
slowdown. Software power-cap events occurred in both variants. No clocks, power limits or driver permissions
were changed. Microbenchmark improvements alone were insufficient to justify the more complex dispatch.

### Reproducing the probes

Run from the repository root, with the SM70 build configured. These are separate experimental executables;
none of their headers are included by production code. FP16 diagnostics use zero-filled integer-valued inputs;
prototype cuBLAS vector stores target the aligned DiT benchmark shapes, not the generic kernel contract.
Some prototypes intentionally own single-device scratch to isolate an idea and are unsuitable for integration.

```sh
python3 experiments/sm70-single/make_fp16_diagnostic.py
python3 experiments/sm70-single/build_probe.py --cutlass build-tools/gemm-fp16-w64.cu
python3 experiments/sm70-single/make_cublas_pipeline.py
python3 experiments/sm70-single/make_cublas_vector_probe.py
python3 experiments/sm70-single/build_gemm_probe.py gemm-cublas-vector
python3 experiments/sm70-single/make_preconvert_attention.py
python3 experiments/sm70-single/make_prefetch_attention.py
python3 experiments/sm70-single/make_coalesced_gemm.py
python3 experiments/sm70-single/make_qkv_norm_probe.py
python3 experiments/sm70-single/build_probe.py --cutlass build-tools/gemm-qkv-norm.cu
```

Use `build_attention_probe.py NAME` for generated attention kernels and `build_gemm_probe.py NAME` for GEMM
replacement kernels; the QKV-fusion source has its own main and uses `build_probe.py --cutlass`. The FP16/cuBLAS probes read `8c21627`, and attention probes read `efa8971`, keeping their starting point
fixed. Coalesced-store and QKV-fusion probes read the current selected native GEMM. Generated CUDA sources,
executables, resource reports and raw timings stay in ignored `build-tools/`.

Final validation: all 12 SM70/build/conversion tests, `sm70_check all` and `big`, memcheck on all INT8 tests,
fused quantized norms and attention (zero errors), and racecheck on mixed safe/fallback tiles, fused norms and
prepared attention (zero errors/warnings). The complete SM70 build links the shared library, server and CLI;
517 selected CPU cases pass with normal missing-fixture skips, as does the CMP0007 verdict wrapper. The
SM70-disabled build passes all nine backend-probe tests. Hardware validation is limited to V100.
