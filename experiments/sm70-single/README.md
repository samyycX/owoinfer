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

## Second stage: exact norm-guarded GEMMs

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
matched the baseline exactly. The integrated caller-workspace version builds and passes all 10 SM70/build/conversion
tests; full integrated-generation and sanitizer validation are still pending at this checkpoint.

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
