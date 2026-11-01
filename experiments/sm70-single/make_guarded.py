"""Build a separately linked prototype of exact norm-guarded no-fold GEMMs."""
from pathlib import Path
import subprocess

s = subprocess.check_output(['git', 'show', 'a8cc10c:src/kernels/sm70/int8_gemm_sm70.cu'], text=True)
s = s.replace('#include "kernels/quant/int8_gemm_impl.cuh"', '#include "kernels/quant/int8_gemm_impl.cuh"\n#include "../experiments/sm70-single/norm_guard_probe.cuh"')
s = s.replace('template <Epilogue E, typename Tile>', 'template <Epilogue E, typename Tile, bool Fast = false>')
s = s.replace('Kernel<E, Tile>', 'Kernel<E, Tile, Fast>')
s = s.replace('using Mma = ExactMma<cutlass::gemm::threadblock::MmaPipelined<', 'using BaseMma = cutlass::gemm::threadblock::MmaPipelined<')
s = s.replace('float, qd::RowMajor, typename Core::MmaPolicy>>;', 'float, qd::RowMajor, typename Core::MmaPolicy>;\n  using Mma = std::conditional_t<Fast, BaseMma, ExactMma<BaseMma>>;')
s = s.replace('qd::SituFactorTables tables, int log_tile) {', 'qd::SituFactorTables tables, int log_tile, const unsigned* bounds) {')
s = s.replace('  if (m0 >= p.m || n0 >= p.n) return;', '''  if (m0 >= p.m || n0 >= p.n) return;
  if (bounds) {
    const bool safe = std::uint64_t(bounds[0]) * bounds[1] <= (std::uint64_t{1} << 48);
    if (safe != Fast) return;
  }''')
s = s.replace('  static_assert(Tile::Warp::kM == 32 && Tile::Warp::kN == 64 && K::Mma::FragmentC::kElements == 64);',
              '  constexpr int WM = Tile::Warp::kM, WN = Tile::Warp::kN;')
s = s.replace('(warp % (BM / 32)) * 32', '(warp % (BM / WM)) * WM')
s = s.replace('(warp / (BM / 32)) * 64', '(warp / (BM / WM)) * WN')
s = s.replace('for (int i = 0; i < 64; ++i)', 'for (int i = 0; i < K::Mma::FragmentC::kElements; ++i)')
s = s.replace('const int r = r0 + ((i >> 3)', 'const int r = r0 + ((i / 32) % (WM / 32)) * 32 + ((i >> 3)')
s = s.replace('const int c = c0 + (i >> 5) * 32', 'const int c = c0 + (i / 32 / (WM / 32)) * 32')
a = s.index('void launch_tile(')
b = s.index('\ntemplate <Epilogue E>\nvoid launch', a)
part = s[a:b]
part = part.replace('cudaStream_t stream) {', 'cudaStream_t stream, const unsigned* bounds = nullptr) {')
part = part.replace('&int8_kernel<E, Tile>', '&int8_kernel<E, Tile, Fast>')
part = part.replace('tables, log_tile);', 'tables, log_tile, bounds);')
s = s[:a] + part + s[b:]
s = s.replace('  if (p.m >= 1024) launch_tile<E, Sm70LargeTile>(p, bias, gate, out, ldo, stream);', '''  if (p.m >= 1024) {
    if constexpr (E == Epilogue::kPlain || E == Epilogue::kSitu) {
      using FastTile = std::conditional_t<E == Epilogue::kSitu, qd::TileCfg<128, 256, 32, 64, 64, 2>,
                                                                  qd::TileCfg<128, 128, 32, 64, 64, 2>>;
      const auto* bounds = probe::prepare(p, stream);
      launch_tile<E, FastTile, true>(p, bias, gate, out, ldo, stream, bounds);
      launch_tile<E, Sm70LargeTile>(p, bias, gate, out, ldo, stream, bounds);
    } else {
      launch_tile<E, Sm70LargeTile>(p, bias, gate, out, ldo, stream);
    }
  }''')
Path('build-tools/gemm-guarded.cu').write_text(s)
