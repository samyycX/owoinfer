"""Generate deliberately non-exact GEMM ceiling probes; never install these as inference kernels.

Run from the repository root. Outputs go to ignored build-tools/; use build_gemm_probe.py in this directory for local timing. Keeping the normal integer-fold kernel is required for arbitrary INT8 data.
"""
from pathlib import Path
import subprocess

source = subprocess.check_output(['git', 'show', 'a8cc10c:src/kernels/sm70/int8_gemm_sm70.cu'], text=True)
start = source.index('template <class Base>')
end = source.index('\nenum class Epilogue', start)
source = source[:start] + 'template <class Base>\nusing ExactMma = Base;\n' + source[end:]
source = source.replace('  static_assert(Tile::Warp::kM == 32 && Tile::Warp::kN == 64 && K::Mma::FragmentC::kElements == 64);',
                        '  constexpr int WM = Tile::Warp::kM, WN = Tile::Warp::kN;')
source = source.replace('(warp % (BM / 32)) * 32', '(warp % (BM / WM)) * WM')
source = source.replace('(warp / (BM / 32)) * 64', '(warp / (BM / WM)) * WN')
source = source.replace('for (int i = 0; i < 64; ++i)', 'for (int i = 0; i < K::Mma::FragmentC::kElements; ++i)')
source = source.replace('const int r = r0 + ((i >> 3)', 'const int r = r0 + ((i / 32) % (WM / 32)) * 32 + ((i >> 3)')
source = source.replace('const int c = c0 + (i >> 5) * 32', 'const int c = c0 + (i / 32 / (WM / 32)) * 32')
for name, tile in [
    ('nofold-w32', '128, 128, 32, 32, 64, 2'),
    ('nofold-w64', '128, 128, 32, 64, 64, 2'),
    ('nofold-n256', '128, 256, 32, 64, 64, 2'),
    ('nofold-m256', '256, 128, 32, 64, 64, 2'),
    ('nofold-w64-k64', '128, 128, 64, 64, 64, 2'),
]:
    Path(f'build-tools/gemm-{name}.cu').write_text(source.replace('128, 128, 32, 32, 64, 2', tile))
