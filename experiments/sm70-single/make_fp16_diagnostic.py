"""FP16-input diagnostic of the same Volta main loop and fused epilogues; zero inputs only, not inference."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
source = subprocess.check_output(['git', 'show', '8c21627:src/kernels/sm70/int8_gemm_sm70.cu'], cwd=root, text=True)
source = source.replace('std::int8_t, qd::RowMajor, 1,', 'cutlass::half_t, qd::RowMajor, 1,')
source = source.replace('std::int8_t, qd::ColumnMajor, 0,', 'cutlass::half_t, qd::ColumnMajor, 0,')
source = source.replace('const_cast<std::int8_t*>(p.x_q)', 'reinterpret_cast<cutlass::half_t*>(const_cast<std::int8_t*>(p.x_q))')
source = source.replace('const_cast<std::int8_t*>(p.w)', 'reinterpret_cast<cutlass::half_t*>(const_cast<std::int8_t*>(p.w))')
a = source.index('  // More rows reuse each weight tile')
b = source.index('\n}\n', a)
bench = (root/'tools/sm70_bench.cu').read_text()
bench = bench[:bench.index('void attention(')] + '''}  // namespace
int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  gemm(8222, 9216, 3072, 0);
  gemm(8222, 3072, 3072, 1);
  gemm(8222, 16384, 3072, 2);
  gemm(8222, 3072, 8192, 1);
}
'''
bench = bench.replace('Buffer x(static_cast<std::size_t>(m) * k), w(static_cast<std::size_t>(n) * k);',
                      'Buffer x(static_cast<std::size_t>(m) * k * 2), w(static_cast<std::size_t>(n) * k * 2);')
for name, tile in [('w32','128,128,32,32,64,2'), ('w64','128,128,32,64,64,2'), ('n256','128,256,32,64,64,2')]:
    body = source[:a] + f'  launch_tile<E, qd::TileCfg<{tile}>, true>(p, bias, gate, out, ldo, stream);' + source[b:]
    (root/f'build-tools/gemm-fp16-{name}.cu').write_text(body + '\n' + bench)
print('FP16-input diagnostic sources generated')
