"""Generate a resident-weight, single-device activation-norm sampler based on the a8cc10c kernel."""
from pathlib import Path
import subprocess

s = subprocess.check_output(['git', 'show', 'a8cc10c:src/kernels/sm70/int8_gemm_sm70.cu'], text=True)
s = s.replace('#include "kernels/quant/int8_gemm_impl.cuh"',
              '#include "kernels/quant/int8_gemm_impl.cuh"\n#include "../experiments/sm70-single/norm_probe.cuh"')
s = s.replace('  // More rows reuse each weight tile',
              '  probe::sample(p, static_cast<int>(E), stream);\n  // More rows reuse each weight tile')
Path('build-tools/gemm-norm-probe.cu').write_text(s)
