"""Generate the separately linked cuBLAS pipeline prototype from the norm-guard checkpoint."""
from pathlib import Path
import subprocess
s = subprocess.check_output(['git','show','8c21627:src/kernels/sm70/int8_gemm_sm70.cu'], text=True)
s=s.replace('#include <string>', '#include <string>\n#include <map>\n#include <memory>\n#include <tuple>\n#include <cublasLt.h>')
a=s.index('template <Epilogue E>\nvoid launch(')
s=s[:a]+'#include "../experiments/sm70-single/cublas_pipeline_probe.cuh"\n\n'+s[a:]
s=s.replace('if constexpr (E == Epilogue::kPlain || E == Epilogue::kSitu)', 'if constexpr (true)')
s=s.replace('        launch_tile<E, FastTile, true>(p, bias, gate, out, ldo, stream, p.norm_bounds);', '''        if constexpr (E != Epilogue::kSitu) pipeline_probe::run<E>(p, bias, gate, out, ldo, stream);
        else launch_tile<E, FastTile, true>(p, bias, gate, out, ldo, stream, p.norm_bounds);''')
Path('build-tools/gemm-cublas-pipeline.cu').write_text(s)
