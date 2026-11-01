"""Build a standalone CUDA probe using the configured sm70_bench compile/link commands and cuBLASLt shim."""
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
build = root / 'build-sm70'
ninja = shutil.which('ninja') or str(root / 'build-tools/ninja')
source = Path(sys.argv[1]).resolve()
binary = root / 'build-tools' / source.stem
obj = binary.with_suffix('.o')
def commands(target):
    return subprocess.check_output([str(ninja), '-C', str(build), '-t', 'commands', target], text=True).strip().splitlines()
args = shlex.split(commands('CMakeFiles/sm70_bench.dir/tools/sm70_bench.cu.o')[-1])
args = args[:args.index('-MD')] + ['-c', str(source), '-o', str(obj)]
subprocess.run(args, cwd=build, check=True)
args = shlex.split(commands('sm70_bench')[-1])[2:-2]
args = [arg for arg in args if not arg.startswith('-Wl,--dependency-file=')]
args[args.index('-o') + 1] = str(binary)
args[args.index('CMakeFiles/sm70_bench.dir/tools/sm70_bench.cu.o')] = str(obj)
subprocess.run(args, cwd=build, check=True)
print(binary)
