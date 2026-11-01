"""Compile and separately link a generated build-tools/ATTENTION_NAME.cu probe against the configured SM70 build."""
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
build = root / 'build-sm70'
name = sys.argv[1]
source = root / 'build-tools' / f'{name}.cu'
obj = source.with_suffix('.o')
ninja = shutil.which('ninja') or str(root / 'build-tools/ninja')

def commands(target):
    return subprocess.check_output([ninja, '-C', str(build), '-t', 'commands', target], text=True).strip().splitlines()

args = shlex.split(commands('CMakeFiles/mossinfer_sm70.dir/src/kernels/sm70/attention_sm70.cu.o')[-1])
args = args[:args.index('-MD')] + ['-Xptxas=-v', '-c', str(source), '-o', str(obj)]
with source.with_suffix('.build.log').open('w') as log:
    result = subprocess.run(args, cwd=build, stdout=log, stderr=subprocess.STDOUT)
if result.returncode:
    print(source.with_suffix('.build.log').read_text())
    sys.exit(result.returncode)
for target in ['sm70_bench', 'sm70_check', 'moss-infer', 'mossinfer_tests']:
    args = shlex.split(commands(target)[-1])[2:-2]
    args[args.index('-o') + 1] = str(root / 'build-tools' / f'{name}-{target}')
    args = [arg for arg in args if not arg.startswith('-Wl,--dependency-file=')]
    args.insert(args.index('libmossinfer_core.a'), str(obj))
    subprocess.run(args, cwd=build, check=True)
print(f'{name}: compiled and linked')
