"""Stage BF16 epilogue results in dead MMA shared storage, then issue 16-byte output stores."""
from pathlib import Path
s=Path('src/kernels/sm70/int8_gemm_sm70.cu').read_text()
s=s.replace('  // CUTLASS\'s Volta FP32 accumulator', '''  __syncthreads();
  auto* staged = reinterpret_cast<__nv_bfloat16*>(buffer);
  constexpr int ON = E == Epilogue::kSitu ? BN / 2 : BN;
  const int out_n0 = E == Epilogue::kSitu ? n0 / 2 : n0;
  // CUTLASS's Volta FP32 accumulator''')
s=s.replace('    out[r * ldo + col] = __float2bfloat16_rn(value);', '    staged[(r - m0) * ON + col - out_n0] = __float2bfloat16_rn(value);')
# Check the exact store spelling before running a timing probe.
if 'staged[(r - m0)' not in s: raise RuntimeError('epilogue store not found')
a=s.index('\n}\n\ntemplate <Epilogue E, typename Tile, bool Fast = false>\nvoid launch_tile')
s=s[:a]+'''
  __syncthreads();
  const int columns = E == Epilogue::kSitu ? p.n / 2 : p.n;
  for (int i = tid * 8; i < BM * ON; i += blockDim.x * 8) {
    const int r = m0 + i / ON, c = out_n0 + i % ON;
    if (r >= p.m || c >= columns) continue;
    if (c + 8 <= columns && reinterpret_cast<std::uintptr_t>(out + std::int64_t(r) * ldo + c) % 16 == 0) {
      *reinterpret_cast<uint4*>(out + std::int64_t(r) * ldo + c) = *reinterpret_cast<const uint4*>(staged + i);
    } else {
      for (int j = 0; j < 8 && c + j < columns; ++j) out[std::int64_t(r) * ldo + c + j] = staged[i + j];
    }
  }
'''+s[a:]
Path('build-tools/gemm-coalesced.cu').write_text(s)
