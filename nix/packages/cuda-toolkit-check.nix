{
  runCommand,
  cudaToolkit,
}:
runCommand "cuda-toolkit-check" {} ''
  set -euo pipefail
  mkdir -p "$out"

  for header in cuda.h cuda_runtime.h; do
    cat > probe.cu <<EOF
  #include <$header>
  #include <nv/target>
  __global__ void write_value(float *out) { out[0] = 1.0f; }
  EOF
    ${cudaToolkit}/bin/nvcc -c probe.cu -o "$out/$header.o"
    test -s "$out/$header.o"
    ${cudaToolkit}/bin/nvcc -ptx probe.cu -o "$out/$header.ptx"
    test -s "$out/$header.ptx"
  done
''
