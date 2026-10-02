{
  pkgs,
  cudaToolkit,
  cudaRuntime,
  gccHost,
  llvm,
  cudaArchitectures,
  withDebugSymbols,
  withNativeTuning,
  enableLto,
  extraCxxFlags,
  extraLdFlags,
  source,
}: let
  cudaIntrinsicGenerator = pkgs.writeText "generate-tvm-cuda-intrinsics.py" (
    builtins.readFile ../../../scripts/generate_cuda_intrinsics.py
  );
  generatorPython = pkgs.python3.withPackages (pythonPackages:
    with pythonPackages; [
      cloudpickle
      ml-dtypes
      numpy
      packaging
      psutil
      scipy
      tornado
      typing-extensions
    ]);
  tvm = pkgs.callPackage ../../packages/tvm.nix {
    inherit
      cudaToolkit
      gccHost
      llvm
      withDebugSymbols
      withNativeTuning
      enableLto
      cudaArchitectures
      extraCxxFlags
      extraLdFlags
      ;
    inherit cudaRuntime;
    inherit (source) src;
    version = source.rev;
    cudaSupport = true;
    withPythonBindings = true;
  };
  tvmCudaIntrinsics =
    pkgs.runCommand "tvm-cuda-intrinsics-${source.rev}" {
      nativeBuildInputs = [generatorPython];
    } ''
      mkdir -p "$out/share/tvm"

      cuda_stubs="$TMPDIR/cuda-stubs"
      mkdir -p "$cuda_stubs"
      ln -s ${cudaToolkit}/lib/stubs/libcuda.so "$cuda_stubs/libcuda.so.1"

      export PYTHONPATH="${tvm.python}/python"
      export TVM_LIBRARY_PATH="${tvm}/lib"
      export LD_LIBRARY_PATH="$cuda_stubs:${tvm}/lib"
      python ${cudaIntrinsicGenerator} \
        "$out/share/tvm/cuda_tensor_intrinsics.json"
    '';
  tvmCpu = pkgs.callPackage ../../packages/tvm.nix {
    inherit
      gccHost
      llvm
      withDebugSymbols
      withNativeTuning
      enableLto
      extraCxxFlags
      extraLdFlags
      ;
    inherit (source) src;
    version = source.rev;
    cudaSupport = false;
  };
in {
  inherit tvm tvmCpu tvmCudaIntrinsics;
}
