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
  pipelineGenerator = pkgs.writeText "generate-tvm-pipeline-contract.py" (
    builtins.readFile ../../../scripts/generate_tvm_pipeline_contract.py
  );
  pipelineContract = pkgs.runCommand "tvm-pipeline-contract-${source.rev}" {
    nativeBuildInputs = [pkgs.python3];
  } ''
    mkdir -p "$out/share/tvm"
    python ${pipelineGenerator} ${source.src} "$out/share/tvm/pipeline-contract.json"
  '';
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
  tvmPackage = pkgs.callPackage ../../packages/tvm.nix {
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
  tvmCpuPackage = pkgs.callPackage ../../packages/tvm.nix {
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
  tvm = tvmPackage.overrideAttrs (old: { passthru = (old.passthru or {}) // { inherit pipelineContract; }; });
  tvmCpu = tvmCpuPackage.overrideAttrs (old: { passthru = (old.passthru or {}) // { inherit pipelineContract; }; });
in {
  inherit tvm tvmCpu tvmCudaIntrinsics;
}
