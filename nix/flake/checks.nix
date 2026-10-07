# nix/flake/checks.nix
{
  perSystem = {
    pkgs,
    config,
    ...
  }: let
    zigrad = config.packages.zigrad;
    zigradXlaCpu = config.packages."xla:cpu";
    zigradIreeCpu = config.packages."iree:cpu";
    zigradTvmCpu = config.packages."tvm:cpu";
    zigradDevCuda = config.packages."xla:cuda+iree:cpu+tvm:cuda+mirage:cuda";
    benchmarkExample = config.packages.zigrad-example-benchmark;
    basicDeploymentCpu = config.packages.zigrad-example-basic-deployment-cpu;
    cifar10Cpu = config.packages.zigrad-example-cifar10-cpu;
    cifar10Tests = config.packages.zigrad-example-cifar10-tests;
    hasBasicDeploymentCuda = config.packages ? zigrad-example-basic-deployment-cuda;
    basicDeploymentCuda = config.packages.zigrad-example-basic-deployment-cuda or null;
    llamaTraining = config.packages.zigrad-example-llama-training;
    mnistExample = config.packages.zigrad-example-mnist;
    mlirCppExample = config.packages.zigrad-example-mlir-cpp;
    dependencyGraphFailures = pkgs.lib.runTests (
      import ../lib/dependency-graph-tests.nix {inherit (pkgs) lib;}
    );
    dependencyGraphTests = assert pkgs.lib.assertMsg (dependencyGraphFailures == [])
    "Zigrad dependency graph tests failed: ${builtins.toJSON dependencyGraphFailures}";
      pkgs.writeText "zigrad-dependency-graph-tests" "passed\n";

    mkExampleCheck = {
      name,
      command,
    }:
      pkgs.runCommand name {} ''
        set -euo pipefail
        export HOME="$TMPDIR"
        export ZG_CACHE_DIR="$TMPDIR/zigrad-cache"

        ${command} >"$TMPDIR/stdout.txt" 2>"$TMPDIR/stderr.txt"

        mkdir -p "$out"
        cp "$TMPDIR/stdout.txt" "$out/stdout.txt"
        cp "$TMPDIR/stderr.txt" "$out/stderr.txt"
      '';

    checkBenchmarkExample = mkExampleCheck {
      name = "check-zigrad-example-benchmark";
      command = ''
        ${benchmarkExample}/bin/benchmark \
          --shapes=2x3x4 \
          --impls=zig_naive \
          --warmup=0 \
          --iters=1
      '';
    };

    checkMnistPjrtExample = mkExampleCheck {
      name = "check-zigrad-example-mnist-pjrt";
      command = "${mnistExample}/bin/mnist --steps=1 --backend=pjrt";
    };

    checkMnistIreeExample = mkExampleCheck {
      name = "check-zigrad-example-mnist-iree";
      command = "${mnistExample}/bin/mnist --steps=1 --backend=iree";
    };

    zigradTests = zigrad.override {
      optimize = "safe";
      runTests = true;
    };

    zigradXlaCpuTests = zigradXlaCpu.override {
      optimize = "safe";
      runTests = true;
    };

    zigradIreeCpuTests = zigradIreeCpu.override {
      optimize = "safe";
      runTests = true;
    };

    zigradTvmCpuTests = zigradTvmCpu.override {
      optimize = "safe";
      runTests = true;
    };

    zigradDevCudaTests = zigradDevCuda.override {
      optimize = "safe";
      runTests = true;
    };

    zigradTvmCpuCompileOnly = zigradTvmCpu.override {
      runtimeInputs = null;
      runtimeEnv = {};
      runtimeLibraryPaths = [];
    };

    checkTvmRuntimePresent =
      pkgs.runCommand "check-zigrad-tvm-runtime-present" {
        nativeBuildInputs = [
          zigradTvmCpu
        ];
      } ''
        set -euo pipefail
        export HOME="$TMPDIR"

        ${zigradTvmCpu}/bin/zigrad tvm symbols > "$TMPDIR/tvm-symbols.txt"
        test -s "$TMPDIR/tvm-symbols.txt"

        mkdir -p "$out"
        cp "$TMPDIR/tvm-symbols.txt" "$out/tvm-symbols.txt"
      '';

    checkTvmRuntimeNoTvm =
      pkgs.runCommand "check-zigrad-tvm-runtime-no-tvm" {
        nativeBuildInputs = [
          zigradTvmCpuCompileOnly
        ];
      } ''
        set -euo pipefail
        export HOME="$TMPDIR"

        if ${zigradTvmCpuCompileOnly}/bin/zigrad tvm symbols > "$TMPDIR/stdout.txt" 2> "$TMPDIR/stderr.txt"; then
          echo "expected 'tvm symbols' to fail without TVM runtime libraries" >&2
          exit 1
        fi

        if ! grep -Eq "(TvmLoadFailed|failed to load TVM FFI runtime|dlopen)" "$TMPDIR/stderr.txt"; then
          echo "expected loader diagnostics in stderr" >&2
          cat "$TMPDIR/stderr.txt" >&2
          exit 1
        fi

        mkdir -p "$out"
        cp "$TMPDIR/stderr.txt" "$out/tvm-missing-stderr.txt"
      '';
  in {
    checks =
      {
        zigrad = zigrad;
        zigrad-build-cli = config.packages.zigrad-build;
        zigrad-autodoc = config.packages.zigrad-autodoc;
        zigrad-unit-tests = zigradTests;
        zigrad-build-configurations = config.packages.zigrad-build-configurations;
        zigrad-dependency-graph = dependencyGraphTests;
        zigrad-example-benchmark = checkBenchmarkExample;
        zigrad-example-basic-deployment-cpu = basicDeploymentCpu;
        zigrad-example-cifar10-cpu = cifar10Cpu;
        zigrad-example-cifar10-tests = cifar10Tests;
        zigrad-example-llama-training = llamaTraining;
        zigrad-example-mlir-cpp = mlirCppExample;
        zigrad-example-mnist-pjrt = checkMnistPjrtExample;
        zigrad-example-mnist-iree = checkMnistIreeExample;
      }
      // pkgs.lib.optionalAttrs hasBasicDeploymentCuda {
        zigrad-example-basic-deployment-cuda = basicDeploymentCuda;
      };

    packages = {
      zigrad-unit-tests = zigradTests;
      zigrad-xla-cpu-unit-tests = zigradXlaCpuTests;
      zigrad-iree-cpu-unit-tests = zigradIreeCpuTests;
      zigrad-tvm-cpu-unit-tests = zigradTvmCpuTests;
      zigrad-dev-cuda-unit-tests = zigradDevCudaTests;
      zigrad-tvm-runtime-check = checkTvmRuntimePresent;
      zigrad-tvm-runtime-missing-check = checkTvmRuntimeNoTvm;
    };
  };
}
