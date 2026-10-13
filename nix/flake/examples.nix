# nix/flake/examples.nix
#
# Packages and development inputs for maintained examples.
{
  perSystem = {
    pkgs,
    config,
    repoRoot,
    zigradCudaArchitectures,
    zigradIree,
    ...
  }: let
    inherit (pkgs) lib;
    buildConfigurations = config.zigrad.resolvedConfigurations;
    fs = lib.fileset;
    zigradSrc = import ../helpers/source-filter.nix {
      inherit lib;
      root = repoRoot;
    };
    zigExampleSrc = name: let
      exampleRoot = repoRoot + "/examples/${name}";
    in
      fs.toSource {
        root = exampleRoot;
        fileset = exampleRoot;
      };
    mkZigApplication = configuration: args:
      pkgs.callPackage ../packages/zigrad-application.nix (
        {
          inherit
            (configuration)
            externalInputs
            needsCudaDriverRunpath
            runtimePolicy
            zigDependencySets
            ;
        }
        // args
      );
    mkZigExample = {
      name,
      mainProgram,
      configuration,
      zigArgs ? [],
    }:
      mkZigApplication configuration {
        inherit mainProgram;
        pname = "zigrad-example-${name}";
        src = zigExampleSrc name;
        inherit zigradSrc;
        inherit zigArgs;
      };
    benchmark = mkZigExample {
      name = "benchmark";
      mainProgram = "benchmark";
      configuration = buildConfigurations.example-benchmark;
    };
    mnist = mkZigExample {
      name = "mnist";
      mainProgram = "mnist";
      configuration = buildConfigurations."xla:cpu+iree:cpu";
    };
    llamaTraining = mkZigExample {
      name = "llama-training";
      mainProgram = "llama-training";
      configuration = buildConfigurations."xla:cuda+iree:cuda";
    };
    llamaTrainingCpu = mkZigExample {
      name = "llama-training";
      mainProgram = "llama-training";
      configuration = buildConfigurations."xla:cpu";
      zigArgs = ["-Dpjrt=true" "-Diree=false"];
    };
    cifar10Train = mkZigExample {
      name = "cifar10-lifecycle";
      mainProgram = "cifar10-train";
      configuration = buildConfigurations."xla:cuda";
      zigArgs = ["-Dmode=train"];
    };
    cifar10Tests = mkZigApplication buildConfigurations.zigrad {
      mainProgram = "cifar10-init";
      pname = "zigrad-example-cifar10-tests";
      src = zigExampleSrc "cifar10-lifecycle";
      inherit zigradSrc;
      zigArgs = ["-Dmode=init"];
      runTests = true;
      testProgram = "cifar10-model-tests";
      withRuntimeEnvironment = false;
    };
    hasIreeCudaArchitecture = builtins.length zigradCudaArchitectures == 1;
    ireeCudaArchitecture = lib.optionalString hasIreeCudaArchitecture (builtins.head zigradCudaArchitectures);
    basicDeploymentSource = zigExampleSrc "basic-deployment";
    mkBasicDeployment = pkgs.callPackage ../../examples/basic-deployment/package.nix {
      inherit mkZigApplication;
      source = basicDeploymentSource;
      emitterConfiguration = buildConfigurations.zigrad;
      inherit zigradSrc;
    };
    embeddedElfCompilerArguments = [
      "--iree-hal-target-device=local"
      "--iree-hal-local-target-device-backends=llvm-cpu"
      "--iree-llvmcpu-debug-symbols=false"
      "--iree-vm-bytecode-module-strip-source-map=true"
      "--iree-vm-emit-polyglot-zip=false"
    ];
    basicDeploymentCpu = mkBasicDeployment {
      name = "cpu";
      backend = "llvm-cpu";
      compilerConfiguration = buildConfigurations."iree:cpu";
      runnerConfiguration = buildConfigurations.iree-cpu-runtime;
      compilerArguments = embeddedElfCompilerArguments;
      compilerEnvironment.IREE_LLVM_EMBEDDED_LINKER_PATH = lib.getExe' config.packages.iree-llvm "ld.lld";
      canExecute = true;
    };
    basicDeploymentCuda = mkBasicDeployment {
      name = "cuda-sm_${ireeCudaArchitecture}";
      backend = "cuda";
      compilerConfiguration = buildConfigurations."iree:cuda";
      runnerConfiguration = buildConfigurations.iree-cuda-runtime;
      compilerArguments = ["--iree-cuda-target=sm_${ireeCudaArchitecture}"];
      runnerArguments = [
        "-Druntime=registered"
        "-Ddriver=cuda"
      ];
      withRuntimeEnvironment = true;
    };
    aarch64Pkgs = pkgs.pkgsCross.aarch64-multiplatform;
    aarch64IreeRuntime = zigradIree.mkRuntimeCpuFor aarch64Pkgs;
    aarch64IreeInputs = pkgs.symlinkJoin {
      name = "zigrad-iree-aarch64-runtime-inputs";
      paths = [aarch64IreeRuntime];
    };
    aarch64RuntimeInputs = aarch64Pkgs.stdenv.cc.libc;
    aarch64CombinedInputs = pkgs.symlinkJoin {
      name = "zigrad-basic-deployment-aarch64-external-inputs";
      paths = [aarch64IreeInputs aarch64RuntimeInputs];
    };
    aarch64ExternalInputs = {
      compile = aarch64IreeInputs;
      runtime = aarch64RuntimeInputs;
      combined = aarch64CombinedInputs;
    };
    basicDeploymentCpuAarch64 = mkBasicDeployment {
      name = "cpu-aarch64";
      backend = "llvm-cpu";
      compilerConfiguration = buildConfigurations."iree:cpu";
      runnerConfiguration = buildConfigurations.iree-cpu-runtime;
      targetPkgs = aarch64Pkgs;
      runnerExternalInputs = aarch64ExternalInputs;
      compilerArguments =
        embeddedElfCompilerArguments
        ++ [
          "--iree-llvmcpu-target-triple=${aarch64Pkgs.stdenv.hostPlatform.config}"
          "--iree-llvmcpu-target-cpu=cortex-a72"
        ];
      compilerEnvironment.IREE_LLVM_EMBEDDED_LINKER_PATH = lib.getExe' config.packages.iree-llvm "ld.lld";
      strip = true;
      canExecute = false;
    };
    cifar10Source = zigExampleSrc "cifar10-lifecycle";
    mkCifar10Deployment = pkgs.callPackage ../../examples/cifar10-lifecycle/package.nix {
      inherit mkZigApplication;
      source = cifar10Source;
      emitterConfiguration = buildConfigurations.zigrad;
      inherit zigradSrc;
    };
    cifar10Cpu = mkCifar10Deployment {
      name = "cpu";
      backend = "llvm-cpu";
      compilerConfiguration = buildConfigurations."iree:cpu";
      runnerConfiguration = buildConfigurations.iree-cpu-runtime;
      compilerArguments = embeddedElfCompilerArguments;
      compilerEnvironment.IREE_LLVM_EMBEDDED_LINKER_PATH = lib.getExe' config.packages.iree-llvm "ld.lld";
      canExecute = true;
    };
    cifar10CpuAarch64 = mkCifar10Deployment {
      name = "cpu-aarch64";
      backend = "llvm-cpu";
      compilerConfiguration = buildConfigurations."iree:cpu";
      runnerConfiguration = buildConfigurations.iree-cpu-runtime;
      targetPkgs = aarch64Pkgs;
      runnerExternalInputs = aarch64ExternalInputs;
      compilerArguments =
        embeddedElfCompilerArguments
        ++ [
          "--iree-llvmcpu-target-triple=${aarch64Pkgs.stdenv.hostPlatform.config}"
          "--iree-llvmcpu-target-cpu=cortex-a72"
        ];
      compilerEnvironment.IREE_LLVM_EMBEDDED_LINKER_PATH = lib.getExe' config.packages.iree-llvm "ld.lld";
      strip = true;
    };
    mlirCpp = pkgs.callPackage ../packages/zigrad-mlir-cpp-example.nix {
      xlaMlirStablehloCapiSdk = config.packages.xla-mlir-stablehlo-capi-sdk;
      llvm = config.packages.llvm;
      src = let
        fs = lib.fileset;
      in
        fs.toSource {
          root = repoRoot + /examples/mlir-cpp;
          fileset = fs.unions [
            (repoRoot + /examples/mlir-cpp/CMakeLists.txt)
            (repoRoot + /examples/mlir-cpp/mlir_ext.cc)
            (repoRoot + /examples/mlir-cpp/zigrad)
            (repoRoot + /examples/mlir-cpp/dev)
            (repoRoot + /examples/mlir-cpp/test)
          ];
        };
    };
    configureMlirCpp = pkgs.writeShellApplication {
      name = "configure-mlir-cpp";
      runtimeInputs = [pkgs.cmake pkgs.git pkgs.ninja];
      text = ''
        root="$(git rev-parse --show-toplevel)"

        cmake \
          -S "$root/examples/mlir-cpp" \
          -B "$root/examples/mlir-cpp/build" \
          -G Ninja \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
          -DZG_ENABLE_DEV_TARGETS=ON \
          -DZG_SDK_INCLUDE=${config.packages.xla-mlir-stablehlo-capi-sdk.dev}/include \
          -DZG_SDK_LIB=${config.packages.xla-mlir-stablehlo-capi-sdk.out}/lib \
          -DZG_LLVM_ROOT=${config.packages.llvm} \
          -DZG_INSTALL_RPATH="${config.packages.xla-mlir-stablehlo-capi-sdk.out}/lib:${config.packages.llvm}/lib"

        cmake --build "$root/examples/mlir-cpp/build" --target zigrad_tblgen
      '';
    };
  in {
    packages =
      {
        configure-mlir-cpp = configureMlirCpp;
        zigrad-example-benchmark = benchmark;
        zigrad-example-basic-deployment-cpu = basicDeploymentCpu;
        zigrad-example-basic-deployment-cpu-aarch64 = basicDeploymentCpuAarch64;
        zigrad-example-cifar10-cpu = cifar10Cpu;
        zigrad-example-cifar10-cpu-aarch64 = cifar10CpuAarch64;
        zigrad-example-cifar10-train = cifar10Train;
        zigrad-example-cifar10-tests = cifar10Tests;
        zigrad-example-llama-training = llamaTraining;
        zigrad-example-llama-training-cpu = llamaTrainingCpu;
        zigrad-example-mnist = mnist;
        zigrad-example-mlir-cpp = mlirCpp;
        zigrad-example-mlir-cpp-dev = mlirCpp.dev;
      }
      // lib.optionalAttrs hasIreeCudaArchitecture {
        zigrad-example-basic-deployment-cuda = basicDeploymentCuda;
      };

    apps =
      {
        configure-mlir-cpp = {
          type = "app";
          program = "${configureMlirCpp}/bin/configure-mlir-cpp";
          meta.description = "Configure the local C++ MLIR example build";
        };
        zigrad-example-benchmark = {
          type = "app";
          program = "${benchmark}/bin/benchmark";
          meta.description = "Run the Zigrad backend benchmark";
        };
        zigrad-example-basic-deployment-cpu = {
          type = "app";
          program = "${basicDeploymentCpu}/bin/basic-deployment";
          meta.description = "Run the Zigrad basic CPU deployment example";
        };
        zigrad-example-mnist = {
          type = "app";
          program = "${mnist}/bin/mnist";
          meta.description = "Run the Zigrad MNIST training example";
        };
        zigrad-example-llama-training = {
          type = "app";
          program = "${llamaTraining}/bin/llama-training";
          meta.description = "Run the Zigrad LLaMA training example";
        };
        zigrad-example-llama-training-cpu = {
          type = "app";
          program = "${llamaTrainingCpu}/bin/llama-training";
          meta.description = "Run the Zigrad LLaMA CPU training example";
        };
        zigrad-example-cifar10-cpu = {
          type = "app";
          program = "${cifar10Cpu}/bin/cifar10-infer";
          meta.description = "Run the Zigrad CIFAR-10 CPU inference example";
        };
      }
      // lib.optionalAttrs hasIreeCudaArchitecture {
        zigrad-example-basic-deployment-cuda = {
          type = "app";
          program = "${basicDeploymentCuda}/bin/basic-deployment";
          meta.description = "Run the Zigrad basic CUDA deployment example";
        };
      };
  };
}
