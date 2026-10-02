{
  lib,
  pkgs,
  zigradSrc,
  version,
  parts,
  configurations,
}: let
  featureDefaults = {
    withPjrt = false;
    withMlir = false;
    withTvm = false;
    withMirage = false;
    withIree = false;
    withNvrtc = false;
    withCudaRuntime = false;
  };

  nodes = {
    "cuda-runtime" = {
      requires = ["cuda-driver"];
      compile = [];
      runtime = [parts.cudaRuntime];
      features.withCudaRuntime = true;
    };

    "cuda-driver" = {
      requires = [];
      compile = [];
      runtime = [];
      features = {};
    };

    nvrtc = {
      requires = ["cuda-runtime"];
      compile = [parts.cudaToolkit];
      runtime = [];
      features.withNvrtc = true;
    };

    "stablehlo-mlir" = {
      requires = [];
      compile = [parts.xlaMlirStablehloCapiSdk.dev];
      runtime = [];
      features.withMlir = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "stablehlo-mlir";
          requirement = "xla-llvm";
          isolation = "in-process";
        }
      ];
    };

    "pjrt-api" = {
      requires = [];
      compile = [
        parts.pjrtHeaders
        parts.xlaProtos
      ];
      runtime = [];
      zigDependencySets = ["protobuf"];
      features.withPjrt = true;
    };

    "pjrt-cpu" = {
      requires = ["pjrt-api"];
      compile = [];
      runtime = [parts.xlaPjrtPlugins];
      features = {};
      conflicts = ["pjrt-cuda"];
    };

    "pjrt-cuda" = {
      requires = [
        "cuda-runtime"
        "pjrt-api"
      ];
      compile = [];
      runtime = [
        parts.xlaCudaRuntime
        parts.xlaPjrtPluginsCuda
      ];
      features = {};
      conflicts = ["pjrt-cpu"];
    };

    iree = {
      requires = [];
      compile = [];
      runtime = [];
      features.withIree = true;
      compatibility = [];
    };

    "iree-runtime-cpu" = {
      requires = ["iree"];
      compile = [parts.ireeRuntimeCpu];
      runtime = [];
      features = {};
      conflicts = ["iree-runtime-cuda"];
    };

    "iree-runtime-cuda" = {
      requires = [
        "cuda-driver"
        "iree"
      ];
      compile = [parts.ireeRuntimeCuda];
      runtime = [];
      features = {};
      conflicts = ["iree-runtime-cpu"];
    };

    "iree-compiler-cpu" = {
      requires = [];
      compile = [];
      runtime = [parts.ireeCompilerCpu];
      features = {};
      conflicts = ["iree-compiler-cuda"];
      compatibility = [
        {
          group = "iree-build-llvm";
          consumer = "iree-compiler";
          requirement = "iree-llvm";
          isolation = "out-of-process";
        }
      ];
    };

    "iree-compiler-cuda" = {
      requires = [];
      compile = [];
      runtime = [parts.ireeCompilerCuda];
      features = {};
      conflicts = ["iree-compiler-cpu"];
      compatibility = [
        {
          group = "iree-build-llvm";
          consumer = "iree-compiler";
          requirement = "iree-llvm";
          isolation = "out-of-process";
        }
      ];
    };

    "tvm-cpu" = {
      requires = [];
      compile = [parts.tvmCpu.dev];
      runtime = [parts.tvmCpu];
      features.withTvm = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "tvm";
          requirement = "tvm-llvm";
          isolation = "in-process";
        }
      ];
      conflicts = [
        "tvm-cuda"
        "tvm-python-cuda"
      ];
    };

    "tvm-cuda" = {
      requires = ["nvrtc"];
      compile = [parts.tvm.dev];
      runtime = [parts.tvm];
      features.withTvm = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "tvm";
          requirement = "tvm-llvm";
          isolation = "in-process";
        }
      ];
      conflicts = [
        "tvm-cpu"
        "tvm-python-cuda"
      ];
    };

    "tvm-python-cuda" = {
      requires = ["nvrtc"];
      compile = [parts.tvm.dev];
      runtime = [parts.tvm parts.tvm.python];
      features.withTvm = true;
      compatibility = [
        {
          group = "host-llvm";
          consumer = "tvm";
          requirement = "tvm-llvm";
          isolation = "in-process";
        }
      ];
      conflicts = [
        "tvm-cpu"
        "tvm-cuda"
      ];
    };

    "mirage-cuda" = {
      requires = [
        "cuda-driver"
        "nvrtc"
      ];
      compile = [
        parts.mirage.dev
        parts.mirageAdapter.dev
      ];
      runtime = [parts.mirageAdapter];
      features.withMirage = true;
    };
  };

  componentDemands = {
    compilers = {
      xla = {
        cpu = ["stablehlo-mlir"];
        cuda = ["stablehlo-mlir"];
      };
      iree = {
        cpu = [
          "iree-compiler-cpu"
          "stablehlo-mlir"
        ];
        cuda = [
          "iree-compiler-cuda"
          "stablehlo-mlir"
        ];
      };
      tvm = {
        cpu = ["tvm-cpu"];
        cuda = ["tvm-cuda"];
      };
    };

    kernelProviders = {
      tvm = {
        cpu = ["tvm-cpu"];
        cuda = ["tvm-cuda"];
      };
      tvm-python.cuda = ["tvm-python-cuda"];
      mirage.cuda = ["mirage-cuda"];
    };

    runtimes = {
      pjrt = {
        cpu = ["pjrt-cpu"];
        cuda = ["pjrt-cuda"];
      };
      iree = {
        cpu = ["iree-runtime-cpu"];
        cuda = ["iree-runtime-cuda"];
      };
      tvm = {
        cpu = ["tvm-cpu"];
        cuda = ["tvm-cuda"];
      };
    };
  };

  componentName = component: "${component.provider}:${component.target}";

  demandsForComponent = role: component:
    componentDemands.${role}.${component.provider}.${component.target}
    or (throw "unknown Zigrad ${role} component '${componentName component}'");

  demandsForConfiguration = definition:
    normalize (
      lib.concatMap (demandsForComponent "compilers") definition.compilers
      ++ lib.concatMap (demandsForComponent "kernelProviders") definition.kernelProviders
      ++ lib.concatMap (demandsForComponent "runtimes") definition.runtimes
    );

  duplicateComponentFor = definition:
    lib.findFirst
    (duplicate: duplicate != null)
    null
    (lib.concatMap
      (role: let
        names = map componentName definition.${role};
      in
        map
        (name:
          if lib.count (candidate: candidate == name) names > 1
          then "${role} contains '${name}' more than once"
          else null)
        names)
      [
        "compilers"
        "kernelProviders"
        "runtimes"
      ]);

  nodeFor = name:
    nodes.${name}
    or (throw "unknown Zigrad build dependency '${name}'");

  dependencyGraph = import ../lib/dependency-graph.nix {
    inherit lib nodes;
  };

  inherit (dependencyGraph) normalize;

  mkExternalInputs = pname: resolved: let
    compilePaths = lib.unique (
      lib.concatMap (node: (nodeFor node).compile or []) resolved
    );
    runtimePaths = lib.unique (
      lib.concatMap (node: (nodeFor node).runtime or []) resolved
    );
  in rec {
    compile = pkgs.symlinkJoin {
      name = "${pname}-external-compile-inputs";
      paths = compilePaths;
    };

    runtime = pkgs.symlinkJoin {
      name = "${pname}-external-runtime-inputs";
      paths = runtimePaths;
    };

    combined = pkgs.symlinkJoin {
      name = "${pname}-external-inputs";
      paths = [
        compile
        runtime
      ];
      passthru = {
        inherit compile runtime;
      };
    };

    inherit compilePaths runtimePaths;
  };

  publicNamePattern = "[a-z][a-z0-9-]*:[a-z][a-z0-9-]*(\\+[a-z][a-z0-9-]*:[a-z][a-z0-9-]*)*";

  mkConfiguration = name: definition: let
    namedComponents = lib.unique (
      map componentName (definition.compilers ++ definition.kernelProviders)
    );
    expectedPublicName = lib.concatStringsSep "+" namedComponents;
    demands = demandsForConfiguration definition;
    resolved = dependencyGraph.close demands;
    conflict = dependencyGraph.conflictFor resolved;
    duplicateComponent = duplicateComponentFor definition;
    validPublicName =
      !definition.expose
      || (
        if namedComponents == []
        then name == "zigrad"
        else
          name
          == expectedPublicName
          && builtins.match publicNamePattern name != null
      );
    pname =
      if definition.pname != null
      then definition.pname
      else "zigrad-${lib.replaceStrings [":" "+"] ["-" "-"] name}";
    externalInputs = mkExternalInputs pname resolved;
    has = node: lib.elem node resolved;
    featureArgs =
      lib.foldl'
      (features: node: features // ((nodeFor node).features or {}))
      featureDefaults
      resolved;
    zigDependencySets = lib.unique (
      lib.concatMap (node: (nodeFor node).zigDependencySets or []) resolved
    );
    zigFeatureArgs = [
      "-Dpjrt=${lib.boolToString featureArgs.withPjrt}"
      "-Dmlir=${lib.boolToString featureArgs.withMlir}"
      "-Dtvm=${lib.boolToString featureArgs.withTvm}"
      "-Dmirage=${lib.boolToString featureArgs.withMirage}"
      "-Diree=${lib.boolToString featureArgs.withIree}"
      "-Dnvrtc=${lib.boolToString featureArgs.withNvrtc}"
      "-Dcuda-runtime=${lib.boolToString featureArgs.withCudaRuntime}"
    ];
    zigFeatureFlags = lib.concatStringsSep " " zigFeatureArgs;
    tvmCompilerFingerprint = builtins.hashString "sha256" (builtins.toJSON {
      zigrad = "${zigradSrc}";
      tvmCpu =
        if has "tvm-cpu"
        then "${parts.tvmCpu}"
        else null;
      tvmCuda =
        if has "tvm-cuda" || has "tvm-python-cuda"
        then "${parts.tvm}"
        else null;
      linker = "${parts.llvm}";
      cudaIntrinsics =
        if has "tvm-cuda" || has "tvm-python-cuda"
        then "${parts.tvmCudaIntrinsics}"
        else null;
      cudaToolkit =
        if has "tvm-cuda" || has "tvm-python-cuda"
        then "${parts.cudaToolkit}"
        else null;
    });
    compatibility = lib.unique (
      lib.concatMap (node: (nodeFor node).compatibility or []) resolved
    );
    runtimeEnv =
      lib.optionalAttrs (has "pjrt-cpu" || has "pjrt-cuda") {
        PJRT_PLUGIN_PATH =
          if has "pjrt-cuda"
          then "${externalInputs.runtime}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
          else "${externalInputs.runtime}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
      }
      // lib.optionalAttrs (has "tvm-cpu" || has "tvm-cuda" || has "tvm-python-cuda") {
        ZG_TVM_COMPILER_FINGERPRINT = tvmCompilerFingerprint;
        ZG_ELF_LINKER_PATH = "${parts.llvm}/bin/ld.lld";
      }
      // lib.optionalAttrs (has "tvm-cuda" || has "tvm-python-cuda") {
        ZG_TVM_CUDA_INTRINSICS_PATH = "${parts.tvmCudaIntrinsics}/share/tvm/cuda_tensor_intrinsics.json";
      }
      // lib.optionalAttrs (has "iree-compiler-cpu" || has "iree-compiler-cuda") {
        ZG_IREE_COMPILER_PATH =
          if has "iree-compiler-cuda"
          then "${parts.ireeCompilerCuda}/bin/iree-compile"
          else "${parts.ireeCompilerCpu}/bin/iree-compile";
        IREE_LLVM_EMBEDDED_LINKER_PATH = lib.getExe' parts.ireeLlvm "ld.lld";
      }
      // lib.optionalAttrs (has "iree-runtime-cpu") {
        ZG_IREE_DRIVER = "local-sync";
        ZG_IREE_TARGET_BACKEND = "llvm-cpu";
      }
      // lib.optionalAttrs (has "iree-runtime-cuda") {
        ZG_IREE_DRIVER = "cuda";
        ZG_IREE_TARGET_BACKEND = "cuda";
      }
      // lib.optionalAttrs (has "nvrtc") {
        CUDA_HOME = "${parts.cudaToolkit}";
        NIX_GLIBC_INCLUDE = "${pkgs.stdenv.cc.libc.dev}/include";
        NIX_GCC_INCLUDE = "${pkgs.stdenv.cc.cc}/lib/gcc/${pkgs.stdenv.hostPlatform.config}/${lib.getVersion pkgs.stdenv.cc.cc}/include";
      }
      // lib.optionalAttrs (has "mirage-cuda") {
        ZG_EXTERNAL_SDK_ROOT = "${externalInputs.combined}";
        ZG_MIRAGE_INCLUDE_DIR = "${parts.mirage.dev}/include";
      };
    runtimeEnvPrefixes = lib.optionalAttrs (has "pjrt-cuda") {
      XLA_FLAGS = "--xla_gpu_cuda_data_dir=${externalInputs.runtime}/runtime/nvidia";
    };
    runtimePolicy = {
      fixed = runtimeEnv;
      defaults = {};
      prefixes = runtimeEnvPrefixes;
    };
    needsCudaDriverRunpath = has "cuda-driver";
    package = pkgs.callPackage ../packages/zigrad.nix {
      inherit
        runtimeEnv
        runtimeEnvPrefixes
        version
        zigradSrc
        ;
      inherit pname;
      compileInputs = externalInputs.compile;
      inherit zigDependencySets zigFeatureArgs;
      inherit needsCudaDriverRunpath;
      runtimeLibraryPaths = lib.optionals (externalInputs.runtimePaths != []) [
        "${externalInputs.runtime}/lib"
        "${externalInputs.runtime}/runtime/sys/lib"
      ];
      runtimeInputs =
        if externalInputs.runtimePaths == []
        then null
        else externalInputs.runtime;
      optimize = "ReleaseFast";
      passthru = {
        inherit
          externalInputs
          needsCudaDriverRunpath
          runtimePolicy
          zigDependencySets
          ;
      };
    };
  in
    assert lib.assertMsg validPublicName
    "Zigrad public configuration '${name}' must be named '${
      if namedComponents == []
      then "zigrad"
      else expectedPublicName
    }' from its compiler and kernel-provider components";
    assert lib.assertMsg (duplicateComponent == null)
    "Zigrad build configuration '${name}' is invalid: ${duplicateComponent}";
    assert lib.assertMsg (conflict == null)
    "Zigrad build configuration '${name}' is invalid: ${conflict}"; {
      inherit
        compatibility
        demands
        externalInputs
        featureArgs
        name
        needsCudaDriverRunpath
        package
        pname
        resolved
        runtimePolicy
        zigDependencySets
        zigFeatureArgs
        zigFeatureFlags
        ;
      inherit (definition) compilers description expose kernelProviders runtimes;
    };

  resolvedConfigurations = lib.mapAttrs mkConfiguration configurations;

  manifest =
    lib.mapAttrs
    (_: configuration: {
      inherit
        (configuration)
        compatibility
        compilers
        demands
        description
        expose
        kernelProviders
        pname
        resolved
        runtimes
        zigDependencySets
        zigFeatureArgs
        ;
      runtimeEnvironment = {
        fixed = builtins.attrNames configuration.runtimePolicy.fixed;
        defaults = builtins.attrNames configuration.runtimePolicy.defaults;
        prefixes = builtins.attrNames configuration.runtimePolicy.prefixes;
      };
    })
    resolvedConfigurations;
in
  assert lib.assertMsg (dependencyGraph.invalidReference == null)
  "Zigrad dependency graph is invalid: ${dependencyGraph.invalidReference}"; {
    inherit
      manifest
      nodes
      resolvedConfigurations
      ;
  }
