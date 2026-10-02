# nix/flake/integrations.nix
#
# External integration derivations and demand-driven build configurations.
{inputs, ...}: let
  zigradVersion = inputs.self.shortRev or inputs.self.dirtyShortRev or "dev";
  zigradRevision = inputs.self.rev or null;
  zigradPreviewRevision =
    if zigradRevision != null
    then zigradRevision
    else builtins.substring 0 40 (inputs.self.dirtyRev or (throw "zigrad-autodoc-preview requires a Git checkout"));
in {
  perSystem = {
    pkgs,
    config,
    cudaCfg,
    repoRoot,
    ...
  }: let
    # Shared settings for source builds that can demand long-running derivations.
    #
    # Defaults and validation live in configuration-options.nix.
    inherit
      (config.zigrad.build)
      withDebugSymbols
      withNativeTuning
      cudaArchitectures
      enableLto
      extraCxxFlags
      extraLdFlags
      extraBazelFlags
      ;
    inherit (pkgs) lib;
    externalSources = import ../external-sources.nix {inherit pkgs;};
    xlaSrc = externalSources.xla.src;
    xlaRevision = externalSources.xla.rev;
    llvmSrc = externalSources.llvm.src;
    llvmRevision = externalSources.llvm.rev;
    llvmVersion = externalSources.llvm.version;
    stablehloSrc = externalSources.stablehlo.src;
    ireeSrc = externalSources.iree.src;
    ireeRevision = externalSources.iree.rev;
    ireeLlvmSrc = externalSources.iree_llvm.src;
    ireeLlvmRevision = externalSources.iree_llvm.rev;
    ireeStablehloSrc = externalSources.iree_stablehlo.src;
    ireeFlatccSrc = externalSources.iree_flatcc.src;
    ireeBenchmarkSrc = externalSources.iree_benchmark.src;

    zigradSrc = import ../helpers/source-filter.nix {
      inherit (pkgs) lib;
      root = repoRoot;
    };

    zigradAutodocDocs = pkgs.callPackage ../packages/zig-autodoc-docs.nix {};
    zigradAutodoc = pkgs.callPackage ../packages/zigrad-autodoc.nix {
      zigAutodocDocs = zigradAutodocDocs;
      zigradSrc = zigradSrc;
    };
    zigradAutodocCandidate = pkgs.callPackage ../packages/zigrad-autodoc-bundle.nix {
      inherit zigradAutodoc;
      revision = zigradRevision;
    };
    zigradAutodocPreview = pkgs.callPackage ../packages/zigrad-autodoc-bundle.nix {
      inherit zigradAutodoc;
      revision = zigradPreviewRevision;
      localPreview = true;
    };

    cuda = import ./integrations/cuda.nix {
      inherit pkgs cudaCfg;
    };
    inherit
      (cuda)
      cudaRuntime
      cudaToolkit
      gccHost
      mkCudaPackage
      ;

    llvm = import ./compiler-support/llvm.nix {
      inherit
        pkgs
        llvmSrc
        llvmRevision
        llvmVersion
        withDebugSymbols
        enableLto
        extraCxxFlags
        extraLdFlags
        ;
    };

    xla = import ./integrations/xla.nix {
      inherit
        pkgs
        cudaCfg
        cudaRuntime
        mkCudaPackage
        cudaArchitectures
        xlaSrc
        xlaRevision
        stablehloSrc
        llvm
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        extraBazelFlags
        ;
    };
    inherit
      (xla)
      pjrtHeaders
      xlaMlirStablehloCapiSdk
      xlaPjrtPlugins
      xlaPjrtPluginsCuda
      xlaCudaRuntime
      xlaProtos
      ;

    mirageParts = import ./integrations/mirage.nix {
      inherit
        pkgs
        lib
        repoRoot
        gccHost
        cudaArchitectures
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        ;
      inherit cudaRuntime cudaToolkit mkCudaPackage;
      cudaVersion = cudaCfg.cudaVersion;
      source = externalSources.mirage;
    };
    inherit (mirageParts) mirage mirageAdapter mirageRustLibs;

    tvmParts = import ./integrations/tvm.nix {
      inherit
        pkgs
        cudaToolkit
        gccHost
        llvm
        cudaArchitectures
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        ;
      inherit cudaRuntime;
      source = externalSources.tvm;
    };
    inherit (tvmParts) tvm tvmCpu tvmCudaIntrinsics;

    iree = import ./integrations/iree.nix {
      inherit
        pkgs
        ireeSrc
        ireeRevision
        ireeLlvmSrc
        ireeLlvmRevision
        ireeStablehloSrc
        ireeFlatccSrc
        ireeBenchmarkSrc
        withDebugSymbols
        withNativeTuning
        enableLto
        extraCxxFlags
        extraLdFlags
        cudaToolkit
        ;
    };
    inherit (iree) ireeCompilerCpu ireeCompilerCuda ireeLlvm ireeRuntimeCpu ireeRuntimeCuda;

    configurationDefinitions = import ./configuration-definitions.nix;
    buildGraph = import ./build-configurations.nix {
      inherit
        lib
        pkgs
        zigradSrc
        ;
      configurations = config.zigrad.configurations;
      version = zigradVersion;
      parts = {
        inherit
          cudaRuntime
          cudaToolkit
          ireeCompilerCpu
          ireeCompilerCuda
          ireeLlvm
          ireeRuntimeCpu
          ireeRuntimeCuda
          llvm
          mirage
          mirageAdapter
          pjrtHeaders
          tvm
          tvmCpu
          tvmCudaIntrinsics
          xlaMlirStablehloCapiSdk
          xlaPjrtPlugins
          xlaPjrtPluginsCuda
          xlaCudaRuntime
          xlaProtos
          ;
      };
    };

    buildConfigurations = buildGraph.resolvedConfigurations;
    configurationPackages =
      lib.concatMapAttrs
      (name: configuration:
        lib.optionalAttrs configuration.expose {
          "${name}" = configuration.package;
        })
      buildConfigurations;
    configurationApps =
      lib.mapAttrs'
      (name: configuration:
        lib.nameValuePair name {
          type = "app";
          program = "${configuration.package}/bin/zigrad";
          meta.description = configuration.description;
        })
      (lib.filterAttrs (_: configuration: configuration.expose) buildConfigurations);
    buildConfigurationManifest = pkgs.writeText "zigrad-build-configurations.json" (
      builtins.toJSON buildGraph.manifest
    );
  in {
    zigrad = {
      configurations = configurationDefinitions;
      resolvedConfigurations = buildConfigurations;
    };
    _module.args.zigradCudaArchitectures = cudaArchitectures;
    _module.args.zigradIree = iree;

    packages =
      configurationPackages
      // {
        mirage-adapter = mirageAdapter;
        mirage-adapter-dev = mirageAdapter.dev;
        inherit mirage;
        mirage-dev = mirage.dev;
        mirage-abstract-subexpr = mirageRustLibs.abstract_subexpr;
        mirage-formal-verifier = mirageRustLibs.formal_verifier;
        zigrad-build-configurations = buildConfigurationManifest;
        zigrad-autodoc = zigradAutodoc;
        zigrad-autodoc-preview = zigradAutodocPreview;
        inherit llvm tvm;
        tvm-dev = tvm.dev;
        tvm-cpu = tvmCpu;
        tvm-python = tvm.python;
        tvm-cuda-intrinsics = tvmCudaIntrinsics;
        cuda-redist = cudaRuntime;
        cuda-redist-dev = cudaToolkit;
        cuda-redist-xla-runtime = xlaCudaRuntime;
        xla-mlir-stablehlo-capi-sdk = xlaMlirStablehloCapiSdk;
        xla-mlir-stablehlo-capi-sdk-dev = xlaMlirStablehloCapiSdk.dev;
        xla-pjrt-plugins = xlaPjrtPlugins;
        xla-pjrt-plugins-cuda = xlaPjrtPluginsCuda;
        iree-llvm = ireeLlvm;
        iree-compiler-cpu = ireeCompilerCpu;
        iree-compiler-cuda = ireeCompilerCuda;
        iree-runtime-cpu = ireeRuntimeCpu;
        iree-runtime-cuda = ireeRuntimeCuda;
        pjrt-headers = pjrtHeaders;
      }
      // lib.optionalAttrs (zigradRevision != null) {
        zigrad-autodoc-candidate = zigradAutodocCandidate;
      };

    apps = configurationApps;
  };
}
