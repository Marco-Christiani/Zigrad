# nix/flake/devshells.nix
#
# Development shells for fast iteration with relaxed hermeticity.
{
  perSystem = {
    pkgs,
    config,
    cudaCfg,
    ...
  }: let
    buildConfigurations = config.zigrad.resolvedConfigurations;
    externalSources = import ../external-sources.nix {inherit pkgs;};
    cudaPackages = pkgs.${cudaCfg.cudaPackagesAttr};
    gccHost = pkgs.${cudaCfg.gccHostAttr};

    # Wrap clangd so it trusts Nix-store gcc/clang wrappers as drivers.
    # Without --query-driver, clangd refuses to query the compiler used in
    #  compile_commands.json and falls back to no system include paths,
    #  producing 'type_traits' file not found cascades in C++ TUs.
    clangdWrapped = pkgs.writeShellScriptBin "clangd" ''
      exec ${pkgs.clang-tools}/bin/clangd \
        --query-driver=/nix/store/*-gcc-wrapper-*/bin/g++,/nix/store/*-clang-wrapper-*/bin/clang++,/nix/store/*-clang-wrapper-*/bin/clang \
        "$@"
    '';

    mlirCppExampleDev = config.packages.zigrad-example-mlir-cpp-dev;
    devCuda = buildConfigurations."xla:cuda+iree:cpu+tvm:cuda+mirage:cuda";

    baseDevShellPkgs = with pkgs; [
      zig
      zls
      zon2nix
      go-task
      binutils
      patchelf
      git
      gccHost
      clang
      cmake
      ninja
      clangdWrapped
      config.packages.zigrad-build
    ];

    lspShadowHook = ''
      export PATH="${clangdWrapped}/bin:$PATH"
    '';

    devShellHook = name: ''
      export ZG_DEVSHELL=${pkgs.lib.escapeShellArg name}

      if [[ ( ''${DIRENV_IN_ENVRC:-0} == 1 || ( $- == *i* && -t 1 ) ) && ''${ZG_DEVSHELL_QUIET:-} != 1 ]]; then
        printf '%s\n' \
          "Zigrad development shell: $ZG_DEVSHELL" \
          "  targets: zig build -l" \
          "  core tests: zig build test --summary all" \
          "  build configurations: zigrad-build config list" \
          "  configured build flags: $ZG_ZIG_BUILD_ARGS"
      fi
    '';

    mlirCppExampleHook = ''
      export PATH="${mlirCppExampleDev}/bin:$PATH"
    '';

    tvmPython = pkgs.python312.withPackages (pythonPackages:
      with pythonPackages; [
        cloudpickle
        ml-dtypes
        numpy
        packaging
        psutil
        scipy
        tornado
        typing-extensions
        xgboost
      ]);

    integrationEnv = configuration: let
      inherit (configuration) externalInputs runtimePolicy;
      combinedInputs = externalInputs.combined;
      externalInputsStr = toString combinedInputs;
      runtimeStr = toString externalInputs.runtime;
    in
      runtimePolicy.fixed
      // runtimePolicy.defaults
      // runtimePolicy.prefixes
      // {
        ZG_EXTERNAL_SDK_ROOT = externalInputsStr;
        ZG_RUNTIME_SDK_ROOT = runtimeStr;
        ZG_ZIG_BUILD_ARGS = configuration.zigFeatureFlags;
        ZG_MIRAGE_RUNTIME_LIBRARY = "${config.packages.mirage}/lib/libmirage_runtime.so";
        ZG_NLOHMANN_JSON_INCLUDE_DIR = "${pkgs.lib.getDev pkgs.nlohmann_json}/include";
        PJRT_CPU_PLUGIN_PATH = "${runtimeStr}/runtime/xla/pjrt/c/pjrt_c_api_cpu_plugin.so";
        PJRT_GPU_PLUGIN_PATH = "${runtimeStr}/runtime/xla/pjrt/c/pjrt_c_api_gpu_plugin.so";

        # vim runtime dirs from the LLVM source tree. Editors can append these
        #  to runtimepath to get LLVM's official syntax/indent/ftplugin/ftdetect
        #  files for .mlir/.ll/.td (handles StableHLO-style assembly that the
        #  community tree-sitter grammar chokes on).
        ZG_MLIR_VIM_RT = "${externalSources.llvm.src}/mlir/utils/vim";
        ZG_LLVM_VIM_RT = "${externalSources.llvm.src}/llvm/utils/vim";
      };

    runtimeLibraryHook = configuration: let
      inherit (configuration) externalInputs;
      runtime = toString externalInputs.runtime;
      paths =
        [
          "${runtime}/lib"
          "${runtime}/runtime/sys/lib"
        ]
        ++ pkgs.lib.optional
        configuration.needsCudaDriverRunpath
        "/run/opengl-driver/lib";
    in ''
      export LD_LIBRARY_PATH="${pkgs.lib.concatStringsSep ":" paths}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    '';
  in {
    devShells = {
      default = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs;
        env = integrationEnv devCuda;
        shellHook =
          devShellHook "default"
          + lspShadowHook
          + runtimeLibraryHook devCuda
          + ''
            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      tvm-python = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [tvmPython];
        env = integrationEnv buildConfigurations.dev-cuda-tvm-python;
        shellHook =
          devShellHook "tvm-python"
          + lspShadowHook
          + runtimeLibraryHook buildConfigurations.dev-cuda-tvm-python
          + ''
            export PYTHONPATH="$ZG_EXTERNAL_SDK_ROOT/python''${PYTHONPATH:+:$PYTHONPATH}"

            # zig is not happy about -fmacro-prefix-map
            unset NIX_CFLAGS_COMPILE
          '';
      };

      profiling = pkgs.mkShellNoCC {
        packages =
          baseDevShellPkgs
          ++ [
            pkgs.perf
            pkgs.gdb
            cudaPackages.nsight_systems
            cudaPackages.nsight_compute
          ];
        env = integrationEnv devCuda;
        shellHook =
          devShellHook "profiling"
          + lspShadowHook
          + runtimeLibraryHook devCuda;
      };

      mlir-cpp = pkgs.mkShellNoCC {
        packages = baseDevShellPkgs ++ [mlirCppExampleDev];
        env = integrationEnv devCuda;
        shellHook =
          devShellHook "mlir-cpp"
          + lspShadowHook
          + mlirCppExampleHook
          + runtimeLibraryHook devCuda;
      };
    };
  };
}
