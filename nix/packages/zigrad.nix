{
  callPackage,
  lib,
  stdenvNoCC,
  zig,
  autoAddDriverRunpath,
  autoPatchelfHook,
  makeWrapper,
  zigradSrc,
  compileInputs,
  runtimeInputs ? null,
  pname ? "zigrad",
  runtimeEnv ? {},
  runtimeEnvDefaults ? {},
  runtimeEnvPrefixes ? {},
  runtimeLibraryPaths ? [],
  passthru ? {},
  zigFeatureArgs ? [],
  zigDependencySets ? [],
  needsCudaDriverRunpath ? false,
  version ? "dev",
  optimize ? "fast",
  runTests ? false,
}: let
  zigDeps = callPackage ./zig-dependencies.nix {
    requestedSets = zigDependencySets;
  };
  featureFlags = lib.concatStringsSep " " ([
      "-Dversion=${version}"
    ]
    ++ zigFeatureArgs);
  runtimeLibraryPath = lib.concatStringsSep ":" runtimeLibraryPaths;
  runtimeWrapperArgs = lib.concatStringsSep " \\\n" (
    (lib.mapAttrsToList (
        name: value: "--set ${lib.escapeShellArg name} ${lib.escapeShellArg (toString value)}"
      )
      runtimeEnv)
    ++ (lib.mapAttrsToList (
        name: value: "--set-default ${lib.escapeShellArg name} ${lib.escapeShellArg (toString value)}"
      )
      runtimeEnvDefaults)
    ++ (lib.mapAttrsToList (
        name: value: "--prefix ${lib.escapeShellArg name} ' ' ${lib.escapeShellArg (toString value)}"
      )
      runtimeEnvPrefixes)
    ++ lib.optional (runtimeLibraryPaths != [])
    "--prefix LD_LIBRARY_PATH : ${lib.escapeShellArg runtimeLibraryPath}"
  );
  runtimeEnvExports = lib.concatStringsSep "\n" (
    lib.mapAttrsToList (
      name: value: "export ${name}=${lib.escapeShellArg (toString value)}"
    )
    (runtimeEnv // runtimeEnvDefaults // runtimeEnvPrefixes)
  );
  hasRuntimeWrapper =
    runtimeEnv
    != {}
    || runtimeEnvDefaults != {}
    || runtimeEnvPrefixes != {}
    || runtimeLibraryPaths != [];
in
  # `stdenvNoCC` is correct here. zig ships its own toolchain and pulling
  #  in nixpkgs's gcc/glibc via plain `stdenv` pollutes the include path
  #  enough to confuse zig's bundled libcxx. The libc choice is forced via
  #  `-Dtarget=native-native-gnu` below so zig uses its bundled glibc
  #  headers instead of falling back to musl.
  stdenvNoCC.mkDerivation {
    inherit pname version;
    src = zigradSrc;

    strictDeps = true;

    nativeBuildInputs =
      [
        zig
        # autoPatchelfHook gives installed executables store-backed ELF
        #  interpreters and dependency RUNPATHs.
        autoPatchelfHook
      ]
      ++ lib.optional needsCudaDriverRunpath autoAddDriverRunpath
      ++ lib.optional hasRuntimeWrapper makeWrapper;

    buildInputs = [compileInputs];
    runtimeDependencies = lib.optional (runtimeInputs != null) runtimeInputs;

    # Invoke Zig directly from the package phases.
    #
    # zig-overlay's binary install has no `setupHook`. The nixpkgs
    #  `zig.passthru.hook` also resolves to the Zig derivation itself.
    configurePhase = ''
      runHook preConfigure
      export ZIG_GLOBAL_CACHE_DIR=$(mktemp -d)
      runHook postConfigure
    '';

    dontBuild = true;

    checkPhase = ''
      runHook preCheck
      testRoot=$(mktemp -d)
      TERM=dumb zig build test-compile \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${optimize} \
        -Dsdk=${compileInputs} \
        -Dtarget=native-native-gnu \
        -Dinstall-runtime-link=false \
        ${featureFlags} \
        -freference-trace=10 \
        --system ${zigDeps} \
        --prefix "$testRoot" \
        --verbose
      autoPatchelf "$testRoot/bin"
      ${lib.optionalString (runtimeLibraryPaths != []) ''
        export LD_LIBRARY_PATH=${lib.escapeShellArg runtimeLibraryPath}
      ''}
      ${runtimeEnvExports}
      "$testRoot/bin/zigrad-tests"
      "$testRoot/bin/zigrad-cli-tests"
      runHook postCheck
    '';

    installPhase = ''
      runHook preInstall
      TERM=dumb zig build install \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${optimize} \
        -Dsdk=${compileInputs} \
        -Dtarget=native-native-gnu \
        -Dinstall-runtime-link=false \
        -Dgen-cli-meta=true \
        ${featureFlags} \
        -freference-trace=10 \
        --system ${zigDeps} \
        --prefix "$out" \
        --verbose
      runHook postInstall
    '';

    doCheck = runTests;

    postFixup =
      lib.optionalString needsCudaDriverRunpath ''
        addDriverRunpath "$out/bin/zigrad"
      ''
      + lib.optionalString hasRuntimeWrapper ''
        wrapProgram "$out/bin/zigrad" \
          ${runtimeWrapperArgs}
      '';

    inherit passthru;

    meta = {
      description = "Zigrad: differentiable computation framework";
      # TODO(release): license, etc.
      license = lib.licenses.asl20;
      mainProgram = "zigrad";
      platforms = lib.platforms.linux;
    };
  }
