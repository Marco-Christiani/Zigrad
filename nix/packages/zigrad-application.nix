# Build and package a Zig executable that imports Zigrad.
{
  autoAddDriverRunpath,
  autoPatchelfHook,
  callPackage,
  lib,
  makeWrapper,
  stdenvNoCC,
  zig,
  externalInputs,
  mainProgram,
  pname,
  src,
  sourceSubdir ? ".",
  zigradSrc,
  runtimePolicy ? {
    fixed = {};
    defaults = {};
    prefixes = {};
  },
  zigDependencySets ? [],
  needsCudaDriverRunpath ? false,
  targetPkgs ? null,
  zigArgs ? [],
  optimize ? "fast",
  runTests ? false,
  testProgram ? null,
  withRuntimeEnvironment ? true,
}: let
  compileInputs = externalInputs.compile;
  runtimeInputs = externalInputs.runtime;
  zigTarget =
    if targetPkgs == null
    then "native-native-gnu"
    else
      lib.concatStringsSep "-" (with targetPkgs.stdenv.hostPlatform.parsed; [
        cpu.name
        kernel.name
        abi.name
      ]);
  runtimeLibraryPaths = [
    "${runtimeInputs}/lib"
    "${runtimeInputs}/runtime/sys/lib"
  ];
  runtimeLibraryPath = lib.concatStringsSep ":" runtimeLibraryPaths;
  zigDeps = callPackage ./zig-dependencies.nix {
    requestedSets = zigDependencySets;
  };
  testProgramName =
    if testProgram != null
    then testProgram
    else "${mainProgram}-tests";
  runtimeWrapperArgs = lib.concatStringsSep " \\\n" (
    (lib.mapAttrsToList (
        name: value: "--set ${lib.escapeShellArg name} ${lib.escapeShellArg (toString value)}"
      )
      runtimePolicy.fixed)
    ++ (lib.mapAttrsToList (
        name: value: "--set-default ${lib.escapeShellArg name} ${lib.escapeShellArg (toString value)}"
      )
      runtimePolicy.defaults)
    ++ (lib.mapAttrsToList (
        name: value: "--prefix ${lib.escapeShellArg name} ' ' ${lib.escapeShellArg (toString value)}"
      )
      runtimePolicy.prefixes)
    ++ ["--prefix LD_LIBRARY_PATH : ${lib.escapeShellArg runtimeLibraryPath}"]
  );
in
  stdenvNoCC.mkDerivation {
    inherit pname;
    version = "dev";

    inherit src;

    strictDeps = true;

    nativeBuildInputs =
      [
        zig
        autoPatchelfHook
      ]
      ++ lib.optional withRuntimeEnvironment makeWrapper
      ++ lib.optional needsCudaDriverRunpath autoAddDriverRunpath;

    buildInputs = [compileInputs];
    runtimeDependencies = [runtimeInputs];

    configurePhase = ''
      runHook preConfigure
      export ZIG_GLOBAL_CACHE_DIR=$(mktemp -d)

      export ZG_ZIG_SYSTEM_PACKAGES="$TMPDIR/zigrad-system-packages"
      mkdir -p "$ZG_ZIG_SYSTEM_PACKAGES"
      cp -a ${zigDeps}/. "$ZG_ZIG_SYSTEM_PACKAGES/"
      chmod u+w "$ZG_ZIG_SYSTEM_PACKAGES"

      ln -s ${zigradSrc} ${lib.escapeShellArg sourceSubdir}/.zigrad-source
      (
        cd ${lib.escapeShellArg sourceSubdir}
        zig fetch --save-exact=zigrad .zigrad-source
      )
      runHook postConfigure
    '';

    dontBuild = true;

    checkPhase = ''
      runHook preCheck
      export ZG_ZIG_SYSTEM_PACKAGES="$TMPDIR/zigrad-system-packages"
      testRoot=$(mktemp -d)
      (
        cd ${lib.escapeShellArg sourceSubdir}
        TERM=dumb zig build test-compile \
          -j"$NIX_BUILD_CORES" \
          -Doptimize=${lib.escapeShellArg optimize} \
          -Dsdk=${compileInputs} \
          -Dtarget=${lib.escapeShellArg zigTarget} \
          ${lib.escapeShellArgs zigArgs} \
          --system "$ZG_ZIG_SYSTEM_PACKAGES" \
          --prefix "$testRoot" \
          --verbose
      )
      autoPatchelf "$testRoot/bin"
      "$testRoot/bin/${testProgramName}"
      runHook postCheck
    '';

    doCheck = runTests;

    installPhase = ''
      runHook preInstall
      export ZG_ZIG_SYSTEM_PACKAGES="$TMPDIR/zigrad-system-packages"
      cd ${lib.escapeShellArg sourceSubdir}
      TERM=dumb zig build install \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=${lib.escapeShellArg optimize} \
        -Dsdk=${compileInputs} \
        -Dtarget=${lib.escapeShellArg zigTarget} \
        ${lib.escapeShellArgs zigArgs} \
        --system "$ZG_ZIG_SYSTEM_PACKAGES" \
        --prefix "$out" \
        --verbose
      runHook postInstall
    '';

    postFixup =
      lib.optionalString (targetPkgs != null) ''
        patchelf --set-interpreter ${lib.escapeShellArg targetPkgs.stdenv.cc.bintools.dynamicLinker} "$out/bin/${mainProgram}"
      ''
      + lib.optionalString needsCudaDriverRunpath ''
        addDriverRunpath "$out/bin/${mainProgram}"
      ''
      + lib.optionalString withRuntimeEnvironment ''
        wrapProgram "$out/bin/${mainProgram}" \
          ${runtimeWrapperArgs}
      '';

    meta = {
      description = "Zigrad application ${pname}";
      # TODO(release): license, etc.
      license = lib.licenses.asl20;
      inherit mainProgram;
      platforms = lib.platforms.linux;
    };
  }
