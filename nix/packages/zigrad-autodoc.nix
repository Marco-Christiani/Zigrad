{
  callPackage,
  lib,
  stdenvNoCC,
  gnutar,
  zig,
  zigAutodocDocs,
  zigradSrc,
}: let
  zigDeps = callPackage ./zig-dependencies.nix {};
in
  stdenvNoCC.mkDerivation {
    pname = "zigrad-autodoc";
    version = "dev";
    src = zigradSrc;

    strictDeps = true;
    nativeBuildInputs = [gnutar zig];

    configurePhase = ''
      runHook preConfigure
      export ZIG_GLOBAL_CACHE_DIR=$(mktemp -d)
      export ZIG_LIB_DIR=${zigAutodocDocs}
      runHook postConfigure
    '';

    dontBuild = true;

    installPhase = ''
      runHook preInstall
      TERM=dumb zig build docs \
        -j"$NIX_BUILD_CORES" \
        -Doptimize=fast \
        -Dtarget=native-native-gnu \
        --system ${zigDeps} \
        --prefix "$out"

      autodoc_dir="$out/autodoc"
      sources_tar="$autodoc_dir/sources.tar"
      sources_tmp=$(mktemp -d)
      mkdir "$sources_tmp/extracted"
      tar -xf "$sources_tar" -C "$sources_tmp/extracted"

      # prune stdlib et al w/ a whitelist policy
      for source_root in zigrad safetensors_zg build_options build_options0; do
        if [ ! -e "$sources_tmp/extracted/$source_root" ]; then
          echo "missing expected autodoc source root: $source_root" >&2
          exit 1
        fi
      done

      tar \
        --create \
        --sort=name \
        --mtime='UTC 1970-01-01' \
        --owner=0 \
        --group=0 \
        --numeric-owner \
        --file "$sources_tmp/sources.tar" \
        --directory "$sources_tmp/extracted" \
        zigrad safetensors_zg build_options build_options0
      mv "$sources_tmp/sources.tar" "$sources_tar"
      runHook postInstall
    '';

    passthru = {
      inherit (zigDeps) autodocSources;
      zigVersion = zig.version;
    };

    meta = {
      description = "Zigrad Zig autodoc bundle";
      license = lib.licenses.asl20;
      platforms = lib.platforms.linux;
    };
  }
