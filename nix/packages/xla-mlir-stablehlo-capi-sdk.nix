# nix/xla-mlir-stablehlo-capi-sdk.nix
#
# Builds the StableHLO C API against a pre-built LLVM/MLIR (from llvm.nix).
# Sources come from the demand-driven external dependency snapshot.
# XLA patches for StableHLO are applied from xlaSrc/third_party/.
{
  lib,
  stdenv,
  runCommand,
  writeText,
  cmake,
  ninja,
  python3,
  perl,
  patch,
  patchelf,
  zlib,
  zstd,
  libxml2,
  ncurses,
  libedit,
  libffi,
  lld,
  binutils,
  # Source inputs.
  xlaSrc,
  xlaRevision,
  stablehloSrc,
  # Pre-built LLVM/MLIR from llvm.nix (shared with TVM).
  llvm,
  # When true: RelWithDebInfo, retain DWARF, don't strip.
  # When false (default, production): Release, NDEBUG, stripped.
  withDebugSymbols ? false,
  # Native CPU codegen has limited effect on the StableHLO C API library.
  withNativeTuning ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
}: let
  stablehloPatches = ["temporary.patch"];

  patchedStablehloSrc = runCommand "stablehlo-src-patched" {nativeBuildInputs = [patch perl];} ''
    set -euo pipefail
    cp -r ${stablehloSrc} "$out"
    chmod -R u+w "$out"
    cd "$out"

    echo "[stablehlo] Verifying patch set"
    expected_patches="${lib.concatStringsSep " " stablehloPatches}"
    actual_patches="$(cd ${xlaSrc}/third_party/stablehlo && ls *.patch | tr '\n' ' ')"
    for p in $actual_patches; do
      case " $expected_patches " in
        *" $p "*) ;;
        *) echo "ERROR: New or unexpected StableHLO patch detected: $p" >&2; exit 1;;
      esac
    done
    for p in $expected_patches; do
      echo "[stablehlo] Applying $p"
      patch -p1 < "${xlaSrc}/third_party/stablehlo/$p"
    done

    # Force ONLY StablehloCAPI to be SHARED without turning on BUILD_SHARED_LIBS globally for StableHLO.
    # This relies on MLIR's add_mlir_public_c_api_library forwarding args into add_mlir_library which supports SHARED.
    if [ -f stablehlo/integrations/c/CMakeLists.txt ]; then
      perl -pi -e 's/add_mlir_public_c_api_library\(StablehloCAPI/add_mlir_public_c_api_library(StablehloCAPI SHARED/' \
        stablehlo/integrations/c/CMakeLists.txt
    fi

    # Sanity check
    grep -n "add_mlir_public_c_api_library(StablehloCAPI" -n stablehlo/integrations/c/CMakeLists.txt || true
    grep -n "StablehloCAPI SHARED" stablehlo/integrations/c/CMakeLists.txt || true

    echo "[stablehlo] post patch stablehlo/integrations/c/CMakeLists.txt:"
    cat stablehlo/integrations/c/CMakeLists.txt
  '';

  # Shell helpers shared by installPhase, postFixup, and out-of-tree checks.
  # `copy_lib` copies a library into a directory without dereferencing
  # symlinks, follows the alias chain to its terminal regular file, and
  # records each source alias it visits in a manifest, whatever the
  # destination holds. `validate_aliases` checks a library directory against
  # such a manifest.
  aliasLib = writeText "xla-mlir-stablehlo-capi-sdk-aliases.sh" ''
    alias_err() { echo "[xla-mlir-stablehlo-capi-sdk] ERROR: $*" >&2; }

    # record_alias MANIFEST LINK: append "name<TAB>target" for a source symlink.
    record_alias() {
      printf '%s\t%s\n' "$(basename "$2")" "$(readlink "$2")" >> "$1"
    }

    # copy_lib SRC DESTDIR MANIFEST: copy SRC and, when it is a symlink, every
    # link in its chain plus the terminal file. Targets must be bare file names
    # so each alias resolves inside DESTDIR.
    copy_lib() {
      local cur="$1" dest="$2" manifest="$3" base target
      while :; do
        base="$(basename "$cur")"
        if [ -L "$cur" ]; then
          target="$(readlink "$cur")"
          case "$target" in
            "" | */*)
              alias_err "alias $cur has non-local target '$target'"
              return 1
              ;;
          esac
          record_alias "$manifest" "$cur"
          if [ ! -e "$dest/$base" ] && [ ! -L "$dest/$base" ]; then
            cp -P "$cur" "$dest/$base"
          fi
          cur="$(dirname "$cur")/$target"
        else
          if [ ! -f "$cur" ]; then
            alias_err "alias chain ends at missing file $cur"
            return 1
          fi
          if [ ! -e "$dest/$base" ]; then
            cp "$cur" "$dest/$base"
          fi
          return 0
        fi
      done
    }

    # validate_aliases LIBDIR MANIFEST: every manifest entry must exist as a
    # symlink with the recorded local target that resolves to a regular file
    # inside LIBDIR, and LIBDIR holds no dangling symlink.
    validate_aliases() {
      local libdir="$1" manifest="$2" root name target path got real rc=0 count=0
      root="$(realpath "$libdir")"
      if [ ! -f "$manifest" ]; then
        alias_err "missing alias manifest $manifest"
        return 1
      fi
      while IFS="$(printf '\t')" read -r name target; do
        count=$((count + 1))
        path="$libdir/$name"
        if [ ! -e "$path" ] && [ ! -L "$path" ]; then
          alias_err "$libdir: expected alias $name is missing"
          rc=1
          continue
        fi
        if [ ! -L "$path" ]; then
          alias_err "$libdir: $name is a regular file, expected symlink to $target"
          rc=1
          continue
        fi
        got="$(readlink "$path")"
        if [ "$got" != "$target" ]; then
          alias_err "$libdir: $name points to '$got', expected '$target'"
          rc=1
          continue
        fi
        if ! real="$(realpath -e "$path" 2>/dev/null)"; then
          alias_err "$libdir: $name is a dangling symlink to $target"
          rc=1
          continue
        fi
        case "$real" in
          "$root"/*) ;;
          *)
            alias_err "$libdir: $name resolves outside the output to $real"
            rc=1
            continue
            ;;
        esac
        if [ ! -f "$real" ] || [ -L "$real" ]; then
          alias_err "$libdir: $name does not resolve to a regular file"
          rc=1
        fi
      done < "$manifest"
      while IFS= read -r path; do
        alias_err "$libdir: dangling symlink $(basename "$path")"
        rc=1
      done < <(find "$libdir" -maxdepth 1 -xtype l)
      if [ "$rc" -eq 0 ]; then
        echo "[xla-mlir-stablehlo-capi-sdk] $libdir: $count aliases valid" >&2
      fi
      return "$rc"
    }
  '';
in
  stdenv.mkDerivation {
    pname = "xla-mlir-stablehlo-capi-sdk";
    version = "xla-${builtins.substring 0 7 xlaRevision}";

    # out: headers + minimal (DT_NEEDED) lib closure for runtime use.
    # dev: full lib closure (all StableHLO + LLVM/MLIR libs) for header
    #      navigation, linking against extra MLIR libs, debugger symbol
    #      resolution. Strict superset of `out`'s lib set.
    outputs = ["out" "dev"];

    strictDeps = true;
    dontUnpack = true;
    dontConfigure = true;
    dontStrip = withDebugSymbols;

    passthru = {inherit aliasLib;};

    nativeBuildInputs = [
      cmake
      ninja
      python3
      patchelf
      lld
      binutils

      # host-tool runtime deps (strictDeps)
      stdenv.cc.cc.lib
      zlib
      zstd
    ];

    buildInputs = [
      zlib
      zstd
      libxml2
      ncurses
      libedit
      libffi
    ];

    # Build StableHLO against the LLVM and MLIR package supplied above.
    buildPhase = ''
      set -euo pipefail

      mkdir -p stablehlo-build
      cxxFlags="${lib.concatStringsSep " " ((lib.optionals withNativeTuning ["-march=native" "-mtune=native"]) ++ extraCxxFlags)}"
      ldFlags="${lib.concatStringsSep " " extraLdFlags}"
      cmake -S ${patchedStablehloSrc} -B stablehlo-build -G Ninja \
        -DCMAKE_BUILD_TYPE=${
        if withDebugSymbols
        then "RelWithDebInfo"
        else "Release"
      } \
        -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
        -DMLIR_DIR="${llvm}/lib/cmake/mlir" \
        -DLLVM_DIR="${llvm}/lib/cmake/llvm" \
        ${lib.optionalString enableLto "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON"} \
        ''${cxxFlags:+-DCMAKE_CXX_FLAGS="$cxxFlags"} \
        ''${ldFlags:+-DCMAKE_SHARED_LINKER_FLAGS="$ldFlags"}

      cmake --build stablehlo-build \
        --parallel "$NIX_BUILD_CORES" \
        --target StablehloCAPI

      # Restore -u to default so fixupPhase's strip-hook doesn't trip on its
      #  own unset variable references when stripping kicks in.
      set +u
    '';

    installPhase = ''
      set -euo pipefail
      log() { echo "[xla-mlir-stablehlo-capi-sdk] $*" >&2; }

      source ${aliasLib}
      out_aliases="$NIX_BUILD_TOP/aliases-out.tsv"
      dev_aliases="$NIX_BUILD_TOP/aliases-dev.tsv"
      : > "$out_aliases"
      : > "$dev_aliases"

      mkdir -p "$out/include" "$out/lib"

      # --- Headers ---
      # MLIR C API and internal headers from pre-built LLVM
      cp -r "${llvm}/include/mlir-c" "$out/include/mlir-c"
      if [ -d "${llvm}/include/mlir" ]; then
        cp -r "${llvm}/include/mlir" "$out/include/mlir"
      fi

      # StableHLO C API headers
      mkdir -p "$out/include/stablehlo/integrations/c"
      cp -r "${patchedStablehloSrc}/stablehlo/integrations/c/"* "$out/include/stablehlo/integrations/c/"

      # XLA PJRT C API headers
      mkdir -p "$out/include/xla/pjrt/c"
      cp -v "${xlaSrc}/xla/pjrt/c/"*.h "$out/include/xla/pjrt/c/"

      # XLA FFI C API headers used by custom-call handlers.
      mkdir -p "$out/include/xla/ffi/api"
      cp -v "${xlaSrc}/xla/ffi/api/"*.h "$out/include/xla/ffi/api/"

      # --- Libraries (out): minimal DT_NEEDED closure for runtime ---
      log "Copying MLIR-C + StablehloCAPI and DT_NEEDED closure to out"

      # Copy the two root DSOs
      for f in "${llvm}/lib/libMLIR-C.so"* stablehlo-build/lib/libStablehloCAPI.so*; do
        if [ -e "$f" ] || [ -L "$f" ]; then
          copy_lib "$f" "$out/lib" "$out_aliases"
        fi
      done

      # Copy transitive deps from pre-built LLVM and StableHLO build tree.
      copy_needed_closure() {
        local search_dirs=("${llvm}/lib" "$PWD/stablehlo-build/lib")
        local -A seen
        local queue=("$@")

        while [ "''${#queue[@]}" -gt 0 ]; do
          local libpath="''${queue[0]}"
          queue=("''${queue[@]:1}")

          [ -f "$libpath" ] || continue
          local base="$(basename "$libpath")"
          if [ -n "''${seen[$base]:-}" ]; then
            continue
          fi
          seen["$base"]=1

          # Ensure it's present in output
          copy_lib "$libpath" "$out/lib" "$out_aliases"

          # Enqueue internal deps that we can find in search_dirs
          while IFS= read -r need; do
            local found=""
            for d in "''${search_dirs[@]}"; do
              if [ -f "$d/$need" ]; then
                found="$d/$need"
                break
              fi
            done
            if [ -n "$found" ]; then
              queue+=("$found")
            fi
          done < <(patchelf --print-needed "$libpath" || true)
        done
      }

      mlir_root="$(ls -1 "${llvm}/lib/libMLIR-C.so."* 2>/dev/null | head -n1 || true)"
      stablehlo_root="$(ls -1 stablehlo-build/lib/libStablehloCAPI.so.* 2>/dev/null | head -n1 || true)"
      copy_needed_closure "$mlir_root" "$stablehlo_root"

      # --- Libraries (dev): full closure for header navigation / extra linking ---
      log "Copying ALL StableHLO + LLVM/MLIR build artifacts to dev"
      mkdir -p "$dev/lib"
      while IFS= read -r f; do
        copy_lib "$f" "$dev/lib" "$dev_aliases"
      done < <(find stablehlo-build \( -type f -o -type l \) \( -name "*.a" -o -name "*.so*" \))
      for f in "${llvm}/lib/"*.so* "${llvm}/lib/"*.a; do
        if [ -f "$f" ] || [ -L "$f" ]; then
          copy_lib "$f" "$dev/lib" "$dev_aliases"
        fi
      done

      # Make copied libraries writable before patching their ELF metadata.
      chmod -R u+w "$out/lib" "$dev/lib"

      # Ensure unversioned linker names exist for the two link-entry DSOs in both outputs.
      for libdir in "$out/lib" "$dev/lib"; do
        manifest="$out_aliases"
        [ "$libdir" = "$dev/lib" ] && manifest="$dev_aliases"
        for name in libMLIR-C libStablehloCAPI; do
          if [ ! -e "$libdir/$name.so" ] && [ ! -L "$libdir/$name.so" ]; then
            so="$(ls -1 "$libdir/$name.so."* 2>/dev/null | head -n1 || true)"
            if [ -n "$so" ]; then
              ln -s "$(basename "$so")" "$libdir/$name.so"
              record_alias "$manifest" "$libdir/$name.so"
            fi
          fi
        done
      done

      # Patch RUNPATH for all shipped DSOs in both outputs.
      rpath="\$ORIGIN:\$ORIGIN/../runtime/sys/lib:${
        lib.makeLibraryPath [
          zlib
          zstd
          libxml2
          ncurses
          libedit
          libffi
          stdenv.cc.cc.lib
        ]
      }"

      patch_rpath_in() {
        local libdir="$1"
        for f in "$libdir/"*.so*; do
          # aliases share their target's ELF; patch each regular file once
          [ -L "$f" ] && continue
          [ -f "$f" ] || continue
          if ! patchelf --set-rpath "$rpath" "$f"; then
            case "$(basename "$f")" in
              libMLIR-C.so*|libStablehloCAPI.so*)
                log "ERROR: patchelf failed for critical DSO: $f"
                exit 1
                ;;
              *)
                log "WARNING: patchelf failed for non-critical DSO: $f"
                ;;
            esac
          fi
        done
      }
      patch_rpath_in "$out/lib"
      patch_rpath_in "$dev/lib"

      # Sanity check... C API dialect handle function should be present when StablehloCAPI is built properly.
      if [ -f "$out/lib/libStablehloCAPI.so" ]; then
        if ! nm -D "$out/lib/libStablehloCAPI.so" 2>/dev/null | grep -q "mlirGetDialectHandle__stablehlo__"; then
          log "WARNING: mlirGetDialectHandle__stablehlo__ not found in libStablehloCAPI.so (symbol export may differ by version/config)"
        fi
      fi

      validate_aliases "$out/lib" "$out_aliases"
      validate_aliases "$dev/lib" "$dev_aliases"

      log "SDK installation complete"

      # Restore -u to default so fixupPhase's strip-hook doesn't trip.
      set +u
    '';

    # fixupPhase strips and rewrites ELF files in both outputs; recheck the
    # alias layout against the manifests captured from the sources.
    postFixup = ''
      source ${aliasLib}
      validate_aliases "$out/lib" "$NIX_BUILD_TOP/aliases-out.tsv"
      validate_aliases "$dev/lib" "$NIX_BUILD_TOP/aliases-dev.tsv"
    '';

    meta = {
      description = "MLIR + StableHLO + PJRT C API SDK";
      # TODO(release): license, etc.
      license = lib.licenses.asl20;
    };
  }
