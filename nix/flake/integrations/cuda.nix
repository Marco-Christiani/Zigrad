{
  pkgs,
  cudaCfg,
}: let
  inherit (pkgs) lib;
  cudaPackages = pkgs.${cudaCfg.cudaPackagesAttr};
  gccHost = cudaPackages.backendStdenv.cc;
  mkCudaPackage = options:
    pkgs.callPackage ../../packages/cuda-redist.nix (
      {inherit (cudaCfg) cudaVersion;}
      // options
    );
  cudaRuntime = mkCudaPackage {
    componentNames = [
      "cuda_cudart"
      "cuda_nvrtc"
      "libnvjitlink"
    ];
    includeSystemRuntime = true;
    nameSuffix = "-runtime";
  };
  # nvcc keeps nixpkgs' backend wrapper and patched compiler headers. Its
  #  first profile header assignment supplies CMake's discovered include paths.
  nvcc = cudaPackages.cuda_nvcc.overrideAttrs (old: {
    postInstall =
      (old.postInstall or "")
      + ''
        substituteInPlace "''${!outputBin:?}/bin/nvcc.profile" \
          --replace-fail \
            "\"-I''${!outputInclude:?}/include\"" \
            "\"-I''${!outputInclude:?}/include\" \"-I''${!outputBin:?}/nvvm/include\" \"-I${cudaPackages.cuda_cudart}/include\" \"-I${cudaPackages.cuda_cccl}/include\""
        cat >> "''${!outputBin:?}/bin/nvcc.profile" <<'EOF'
        LIBRARIES =+ $(_SPACE_) "-L${cudaPackages.cuda_cudart}/lib/stubs" "-L${cudaPackages.cuda_cudart}/lib"
        EOF
      '';
  });
  toolkit =
    pkgs.runCommand "cuda-${cudaCfg.cudaVersion}-toolkit" {
      nativeBuildInputs = [pkgs.cmake];
      passthru = {inherit gccHost;};
    } ''
      set -euo pipefail
      mkdir -p "$out"/{bin,include,lib/stubs,nvvm}

      # Merge directories and reject conflicting files before creating links.
      merge_tree() {
        local source="$1" destination="$2" entry relative target
        test -d "$source"
        while IFS= read -r -d $'\0' entry; do
          relative="''${entry#"$source/"}"
          target="$destination/$relative"
          if [ -d "$entry" ]; then
            if [ -e "$target" ] && [ ! -d "$target" ]; then
              echo "CUDA directory collision: $target from $entry" >&2
              exit 1
            fi
            mkdir -p "$target"
          elif [ -e "$target" ] || [ -L "$target" ]; then
            if ! cmp -s "$entry" "$target"; then
              echo "CUDA file collision: $target from $entry" >&2
              exit 1
            fi
          else
            ln -s "$entry" "$target"
          fi
        done < <(find -L "$source" -mindepth 1 -print0 | sort -z)
      }

      merge_tree ${nvcc}/bin "$out/bin"
      merge_tree ${nvcc}/nvvm "$out/nvvm"
      merge_tree ${nvcc}/include "$out/include"
      merge_tree ${cudaPackages.cuda_cudart}/include "$out/include"
      merge_tree ${cudaPackages.cuda_cudart}/lib "$out/lib"
      merge_tree ${cudaPackages.cuda_cccl}/include "$out/include"
      merge_tree ${lib.getOutput "include" cudaPackages.cuda_nvrtc}/include "$out/include"
      merge_tree ${lib.getLib cudaPackages.cuda_nvrtc}/lib "$out/lib"
      merge_tree ${lib.getOutput "include" cudaPackages.libnvjitlink}/include "$out/include"
      merge_tree ${lib.getLib cudaPackages.libnvjitlink}/lib "$out/lib"

      # Package setup hooks assemble CMake roots and compiler flags. This flat
      #  toolkit supplies those paths through its layout and nvcc profile.
      for required in bin/nvcc bin/nvcc.profile bin/cudafe++ nvvm/bin/cicc \
        nvvm/libdevice/libdevice.10.bc include/cuda.h include/cuda_runtime.h \
        include/crt/math_functions.h include/nv/target lib/libcudart.so \
        lib/libcudart_static.a lib/libcudadevrt.a lib/stubs/libcuda.so \
        lib/libnvrtc.so lib/libnvJitLink.so; do
        test -s "$out/$required"
      done

      mkdir -p "$TMPDIR/cmake-probe"
      cat > "$TMPDIR/cmake-probe/CMakeLists.txt" <<'EOF'
      cmake_minimum_required(VERSION 3.24)
      project(cuda_toolkit_discovery LANGUAGES NONE)
      find_package(CUDAToolkit REQUIRED)
      foreach(header IN ITEMS cuda.h cuda_runtime.h crt/math_functions.h nv/target)
        unset(header_path)
        find_file(header_path "''${header}"
          PATHS ''${CUDAToolkit_INCLUDE_DIRS}
          NO_DEFAULT_PATH NO_CACHE REQUIRED)
      endforeach()
      EOF
      PATH="$out/bin:$PATH" cmake \
        -S "$TMPDIR/cmake-probe" -B "$TMPDIR/cmake-probe/build"
    '';
in {
  inherit gccHost cudaRuntime mkCudaPackage;
  cudaToolkit = toolkit;
}
