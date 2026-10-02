{
  lib,
  stdenv,
  cmake,
  ninja,
  python3,
  patch,
  # Standard nixpkgs LLVM (used when llvm is null)
  llvmPackages ? null,
  # Shared LLVM used when TVM loads with other compiler integrations.
  llvm ? null,
  pkg-config,
  git,
  patchelf,
  zlib,
  ncurses,
  libxml2,
  autoAddDriverRunpath,
  # Build-time CUDA toolkit (cudatoolkit-style layout: bin/nvcc, include/,
  #  link-time libs in lib/, lib/stubs/libcuda.so). From cuda-redist.dev.
  cudaToolkit ? null,
  # Runtime CUDA layout, pulled into buildInputs so autoPatchelfHook resolves
  #  libtvm.so's NEEDED libs (libcudart, libcuda, libnvrtc) against this path
  #  and the resulting rpath references runtime artifacts only. This keeps
  #  cudaToolkit (build-time) out of TVM's runtime closure. From cuda-redist.out.
  cudaRuntime ? null,
  gccHost ? null,
  cudaSupport ? true,
  cudaArchitectures ? [],
  enableCublas ? false,
  enableCudnn ? false,
  enableCutlass ? false,
  # Build and install TVM's Python package in a separate output.
  withPythonBindings ? false,
  # When true: RelWithDebInfo, retain DWARF, don't strip.
  # When false (default, production): Release, NDEBUG, stripped.
  withDebugSymbols ? false,
  # Native CPU codegen for TVM's host-side runtime. Default true for TVM
  #  specifically, ships fused/lowered kernels and benefits notably
  #  from native tuning on the build host.
  withNativeTuning ? false,
  enableLto ? false,
  extraCxxFlags ? [],
  extraLdFlags ? [],
  src,
  version,
}: let
  boolToCmake = v:
    if v
    then "ON"
    else "OFF";
  cudaEnabled = cudaSupport && cudaToolkit != null;
  cudaArchStr = lib.concatStringsSep ";" cudaArchitectures;

  # Use the shared LLVM build when provided.
  # When using nixpkgs llvmPackages, use static linking to isolate symbols.
  useCustomLlvm = llvm != null;
  llvmDev =
    if useCustomLlvm
    then llvm
    else llvmPackages.llvm.dev;
  llvmLib =
    if useCustomLlvm
    then llvm
    else llvmPackages.llvm.lib;
  llvmConfigCmd =
    if useCustomLlvm
    then "${llvm}/bin/llvm-config" # Shared linking
    else "${llvmPackages.llvm.dev}/bin/llvm-config --link-static"; # Static isolation
in
  assert lib.assertMsg (!cudaSupport || cudaToolkit != null) "tvm: cudaSupport=true requires cudaToolkit";
  assert lib.assertMsg (llvm != null || llvmPackages != null) "tvm: requires either llvm or llvmPackages";
    stdenv.mkDerivation {
      pname = "tvm";
      inherit src version;
      outputs =
        ["out" "dev"]
        ++ lib.optional withPythonBindings "python";

      strictDeps = true;
      # Strip libtvm.so etc. so debug-info path strings (e.g. cmake's CUDA
      #  include dir from cudaToolkit = cuda-redist.dev) don't leave
      #  references to dev paths in the runtime closure. Flip via
      #  withDebugSymbols=true if you need DWARF for backtraces.
      dontStrip = withDebugSymbols;

      postPatch = ''
        ${lib.optionalString useCustomLlvm ''
          # Apply LLVM 22 API compatibility patch
          patch -p1 < ${../patches/tvm-llvm22.patch}
        ''}

        ${lib.optionalString withPythonBindings ''
          # Enable Python module build in tvm-ffi subproject
          # By default, tvm-ffi skips Python module when used as a subdirectory
          patch -p1 < ${../patches/tvm-ffi-python.patch}
        ''}
      '';

      nativeBuildInputs =
        [
          cmake
          ninja
          python3
          pkg-config
          git
          patchelf
          patch
          llvmDev # Provides llvm-config (either custom LLVM or llvmPackages.llvm.dev)
        ]
        ++ lib.optional withPythonBindings python3.pkgs.cython
        ++ lib.optionals cudaEnabled [
          autoAddDriverRunpath # automatically patches rpath to include /run/opengl-driver for libcuda.so
          # nvcc invokes cudafe++/cicc/ptxas/etc. as bare command names via PATH.
          #  cudaToolkit must be in nativeBuildInputs so its bin/ lands on PATH
          #  during build (buildInputs only contributes library paths).
          cudaToolkit
        ];

      buildInputs =
        [
          # LLVM runtime library (either custom LLVM or llvmPackages.llvm.lib)
          llvmLib
          # LLVM's dependencies (required for linking)
          zlib
          ncurses # provides libtinfo
          libxml2
        ]
        ++ lib.optionals cudaEnabled (
          # Order matters: cudaRuntime FIRST so autoPatchelfHook resolves NEEDED
          #  libs (libcudart, libcuda, libnvrtc) against the runtime layout,
          #  baking that path into libtvm.so's rpath. cudaToolkit second so
          #  build-time linking still has access to its libs (cudaRuntime is a
          #  symlink farm into the same files, but cudaToolkit also has the
          #  static .a archives that cmake's CUDA probe needs).
          lib.optional (cudaRuntime != null) cudaRuntime
          ++ [cudaToolkit]
        );

      # Configure the patched source tree in place.
      configurePhase = ''
        set -euo pipefail

        mkdir -p build
        cp cmake/config.cmake build/config.cmake
        chmod u+w build/config.cmake

        cat >> build/config.cmake <<EOF
        set(CMAKE_BUILD_TYPE ${
          if withDebugSymbols
          then "RelWithDebInfo"
          else "Release"
        })
        set(CMAKE_CXX_STANDARD 17)
        set(USE_LLVM "${llvmConfigCmd}")
        set(HIDE_PRIVATE_SYMBOLS ON)
        set(USE_CUDA ${boolToCmake cudaEnabled})
        set(USE_METAL OFF)
        set(USE_VULKAN OFF)
        set(USE_OPENCL OFF)
        set(USE_CUBLAS ${boolToCmake enableCublas})
        set(USE_CUDNN ${boolToCmake enableCudnn})
        set(USE_CUTLASS ${boolToCmake enableCutlass})
        ${lib.optionalString withPythonBindings "set(TVM_FFI_BUILD_PYTHON_MODULE ON)"}
        ${lib.optionalString enableLto "set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)"}
        ${lib.optionalString (withNativeTuning || extraCxxFlags != []) ''
          set(CMAKE_CXX_FLAGS "${lib.concatStringsSep " " ((lib.optionals withNativeTuning ["-march=native" "-mtune=native"]) ++ extraCxxFlags)} \''${CMAKE_CXX_FLAGS}")
        ''}
        ${lib.optionalString (extraLdFlags != []) ''
          set(CMAKE_SHARED_LINKER_FLAGS "${lib.concatStringsSep " " extraLdFlags} \''${CMAKE_SHARED_LINKER_FLAGS}")
        ''}
        EOF

        ${lib.optionalString cudaEnabled ''
          cat >> build/config.cmake <<EOF
          set(CUDAToolkit_ROOT "${cudaToolkit}")
          set(CUDA_TOOLKIT_ROOT_DIR "${cudaToolkit}")
          set(CMAKE_CUDA_COMPILER "${cudaToolkit}/bin/nvcc")
          set(CUDA_CUDA_LIBRARY "${cudaToolkit}/lib/stubs/libcuda.so")
          EOF

          # CMake 4.x requires CMAKE_CUDA_ARCHITECTURES set before
          #  enable_language(CUDA). Use the caller's pin if provided, else
          #  cmake's "all-major" sentinel which compiles for representative
          #  archs across recent generations. TVM doesn't AOT-compile CUDA
          #  itself (kernels go through NVRTC at runtime), so this only
          #  affects cmake's toolchain probe, not the produced artifact.
          cat >> build/config.cmake <<EOF
          set(CMAKE_CUDA_ARCHITECTURES "${
            if cudaArchitectures != []
            then cudaArchStr
            else "all-major"
          }")
          EOF

          ${lib.optionalString (gccHost != null) ''
            cat >> build/config.cmake <<EOF
            set(CMAKE_CUDA_HOST_COMPILER "${gccHost}/bin/g++")
            EOF
          ''}
        ''}

        cmake -S . -B build -G Ninja

        # Restore -u to default before phase exits so it doesn't leak into
        #  later phases (e.g. fixupPhase's strip-hook references an exit_code
        #  that's unset on some paths and would fail under -u).
        set +u
      '';

      buildPhase = ''
        cmake --build build --parallel $NIX_BUILD_CORES
      '';

      installPhase = ''
        set -euo pipefail

        mkdir -p $out/lib $dev/include

        for f in build/libtvm* build/libtvm_runtime*; do
          if [ -e "$f" ]; then
            cp -v "$f" $out/lib/
          fi
        done

        # Copy tvm_ffi_testing.so if it exists (needed by Python Cython extension)
        echo "Searching for libtvm_ffi_testing.so..."
        find build -name "libtvm_ffi_testing.so*" -type f
        testing_lib="$(find build -name "libtvm_ffi_testing.so" -type f | head -n1 || true)"
        if [ -n "$testing_lib" ] && [ -f "$testing_lib" ]; then
          echo "Found testing library at: $testing_lib"
          cp -v "$testing_lib" $out/lib/
        else
          echo "WARNING: libtvm_ffi_testing.so not found in build directory"
          echo "This may cause Python imports to fail if core.abi3.so depends on it"
        fi

        # TVM FFI may place its shared library under its subproject build tree.
        if [ ! -e "$out/lib/libtvm_ffi.so" ]; then
          found="$(find build -type f -name 'libtvm_ffi.so*' | head -n1 || true)"
          if [ -n "$found" ]; then
            cp -v "$found" $out/lib/
          fi
        fi

        cp -r --no-preserve=mode,ownership include/. $dev/include/
        chmod -R u+w $dev/include

        if [ -d 3rdparty/tvm-ffi/include ]; then
          cp -r --no-preserve=mode,ownership 3rdparty/tvm-ffi/include/. $dev/include/
        fi

        # DLPack headers are a TVM dependency used by the runtime C API.
        if [ -d 3rdparty/dlpack/include/dlpack ]; then
          mkdir -p $dev/include/dlpack
          cp -r 3rdparty/dlpack/include/dlpack/. $dev/include/dlpack/
        elif [ -d 3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack ]; then
          mkdir -p $dev/include/dlpack
          cp -r 3rdparty/tvm-ffi/3rdparty/dlpack/include/dlpack/. $dev/include/dlpack/
        fi

        # DMLC headers are required by TVM C++ APIs.
        if [ -d 3rdparty/dmlc-core/include/dmlc ]; then
          mkdir -p $dev/include/dmlc
          cp -r 3rdparty/dmlc-core/include/dmlc/. $dev/include/dmlc/
        fi

        # Build rpath for TVM libs.
        # - Include LLVM library in rpath so libtvm.so finds libLLVM.so.
        # - Always include LLVM's runtime deps (zlib, ncurses, libxml2).
        # - Note: libcuda.so.1 is provided by the driver and resolved via autoAddDriverRunpath or LD_LIBRARY_PATH.
        # Use cudaRuntime (cuda-redist.out's flat lib symlink farm) for the
        #  rpath, NOT cudaToolkit (= cuda-redist.dev). cudaToolkit is a build-time
        #  artifact (headers + nvcc + static .a archives). Pointing libtvm.so's
        #  rpath at it would drag dev into the runtime closure unnecessarily.
        rpath="\$ORIGIN:${lib.makeLibraryPath (
          [stdenv.cc.cc.lib zlib ncurses libxml2 llvmLib]
          ++ lib.optional (cudaEnabled && cudaRuntime != null) cudaRuntime
          ++ lib.optional (cudaEnabled && cudaRuntime == null) cudaToolkit
        )}"
        for f in $out/lib/*.so*; do
          [ -e "$f" ] || continue
          patchelf --set-rpath "$rpath" "$f"
        done

        ${lib.optionalString withPythonBindings ''
          echo "Installing Python bindings"
          mkdir -p $python/python

          cp -r python/tvm $python/python/

          if [ -d 3rdparty/tvm-ffi/python/tvm_ffi ]; then
            cp -r 3rdparty/tvm-ffi/python/tvm_ffi $python/python/
          fi

          echo "Searching for Cython core extension..."
          find build -name "core*.so" -type f
          core_so="$(find build -name "core*.so" -type f | head -n1 || true)"
          if [ -n "$core_so" ] && [ -f "$core_so" ]; then
            cp -v "$core_so" $python/python/tvm_ffi/

            patchelf --set-rpath "$out/lib:$rpath" "$python/python/tvm_ffi/$(basename "$core_so")"

            echo "Installed Cython core extension: $core_so"
          else
            echo "TVM FFI Python extension was not built" >&2
            exit 1
          fi

          chmod -R u+w $python/python
        ''}

        # Restore -u to default so fixupPhase's strip-hook (which references an
        #  exit_code variable unset on some paths) doesn't trip.
        set +u
      '';

      meta = {
        description = "Apache TVM compiler stack";
        # TODO(release): license, etc.
        license = lib.licenses.asl20;
        platforms = lib.platforms.linux;
      };
    }
