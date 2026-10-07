{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/bbbf018e74b80af4a4692a4b3fca8e91bb0ed7ee";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    zig-overlay = {
      url = "github:mitchellh/zig-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pog = {
      url = "github:jpetrucciani/pog/d5327fff16292e065d4263df10f20cd1f2688836";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {
    flake-parts,
    treefmt-nix,
    ...
  }: let
    zigVersion = "0.17.0";
    zigOverlay = final: prev: {
      zig = inputs.zig-overlay.packages.${prev.stdenv.hostPlatform.system}.${zigVersion};
      zls = prev.stdenvNoCC.mkDerivation (finalAttrs: {
        pname = "zls";
        version = "0.16.0";

        src = let
          assets = {
            x86_64-linux = {
              url = "https://github.com/zigtools/zls/releases/download/${finalAttrs.version}/zls-x86_64-linux.tar.xz";
              hash = "sha256-3tbVYqC4buh4sd33D/qyeXzjzco7AtYHdUj51W3/lrY=";
            };
            aarch64-linux = {
              url = "https://github.com/zigtools/zls/releases/download/${finalAttrs.version}/zls-aarch64-linux.tar.xz";
              hash = "sha256-QwzSk9IB63CuJRnbyWyFS/h5G433/JOS6NLcloCivtc=";
            };
          };
          asset =
            assets.${prev.stdenvNoCC.hostPlatform.system}
            or (throw "zls-bin: unsupported system ${prev.stdenvNoCC.hostPlatform.system}");
        in
          prev.fetchurl asset;

        sourceRoot = ".";

        installPhase = ''
          runHook preInstall

          install -Dm755 zls "$out/bin/zls"
          install -Dm644 README.md "$out/share/doc/zls/README.md"
          install -Dm644 LICENSE "$out/share/licenses/zls/LICENSE"

          runHook postInstall
        '';

        meta = {
          description = "Zig LSP implementation and Zig language server";
          mainProgram = "zls";
          homepage = "https://github.com/zigtools/zls";
          license = prev.lib.licenses.mit;
          platforms = [
            "x86_64-linux"
            "aarch64-linux"
          ];
        };
      });
    };

    # Shared CUDA package and target configuration.
    cudaCfg = {
      gccHostAttr = "gcc14";
      cudaPackagesAttr = "cudaPackages_12_9";
      cudaVersion = "12.9.1";
      cudnnVersion = "9.8.0";
      ncclVersion = "2.27.7";
      nvshmemVersion = "3.2.5";
    };

    # Local overrides for external integration builds.
    #
    #  Per-user overrides live in ./local-build-cfg.nix (gitignored).
    #  Copy ./local-build-cfg.example.nix to start. Impure evaluation from the
    #  repository loads this file. Pure evaluation uses the portable defaults.
    #
    # See the documentation on what is supported.
    localBuildCfg = let
      # impure eval supplies PWD for local cfg overrides
      pwd = builtins.getEnv "PWD";
      localFile =
        if pwd != ""
        then "${pwd}/local-build-cfg.nix"
        else null;
      localOverrides =
        if localFile != null && builtins.pathExists localFile
        then import localFile
        else {};
    in
      localOverrides;
  in
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      flake = {
        overlays.default = zigOverlay;
        lib = (import ./nix/lib/consumer.nix {
          zigradSrc = ./.;
        }) // {inherit zigVersion;};
        templates.mnist = {
          path = ./examples/mnist;
          description = "Zigrad MNIST application with interchangeable CPU backends";
        };
      };

      imports = [
        ./nix/flake/configuration-options.nix
        ./nix/flake/integrations.nix
        ./nix/flake/examples.nix
        ./nix/flake/tools.nix
        ./nix/flake/devshells.nix
        ./nix/flake/checks.nix
      ];

      perSystem = {system, ...}: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [
            zigOverlay
            inputs.pog.overlays.default
          ];
          config = {
            allowUnfreePredicate = package:
              builtins.elem (inputs.nixpkgs.lib.getName package) [
                "nsight_compute"
                "nsight_systems"
              ];
            # Enabling cudaSupport globally changes unrelated nixpkgs packages.
            # cudaSupport = true;
          };
        };
        treefmt = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;

        # cudaCfg owns NVIDIA artifact versions and the nixpkgs toolkit used by
        #  repository tools. Assert that both CUDA selections have the same
        #  major and minor version.
        pkgsCudartVersion = pkgs.${cudaCfg.cudaPackagesAttr}.cuda_cudart.version;
        cudaVerMM = pkgs.lib.versions.majorMinor cudaCfg.cudaVersion;
        pkgsCudartVerMM = pkgs.lib.versions.majorMinor pkgsCudartVersion;
      in
        assert pkgs.lib.assertMsg
        (cudaVerMM == pkgsCudartVerMM)
        ''
          cudaCfg drift: cudaVersion=${cudaCfg.cudaVersion} (major.minor ${cudaVerMM})
            disagrees with nixpkgs ${cudaCfg.cudaPackagesAttr}.cuda_cudart.version=${pkgsCudartVersion} (major.minor ${pkgsCudartVerMM}).
            Update flake.nix cudaCfg or the nixpkgs lock.
        ''; {
          _module.args = {
            inherit pkgs cudaCfg;
            repoRoot = ./.;
          };
          zigrad.build = localBuildCfg;
          formatter = treefmt.config.build.wrapper;
          checks.formatting = treefmt.config.build.check inputs.self;
        };
    };
}
