{
  fetchgit,
  fetchzip,
  lib,
  linkFarm,
  stdenv,
  requestedSets ? [],
}: let
  safetensorsSource = {
    repository = "https://github.com/Marco-Christiani/safetensors-zg";
    revision = "2537a1ae8c8a5e3f347c2605eae905f26db24f5e";
    revision_url = "https://github.com/Marco-Christiani/safetensors-zg/commit/2537a1ae8c8a5e3f347c2605eae905f26db24f5e";
    file_url_template = "https://github.com/Marco-Christiani/safetensors-zg/blob/2537a1ae8c8a5e3f347c2605eae905f26db24f5e/src/{path}";
  };
  safetensors = {
    name = "safetensors_zg-0.0.1-dRXUiNjwAACWR1_WcxxaEM_wzkIqASiNvP_dgwuIPnGO";
    path = fetchgit {
      url = safetensorsSource.repository;
      rev = safetensorsSource.revision;
      hash = "sha256-qCiN+72qtxCYgxdiJH/gdZOolEiwLGZcymcQb3ENys8=";
    };
  };

  protobuf = {
    name = "protobuf-5.0.0-0e82ak1NKACDoOz0E7W7RJ2cPQLQP9pIEHS6RkDhNGAz";
    path = fetchgit {
      url = "https://github.com/Marco-Christiani/zig-protobuf";
      rev = "a4c1e492e5d5afe947724ec0ee95f09f1b828881";
      hash = "sha256-nooYVDBBXz6hG+i1qEtQu3ajJirGtpI099tRRp6uguU=";
    };
  };

  protocAssets = {
    x86_64-linux = {
      name = "N-V-__8AAGKbngAmNuaBMSXq_WgmQi6N8WVWVKp0moFSTvoJ";
      url = "https://github.com/protocolbuffers/protobuf/releases/download/v32.1/protoc-32.1-linux-x86_64.zip";
      hash = "sha256-+nzX+bcCBfIewSNHxE/bez8STbL+LyIdXOfxrNAyknE=";
    };
    aarch64-linux = {
      name = "N-V-__8AAJKMngA8y82sENkRg-JF100BtRa7GQxoBIfU3c3_";
      url = "https://github.com/protocolbuffers/protobuf/releases/download/v32.1/protoc-32.1-linux-aarch_64.zip";
      hash = "sha256-HPNufPYkFYW9c75ZuVwQC9I7hC5pMty2P0CBrIqxud0=";
    };
  };

  protocAsset =
    protocAssets.${stdenv.hostPlatform.system}
    or (throw "zig-dependencies: unsupported protoc host ${stdenv.hostPlatform.system}");
  protoc = {
    inherit (protocAsset) name;
    path = fetchzip {
      inherit (protocAsset) url hash;
      stripRoot = false;
    };
  };

  dependencySets = {
    protobuf = [
      protobuf
      protoc
    ];
  };
  selectedPackages = lib.concatMap (
    name:
      dependencySets.${name}
      or (throw "zig-dependencies: unknown dependency set '${name}'")
  ) (lib.unique requestedSets);
in
  (linkFarm "zig-packages" (
    [safetensors]
    ++ selectedPackages
  )).overrideAttrs (_: {
    passthru.autodocSources.safetensors_zg = safetensorsSource;
  })
