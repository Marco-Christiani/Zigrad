{
  lib,
  root,
}: let
  fs = lib.fileset;
in
  fs.toSource {
    inherit root;
    fileset = fs.unions [
      (root + /build.zig)
      (root + /build.zig.zon)
      (root + /assets)
      (root + /src)
      (root + /tools)
      (root + /scripts/generate_tvm_pipeline_contract.py)
    ];
  }
